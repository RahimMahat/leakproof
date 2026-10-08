"""Thin Kafka (Redpanda) helpers: a fresh single-partition topic, produce, and consume to the end."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable, Iterator

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from leakproof.config import KAFKA_BOOTSTRAP
from leakproof.stream.events import Event

END = {"kind": "end"}  # marker so a consumer knows the replay is complete


def fresh_topic(prefix: str) -> str:
    """Create a uniquely named topic with one partition, so events keep their order."""
    name = f"{prefix}-{uuid.uuid4().hex[:8]}"
    admin = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP})
    for future in admin.create_topics([NewTopic(name, num_partitions=1, replication_factor=1)]).values():
        future.result(timeout=30)
    return name


def delete_topic(name: str) -> None:
    admin = AdminClient({"bootstrap.servers": KAFKA_BOOTSTRAP})
    for future in admin.delete_topics([name]).values():
        future.result(timeout=30)


def produce(topic: str, events: Iterable[Event], speed: float | None = None) -> int:
    """Send events in order, then the END marker. `speed` replays event time (e.g. 3600 = one hour
    of data per second); None sends as fast as possible."""
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP, "linger.ms": 20, "acks": "all"})
    n, first_t, started = 0, None, time.monotonic()
    for ev in events:
        if speed:
            first_t = ev["t"] if first_t is None else first_t
            wait = (ev["t"] - first_t) / speed - (time.monotonic() - started)
            if wait > 0:
                time.sleep(wait)
        while True:
            try:
                producer.produce(topic, json.dumps(ev).encode(), key=str(ev["card_id"]).encode())
                break
            except BufferError:  # local queue full: let it drain
                producer.poll(0.5)
        n += 1
        if n % 10_000 == 0:
            producer.poll(0)
    producer.produce(topic, json.dumps(END).encode())
    producer.flush(60)
    return n


def consume(topic: str, timeout_s: float = 60.0) -> Iterator[Event]:
    """Yield events from the start of the topic until the END marker."""
    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": f"leakproof-{uuid.uuid4().hex[:8]}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    try:
        idle_since = time.monotonic()
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                if time.monotonic() - idle_since > timeout_s:
                    raise TimeoutError(f"no message on {topic} for {timeout_s:.0f}s and no END marker")
                continue
            err = msg.error()
            if err is not None:
                if err.code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(err)
            idle_since = time.monotonic()
            ev = json.loads(msg.value() or b"{}")
            if ev.get("kind") == "end":
                return
            yield ev
    finally:
        consumer.close()
