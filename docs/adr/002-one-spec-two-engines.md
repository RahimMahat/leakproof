# 002: One feature spec, computed offline in DuckDB and online in Redis

## Context
Training needs features for 590k rows at once. Serving needs them for one transaction in
milliseconds. Two implementations of the same features tend to drift apart.

## Decision
- `features/spec.py` lists every window aggregate (function, window length) and holds the one
  function that computes derived features. Both paths import it.
- **Offline:** DuckDB window functions, `range between N preceding and 1 preceding` per card.
- **Online:** per card, a Redis sorted set of transactions scored by time and a hash of label
  counters. A read takes transactions up to `t - 1` and aggregates them.
- **One stream:** transactions and label arrivals go to one single-partition Redpanda topic,
  ordered by time with labels first at equal times. The consumer reads a transaction's features
  before recording it.
- A parity check streams the holdout period through the real containers and compares every
  feature with its offline value.

## Alternatives
- **A feature store (Feast):** it would manage the two stores but not prove they agree, which is
  the point here.
- **Separate topics for transactions and labels:** their relative order would then need
  event-time watermarks.

## Consequences
- Parity on the holdout period: 0 mismatched rows of 88,581.
- A card's full history is read on each lookup. That is fine for this data (longest card about
  1,400 transactions) and would need trimming for long-lived cards.
- One partition means one consumer. Scaling out needs partitioning by card.
