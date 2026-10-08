"""``leakproof`` command-line entry point."""

from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True, help="Leakproof: fraud detection that measures its own leakage.")
data_app = typer.Typer(no_args_is_help=True, help="Download and prepare the dataset.")
app.add_typer(data_app, name="data")


@data_app.command("download")
def download(force: bool = False) -> None:
    """Download the IEEE-CIS files from Kaggle into data/raw."""
    from leakproof.data.download import download_raw

    typer.echo(f"raw data at {download_raw(force=force)}")


@data_app.command("build")
def build() -> None:
    """Join the raw files, assign periods, simulate label delay, and print the split summary."""
    from leakproof.config import EXPECTED_ROWS
    from leakproof.data.split import build_dataset, summarize

    path = build_dataset()
    summary = summarize(path)
    typer.echo(f"dataset at {path}\n")
    typer.echo(summary.to_string(index=False))
    if (rows := int(summary["rows"].sum())) != EXPECTED_ROWS:
        typer.echo(f"\nrow count {rows:,} differs from the official {EXPECTED_ROWS:,}", err=True)
        raise typer.Exit(1)


run_app = typer.Typer(no_args_is_help=True, help="Run experiments.")
app.add_typer(run_app, name="run")


@run_app.command("naive")
def run_naive(mlflow: bool = typer.Option(True, help="Also log the runs to the local MLflow store.")) -> None:
    """Train the leakage ladder on the development data and record the offline scores."""
    import pandas as pd

    from leakproof.model import naive_run

    results = naive_run.run_ladder()
    typer.echo(pd.DataFrame(results).to_string(index=False))
    typer.echo(f"\nsaved to {naive_run.save(results)}")
    if mlflow:
        naive_run.log_to_mlflow(results)


@run_app.command("honest")
def run_honest(
    mlflow: bool = typer.Option(True, help="Also log the runs to the local MLflow store."),
) -> None:
    """Train the point-in-time pipeline (and two label-timing shortcuts) and record the scores."""
    import pandas as pd

    from leakproof.model import honest_run, naive_run

    results = honest_run.run_ladder()
    typer.echo(pd.DataFrame(results).to_string(index=False))
    typer.echo(f"\nsaved to {naive_run.save(results, honest_run.RESULTS_PATH)}")
    if mlflow:
        naive_run.log_to_mlflow(results, experiment="point-in-time")


@run_app.command("headline")
def run_headline() -> None:
    """Train both final models and score the holdout period on live features: offline vs live."""
    import pandas as pd

    from leakproof.model import final, naive_run

    results = final.run_headline(honest_offline=final.honest_offline_estimate())
    typer.echo(pd.DataFrame(results).to_string(index=False))
    typer.echo(f"\nsaved to {naive_run.save(results, final.RESULTS_PATH)}")


@run_app.command("decisions")
def run_decisions(
    review_cost: float = typer.Option(None, help="Cost of one manual review (default: config)."),
) -> None:
    """Cost-based alert threshold, feature drift and daily performance on the holdout period."""
    import pandas as pd

    from leakproof import monitor
    from leakproof.config import REVIEW_COST
    from leakproof.model import final, naive_run, threshold

    cost = REVIEW_COST if review_cost is None else review_cost
    f = final.scored_frames()
    holdout = f["holdout"]
    rows = threshold.run_decisions(f["honest_select"], f["naive_select"], holdout, cost)
    typer.echo(f"review cost per alert: {cost}")
    typer.echo(pd.DataFrame(rows).to_string(index=False))
    naive_run.save(rows, threshold.RESULTS_PATH)

    drift = monitor.drift_report(f["reference"], f["current"])
    score_psi = monitor.psi(f["honest_select"]["score"], holdout["score_honest"])
    typer.echo(
        f"\nmodel score PSI: {score_psi:.4f}   features by level: {drift['level'].value_counts().to_dict()}"
    )
    typer.echo(drift.head(10).to_string(index=False))
    naive_run.save(
        [{"feature": "model_score", "psi": round(score_psi, 4)}, *monitor.to_records(drift)],
        monitor.DRIFT_PATH,
    )

    chosen = next(r["threshold"] for r in rows if r["policy"] == "honest_cost_threshold")
    daily = monitor.daily_report(
        holdout.assign(score=holdout["score_honest"]), chosen, as_of=float(holdout["TransactionDT"].max())
    )
    typer.echo(f"\ndaily view at threshold {chosen}, labels as known on the last holdout day:")
    typer.echo(daily.to_string(index=False))
    naive_run.save(monitor.to_records(daily), monitor.DAILY_PATH)


stream_app = typer.Typer(no_args_is_help=True, help="Streaming path (needs `docker compose up -d`).")
app.add_typer(stream_app, name="stream")


@stream_app.command("parity")
def stream_parity(
    limit: int = typer.Option(None, help="Only the first N holdout events (smoke run)."),
) -> None:
    """Stream the holdout period through Redpanda and Redis; compare online with offline features."""
    from leakproof.stream import parity

    report = parity.run_parity(limit=limit)
    typer.echo(
        f"rows compared: {report['rows']:,}   mismatched rows: {report['mismatched_rows']:,} "
        f"({report['mismatch_rate']:.4%})"
    )
    typer.echo(
        f"history events backfilled: {report['history_events']:,}   live events: {report['live_events']:,}"
    )
    typer.echo(f"timings: {report['timings']}   consumer throughput: {report['events_per_s']:,} events/s")
    bad = {f: v for f, v in report["by_feature"].items() if v["mismatches"]}
    if bad:
        typer.echo(f"features with mismatches: {bad}")
    if not limit:
        typer.echo(f"saved to {parity.save(report)}")
    if report["mismatched_rows"]:
        raise typer.Exit(1)


serve_app = typer.Typer(no_args_is_help=True, help="Online scoring service.")
app.add_typer(serve_app, name="serve")


@serve_app.command("run")
def serve_run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the scoring API with the honest model (needs `leakproof run headline` first)."""
    import uvicorn

    uvicorn.run("leakproof.serve.app:app_from_env", factory=True, host=host, port=port)


@serve_app.command("replay")
def serve_replay(limit: int = typer.Option(2000, help="Holdout transactions to send.")) -> None:
    """Replay the start of the holdout period through the HTTP service; report latency."""
    from leakproof.model import naive_run
    from leakproof.serve import replay

    report = replay.run_replay(limit=limit)
    for key, value in report.items():
        typer.echo(f"{key}: {value}")
    typer.echo(f"saved to {naive_run.save([report], replay.RESULTS_PATH)}")


if __name__ == "__main__":
    app()
