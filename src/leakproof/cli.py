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


if __name__ == "__main__":
    app()
