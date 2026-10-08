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


if __name__ == "__main__":
    app()
