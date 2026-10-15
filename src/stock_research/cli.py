"""Command-line entry point (``stock-research``)."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from stock_research.config import DEFAULT_CONFIG_PATH, ConfigError, Secrets, load_config

app = typer.Typer(help="Research assistant for Taiwan-listed stocks.", no_args_is_help=True)
config_app = typer.Typer(help="Inspect the configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")

ConfigOption = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to config.yaml.")


@config_app.command("show")
def config_show(config: Path = ConfigOption) -> None:
    """Validate the config file and print it, plus which secrets are set (never their values)."""
    try:
        cfg = load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True))
    secrets = Secrets()
    typer.echo("secrets:")
    for name in Secrets.NAMES:
        typer.echo(f"  {name}: {'set' if secrets.is_set(name) else 'unset'}")
