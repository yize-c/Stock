"""Command-line entry point (``stock-research``)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
import yaml

from stock_research.config import (
    DEFAULT_CONFIG_PATH,
    AppConfig,
    ConfigError,
    MissingSecretError,
    Secrets,
    load_config,
)
from stock_research.data.cache import Cache
from stock_research.data.market_data import MarketData
from stock_research.report.health import render_health_report

app = typer.Typer(help="Research assistant for Taiwan-listed stocks.", no_args_is_help=True)
config_app = typer.Typer(help="Inspect the configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")

ConfigOption = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to config.yaml.")
AsOfOption = typer.Option(
    None, "--as-of", help="Only use data published on or before this date (YYYY-MM-DD)."
)


def _load(config: Path) -> AppConfig:
    try:
        return load_config(config)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


def open_market_data(cfg: AppConfig) -> MarketData:
    """Real data access: FinMind, Yahoo and TPEx behind the local cache."""
    cache = Cache(cfg.data_sources.cache_dir / "cache.sqlite")
    return MarketData.open(cfg, Secrets(), cache)


@config_app.command("show")
def config_show(config: Path = ConfigOption) -> None:
    """Validate the config file and print it, plus which secrets are set (never their values)."""
    cfg = _load(config)
    typer.echo(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True))
    secrets = Secrets()
    typer.echo("secrets:")
    for name in Secrets.NAMES:
        typer.echo(f"  {name}: {'set' if secrets.is_set(name) else 'unset'}")


@app.command("validate")
def validate(
    ticker: str = typer.Argument(..., help="Stock code, e.g. 2330."),
    as_of: str | None = AsOfOption,
    config: Path = ConfigOption,
) -> None:
    """Run every data check for one stock and print the data-health report."""
    cfg = _load(config)
    try:
        market_data = open_market_data(cfg)
    except MissingSecretError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    today = market_data.today()
    try:
        day = date.fromisoformat(as_of) if as_of else today
    except ValueError as exc:
        typer.echo(f"--as-of must be YYYY-MM-DD, got {as_of!r}", err=True)
        raise typer.Exit(code=2) from exc
    if day > today:
        typer.echo(f"--as-of {day} is in the future (today is {today})", err=True)
        raise typer.Exit(code=2)
    report = market_data.health_report(ticker.strip(), day)
    typer.echo(render_health_report(report))
    if not report.passed:
        raise typer.Exit(code=1)
