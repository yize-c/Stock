"""Command-line entry point (``stock-research``)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
import yaml

from stock_research.analysis.breakpoints import BreakResult
from stock_research.analysis.inputs import load_stock_data
from stock_research.analysis.pipeline import AnalysisBlocked, run_analysis
from stock_research.analysis.stability_inputs import stability_report
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
from stock_research.data.windows import MAIN_WINDOWS, WindowName
from stock_research.report.analysis import render_analysis, render_comparison
from stock_research.report.health import render_health_report

app = typer.Typer(help="Research assistant for Taiwan-listed stocks.", no_args_is_help=True)
config_app = typer.Typer(help="Inspect the configuration.", no_args_is_help=True)
app.add_typer(config_app, name="config")

ConfigOption = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to config.yaml.")
AsOfOption = typer.Option(
    None, "--as-of", help="Only use data published on or before this date (YYYY-MM-DD)."
)

TickersArgument = typer.Argument(..., help="One or more stock codes, e.g. 2330 2454.")
WindowOption = typer.Option(
    None, "--window", help="Report a single window only (no stability verdicts)."
)
OutputDirOption = typer.Option(None, "--output-dir", help="Default: report.output_dir.")


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


def _open(cfg: AppConfig) -> MarketData:
    try:
        return open_market_data(cfg)
    except MissingSecretError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


def _as_of(market_data: MarketData, as_of: str | None) -> date:
    today = market_data.today()
    try:
        day = date.fromisoformat(as_of) if as_of else today
    except ValueError as exc:
        typer.echo(f"--as-of must be YYYY-MM-DD, got {as_of!r}", err=True)
        raise typer.Exit(code=2) from exc
    if day > today:
        typer.echo(f"--as-of {day} is in the future (today is {today})", err=True)
        raise typer.Exit(code=2)
    return day


@app.command("validate")
def validate(
    ticker: str = typer.Argument(..., help="Stock code, e.g. 2330."),
    as_of: str | None = AsOfOption,
    config: Path = ConfigOption,
) -> None:
    """Run every data check for one stock and print the data-health report."""
    cfg = _load(config)
    market_data = _open(cfg)
    day = _as_of(market_data, as_of)
    report = market_data.health_report(ticker.strip(), day)
    typer.echo(render_health_report(report))
    if not report.passed:
        raise typer.Exit(code=1)


@app.command("analyze")
def analyze(
    tickers: list[str] = TickersArgument,
    as_of: str | None = AsOfOption,
    window: WindowName | None = WindowOption,
    compare: bool = typer.Option(False, "--compare", help="Also write a comparison table."),
    only_unstable: bool = typer.Option(
        False, "--only-unstable", help="Show only REGIME_SHIFT / UNSTABLE items."
    ),
    peers: bool = typer.Option(
        True, "--peers/--no-peers", help="Load industry peers for the industry medians."
    ),
    llm: bool = typer.Option(
        False, "--llm/--no-llm", help="Run the LLM tasks on documents in data/documents/."
    ),
    output_dir: Path | None = OutputDirOption,
    config: Path = ConfigOption,
) -> None:
    """Analyse stocks and write one Traditional Chinese report per stock."""
    if window is not None and window not in MAIN_WINDOWS:
        typer.echo(f"--window must be one of {[w.value for w in MAIN_WINDOWS]}", err=True)
        raise typer.Exit(code=2)
    cfg = _load(config)
    market_data = _open(cfg)
    day = _as_of(market_data, as_of)
    folder = output_dir or cfg.report.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    done, blocked = [], False
    for code in (t.strip() for t in tickers):
        try:
            result = run_analysis(
                market_data,
                cfg,
                code,
                day,
                with_peers=peers,
                with_llm=llm,
                progress=lambda message: typer.echo(message, err=True),
            )
        except AnalysisBlocked as exc:
            typer.echo(render_health_report(exc.report))
            typer.echo(f"{code}: data checks failed; no analysis was run.", err=True)
            blocked = True
            continue
        path = folder / f"{code}_{day.isoformat()}.md"
        path.write_text(
            render_analysis(result, cfg, window=window, only_unstable=only_unstable),
            encoding="utf-8",
        )
        typer.echo(f"{code}: {path}")
        done.append(result)
    if compare and len(done) > 1:
        path = folder / f"compare_{'_'.join(a.info.ticker.code for a in done)}_{day.isoformat()}.md"
        path.write_text(render_comparison(done), encoding="utf-8")
        typer.echo(f"compare: {path}")
    if blocked:
        raise typer.Exit(code=1)


@app.command("breakpoints")
def breakpoints(
    ticker: str = typer.Argument(..., help="Stock code, e.g. 2330."),
    as_of: str | None = AsOfOption,
    config: Path = ConfigOption,
) -> None:
    """Quandt-Andrews break detection only (information; never changes a verdict)."""
    cfg = _load(config)
    market_data = _open(cfg)
    day = _as_of(market_data, as_of)
    data = load_stock_data(market_data, ticker.strip(), day, with_peers=False)
    for result in stability_report(data, cfg).breakpoints:
        if isinstance(result, BreakResult):
            typer.echo(
                f"{result.model}: {result.break_date} (sup-F {result.sup_f:.2f}, "
                f"5% critical {result.critical_value}, "
                f"{'significant' if result.significant else 'not significant'}, "
                f"{result.days_from:+d} days from the regime split)"
            )
        else:
            typer.echo(f"{result.metric}: insufficient ({result.reason})")
