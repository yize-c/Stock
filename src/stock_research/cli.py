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
OutOption = typer.Option(Path("site"), "--out", help="Output folder (data/ is written inside).")
CodesArgumentApp = typer.Argument(None, help="Stock codes (default: app.watchlist).")
ForwardLogOption = typer.Option(
    True,
    "--forward-log/--no-forward-log",
    help="Record today's verdicts in validation/forward_log/ (runs at today's date only).",
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
    forward_log: bool = ForwardLogOption,
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
        tracking = cfg.validation.forward_tracking.enabled and forward_log
        if tracking and window is None and day == market_data.today():
            from stock_research.validation.forward_log import write_entry

            entry = write_entry(
                cfg.validation.directory, code, result.stability, cfg, config, market_data.now
            )
            typer.echo(f"{code}: forward log {entry}")
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


@app.command("export")
def export(
    out: Path = OutOption,
    codes: list[str] | None = CodesArgumentApp,
    config: Path = ConfigOption,
) -> None:
    """Analyse the app watchlist at today's date and write JSON for the web app."""
    from stock_research.export.site import export_site

    cfg = _load(config)
    market_data = _open(cfg)
    summary = export_site(
        market_data, cfg, out, codes=[c.strip() for c in codes] if codes else None, progress=_echo
    )
    typer.echo(
        f"as_of {summary.as_of}: {len(summary.ok)} ok, {len(summary.blocked)} blocked by data "
        f"checks, {len(summary.errors)} errors"
    )
    for code, reason in {**summary.blocked, **summary.errors}.items():
        typer.echo(f"  {code}: {reason}")
    if not summary.ok:
        typer.echo("no stock could be analysed; nothing should be published", err=True)
        raise typer.Exit(code=1)


# --- validation (spec §14) -------------------------------------------------------------------

validation_app = typer.Typer(help="Validate the tool itself (spec §14).", no_args_is_help=True)
run_app = typer.Typer(help="Run a validation.", no_args_is_help=True)
app.add_typer(validation_app, name="validation")
validation_app.add_typer(run_app, name="run")

PhaseOption = typer.Option(..., "--phase", help="dev (tuning allowed) or holdout (runs once).")
SetOption = typer.Option(
    None, "--set", help="Dev only: override a tunable config value, e.g. stability.alpha=0.1."
)
CodesArgument = typer.Argument(None, help="Stock codes (default: the walk-forward universe).")
DateOption = typer.Option(None, "--date", help="As-of date(s) to audit (repeatable).")
FolderArgument = typer.Argument(..., help="An expert_forms/<date> folder with filled_*.csv.")


def _write_result(folder: Path, name: str, text: str) -> Path:
    """Results are never overwritten."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _echo(message: str) -> None:
    typer.echo(message, err=True)


@run_app.command("known-cases")
def run_known_cases_command(config: Path = ConfigOption) -> None:
    """Known-case test: expected aspect changes vs the tool (spec §14.3)."""
    from stock_research.validation.known_cases import load_known_cases, run_known_cases
    from stock_research.validation.report import render_known_cases

    cfg = _load(config)
    cases = load_known_cases(cfg.validation.known_cases.file)
    market_data = _open(cfg)
    result = run_known_cases(market_data, cfg, cases, progress=_echo)
    text = render_known_cases(result, cases.labeled_by)
    stamp = market_data.now.strftime("%Y%m%dT%H%M%S")
    folder = cfg.validation.directory / "results"
    path = _write_result(folder, f"known_cases_{result.as_of}_{stamp}.md", text)
    result.table.to_csv(folder / f"known_cases_{result.as_of}_{stamp}.csv", index=False)
    typer.echo(text)
    typer.echo(f"written: {path}")


@run_app.command("walk-forward")
def run_walk_forward_command(
    phase: str = PhaseOption,
    overrides: list[str] | None = SetOption,
    config: Path = ConfigOption,
) -> None:
    """Monthly walk-forward of the stability verdicts (spec §14.4)."""
    from stock_research.validation import walk_forward as wf
    from stock_research.validation.preregistration import (
        HoldoutBlocked,
        acquire_holdout_lock,
        check_holdout_allowed,
        git_commit,
    )
    from stock_research.validation.report import render_walk_forward

    if phase not in ("dev", "holdout"):
        typer.echo("--phase must be dev or holdout", err=True)
        raise typer.Exit(code=2)
    cfg = _load(config)
    settings = cfg.validation.walk_forward
    directory = cfg.validation.directory
    attempt = None
    applied: dict = {}
    if phase == "holdout":
        if overrides:
            typer.echo("--set is not allowed in the holdout phase", err=True)
            raise typer.Exit(code=2)
        try:
            registration = check_holdout_allowed(directory, config)
        except HoldoutBlocked as exc:
            typer.echo(f"holdout blocked: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    else:
        try:
            applied = wf.parse_overrides(overrides or [], settings.tunable)
            cfg = wf.apply_overrides(cfg, applied)
            attempt = wf.next_attempt(directory, settings.max_tuning_attempts)
        except (ValueError, ConfigError, wf.TuningLimitReached) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
    market_data = _open(cfg)
    started = market_data.now
    if phase == "holdout":
        acquire_holdout_lock(directory, registration, started)
    run = wf.run_walk_forward(market_data, cfg, phase, attempt=attempt, progress=_echo)  # type: ignore[arg-type]
    wf.append_rows(directory, run.rows)
    attempts = None
    if phase == "dev":
        commit, dirty = git_commit()
        wf.append_tuning_log(directory, wf.tuning_row(run, applied, started, commit, dirty))
        attempts = wf.tuning_attempts(directory)
    text = render_walk_forward(run, settings.confidence, attempts)
    name = f"walk_forward_dev_attempt_{attempt:02d}.md" if attempt else "walk_forward_holdout.md"
    path = _write_result(directory / "results", name, text)
    typer.echo(text)
    typer.echo(f"written: {path}")


@run_app.command("point-in-time")
def run_point_in_time_command(
    codes: list[str] | None = CodesArgument,
    dates: list[str] | None = DateOption,
    config: Path = ConfigOption,
) -> None:
    """Check that data loaded for a date contains nothing dated after it."""
    from stock_research.validation.point_in_time import check_point_in_time
    from stock_research.validation.walk_forward import base_dates

    cfg = _load(config)
    market_data = _open(cfg)
    settings = cfg.validation.walk_forward
    if dates:
        days = [_as_of(market_data, d) for d in dates]
    else:
        dev = base_dates(settings.dev, settings.horizon_months, market_data.today())
        days = [dev[0], dev[-1]] if dev else [market_data.today()]
    failed = False
    for code in codes or settings.universe:
        for day in days:
            problems = check_point_in_time(market_data, code, day)
            failed |= bool(problems)
            typer.echo(f"{code} {day}: " + ("OK" if not problems else "; ".join(problems)))
    if failed:
        raise typer.Exit(code=1)


@validation_app.command("forward-compare")
def forward_compare_command(
    months: int = typer.Option(6, "--months", help="Compare entries at least this old."),
    config: Path = ConfigOption,
) -> None:
    """Compare forward-log verdicts with what happened afterwards (spec §14.5)."""
    from stock_research.validation.forward_log import compare_entries
    from stock_research.validation.report import render_forward

    cfg = _load(config)
    market_data = _open(cfg)
    comparison = compare_entries(market_data, cfg, cfg.validation.directory, months, progress=_echo)
    text = render_forward(comparison)
    stamp = market_data.now.strftime("%Y%m%dT%H%M%S")
    path = _write_result(
        cfg.validation.directory / "results", f"forward_{months}m_{stamp}.md", text
    )
    typer.echo(text)
    typer.echo(f"written: {path}")


@validation_app.command("export-expert-forms")
def export_expert_forms_command(
    codes: list[str] = TickersArgument,
    config: Path = ConfigOption,
) -> None:
    """Blank expert forms plus the tool's verdicts at the same date (spec §14.6)."""
    from stock_research.validation.experts import export_forms

    cfg = _load(config)
    market_data = _open(cfg)
    try:
        folder = export_forms(market_data, cfg, [c.strip() for c in codes], progress=_echo)
    except FileExistsError as exc:
        typer.echo(f"forms for today already exist: {exc.filename}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"written: {folder}")


@validation_app.command("score-experts")
def score_experts_command(
    folder: Path = FolderArgument,
    config: Path = ConfigOption,
) -> None:
    """Kappa between experts and between the tool and the expert majority."""
    from stock_research.validation.experts import FormError, score_experts
    from stock_research.validation.report import render_experts

    cfg = _load(config)
    try:
        scores = score_experts(folder, cfg)
    except FormError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    text = render_experts(scores)
    (folder / "scores.md").write_text(text, encoding="utf-8")
    typer.echo(text)
