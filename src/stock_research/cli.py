"""Command-line entry point (``stock-research``)."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

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


def open_market_data(cfg: AppConfig, *, wait_on_quota: bool = False) -> MarketData:
    """Real data access: FinMind, Yahoo and TPEx behind the local cache."""
    cache = Cache(cfg.data_sources.cache_dir / "cache.sqlite")
    return MarketData.open(cfg, Secrets(), cache, wait_on_quota=wait_on_quota)


@config_app.command("show")
def config_show(config: Path = ConfigOption) -> None:
    """Validate the config file and print it, plus which secrets are set (never their values)."""
    cfg = _load(config)
    typer.echo(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True))
    secrets = Secrets()
    typer.echo("secrets:")
    for name in Secrets.NAMES:
        typer.echo(f"  {name}: {'set' if secrets.is_set(name) else 'unset'}")


def _open(cfg: AppConfig, *, batch: bool = False) -> MarketData:
    try:
        return open_market_data(cfg, wait_on_quota=True) if batch else open_market_data(cfg)
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


NoMarketOption = typer.Option(
    False, "--no-market", help="Skip official closes, events and history (local previews)."
)


@app.command("export")
def export(
    out: Path = OutOption,
    codes: list[str] | None = CodesArgumentApp,
    no_market: bool = NoMarketOption,
    config: Path = ConfigOption,
) -> None:
    """Analyse the app watchlist at today's date and write JSON for the web app.

    Also extends the published site's history of official closes, ex-rights events and
    scores for the holdings page (reads ``portfolio.site_url``; ``--no-market`` skips it)."""
    from stock_research.export.market import http_fetch
    from stock_research.export.site import export_site

    cfg = _load(config)
    market_data = _open(cfg, batch=True)
    summary = export_site(
        market_data,
        cfg,
        out,
        codes=[c.strip() for c in codes] if codes else None,
        progress=_echo,
        previous_site=None if no_market else http_fetch(cfg.data_sources.request_timeout_seconds),
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


# --- calibration of the 0-100 score (spec v2 §4.2) -------------------------------------------

calibration_app = typer.Typer(
    help="Full-market calibration of the 0-100 score (runs on GitHub Actions).",
    no_args_is_help=True,
)
app.add_typer(calibration_app, name="calibration")


def _calibration_market(cfg: AppConfig) -> MarketData:
    from stock_research.calibration.pipeline import calibration_market

    try:
        return calibration_market(cfg)
    except MissingSecretError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


@calibration_app.command("fetch")
def calibration_fetch(
    minutes: float = typer.Option(320, "--minutes", help="Stop fetching after this long."),
    config: Path = ConfigOption,
) -> None:
    """Fill the calibration cache (resumable); write the liquidity screen when complete."""
    import json
    from datetime import UTC, datetime, timedelta

    from stock_research.calibration import pipeline

    cfg = _load(config)
    market_data = _calibration_market(cfg)
    deadline = datetime.now(UTC) + timedelta(minutes=minutes)
    status = pipeline.fetch(market_data, cfg, deadline, progress=_echo)
    work = cfg.calibration.work_dir
    work.mkdir(parents=True, exist_ok=True)
    if status["complete"]:
        found = pipeline.universe(market_data, cfg)
        pipeline.liquidity(market_data, cfg, found).to_parquet(
            work / pipeline.LIQUIDITY_FILE, index=False
        )
    (work / pipeline.FETCH_STATUS_FILE).write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    typer.echo(json.dumps({k: v for k, v in status.items() if k != "failed"}))
    for code, reason in status.get("failed", {}).items():
        typer.echo(f"  failed {code}: {reason}")


@calibration_app.command("liquidity")
def calibration_liquidity(config: Path = ConfigOption) -> None:
    """Rebuild the liquidity screen from the cache (prices only)."""
    from stock_research.calibration import pipeline

    cfg = _load(config)
    market_data = _calibration_market(cfg)
    found = pipeline.universe(market_data, cfg)
    screen = pipeline.liquidity(market_data, cfg, found)
    work = cfg.calibration.work_dir
    work.mkdir(parents=True, exist_ok=True)
    screen.to_parquet(work / pipeline.LIQUIDITY_FILE, index=False)
    typer.echo(f"{len(found.codes)} codes, {int(screen['kept'].sum())} kept stock-months")


@calibration_app.command("score")
def calibration_score(
    shard: int = typer.Option(..., "--shard", min=0),
    shards: int = typer.Option(..., "--shards", min=1),
    config: Path = ConfigOption,
) -> None:
    """Score one shard of the kept (stock, base date) pairs from the cache."""
    import pandas as pd

    from stock_research.calibration import pipeline

    cfg = _load(config)
    if shard >= shards:
        typer.echo("--shard must be below --shards", err=True)
        raise typer.Exit(code=2)
    market_data = _calibration_market(cfg)
    work = cfg.calibration.work_dir
    screen = pd.read_parquet(work / pipeline.LIQUIDITY_FILE)
    found = pipeline.universe(market_data, cfg)
    rows = pipeline.score_shard(
        market_data, cfg, screen.loc[screen["kept"]], found.delisted, shard, shards, progress=_echo
    )
    path = work / f"rows_{shard:02d}.parquet"
    rows.to_parquet(path, index=False)
    typer.echo(rows["status"].value_counts().to_string())
    typer.echo(f"written: {path}")


RowsDirOption = typer.Option(None, "--rows-dir", help="Folder with rows_*.parquet.")


@calibration_app.command("summarize")
def calibration_summarize(
    rows_dir: Path | None = RowsDirOption,
    config: Path = ConfigOption,
) -> None:
    """Score groups, caps and the gate from all shards; writes calibration.output."""
    import pandas as pd

    from stock_research.calibration import pipeline, summary

    cfg = _load(config)
    market_data = _calibration_market(cfg)
    work = cfg.calibration.work_dir
    files = sorted((rows_dir or work).glob("rows_*.parquet"))
    if not files:
        typer.echo("no rows_*.parquet found", err=True)
        raise typer.Exit(code=1)
    rows = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows["base_date"] = pd.to_datetime(rows["base_date"]).dt.date
    screen = pd.read_parquet(work / pipeline.LIQUIDITY_FILE)
    found = pipeline.universe(market_data, cfg)
    bench = summary.benchmarks(market_data, cfg, pipeline.base_dates(cfg.calibration))
    result = summary.summarize(
        rows,
        screen,
        bench,
        cfg,
        market_data.now,
        extra={"universe_codes": len(found.codes), "delisted_codes": len(found.delisted)},
    )
    summary.save(result, cfg.calibration.output)
    for key, per in result["preferences"].items():
        gate = per["gate"]
        typer.echo(f"[{key}] gate passed: {per['passed']} ({gate.get('reason') or 'ok'})")
        if per["overfit"]["warning"]:
            typer.echo(f"[{key}] overfitting warning: {per['overfit']['message']}")
        tier = gate["tier"]
        for entry in per["table"]:
            oos = entry["out_of_sample"]["tiers"][tier]
            mean = "—" if oos["mean"] is None else f"{oos['mean']:+.2%}"
            typer.echo(f"  group {entry['group']:2d}: out-of-sample {mean} (n={oos['n']})")
    typer.echo(f"written: {cfg.calibration.output}")


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
RecordArgument = typer.Argument(..., help="An expert record (validation/expert_forms/*.yaml).")
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
    market_data = _open(cfg, batch=True)
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
    from stock_research.validation.statistics import overfit_check

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
    market_data = _open(cfg, batch=True)
    started = market_data.now
    # The code that runs is the code at the start; a long run must not log a later commit.
    commit, dirty = git_commit()
    if phase == "holdout":
        acquire_holdout_lock(directory, registration, started)
    run = wf.run_walk_forward(market_data, cfg, phase, attempt=attempt, progress=_echo)  # type: ignore[arg-type]
    wf.append_rows(directory, run.rows)
    attempts = None
    if phase == "dev":
        wf.append_tuning_log(directory, wf.tuning_row(run, applied, started, commit, dirty))
        attempts = wf.tuning_attempts(directory)
    overfit = None
    if phase == "holdout":
        dev_auc = wf.attempt_auc(directory, registration.selected_attempt)
        holdout_auc = run.evaluation.tool_auc.estimate
        check = overfit_check(
            None if dev_auc is None else dev_auc - 0.5,
            None if holdout_auc is None else holdout_auc - 0.5,
            cfg.robustness.overfit_ratio,
        )
        overfit = (check, registration.selected_attempt)
    text = render_walk_forward(run, settings.confidence, attempts, overfit)
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


@validation_app.command("metric-walk-forward")
def metric_walk_forward_command(config: Path = ConfigOption) -> None:
    """Per-metric walk-forward result of the latest development attempt (score part B)."""
    from datetime import UTC, datetime

    import pandas as pd

    from stock_research.validation import walk_forward as wf
    from stock_research.validation.metric_walk_forward import evaluate_metrics, save

    cfg = _load(config)
    rows = pd.read_parquet(cfg.validation.directory / wf.PARQUET)
    summary = evaluate_metrics(rows, cfg.validation.walk_forward)
    path = cfg.research_score.consistency.walk_forward_results
    save(summary, path, datetime.now(UTC))
    for metric, result in summary["metrics"].items():
        mark = "正向" if result["positive"] else "否"
        auc_value, baseline = result["tool_auc"], result["baseline_b_quantile"]
        typer.echo(f"{metric}: AUC {auc_value:.3f}，基準 B 95% {baseline:.3f}，{mark}")
    typer.echo(f"written: {path}")


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


@validation_app.command("compare-expert")
def compare_expert_command(
    record: Path = RecordArgument,
    config: Path = ConfigOption,
) -> None:
    """The tool's growth, risk and quadrant next to each judgement in an expert record
    (成長 × 風險 §3); writes a new dated comparison file next to the record."""
    from stock_research.analysis.growth_risk import assess
    from stock_research.analysis.inputs import load_stock_data
    from stock_research.analysis.stability_inputs import stability_report
    from stock_research.validation.expert_record import (
        RecordError,
        compare,
        load_record,
        output_path,
        render,
    )

    cfg = _load(config)
    try:
        body, judgements = load_record(record)
    except RecordError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    market_data = _open(cfg)
    run_at = market_data.now.astimezone(ZoneInfo(cfg.general.timezone))
    run_date = run_at.date()
    target = output_path(record, run_at)
    if target.exists():
        typer.echo(f"{target} already exists (comparisons are append-only)", err=True)
        raise typer.Exit(code=1)
    comparisons = []
    for judgement in judgements:
        _echo(f"{judgement.code} at {judgement.as_of}")
        health = market_data.health_report(judgement.code, judgement.as_of)
        if not health.passed:
            failed = ", ".join(c.name for c in health.checks if not c.passed)
            typer.echo(f"{judgement.code}: data checks failed ({failed})", err=True)
            raise typer.Exit(code=1)
        data = load_stock_data(market_data, judgement.code, judgement.as_of, with_peers=False)
        report = stability_report(data, cfg, with_breakpoints=False)
        comparisons.append(compare(judgement, assess(market_data, data, report, cfg)))
    text = render(body, comparisons, run_date)
    target.write_text(text, encoding="utf-8")
    typer.echo(text)
    typer.echo(f"written: {target}")


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
    with (folder / "disagreements.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["code", "question", "majority", "yes", "no", "tool_verdict", "tool_answer"]
        )
        for d in scores.disagreements:
            writer.writerow(
                [d.code, d.question, d.majority, d.yes, d.no, d.tool_verdict, d.tool_answer]
            )
    typer.echo(text)
