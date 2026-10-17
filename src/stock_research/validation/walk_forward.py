"""Monthly walk-forward validation of the stability verdicts (spec §14.4).

For every base date and stock in the configured universe, the tool runs at ``as_of = base``
(data checks first), the verdicts of the §14.1 aspect metrics are recorded, and the outcome
after the horizon is measured (``outcomes``). Dev runs may override tunable parameters; each
dev run is one tuning attempt, appended to ``results/tuning_log.csv`` and capped. The holdout
phase runs once, with the preregistered config and no overrides.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from stock_research.analysis.inputs import StockData, load_stock_data
from stock_research.analysis.stability import MetricVerdict
from stock_research.analysis.stability_inputs import metric_specs, stability_report
from stock_research.config import AppConfig, ConfigError, DateRange, add_months
from stock_research.data.errors import DataValidationError
from stock_research.data.market_data import MarketData
from stock_research.data.windows import WindowName
from stock_research.validation.aspects import ASPECTS, aspect_metrics
from stock_research.validation.outcomes import outcomes
from stock_research.validation.statistics import Evaluation, evaluate

Phase = Literal["dev", "holdout"]
RESULTS = Path("results")
TUNING_LOG = RESULTS / "tuning_log.csv"
PARQUET = RESULTS / "walk_forward.parquet"
TUNING_COLUMNS = [
    "attempt", "started_at", "git_commit", "git_dirty", "overrides", "rows", "unstable",
    "tool_auc", "tool_auc_low", "tool_auc_high", "baseline_b_quantile", "mwu_effect",
    "mwu_p", "passed",
]  # fmt: skip
ROW_COLUMNS = [
    "phase", "attempt", "code", "base_date", "aspect", "metric", "status", "verdict",
    "reference", "future", "outcome_date", "normalized", "exceeded", "reason",
]  # fmt: skip


class TuningLimitReached(Exception):
    pass


def base_dates(span: DateRange, horizon_months: int, today: date) -> list[date]:
    """Month ends inside ``span`` whose horizon has fully passed by ``today``."""
    ends = pd.date_range(span.start, span.end, freq="ME")
    return [d.date() for d in ends if add_months(d.date(), horizon_months) <= today]


# --- tuning overrides ----------------------------------------------------------------------


def parse_overrides(items: list[str], tunable: list[str]) -> dict[str, Any]:
    """``path=value`` pairs (value parsed as YAML); only tunable paths are accepted."""
    out: dict[str, Any] = {}
    for item in items:
        path, sep, value = item.partition("=")
        if not sep or not path:
            raise ValueError(f"override must look like path=value, got {item!r}")
        if not any(path == t or path.startswith(t + ".") for t in tunable):
            raise ValueError(f"{path} is not tunable (validation.walk_forward.tunable: {tunable})")
        out[path] = yaml.safe_load(value)
    return out


def apply_overrides(config: AppConfig, overrides: dict[str, Any]) -> AppConfig:
    raw = config.model_dump(mode="json")
    for path, value in overrides.items():
        node = raw
        *parents, leaf = path.split(".")
        for key in parents:
            if not isinstance(node, dict) or key not in node:
                raise ValueError(f"unknown config path {path}")
            node = node[key]
        if not isinstance(node, dict) or leaf not in node:
            raise ValueError(f"unknown config path {path}")
        node[leaf] = value
    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(f"overrides make the config invalid: {exc}") from exc


# --- tuning log (append-only) ----------------------------------------------------------------


def tuning_attempts(directory: Path) -> int:
    path = directory / TUNING_LOG
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def attempt_auc(directory: Path, attempt: int) -> float | None:
    """The tool AUC a development attempt logged; None if the attempt is not in the log or
    had no AUC."""
    path = directory / TUNING_LOG
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["attempt"]) == attempt:
                return float(row["tool_auc"]) if row["tool_auc"] else None
    return None


def next_attempt(directory: Path, limit: int) -> int:
    done = tuning_attempts(directory)
    if done >= limit:
        raise TuningLimitReached(
            f"{done} tuning attempts already logged (limit {limit}); choose among them"
        )
    return done + 1


def append_tuning_log(directory: Path, row: dict[str, Any]) -> None:
    path = directory / TUNING_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TUNING_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in TUNING_COLUMNS})


def append_rows(directory: Path, rows: pd.DataFrame) -> Path:
    """Add this run's rows to walk_forward.parquet (earlier runs are kept)."""
    path = directory / PARQUET
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        rows = pd.concat([pd.read_parquet(path), rows], ignore_index=True)
    rows.to_parquet(path, index=False)
    return path


# --- the run ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardRun:
    phase: Phase
    attempt: int | None
    base_dates: list[date]
    rows: pd.DataFrame
    evaluation: Evaluation


def _rows_for(
    code: str,
    base: date,
    metrics: list[str],
    status: str,
    reason: str | None = None,
) -> list[dict[str, Any]]:
    return [
        {"code": code, "base_date": base, "metric": m, "status": status, "reason": reason}
        for m in metrics
    ]


def _stock_rows(
    market_data: MarketData,
    config: AppConfig,
    code: str,
    dates: list[date],
    limit: date,
    later: StockData | Exception,
    progress: Callable[[str], None],
) -> list[dict[str, Any]]:
    wf = config.validation.walk_forward
    keys = aspect_metrics(config.validation)
    specs = metric_specs(config)
    rows: list[dict[str, Any]] = []
    listed = market_data.listing_date(market_data.resolve(code))
    for base in dates:
        if listed is not None and listed > base:
            rows += _rows_for(code, base, keys, "NOT_LISTED", f"listed {listed}")
            continue
        try:
            health = market_data.health_report(code, base)
            if not health.passed:
                failed = ", ".join(c.name for c in health.checks if not c.passed)
                rows += _rows_for(code, base, keys, "DATA_CHECK_FAILED", failed)
                continue
            decision = load_stock_data(market_data, code, base, with_peers=False)
            report = stability_report(decision, config, with_breakpoints=False)
        except (DataValidationError, ValueError, KeyError) as exc:
            rows += _rows_for(code, base, keys, "ERROR", f"{type(exc).__name__}: {exc}")
            continue
        verdicts: dict[str, MetricVerdict] = {m.spec.key: m for m in report.metrics}
        references = {k: verdicts[k].values[WindowName.POST_2020] for k in keys}
        if isinstance(later, Exception):
            found = {}
            later_problem = f"outcome data failed: {later}"
        else:
            found = outcomes(
                {k: specs[k] for k in keys},
                references,
                decision,
                later,
                base,
                wf.horizon_months,
                limit,
                config,
                wf.min_forward_returns,
            )
            later_problem = None
        for key in keys:
            outcome = found.get(key)
            rows.append(
                {
                    "code": code,
                    "base_date": base,
                    "metric": key,
                    "status": "OK",
                    "verdict": verdicts[key].verdict.value,
                    "reference": outcome.reference if outcome else None,
                    "future": outcome.future if outcome else None,
                    "outcome_date": outcome.outcome_date if outcome else None,
                    "normalized": outcome.normalized if outcome else None,
                    "exceeded": outcome.exceeded if outcome else None,
                    "reason": outcome.reason if outcome else later_problem,
                }
            )
        progress(f"{code} {base}: done")
    return rows


def run_walk_forward(
    market_data: MarketData,
    config: AppConfig,
    phase: Phase,
    *,
    attempt: int | None,
    progress: Callable[[str], None] = lambda _: None,
) -> WalkForwardRun:
    wf = config.validation.walk_forward
    today = market_data.today()
    span = wf.dev if phase == "dev" else wf.holdout
    dates = base_dates(span, wf.horizon_months, today)
    if not dates:
        raise ValueError(f"no {phase} base date has a completed {wf.horizon_months}-month horizon")
    limit = min(wf.outcome_limit(phase), today)
    aspect_of = {m: a for a in ASPECTS for m in config.validation.aspects[a]}  # type: ignore[index]
    rows: list[dict[str, Any]] = []
    for code in wf.universe:
        progress(f"{code}: loading outcome data as of {limit}")
        later: StockData | Exception
        try:
            later = load_stock_data(market_data, code, limit, with_peers=False)
        except (DataValidationError, ValueError, KeyError) as exc:
            later = exc
        rows += _stock_rows(market_data, config, code, dates, limit, later, progress)
    frame = pd.DataFrame(rows).reindex(columns=ROW_COLUMNS)
    frame["phase"] = phase
    frame["attempt"] = attempt
    frame["aspect"] = frame["metric"].map(aspect_of)
    decided = frame.loc[frame["status"] == "OK"].copy()
    evaluation = evaluate(
        decided,
        draws_baseline=wf.baseline_draws,
        draws_bootstrap=wf.bootstrap_draws,
        confidence=wf.confidence,
        alpha=wf.alpha,
        seed=wf.seed,
    )
    return WalkForwardRun(phase, attempt, dates, frame, evaluation)


def tuning_row(
    run: WalkForwardRun,
    overrides: dict[str, Any],
    started: datetime,
    commit: str | None,
    dirty: bool,
) -> dict[str, Any]:
    e = run.evaluation.as_dict()
    return {
        "attempt": run.attempt,
        "started_at": started.isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "overrides": json.dumps(overrides, ensure_ascii=False, sort_keys=True),
        **{k: e[k] for k in TUNING_COLUMNS if k in e},
    }
