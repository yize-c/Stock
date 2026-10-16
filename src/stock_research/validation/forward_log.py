"""Prospective tracking (spec §14.5).

Every analysis run at today's date stores its verdicts, the git commit and the config hash in
``forward_log/<as_of>_<code>.json``; existing files are never overwritten (a second run the
same day gets ``_2``, ``_3``...). ``compare_entries`` measures, 6 or 12 months later, what
happened to each logged metric with the same outcome rules as the walk-forward.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stock_research.analysis.inputs import load_stock_data
from stock_research.analysis.stability_inputs import StabilityReport, metric_specs
from stock_research.config import AppConfig, add_months
from stock_research.data.errors import DataValidationError
from stock_research.data.market_data import MarketData
from stock_research.data.windows import WindowName
from stock_research.validation.aspects import aspect_metrics, aspect_verdicts
from stock_research.validation.outcomes import outcomes
from stock_research.validation.preregistration import file_sha256, git_commit

FOLDER = Path("forward_log")
MAX_ENTRIES_PER_DAY = 999


def _value(value: Any) -> Any:
    if isinstance(value, float) and value != value:  # NaN is not valid JSON
        return None
    return value


def write_entry(
    directory: Path,
    code: str,
    report: StabilityReport,
    config: AppConfig,
    config_path: Path,
    now: datetime,
) -> Path:
    folder = directory / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_commit()
    aspects = aspect_verdicts(report.metrics, config.validation)
    entry = {
        "code": code,
        "as_of": report.as_of.isoformat(),
        "created_at": now.isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "config_sha256": file_sha256(config_path) if config_path.exists() else None,
        "overall": report.overall.verdict.value,
        "aspects": {k: v.value for k, v in aspects.items()},
        "metrics": [
            {
                "key": m.spec.key,
                "verdict": m.verdict.value,
                "v1": _value(m.values[WindowName.POST_2020]),
                "v3": _value(m.values[WindowName.PRE_2020]),
            }
            for m in report.metrics
        ],
    }
    stem = f"{report.as_of.isoformat()}_{code}"
    for n in range(1, MAX_ENTRIES_PER_DAY + 1):
        path = folder / (f"{stem}.json" if n == 1 else f"{stem}_{n}.json")
        try:
            with path.open("x", encoding="utf-8") as handle:  # never overwrite
                json.dump(entry, handle, ensure_ascii=False, indent=2)
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"too many forward-log entries for {stem}")


def read_entries(directory: Path) -> list[dict[str, Any]]:
    folder = directory / FOLDER
    if not folder.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]


@dataclass(frozen=True)
class ForwardComparison:
    months: int
    rows: pd.DataFrame  # code, as_of, metric, verdict, reference, future, normalized, ...
    pending: int  # entries whose horizon has not passed yet
    failures: dict[str, str]


def compare_entries(
    market_data: MarketData,
    config: AppConfig,
    directory: Path,
    months: int,
    *,
    progress: Callable[[str], None] = lambda _: None,
) -> ForwardComparison:
    today = market_data.today()
    keys = aspect_metrics(config.validation)
    specs = metric_specs(config)
    rows, pending, failures = [], 0, {}
    for entry in read_entries(directory):
        as_of = date.fromisoformat(entry["as_of"])
        if add_months(as_of, months) > today:
            pending += 1
            continue
        code = entry["code"]
        progress(f"{code} {as_of}")
        logged = {m["key"]: m for m in entry["metrics"]}
        try:
            decision = load_stock_data(market_data, code, as_of, with_peers=False)
            later = load_stock_data(market_data, code, today, with_peers=False)
            found = outcomes(
                {k: specs[k] for k in keys},
                {k: logged[k]["v1"] for k in keys if k in logged},
                decision,
                later,
                as_of,
                months,
                today,
                config,
                config.validation.walk_forward.min_forward_returns,
            )
        except (DataValidationError, ValueError, KeyError) as exc:
            failures[f"{code} {as_of}"] = f"{type(exc).__name__}: {exc}"
            continue
        for key in keys:
            o = found[key]
            rows.append(
                {
                    "code": code,
                    "as_of": as_of,
                    "metric": key,
                    "verdict": logged.get(key, {}).get("verdict"),
                    "reference": o.reference,
                    "future": o.future,
                    "normalized": o.normalized,
                    "exceeded": o.exceeded,
                    "reason": o.reason,
                }
            )
    return ForwardComparison(months, pd.DataFrame(rows), pending, failures)
