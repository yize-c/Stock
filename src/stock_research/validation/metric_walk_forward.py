"""Walk-forward evidence per metric, for the score's consistency part (v2 §4.1 B).

The user chose to judge "walk-forward 正向" metric by metric: a metric's V1/V3 agreement counts
toward the score only when, in the latest development attempt, that metric's stability
verdicts predicted its later changes (the same test as the whole walk-forward: AUC above
baseline B's quantile and a significant Mann-Whitney effect; see ``statistics.evaluate``).

Only development-phase rows are used (the holdout stays untouched). The attempt's base dates
(2023-12 .. 2025-06) overlap the calibration's out-of-sample period, so which metrics count is
known from data inside that period; the calibration output says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stock_research.config import WalkForwardConfig
from stock_research.validation.statistics import evaluate

SCHEMA_VERSION = 1


def evaluate_metrics(rows: pd.DataFrame, settings: WalkForwardConfig) -> dict[str, Any]:
    """Per-metric evaluation of the latest development attempt in ``rows``."""
    dev = rows.loc[(rows["phase"] == "dev") & (rows["status"] == "OK")]
    if dev.empty:
        raise ValueError("no development-phase walk-forward rows")
    attempt = int(dev["attempt"].max())
    latest = dev.loc[dev["attempt"] == attempt]
    metrics: dict[str, Any] = {}
    for metric, group in latest.groupby("metric", sort=True):
        result = evaluate(
            group,
            draws_baseline=settings.baseline_draws,
            draws_bootstrap=settings.bootstrap_draws,
            confidence=settings.confidence,
            alpha=settings.alpha,
            seed=settings.seed,
        )
        metrics[str(metric)] = {**result.as_dict(), "positive": bool(result.passed)}
    return {
        "schema": SCHEMA_VERSION,
        "phase": "dev",
        "attempt": attempt,
        "base_dates": [str(min(latest["base_date"])), str(max(latest["base_date"]))],
        "metrics": metrics,
    }


def save(summary: dict[str, Any], path: Path, now: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {**summary, "generated_at": now.isoformat(timespec="seconds")}
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class WalkForwardEvidence:
    evaluated: frozenset[str]
    positive: frozenset[str]
    source: str


def walk_forward_evidence(path: Path) -> WalkForwardEvidence | None:
    """Which metrics were evaluated, which are positive, and where that comes from; None if
    the file has not been produced."""
    if not path.exists():
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    metrics = body["metrics"]
    first, last = body["base_dates"]
    return WalkForwardEvidence(
        frozenset(metrics),
        frozenset(m for m, r in metrics.items() if r["positive"]),
        f"walk-forward 開發期第 {body['attempt']} 次（{first}～{last}）",
    )


def positive_metrics(path: Path) -> tuple[frozenset[str], str] | None:
    """The metrics whose walk-forward is positive, and where that comes from; None if the
    file has not been produced."""
    evidence = walk_forward_evidence(path)
    return None if evidence is None else (evidence.positive, evidence.source)
