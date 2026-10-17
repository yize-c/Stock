"""Analyse the watchlist once and write JSON for the web app.

Output under ``<out>/data/``:

* ``index.json``: one row per stock (verdicts, composite score, raw close) plus the dataset's
  ``as_of`` and ``generated_at``.
* ``stocks/<code>.json``: the detail page, including the full Traditional Chinese report.

Every number comes from the analysis pipeline; this module only formats it. A stock that fails
the data checks is published as ``blocked`` with the failed checks, and a stock whose analysis
raised is ``error`` with the message: neither is ever filled with older data.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from stock_research.analysis.pipeline import FUNDAMENTALS, Analysis, AnalysisBlocked, run_analysis
from stock_research.analysis.risk import RiskMetrics
from stock_research.analysis.scoring import DISCLAIMER as SCORE_DISCLAIMER
from stock_research.analysis.stability import Gap
from stock_research.analysis.technicals import DISPUTED
from stock_research.analysis.weighting import WeightedStat
from stock_research.backtest.score import load_summary, reference_for
from stock_research.config import AppConfig
from stock_research.data.errors import DataValidationError
from stock_research.data.market_data import MarketData
from stock_research.data.validation import HealthReport
from stock_research.data.windows import WindowName
from stock_research.report.analysis import (
    COMPARE_ORDER,
    FLAG_LABEL,
    FRAMEWORK_LABEL,
    HISTORICAL,
    INTERNAL_LABEL,
    OUTCOME_LABEL,
    VERDICT_LABEL,
    _metric_name,
    comparison_rows,
    cost_note,
    metric_value,
    num,
    pct,
    render_analysis,
)
from stock_research.report.health import CHECK_NAMES, DISCLAIMER
from stock_research.validation.aspects import aspect_verdicts

SCHEMA_VERSION = 1
WINDOW_KEYS = {WindowName.POST_2020: "V1", WindowName.PRE_2020: "V3", WindowName.FULL: "V2"}
WINDOW_NAMES = {"V1": "2020 後", "V3": "2020 前", "V2": "全期（參考）"}
ASPECT_NAMES = {"revenue": "營收", "profitability": "獲利", "risk": "風險"}


def _clean(value: Any) -> Any:
    """JSON-safe: NaN and infinities become None; dates become ISO strings."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def _verdict(v) -> dict[str, str]:
    return {"code": v.value, "label": VERDICT_LABEL[v]}


def _health(report: HealthReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "checks": [
            {"name": CHECK_NAMES.get(c.name, c.name), "passed": c.passed, "detail": c.detail}
            for c in report.checks
        ],
        "limitations": list(report.limitations),
    }


def _risk(a: Analysis, config: AppConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for window in COMPARE_ORDER:
        m = a.risk[window]
        key = WINDOW_KEYS[window]
        if not isinstance(m, RiskMetrics):
            out[key] = {"available": False, "reason": m.reason}
            continue
        sharpe = a.sharpe[window]
        out[key] = {
            "available": True,
            "label": HISTORICAL.format(start=m.start, end=m.end),
            "rows": [
                ["年化平均報酬", pct(m.annualized_return)],
                ["年化波動度", pct(m.volatility)],
                [
                    "最大回撤",
                    f"{pct(m.max_drawdown.depth)}（{m.max_drawdown.peak} → {m.max_drawdown.trough}）",
                ],
                ["Beta", num(m.beta)],
                [f"歷史 VaR {m.var_confidence:.0%}（單日）", pct(m.var)],
                ["Sharpe", f"{sharpe.value:.2f}" if not isinstance(sharpe, str) else "無法計算"],
            ],
            "sharpe_note": sharpe if isinstance(sharpe, str) else None,
            "comparison": comparison_rows(a.returns[window], cost_note(config)),
        }
    return out


def _fundamentals(a: Analysis) -> list[dict[str, Any]]:
    rows = []
    for key, name in FUNDAMENTALS:
        values = {}
        for window in COMPARE_ORDER:
            stat = a.fundamentals[key][window]
            if isinstance(stat, WeightedStat):
                shown = metric_value(key, _gap_for(a, key), stat.unweighted)
            else:
                shown = "資料不足"
            values[WINDOW_KEYS[window]] = shown
        rows.append({"key": key, "name": name, "values": values})
    yoy = {}
    for window in COMPARE_ORDER:
        stat = a.revenue_yoy[window]
        yoy[WINDOW_KEYS[window]] = (
            pct(stat.unweighted) if isinstance(stat, WeightedStat) else "資料不足"
        )
    rows.append({"key": "revenue_yoy", "name": "月營收年增率（平均）", "values": yoy})
    return rows


def _gap_for(a: Analysis, key: str):
    spec = next((m.spec for m in a.stability.metrics if m.spec.key == key), None)
    return spec.gap if spec is not None else Gap.RELATIVE


def _metrics(a: Analysis) -> list[dict[str, Any]]:
    out = []
    for m in a.stability.metrics:
        if m.spec.framework is not None:
            continue  # framework items appear under their framework
        out.append(
            {
                "key": m.spec.key,
                "name": _metric_name(m),
                "category": m.spec.category,
                "verdict": _verdict(m.verdict),
                "internal": INTERNAL_LABEL[m.internal],
                "flags": [FLAG_LABEL[f] for f in sorted(m.flags)],
                "values": {
                    WINDOW_KEYS[w]: metric_value(m.spec.key, m.spec.gap, m.values[w])
                    for w in COMPARE_ORDER
                },
                "note": m.step_a.note or None,
            }
        )
    return out


def _frameworks(a: Analysis) -> list[dict[str, Any]]:
    st = a.stability
    by_window = {w: {r.framework: r for r in st.framework_results[w]} for w in COMPARE_ORDER}
    out = []
    for name in FRAMEWORK_LABEL:
        v1 = by_window[WindowName.POST_2020].get(name)
        if v1 is None:
            continue
        items = []
        for item in v1.items:
            outcomes = {}
            for w in COMPARE_ORDER:
                found = next((i for i in by_window[w][name].items if i.key == item.key), None)
                outcomes[WINDOW_KEYS[w]] = OUTCOME_LABEL[found.outcome.value] if found else "—"
            items.append({"name": item.name, "outcomes": outcomes, "detail": item.detail})
        aggregate = st.frameworks.get(name)
        out.append(
            {
                "key": name,
                "name": FRAMEWORK_LABEL[name],
                "scores": {WINDOW_KEYS[w]: by_window[w][name].score for w in COMPARE_ORDER},
                "stability": _verdict(aggregate.verdict) if aggregate else None,
                "not_applicable": v1.not_applicable,
                "items": items,
            }
        )
    return out


def stock_payload(
    a: Analysis, config: AppConfig, backtest: dict[str, Any] | None = None
) -> dict[str, Any]:
    st = a.stability
    raw = a.data.prices.raw.dropna(subset=["close"])
    last = raw.iloc[-1]
    names = {m.spec.key: _metric_name(m) for m in st.metrics}
    aspects = aspect_verdicts(st.metrics, config.validation)
    return _clean(
        {
            "schema": SCHEMA_VERSION,
            "status": "ok",
            "code": a.info.ticker.code,
            "name": a.info.name,
            "market": a.info.ticker.market.value,
            "industry": a.data.industry,
            "as_of": a.as_of,
            "close": {
                "value": float(last["close"]),
                "date": last["date"],
                "note": "原始收盤價（未還原）",
            },
            "overall": _verdict(st.overall.verdict),
            "drivers": [names[k] for k in st.overall.drivers],
            "aspects": {k: {"name": ASPECT_NAMES[k], **_verdict(aspects[k])} for k in ASPECT_NAMES},
            "composite": {WINDOW_KEYS[w]: st.composites[w].score for w in COMPARE_ORDER},
            "composite_notes": sorted({n for w in COMPARE_ORDER for n in st.composites[w].notes}),
            "windows": WINDOW_NAMES,
            "metrics": _metrics(a),
            "fundamentals": _fundamentals(a),
            "frameworks": _frameworks(a),
            "risk": _risk(a, config),
            "technicals": {
                "note": DISPUTED,
                "descriptions": list(a.technicals.descriptions) if a.technicals else [],
            },
            "health": _health(a.health),
            "backtest_reference": reference_for(
                st.composites[WindowName.POST_2020].score, backtest, config
            ),
            "report_markdown": render_analysis(a, config),
            "disclaimer": [DISCLAIMER, SCORE_DISCLAIMER],
        }
    )


@dataclass
class ExportSummary:
    as_of: date
    ok: list[str] = field(default_factory=list)
    blocked: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def export_site(
    market_data: MarketData,
    config: AppConfig,
    out_dir: Path,
    *,
    codes: list[str] | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> ExportSummary:
    as_of = market_data.today()
    summary = ExportSummary(as_of)
    backtest = load_summary(config.backtest.output)
    rows = []
    for code in codes or config.app.watchlist:
        progress(f"{code}: analysing as of {as_of}")
        row: dict[str, Any] = {"code": code}
        try:
            analysis = run_analysis(
                market_data, config, code, as_of, with_peers=config.app.with_peers
            )
        except AnalysisBlocked as exc:
            failed = [c for c in exc.report.checks if not c.passed]
            reason = "、".join(CHECK_NAMES.get(c.name, c.name) for c in failed)
            summary.blocked[code] = reason
            row |= {"status": "blocked", "problem": f"資料檢查未通過：{reason}"}
            payload = {
                "schema": SCHEMA_VERSION,
                "status": "blocked",
                "code": code,
                "as_of": as_of,
                "problem": row["problem"],
                "health": _health(exc.report),
                "disclaimer": [DISCLAIMER],
            }
            row["name"] = _name(market_data, code)
            payload["name"] = row["name"]
            _write(out_dir / "data" / "stocks" / f"{code}.json", _clean(payload))
            rows.append(row)
            continue
        except (DataValidationError, ValueError, KeyError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            summary.errors[code] = message
            row |= {"status": "error", "name": _name(market_data, code), "problem": message}
            _write(
                out_dir / "data" / "stocks" / f"{code}.json",
                _clean(
                    {"schema": SCHEMA_VERSION, "as_of": as_of, **row, "disclaimer": [DISCLAIMER]}
                ),
            )
            rows.append(row)
            continue
        payload = stock_payload(analysis, config, backtest)
        _write(out_dir / "data" / "stocks" / f"{code}.json", payload)
        summary.ok.append(code)
        rows.append(
            {
                "code": code,
                "status": "ok",
                "name": payload["name"],
                "industry": payload["industry"],
                "close": payload["close"],
                "overall": payload["overall"],
                "aspects": payload["aspects"],
                "composite_v1": payload["composite"]["V1"],
                "reference_label": payload["backtest_reference"].get("label"),
            }
        )
    _write(
        out_dir / "data" / "index.json",
        _clean(
            {
                "schema": SCHEMA_VERSION,
                "as_of": as_of,
                "generated_at": market_data.now,
                "stale_after_hours": config.app.stale_after_hours,
                "stocks": rows,
                "disclaimer": [DISCLAIMER, SCORE_DISCLAIMER],
            }
        ),
    )
    return summary


def _name(market_data: MarketData, code: str) -> str:
    try:
        return market_data.stock_info(code).name
    except (DataValidationError, KeyError):
        return ""
