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
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from stock_research.analysis import research_score as rs
from stock_research.analysis.pipeline import FUNDAMENTALS, Analysis, AnalysisBlocked, run_analysis
from stock_research.analysis.risk import RiskMetrics
from stock_research.analysis.scoring import DISCLAIMER as SCORE_DISCLAIMER
from stock_research.analysis.stability import Gap
from stock_research.analysis.technicals import DISPUTED
from stock_research.analysis.weighting import WeightedStat
from stock_research.calibration.summary import for_preference
from stock_research.calibration.summary import load as load_calibration
from stock_research.config import AppConfig
from stock_research.data.errors import DataValidationError
from stock_research.data.market_data import MarketData
from stock_research.data.validation import HealthReport
from stock_research.data.windows import WindowName
from stock_research.export import market
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
    factor_label,
    metric_value,
    num,
    pct,
    principle_summary,
    render_analysis,
    sensitivity_note,
    walk_forward_note,
)
from stock_research.report.health import ATTRIBUTION, CHECK_NAMES, DISCLAIMER
from stock_research.validation.aspects import aspect_verdicts
from stock_research.validation.metric_walk_forward import positive_metrics

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


def _walk_forward(a: Analysis, key: str) -> str | None:
    evidence = a.walk_forward
    if evidence is None or key not in evidence.evaluated:
        return None
    return "正向" if key in evidence.positive else "未正向（不計入一致性分數）"


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
                "walk_forward": _walk_forward(a, m.spec.key),
            }
        )
    return out


def _sensitivity(a: Analysis, config: AppConfig) -> dict[str, Any]:
    """V1 conclusions under other half-lives (v2 P1-8): the composite and what moved."""
    s = a.sensitivity
    rows = [
        {
            "name": f"{FRAMEWORK_LABEL[name]}：框架分數",
            "values": [pct(score.values[f], 0) for f in s.multipliers],
            "note": sensitivity_note(score.recent, config),
        }
        for name, score in s.frameworks.items()
        if score.sensitive
    ] + [
        {
            "name": f"{FRAMEWORK_LABEL[item.framework]}：{item.name}",
            "values": [OUTCOME_LABEL[item.outcomes[f].value] for f in s.multipliers],
            "note": sensitivity_note(item.recent, config),
        }
        for item in s.items
    ]
    return {
        "multipliers": [factor_label(f) for f in s.multipliers],
        "composite": {
            "values": [pct(s.composite.values[f], 0) for f in s.multipliers],
            "note": sensitivity_note(s.composite.recent, config) if s.composite.sensitive else None,
        },
        "rows": rows,
    }


def _composite_note(a: Analysis, config: AppConfig) -> str | None:
    s = a.sensitivity.composite
    if not s.sensitive:
        return None
    values = "／".join(
        f"{factor_label(f)} {pct(s.values[f], 0)}" for f in a.sensitivity.multipliers
    )
    return f"投資大師原則（A 項）隨半衰期改變（{values}）：{sensitivity_note(s.recent, config)}"


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


def research_payload(
    market_data: MarketData,
    a: Analysis,
    config: AppConfig,
    calibration: dict[str, Any] | None,
    positive: tuple[frozenset[str], str] | None,
) -> dict[str, Any]:
    """The 條件符合度 (0-100) per capital tier (spec v2 §4) for every investor preference
    (成長 × 風險 §1.4), with every validation's result next to it (user decision 2026-10-18:
    shown even when none passed, then marked 僅供參考). The top level is the default
    preference's, so readers that know no preferences keep working."""
    if calibration is None:
        return {"available": False, "reason": "全市場歷史校準尚未完成，條件符合度暫不顯示"}
    preferences = config.research_score.preferences
    bases = rs.score_stock(market_data, a.data, a.stability, config, positive, a.growth_risk)
    raw = a.data.prices.raw.dropna(subset=["close"])
    price = float(raw["close"].iloc[-1])  # internal: positions and costs, never published
    by_preference = {
        key: {
            "label": option.label,
            **_preference_score(a, config, bases[key], for_preference(calibration, key), price),
        }
        for key, option in preferences.options.items()
    }
    return {
        **by_preference[preferences.default],
        "default_preference": preferences.default,
        "preferences": by_preference,
    }


def _preference_score(
    a: Analysis,
    config: AppConfig,
    base: rs.BaseScore,
    calibration: dict[str, Any] | None,
    price: float,
) -> dict[str, Any]:
    if calibration is None:
        return {"available": False, "reason": "校準結果沒有這個投資人偏好（需重跑校準）"}
    if base.score is None:
        return {"available": False, "reason": base.problem}
    group = rs.group_of(base.score, calibration["cutoffs"])
    history = calibration["table"][group - 1]["out_of_sample"]
    tiers = rs.tier_scores(base, price, calibration, config)
    validations = validation_summaries(calibration)
    validated = bool(calibration.get("passed_any"))
    return {
        "available": True,
        "name": SCORE_NAME,
        "aim": SCORE_AIM,
        "validated": validated,
        "validation_note": None if validated else NOT_VALIDATED,
        "validations": validations,
        "default_capital": config.research_score.default_capital,
        "base": base.score,
        "group": group,
        "groups": calibration["groups"],
        "parts": [
            {"name": p.name, "points": p.points, "max": p.max_points, "detail": p.detail}
            for p in base.parts
        ],
        "tiers": [
            {
                "capital": t.capital,
                "holdings": t.holdings,
                "capital_label": capital_label(t.capital, t.holdings),
                "score": t.final,
                "label": t.label,
                "label_note": t.label_note,
                "factor": t.factor,
                "cost_ratio": t.cost_ratio,
                "odd_lot": t.odd_lot,
                "cap": t.cap,
                "cap_reason": t.cap_reason,
                "history": history["tiers"][str(t.capital)],
            }
            for t in tiers
        ],
        "calibration": {
            "data_as_of": calibration["data_as_of"],
            "out_of_sample": [
                calibration["base_dates"]["out_of_sample_start"],
                calibration["base_dates"]["last"],
            ],
            "horizon_months": calibration["horizon_months"],
            "months": history["months"],
        },
        "notes": [
            n
            for n in (_composite_note(a, config), (calibration.get("overfit") or {}).get("message"))
            if n
        ],
        "disclaimer": rs.DISCLAIMER,
    }


SCORE_NAME = "條件符合度"
SCORE_AIM = "追求比 0050 抗跌"
NOT_VALIDATED = "未通過任何驗證，僅供參考"
VALIDATION_NAMES = {
    "drawdown": "回撤較小",
    "eps_growth": "EPS 成長",
    "revenue_growth": "營收成長",
    "strategy": "策略回測",
    "returns": "報酬（僅記錄）",
}


def validation_summaries(calibration: dict[str, Any]) -> list[dict[str, Any]]:
    """Each validation's result as the site shows it: name, passed, what it tested, and the
    values and periods (formatted here; the page only lays them out)."""
    out = []
    oos = calibration["base_dates"]
    period = (
        f"樣本外基準日 {oos['out_of_sample_start']}～{oos['last']}，每月 {calibration['groups']} 組"
    )
    for key, check in (calibration.get("validations") or {}).items():
        lines = [check.get("label", "")]
        if key == "strategy":
            for test in (check.get("criteria") or {}).values():
                mark = "通過" if test["passed"] else "未通過"
                lines.append(f"{test['name']}：{mark}（{test['detail']}）")
            periods = check.get("periods") or {}
            if "out_of_sample" in periods:
                p = periods["out_of_sample"]["strategy"]
                lines.append(f"樣本外期間 {p['start']}～{p['end']}；歷史回測，不代表未來。")
            if (check.get("overfit") or {}).get("warning"):
                lines.append("樣本內明顯優於樣本外，可能過度擬合；以樣本外結果為準。")
        else:
            lines.append(period)
            rho = check.get("rank_correlation")
            if rho is not None:
                need = check.get("min_rank_correlation")
                lines.append(f"分數組與結果的等級相關 {rho:.2f}（門檻 {need:g}）")
            if key == "returns":
                lines.append("只記錄、不作為上線條件；歷史回測平均，不是預期報酬。")
            elif check.get("top_median") is not None:
                lines.append(
                    f"最高分組中位數 {check['top_median']:+.1%}、"
                    f"最低分組 {check['bottom_median']:+.1%}"
                )
        if check.get("reason"):
            lines.append(f"未通過原因：{check['reason']}")
        out.append(
            {
                "key": key,
                "name": VALIDATION_NAMES.get(key, key),
                "passed": bool(check.get("passed")),
                "gating": bool(check.get("gating")),
                "lines": [line for line in lines if line],
            }
        )
    return out


def stock_payload(
    a: Analysis, config: AppConfig, research: dict[str, Any] | None = None
) -> dict[str, Any]:
    st = a.stability
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
            "overall": _verdict(st.overall.verdict),
            "drivers": [names[k] for k in st.overall.drivers],
            "aspects": {k: {"name": ASPECT_NAMES[k], **_verdict(aspects[k])} for k in ASPECT_NAMES},
            "composite": {WINDOW_KEYS[w]: st.composites[w].score for w in COMPARE_ORDER},
            "composite_notes": sorted({n for w in COMPARE_ORDER for n in st.composites[w].notes}),
            "windows": WINDOW_NAMES,
            "metrics": _metrics(a),
            "walk_forward_note": walk_forward_note(a),
            "principles": [asdict(p) for p in principle_summary(a)],
            "half_life_sensitivity": _sensitivity(a, config),
            "fundamentals": _fundamentals(a),
            "frameworks": _frameworks(a),
            "risk": _risk(a, config),
            "technicals": {
                "note": DISPUTED,
                "descriptions": list(a.technicals.descriptions) if a.technicals else [],
            },
            "health": _health(a.health),
            "research_score": research or {"available": False, "reason": "未計算"},
            "report_markdown": render_analysis(a, config),
            "disclaimer": [DISCLAIMER, SCORE_DISCLAIMER, ATTRIBUTION],
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
    previous_site: market.Fetch | None = None,
) -> ExportSummary:
    """``previous_site`` reads the published site, whose history archive this export extends
    with today's official closes, ex-rights events and scores (the holdings page). Without it
    no market files are written."""
    as_of = market_data.today()
    summary = ExportSummary(as_of)
    archive = closes = events = None
    if previous_site is not None:  # read first: a failure must stop before the long analysis
        archive = market.load_previous(previous_site, config.portfolio.site_url)
        closes = market_data.official_closes()
        events = market_data.official_events()
    calibration = load_calibration(config.calibration.output)
    positive = positive_metrics(config.research_score.consistency.walk_forward_results)
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
                "disclaimer": [DISCLAIMER, ATTRIBUTION],
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
                    {
                        "schema": SCHEMA_VERSION,
                        "as_of": as_of,
                        **row,
                        "disclaimer": [DISCLAIMER, ATTRIBUTION],
                    }
                ),
            )
            rows.append(row)
            continue
        try:
            research = research_payload(market_data, analysis, config, calibration, positive)
        except DataValidationError as exc:
            research = {"available": False, "reason": f"分數無法計算：{exc}"}
        payload = stock_payload(analysis, config, research)
        _write(out_dir / "data" / "stocks" / f"{code}.json", payload)
        summary.ok.append(code)
        rows.append(
            {
                "code": code,
                "status": "ok",
                "name": payload["name"],
                "industry": payload["industry"],
                "overall": payload["overall"],
                "aspects": payload["aspects"],
                "composite_v1": payload["composite"]["V1"],
                "score": _list_score(payload["research_score"]),
            }
        )
    market_index: dict[str, Any] = {}
    market_note: list[str] = []
    if archive is not None:
        day = market.add_closes(archive, closes)
        market.add_events(archive, *events)
        market.add_scores(archive, as_of, rows)
        market.write_market(out_dir, archive, day, config)
        market_index = {"archive": market.ARCHIVE, "closes_date": day}
        market_note = [market.SOURCE_NOTE]
    _write(
        out_dir / "data" / "index.json",
        _clean(
            {
                "schema": SCHEMA_VERSION,
                "as_of": as_of,
                "generated_at": market_data.now,
                "stale_after_hours": config.app.stale_after_hours,
                "default_capital": config.research_score.default_capital,
                "capital_tiers": [
                    {"capital": t.capital, "label": capital_label(t.capital, t.holdings)}
                    for t in config.research_score.capital_tiers
                ],
                "stocks": rows,
                **market_index,
                "disclaimer": [DISCLAIMER, SCORE_DISCLAIMER, ATTRIBUTION, *market_note],
            }
        ),
    )
    return summary


def capital_label(capital: int, holdings: int) -> str:
    """'30 萬・5 檔': the capital tier as the site shows it (formatted here, not in JS)."""
    return f"{capital / 10000:g} 萬・{holdings} 檔"


def _list_score(research: dict[str, Any]) -> dict[str, Any]:
    """Every tier's score and label for the list (the page switches tiers and preferences),
    or why there is none. The cap reason lets the capital page leave out stocks capped for
    doing worse than the comparison ETF. The top level is the default preference's."""
    entry = _list_tiers(research)
    if research.get("preferences"):
        entry["preferences"] = {k: _list_tiers(v) for k, v in research["preferences"].items()}
    return entry


def _list_tiers(research: dict[str, Any]) -> dict[str, Any]:
    if not research.get("available"):
        return {"available": False, "reason": research.get("reason")}
    return {
        "available": True,
        "tiers": {
            str(t["capital"]): {
                "value": t["score"],
                "label": t["label"],
                "cap_reason": t["cap_reason"],
            }
            for t in research["tiers"]
        },
    }


def _name(market_data: MarketData, code: str) -> str:
    try:
        return market_data.stock_info(code).name
    except (DataValidationError, KeyError):
        return ""
