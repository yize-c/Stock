"""Build stability samples from one stock's data and assemble the full stability report.

Samples are unweighted (spec §4.3). Windows: V1, V3, V2 (reference only) and V1's halves.
The crash-excluded variant drops daily returns inside ``crash_exclusion.price_start..end``
and TTM / YoY values whose period covers a crash quarter; framework items, scores and
single-date valuation metrics have no such variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from stock_research.analysis import breakpoints as bp
from stock_research.analysis.frameworks import run_all
from stock_research.analysis.frameworks.base import FrameworkResult, Outcome
from stock_research.analysis.fundamentals import Valuation, ttm_table
from stock_research.analysis.inputs import StockData, inputs_for
from stock_research.analysis.results import InsufficientHistory
from stock_research.analysis.revenue import gap_months
from stock_research.analysis.risk import daily_returns
from stock_research.analysis.scoring import CompositeScore, composite
from stock_research.analysis.stability import (
    Aggregate,
    Gap,
    MetricInput,
    MetricSpec,
    MetricVerdict,
    Sample,
    StatTest,
    aggregate,
    evaluate,
)
from stock_research.config import AppConfig
from stock_research.data.windows import TimeWindow, WindowName, build_window

STEP_WINDOWS = (
    WindowName.POST_2020,
    WindowName.PRE_2020,
    WindowName.FULL,
    WindowName.POST_2020_EARLY,
    WindowName.POST_2020_LATE,
)

TTM_METRICS = {
    # key: (name, gap, tolerance key or None for relative)
    "roe": ("ROE", Gap.ABSOLUTE, "roe"),
    "roic": ("ROIC", Gap.ABSOLUTE, "roic"),
    "gross_margin": ("毛利率", Gap.ABSOLUTE, "gross_margin"),
    "operating_margin": ("營業利益率", Gap.ABSOLUTE, "operating_margin"),
    "net_margin": ("淨利率", Gap.ABSOLUTE, "net_margin"),
    "eps": ("EPS（近四季）", Gap.RELATIVE, None),
    "free_cash_flow": ("自由現金流（近四季）", Gap.RELATIVE, None),
    "interest_coverage": ("利息保障倍數", Gap.RELATIVE, None),
    "current_ratio": ("流動比率", Gap.RELATIVE, None),
    "borrowings_to_equity": ("有息負債權益比", Gap.RELATIVE, None),
}


def _quarters(labels: list[str]) -> list[tuple[date, date]]:
    out = []
    for label in labels:
        year, q = int(label[:4]), int(label[-1])
        start = date(year, 3 * q - 2, 1)
        out.append((start, (pd.Timestamp(start) + pd.offsets.QuarterEnd(0)).date()))
    return out


def _drop_crash_periods(rows: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    crash = _quarters(config.stability.crash_exclusion.quarters)
    keep = [
        not any(start <= q_end and end >= q_start for q_start, q_end in crash)
        for start, end in zip(rows["period_start"], rows["period_end"], strict=True)
    ]
    return rows.loc[keep]


def _drop_crash_days(series: pd.Series | pd.DataFrame, config: AppConfig):
    crash = config.stability.crash_exclusion
    keep = [not crash.price_start <= d <= crash.price_end for d in series.index]
    return series.loc[keep]


@dataclass(frozen=True)
class WindowData:
    returns: pd.Series
    paired: pd.DataFrame
    ttm: pd.DataFrame
    yoy: pd.DataFrame


def _window_data(data: StockData, window: TimeWindow, ttm: pd.DataFrame, crash: bool, config):
    prices = window.filter_prices(data.prices.adjusted)
    returns = daily_returns(prices, "adj_close")
    both = prices.merge(window.filter_prices(data.benchmark), on="date", how="inner")
    paired = pd.concat(
        [daily_returns(both, "adj_close"), daily_returns(both, "total_return_index")],
        axis=1,
        keys=["stock", "index"],
    ).dropna()
    ttm_rows = window.filter_financials(ttm)
    yoy = window.filter_financials(data.revenue_yoy)
    if crash:
        returns = _drop_crash_days(returns, config)
        paired = _drop_crash_days(paired, config)
        ttm_rows = _drop_crash_periods(ttm_rows, config)
        # A missing revenue month still marks the window, whichever periods are dropped.
        kept = _drop_crash_periods(yoy, config).index
        yoy = yoy.loc[yoy.index.isin(kept) | yoy.get("gap", False)]
    return WindowData(returns, paired, ttm_rows, yoy)


def _risk_samples(w: WindowData, config: AppConfig) -> dict[str, Sample]:
    r = w.returns.to_numpy(dtype=float)
    days = config.risk.annualization_days
    if len(r) < 2:
        missing = Sample.missing(f"{len(r)} daily return(s)")
        return dict.fromkeys(("mean_return", "volatility", "beta", "max_drawdown", "var"), missing)
    wealth = np.cumprod(1 + r)
    drawdown = float(-(wealth / np.maximum.accumulate(wealth) - 1).min())
    paired = w.paired
    beta = (
        Sample(
            float(paired["stock"].cov(paired["index"]) / paired["index"].var()),
            len(paired),
            paired,
        )
        if len(paired) >= 2
        else Sample.missing(f"{len(paired)} paired return(s)")
    )
    return {
        "mean_return": Sample(float(r.mean() * days), len(r), r),
        "volatility": Sample(float(r.std(ddof=1) * np.sqrt(days)), len(r), r),
        "beta": beta,
        "max_drawdown": Sample(drawdown, len(r)),
        "var": Sample(float(-np.quantile(r, 1 - config.risk.var_confidence)), len(r)),
    }


def _ttm_sample(rows: pd.DataFrame, column: str) -> Sample:
    values = rows[column].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return Sample.missing("no value in the window")
    return Sample(float(values.mean()), len(values))


def _yoy_sample(rows: pd.DataFrame) -> Sample:
    holes = gap_months(rows)
    if holes:  # a window with a missing revenue month has no YoY sample (user, 2026-10-19)
        return Sample.missing(f"missing revenue month(s): {holes[0]} and {len(holes) - 1} more")
    values = rows["yoy"].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return Sample.missing("no YoY month in the window")
    return Sample(float(values.mean()), len(values), values)


def _specs(config: AppConfig) -> list[MetricSpec]:
    st = config.stability
    tol = st.absolute_tolerance
    samples = st.min_samples
    econ = samples.econ_observations
    specs = [
        MetricSpec("mean_return", "年化平均報酬", "risk", Gap.ABSOLUTE,
                   tol["annualized_return"], econ, StatTest.WELCH),
        MetricSpec("volatility", "年化波動度", "risk", Gap.ABSOLUTE,
                   tol["annualized_volatility"], econ, StatTest.BROWN_FORSYTHE),
        MetricSpec("beta", "Beta", "risk", Gap.RELATIVE, st.relative_tolerance, econ,
                   StatTest.CHOW_HAC),
        MetricSpec("max_drawdown", "最大回撤", "risk", Gap.ABSOLUTE, tol["max_drawdown"], econ),
        MetricSpec("var", "歷史 VaR", "risk", Gap.ABSOLUTE, tol["var"], econ),
        MetricSpec("revenue_yoy", "月營收年增率", "revenue", Gap.ABSOLUTE, tol["revenue_yoy"],
                   econ, StatTest.HAC_DUMMY),
    ]  # fmt: skip
    for key, (name, gap, tol_key) in TTM_METRICS.items():
        tolerance = tol[tol_key] if tol_key else st.relative_tolerance
        specs.append(MetricSpec(key, name, "fundamental", gap, tolerance, samples.ttm_quarters))
    specs += [
        MetricSpec("pe", "本益比", "valuation", Gap.RELATIVE, st.relative_tolerance, 1,
                   point_in_time=True),
        MetricSpec("pb", "股價淨值比", "valuation", Gap.RELATIVE, st.relative_tolerance, 1,
                   point_in_time=True),
        MetricSpec("dividend_yield", "現金殖利率", "valuation", Gap.ABSOLUTE,
                   tol["dividend_yield"], 1, point_in_time=True),
    ]  # fmt: skip
    return specs


def _valuation_sample(results_inputs, key: str) -> Sample:
    v = results_inputs.valuation
    if v is None:
        return Sample.missing(results_inputs.valuation_problem or "no valuation")
    value = getattr(v, key)
    if value is None or np.isnan(value):
        return Sample.missing(f"{key} undefined (EPS or equity <= 0)")
    return Sample(float(value), 1)


def _outcome_sample(outcome: Outcome, detail: str) -> Sample:
    if outcome in (Outcome.PASS, Outcome.FAIL):
        return Sample(outcome.value, 1)
    return Sample.missing(f"{outcome.value}: {detail}")


@dataclass(frozen=True)
class StabilityReport:
    as_of: date
    windows: dict[WindowName, TimeWindow]
    metrics: list[MetricVerdict]
    frameworks: dict[str, Aggregate]
    overall: Aggregate
    framework_results: dict[WindowName, list[FrameworkResult]]
    composites: dict[WindowName, CompositeScore]
    breakpoints: list[bp.BreakResult | InsufficientHistory]
    valuations: dict[WindowName, Valuation | None]
    valuation_problems: dict[WindowName, str | None]


def stability_report(
    data: StockData, config: AppConfig, *, with_breakpoints: bool = True
) -> StabilityReport:
    """``with_breakpoints=False`` skips the (slow, information-only) break scan."""
    windows = {name: build_window(name, config, data.as_of) for name in STEP_WINDOWS}
    ttm = ttm_table(data.quarterly, data.basis, data.as_of)
    plain = {n: _window_data(data, w, ttm, False, config) for n, w in windows.items()}
    ex_crash = {
        n: _window_data(data, windows[n], ttm, True, config)
        for n in (WindowName.POST_2020_EARLY, WindowName.POST_2020_LATE)
    }
    framework_inputs = {n: inputs_for(data, w, config) for n, w in windows.items()}
    results = {n: run_all(i) for n, i in framework_inputs.items()}
    composites = {n: composite(r, config.scoring) for n, r in results.items()}

    def samples_for(by_window: dict[WindowName, WindowData]) -> dict[WindowName, dict]:
        out = {}
        for n, w in by_window.items():
            row = _risk_samples(w, config)
            row["revenue_yoy"] = _yoy_sample(w.yoy)
            for key in TTM_METRICS:
                row[key] = _ttm_sample(w.ttm, key)
            for key in ("pe", "pb", "dividend_yield"):
                row[key] = _valuation_sample(framework_inputs[n], key)
            out[n] = row
        return out

    s = samples_for(plain)
    sx = samples_for(ex_crash)
    early, late = WindowName.POST_2020_EARLY, WindowName.POST_2020_LATE
    inputs: list[MetricInput] = []
    for spec in _specs(config):
        has_crash_variant = spec.category != "valuation"
        inputs.append(
            MetricInput(
                spec,
                v1=s[WindowName.POST_2020][spec.key],
                v3=s[WindowName.PRE_2020][spec.key],
                v2=s[WindowName.FULL][spec.key],
                halves=(s[early][spec.key], s[late][spec.key]),
                halves_ex_crash=(sx[early][spec.key], sx[late][spec.key])
                if has_crash_variant
                else None,
            )
        )
    inputs += _framework_inputs(results, composites, config)
    verdicts = evaluate(inputs, config.stability)

    by_framework: dict[str, list[MetricVerdict]] = {}
    for v in verdicts:
        if v.spec.framework is not None:
            by_framework.setdefault(v.spec.framework, []).append(v)
    return StabilityReport(
        as_of=data.as_of,
        windows=windows,
        metrics=verdicts,
        frameworks={name: aggregate(items) for name, items in by_framework.items()},
        overall=aggregate(verdicts),
        framework_results=results,
        composites=composites,
        breakpoints=_breakpoints(plain[WindowName.FULL], config) if with_breakpoints else [],
        valuations={n: i.valuation for n, i in framework_inputs.items()},
        valuation_problems={n: i.valuation_problem for n, i in framework_inputs.items()},
    )


def _framework_inputs(
    results: dict[WindowName, list[FrameworkResult]],
    composites: dict[WindowName, CompositeScore],
    config: AppConfig,
) -> list[MetricInput]:
    st = config.stability
    by = {n: {r.framework: r for r in rs} for n, rs in results.items()}
    early, late = WindowName.POST_2020_EARLY, WindowName.POST_2020_LATE
    out: list[MetricInput] = []
    for framework, v1 in by[WindowName.POST_2020].items():
        for item in v1.items:
            if item.outcome is Outcome.INFO:
                continue
            spec = MetricSpec(
                f"{framework}.{item.key}", item.name, "framework_item", Gap.CATEGORY, 0.0, 1,
                framework=framework,
            )  # fmt: skip

            def sample(window: WindowName, key: str = item.key, fw: str = framework) -> Sample:
                result = by[window][fw]
                if result.not_applicable:
                    return Sample.missing(result.not_applicable)
                found = next(i for i in result.items if i.key == key)
                return _outcome_sample(found.outcome, found.detail)

            out.append(
                MetricInput(
                    spec,
                    sample(WindowName.POST_2020),
                    sample(WindowName.PRE_2020),
                    sample(WindowName.FULL),
                    (sample(early), sample(late)),
                )
            )
        spec = MetricSpec(
            f"{framework}.score", "框架分數", "framework_score", Gap.SCORE, st.score_tolerance, 1,
            framework=framework,
        )  # fmt: skip

        def score(window: WindowName, fw: str = framework) -> Sample:
            value = by[window][fw].score
            return Sample(value, 1) if value is not None else Sample.missing("score N/A")

        out.append(
            MetricInput(
                spec,
                score(WindowName.POST_2020),
                score(WindowName.PRE_2020),
                score(WindowName.FULL),
                (score(early), score(late)),
            )
        )

    def total(window: WindowName) -> Sample:
        value = composites[window].score
        return Sample(value, 1) if value is not None else Sample.missing("composite N/A")

    out.append(
        MetricInput(
            MetricSpec("composite", "綜合分數", "composite", Gap.SCORE, st.score_tolerance, 1),
            total(WindowName.POST_2020),
            total(WindowName.PRE_2020),
            total(WindowName.FULL),
            (total(early), total(late)),
        )
    )
    return out


def _breakpoints(full: WindowData, config: AppConfig) -> list[bp.BreakResult | InsufficientHistory]:
    trimming = config.stability.breakpoints.trimming
    alpha = config.stability.alpha
    reference = config.general.regime_split_date
    minimum = 2 * config.stability.min_samples.daily_observations
    out: list[bp.BreakResult | InsufficientHistory] = []
    cases = (
        (bp.BreakModel.MEAN, full.returns.to_frame("stock")),
        (bp.BreakModel.BETA, full.paired),
    )
    for model, series in cases:
        if len(series) < minimum:
            out.append(
                InsufficientHistory(
                    WindowName.FULL, f"breakpoint.{model}", f"{len(series)} returns, need {minimum}"
                )
            )
            continue
        out.append(
            bp.quandt_andrews(series, model, trimming=trimming, alpha=alpha, reference=reference)
        )
    return out


def metric_specs(config: AppConfig) -> dict[str, MetricSpec]:
    """The numeric metrics' specs (gap rule and tolerance) by key."""
    return {spec.key: spec for spec in _specs(config)}
