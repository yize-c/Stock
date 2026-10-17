"""Typed, validated configuration loaded from ``config.yaml``.

Every threshold the analysis uses comes from here. Loading fails loudly on unknown keys,
missing keys or inconsistent values; no silent defaults are applied to a section that
appears in the file. Secrets are read separately from the environment (see ``Secrets``).
"""

from __future__ import annotations

import math
import os
import re
from datetime import date, time
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_CONFIG_PATH = Path("config.yaml")

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]

_QUARTER_RE = re.compile(r"^\d{4}Q[1-4]$")
# Tolerance for comparing weight sums that come from decimal literals in YAML.
_WEIGHT_TOLERANCE = 1e-9


class ConfigError(Exception):
    """Raised when the configuration file is missing, malformed or inconsistent."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- general -----------------------------------------------------------------------------


class GeneralConfig(_Model):
    timezone: str
    default_market: Literal["tw", "us"]
    regime_split_date: date
    ifrs_start_year: Annotated[int, Field(ge=1900)]

    @model_validator(mode="after")
    def _check_timezone(self) -> GeneralConfig:
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone: {self.timezone!r}") from exc
        return self


# --- markets -----------------------------------------------------------------------------


class ExchangeConfig(_Model):
    yfinance_suffix: str


class PriceLimitRule(_Model):
    effective_from: date | None
    limit: PositiveFloat


class PriceLimitConfig(_Model):
    rules: list[PriceLimitRule] = Field(min_length=1)
    rounding_tolerance: Fraction
    new_listing_exempt_days: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _check_rule_order(self) -> PriceLimitConfig:
        if self.rules[0].effective_from is not None:
            raise ValueError("the first price-limit rule must have effective_from: null")
        dated = [r.effective_from for r in self.rules[1:]]
        if any(d is None for d in dated):
            raise ValueError("only the first price-limit rule may have effective_from: null")
        if dated != sorted(dated) or len(set(dated)) != len(dated):  # type: ignore[type-var]
            raise ValueError("price-limit rules must have strictly increasing effective_from")
        return self


class FilingDeadline(_Model):
    month: Annotated[int, Field(ge=1, le=12)]
    day: Annotated[int, Field(ge=1, le=31)]
    year_offset: Annotated[int, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def _check_real_day(self) -> FilingDeadline:
        # Use a leap year so Feb 29 is accepted; real deadlines are checked per year later.
        try:
            date(2024, self.month, self.day)
        except ValueError as exc:
            raise ValueError(f"invalid deadline {self.month}/{self.day}") from exc
        return self


class FilingDeadlineConfig(_Model):
    quarterly: dict[Literal["Q1", "Q2", "Q3", "Q4"], FilingDeadline]
    monthly_revenue_day: Annotated[int, Field(ge=1, le=28)]

    @model_validator(mode="after")
    def _check_all_quarters(self) -> FilingDeadlineConfig:
        missing = {"Q1", "Q2", "Q3", "Q4"} - set(self.quarterly)
        if missing:
            raise ValueError(f"filing deadlines missing for {sorted(missing)}")
        return self


class ManualClosure(_Model):
    date: date
    reason: str = Field(min_length=1)


class TradingCalendarConfig(_Model):
    historical_source: Literal["official_index_closes"]
    manual_closures: list[ManualClosure]
    session_close_time: time
    daily_data_ready_time: time

    @model_validator(mode="after")
    def _check(self) -> TradingCalendarConfig:
        dates = [c.date for c in self.manual_closures]
        if len(set(dates)) != len(dates):
            raise ValueError("manual_closures contains duplicate dates")
        if self.daily_data_ready_time < self.session_close_time:
            raise ValueError("daily_data_ready_time must not be before session_close_time")
        return self


class KnownSuspension(_Model):
    ticker: Annotated[str, Field(pattern=r"^[1-9]\d{3}$")]
    start: date
    end: date
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> KnownSuspension:
        if self.end < self.start:
            raise ValueError(f"suspension of {self.ticker} ends before it starts")
        return self

    def covers(self, day: date) -> bool:
        return self.start <= day <= self.end


class PerShareConfig(_Model):
    default_par_value: PositiveFloat
    share_multiplier_tolerance: Fraction
    share_count_tolerance: Fraction


class IndustryOverride(_Model):
    ticker: Annotated[str, Field(pattern=r"^[1-9]\d{3}$")]
    industry: str = Field(min_length=1)


class PeerConfig(_Model):
    min_peers: PositiveInt
    cache_max_age_days: PositiveFloat


class IndustryConfig(_Model):
    ignored_categories: list[str]
    aliases: dict[str, str]
    overrides: list[IndustryOverride]
    groups: dict[str, list[str]]
    peers: PeerConfig

    @model_validator(mode="after")
    def _check(self) -> IndustryConfig:
        tickers = [o.ticker for o in self.overrides]
        if len(set(tickers)) != len(tickers):
            raise ValueError("industries.overrides contains duplicate tickers")
        aliased = sorted(set(self.aliases) & {i for g in self.groups.values() for i in g})
        if aliased:
            raise ValueError(f"industry groups must use canonical names, not aliases: {aliased}")
        return self


class ListingOverride(_Model):
    ticker: Annotated[str, Field(pattern=r"^[1-9]\d{3}$")]
    listing_date: date
    source: str = Field(min_length=1)


class PriceHistoryOverride(_Model):
    """A later price start for one stock, when an earlier capital change lacks the reference
    price needed to adjust its prices (FinMind's capital-reduction data starts in 2011)."""

    ticker: Annotated[str, Field(pattern=r"^[1-9]\d{3}$")]
    start: date
    reason: str = Field(min_length=1)


class ListingConfig(_Model):
    emerging_board_start: date
    overrides: list[ListingOverride]

    @model_validator(mode="after")
    def _check_unique(self) -> ListingConfig:
        tickers = [o.ticker for o in self.overrides]
        if len(set(tickers)) != len(tickers):
            raise ValueError("listing.overrides contains duplicate tickers")
        return self


class TwBenchmarkConfig(_Model):
    primary: str
    fallback_ticker: str
    fallback_uses_adjusted_prices: bool
    compare_etf: str


class RiskFreeRate(_Model):
    start: date
    end: date
    rate: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def _check_range(self) -> RiskFreeRate:
        if self.end < self.start:
            raise ValueError(f"risk-free period ends before it starts: {self.start}..{self.end}")
        return self


class CbcBankMembers(_Model):
    """The CBC's five largest banks from ``start`` until the next entry's start."""

    start: date
    banks: Annotated[list[str], Field(min_length=1)]


class CbcRatesConfig(_Model):
    url: str
    one_year_fixed_column: NonNegativeInt  # 0-based (column N = 13)
    start: date
    members: Annotated[list[CbcBankMembers], Field(min_length=1)]
    cache_max_age_hours: PositiveInt


class RiskFreeConfig(_Model):
    source: str
    cbc: CbcRatesConfig
    # Days after the last month with a rate that may reuse it (marked in the output).
    carry_forward_max_days: NonNegativeInt
    fallback_rates: list[RiskFreeRate]

    @model_validator(mode="after")
    def _check_no_overlap(self) -> RiskFreeConfig:
        periods = sorted(self.fallback_rates, key=lambda r: r.start)
        for prev, cur in zip(periods, periods[1:], strict=False):
            if cur.start <= prev.end:
                raise ValueError(f"risk-free fallback periods overlap at {cur.start}")
        return self


class KnownSourceError(_Model):
    """A secondary source's close found wrong against the official TWSE/TPEx daily data."""

    source: Literal["yahoo"]
    ticker: Annotated[str, Field(pattern=r"^[1-9]\d{3}$")]
    date: date
    official_close: PositiveFloat
    evidence: str = Field(min_length=1)


class DataValidationConfig(_Model):
    # A halt of at least this many trading days before a beyond-limit close, earlier than
    # FinMind's capital-reduction coverage, marks an uncovered capital change.
    min_halt_trading_days: PositiveInt
    cross_source_tolerance: Fraction
    known_source_errors: list[KnownSourceError]
    cross_source_lookback_trading_days: PositiveInt
    balance_sheet_tolerance: Fraction
    revenue_yoy_abs_max: PositiveFloat


class TwMarketConfig(_Model):
    enabled: bool
    currency: Literal["TWD"]
    exchanges: dict[Literal["TWSE", "TPEX"], ExchangeConfig]
    price_limits: PriceLimitConfig
    filing_deadlines: FilingDeadlineConfig
    trading_calendar: TradingCalendarConfig
    known_suspensions: list[KnownSuspension]
    per_share: PerShareConfig
    industries: IndustryConfig
    listing: ListingConfig
    price_history_overrides: list[PriceHistoryOverride]
    benchmark: TwBenchmarkConfig
    risk_free: RiskFreeConfig
    industry_classification: str
    data_validation: DataValidationConfig


class UsBenchmarkConfig(_Model):
    primary: str


class UsMarketConfig(_Model):
    enabled: bool
    currency: Literal["USD"]
    benchmark: UsBenchmarkConfig


class MarketsConfig(_Model):
    tw: TwMarketConfig
    us: UsMarketConfig


# --- weighting & stability ----------------------------------------------------------------


class HalfLifeConfig(_Model):
    daily_returns: PositiveFloat
    monthly_revenue: PositiveFloat
    quarterly_financials: PositiveFloat
    annual_financials: PositiveFloat
    news_text: PositiveFloat


class WeightingConfig(_Model):
    half_life_days: HalfLifeConfig


class MinSamplesConfig(_Model):
    econ_observations: PositiveInt
    ttm_quarters: PositiveInt
    daily_observations: PositiveInt
    monthly_observations: PositiveInt
    low_sample_threshold: PositiveInt


class InternalSplitConfig(_Model):
    mode: Literal["midpoint", "fixed"]
    fixed_date: date | None

    @model_validator(mode="after")
    def _check_fixed_date(self) -> InternalSplitConfig:
        if self.mode == "fixed" and self.fixed_date is None:
            raise ValueError("internal_split.mode 'fixed' requires fixed_date")
        if self.mode == "midpoint" and self.fixed_date is not None:
            raise ValueError("internal_split.fixed_date must be null when mode is 'midpoint'")
        return self


class CrashExclusionConfig(_Model):
    price_start: date
    price_end: date
    annual_years: list[int]
    quarters: list[str]

    @model_validator(mode="after")
    def _check(self) -> CrashExclusionConfig:
        if self.price_end < self.price_start:
            raise ValueError("crash_exclusion.price_end is before price_start")
        bad = [q for q in self.quarters if not _QUARTER_RE.match(q)]
        if bad:
            raise ValueError(f"crash_exclusion.quarters must look like '2020Q1', got {bad}")
        return self


class BreakpointConfig(_Model):
    trimming: Annotated[float, Field(gt=0.0, lt=0.5)]


class StabilityConfig(_Model):
    relative_tolerance: PositiveFloat
    absolute_tolerance: dict[str, PositiveFloat]
    score_tolerance: Fraction
    alpha: Annotated[float, Field(gt=0.0, lt=1.0)]
    multiple_testing: Literal["benjamini_hochberg"]
    min_samples: MinSamplesConfig
    internal_split: InternalSplitConfig
    crash_exclusion: CrashExclusionConfig
    breakpoints: BreakpointConfig


# --- frameworks & scoring -----------------------------------------------------------------


class FrameworkDefinitions(_Model):
    recent_years: PositiveInt
    increasing_min_years: PositiveInt
    new_high_lookback_months: PositiveInt

    @model_validator(mode="after")
    def _check(self) -> FrameworkDefinitions:
        if self.increasing_min_years > self.recent_years:
            raise ValueError("increasing_min_years cannot exceed recent_years")
        return self


class WarrenChouConfig(_Model):
    max_recent_quarter_eps_yoy_decline: Fraction
    reference_yield_range: tuple[Fraction, Fraction]

    @model_validator(mode="after")
    def _check(self) -> WarrenChouConfig:
        low, high = self.reference_yield_range
        if low > high:
            raise ValueError("reference_yield_range must be [low, high]")
        return self


class GuyuConfig(_Model):
    core_profit_ratio_min: Fraction
    interest_coverage_min: PositiveFloat
    roe_min: float
    operating_margin_min: float
    pe_max: PositiveFloat
    dividend_yield_min: Fraction
    excluded_industries: list[str]
    expected_roe_by_industry: dict[str, PositiveFloat]


class ChenChungMingConfig(_Model):
    preferred_industries: list[str]


class BuffettConfig(_Model):
    roe_min: float
    debt_to_equity_max: PositiveFloat
    fcf_positive_year_ratio_min: Fraction


class GrahamConfig(_Model):
    pe_times_pb_max: PositiveFloat
    current_ratio_min: PositiveFloat


class LynchGrowthCategories(_Model):
    slow_grower_max: float
    stalwart_max: float

    @model_validator(mode="after")
    def _check(self) -> LynchGrowthCategories:
        if self.slow_grower_max >= self.stalwart_max:
            raise ValueError("slow_grower_max must be below stalwart_max")
        return self


class LynchConfig(_Model):
    peg_max: PositiveFloat
    growth_categories: LynchGrowthCategories


class MungerConfig(_Model):
    roic_min: float


class FrameworksConfig(_Model):
    min_years: PositiveInt
    original_years: PositiveInt
    definitions: FrameworkDefinitions
    warren_chou: WarrenChouConfig
    guyu: GuyuConfig
    chen_chung_ming: ChenChungMingConfig
    buffett: BuffettConfig
    graham: GrahamConfig
    lynch: LynchConfig
    munger: MungerConfig

    @model_validator(mode="after")
    def _check(self) -> FrameworksConfig:
        if self.min_years > self.original_years:
            raise ValueError("frameworks.min_years cannot exceed original_years")
        return self


FRAMEWORK_NAMES = frozenset(
    {"warren_chou", "guyu", "chen_chung_ming", "buffett", "graham", "lynch", "munger"}
)


class ScoringGroup(_Model):
    weight: Fraction
    members: dict[str, Fraction] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_member_sum(self) -> ScoringGroup:
        total = sum(self.members.values())
        if not math.isclose(total, self.weight, abs_tol=_WEIGHT_TOLERANCE):
            raise ValueError(f"member weights sum to {total}, expected group weight {self.weight}")
        return self


class ScoringConfig(_Model):
    groups: dict[str, ScoringGroup] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_groups(self) -> ScoringConfig:
        total = sum(g.weight for g in self.groups.values())
        if not math.isclose(total, 1.0, abs_tol=_WEIGHT_TOLERANCE):
            raise ValueError(f"group weights sum to {total}, expected 1.0")
        seen: list[str] = [m for g in self.groups.values() for m in g.members]
        duplicates = sorted({m for m in seen if seen.count(m) > 1})
        if duplicates:
            raise ValueError(f"frameworks assigned to more than one group: {duplicates}")
        unknown = sorted(set(seen) - FRAMEWORK_NAMES)
        if unknown:
            raise ValueError(f"unknown frameworks in scoring groups: {unknown}")
        missing = sorted(FRAMEWORK_NAMES - set(seen))
        if missing:
            raise ValueError(f"frameworks missing from scoring groups: {missing}")
        return self


# --- data sources, risk, technicals, llm, report ------------------------------------------


class FinMindConfig(_Model):
    max_requests_per_hour: PositiveInt
    # Batch export only: on an exhausted quota (HTTP 402) wait this long and retry.
    quota_retries: Annotated[int, Field(ge=0)]
    # First date of TaiwanStockCapitalReductionReferencePrice; earlier capital changes are
    # detected from the prices instead (see data/early_changes.py).
    capital_reduction_coverage_start: date
    quota_wait_minutes: PositiveFloat
    history_start: date
    price_history_start: date
    valuation_coverage_start: date
    # Institutional flows and margin balances are read from here (score part E).
    chips_history_start: date

    @model_validator(mode="after")
    def _check_order(self) -> FinMindConfig:
        if not self.history_start <= self.price_history_start:
            raise ValueError("price_history_start must not be before history_start")
        return self


class OfficialSourceConfig(_Model):
    min_request_interval_seconds: Annotated[float, Field(ge=0.0)]


class YahooSourceConfig(_Model):
    min_request_interval_seconds: Annotated[float, Field(ge=0.0)]


class DataSourcesConfig(_Model):
    cache_dir: Path
    finmind: FinMindConfig
    official: OfficialSourceConfig
    yahoo: YahooSourceConfig
    cache_max_age_hours: PositiveFloat
    request_timeout_seconds: PositiveFloat
    scrape_reference_sites: bool


class MajorEvent(_Model):
    name: str = Field(min_length=1)
    start: date
    end: date

    @model_validator(mode="after")
    def _check(self) -> MajorEvent:
        if self.end < self.start:
            raise ValueError(f"major event '{self.name}' ends before it starts")
        return self


class RiskConfig(_Model):
    var_confidence: Annotated[float, Field(gt=0.0, lt=1.0)]
    annualization_days: PositiveInt
    major_events: list[MajorEvent]


class MacdConfig(_Model):
    fast: PositiveInt
    slow: PositiveInt
    signal: PositiveInt

    @model_validator(mode="after")
    def _check(self) -> MacdConfig:
        if self.fast >= self.slow:
            raise ValueError("macd.fast must be shorter than macd.slow")
        return self


class TechnicalsConfig(_Model):
    sma_windows: list[PositiveInt] = Field(min_length=1)
    rsi_window: PositiveInt
    macd: MacdConfig


class LlmConfig(_Model):
    provider: Literal["anthropic"]
    model: str = Field(min_length=1)
    effort: Literal["low", "medium", "high", "xhigh", "max"]
    max_output: PositiveInt
    max_retries: Annotated[int, Field(ge=0)]
    log_dir: Path
    documents_dir: Path
    max_document_chars: PositiveInt
    min_document_chars: PositiveInt
    min_cjk_ratio: Fraction
    max_replacement_ratio: Fraction
    number_tolerance: Fraction


class ReferenceLink(_Model):
    name: str = Field(min_length=1)
    url: str = Field(pattern=r"^https://")


class ReportConfig(_Model):
    language: Literal["zh-TW"]
    output_dir: Path
    reference_links: list[ReferenceLink]


# --- validation (spec §14) -----------------------------------------------------------------

Aspect = Literal["revenue", "profitability", "risk"]


class DateRange(_Model):
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> DateRange:
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        return self


def add_months(day: date, months: int) -> date:
    """Same day ``months`` later; month ends map to month ends (2023-11-30 + 3 = 2024-02-29)."""
    month = day.month - 1 + months
    year, month = day.year + month // 12, month % 12 + 1
    last = (date(year + month // 12, month % 12 + 1, 1) - date.resolution).day
    is_month_end = (day + date.resolution).month != day.month
    return date(year, month, last if is_month_end else min(day.day, last))


class KnownCasesConfig(_Model):
    file: Path
    as_of: date


class WalkForwardConfig(_Model):
    horizon_months: PositiveInt
    dev: DateRange
    quarantine: DateRange
    holdout: DateRange
    universe: list[str] = Field(min_length=1)
    min_forward_returns: PositiveInt
    max_tuning_attempts: PositiveInt
    tunable: list[str]
    baseline_draws: PositiveInt
    bootstrap_draws: PositiveInt
    confidence: Annotated[float, Field(gt=0.5, lt=1.0)]
    alpha: Annotated[float, Field(gt=0.0, lt=0.5)]
    seed: int

    @model_validator(mode="after")
    def _phases(self) -> WalkForwardConfig:
        if not self.dev.end < self.quarantine.start <= self.quarantine.end < self.holdout.start:
            raise ValueError("phases must be ordered: dev < quarantine < holdout")
        if add_months(self.dev.end, self.horizon_months) > self.quarantine.end:
            raise ValueError(
                "dev outcomes would reach past the quarantine: dev.end + horizon_months "
                f"= {add_months(self.dev.end, self.horizon_months)} > {self.quarantine.end}"
            )
        if len(set(self.universe)) != len(self.universe):
            raise ValueError("walk_forward.universe has duplicates")
        return self

    def outcome_limit(self, phase: Literal["dev", "holdout"]) -> date:
        """Last date a phase's outcomes may use."""
        if phase == "dev":
            return self.quarantine.end
        return add_months(self.holdout.end, self.horizon_months)


class ForwardTrackingConfig(_Model):
    enabled: bool
    compare_after_months: list[PositiveInt]


class ExpertsConfig(_Model):
    kappa_min: Annotated[float, Field(ge=-1.0, le=1.0)]
    kappa_margin: Fraction


class SuccessConfig(_Model):
    known_cases_accuracy: Fraction
    report_score_margin: PositiveFloat


class ValidationConfig(_Model):
    directory: Path
    aspects: dict[Aspect, list[str]]
    known_cases: KnownCasesConfig
    walk_forward: WalkForwardConfig
    forward_tracking: ForwardTrackingConfig
    experts: ExpertsConfig
    success: SuccessConfig

    @model_validator(mode="after")
    def _all_aspects(self) -> ValidationConfig:
        missing = {"revenue", "profitability", "risk"} - set(self.aspects)
        if missing or any(not keys for keys in self.aspects.values()):
            raise ValueError(f"every aspect needs at least one metric (missing: {missing})")
        return self


# --- app export ---------------------------------------------------------------------------


class AppExportConfig(_Model):
    watchlist: list[Annotated[str, Field(pattern=r"^[1-9]\d{3}$")]] = Field(min_length=1)
    with_peers: bool
    stale_after_hours: PositiveInt

    @model_validator(mode="after")
    def _unique(self) -> AppExportConfig:
        if len(set(self.watchlist)) != len(self.watchlist):
            raise ValueError("app.watchlist has duplicates")
        return self


# --- backtest reference (user rule change 2026-10-17) ---------------------------------------


# --- trading costs (spec v2 §3) --------------------------------------------------------------


class CostsConfig(_Model):
    fee_rate: Fraction
    fee_discount: Annotated[float, Field(gt=0.0, le=1.0)]
    min_fee: Annotated[float, Field(ge=0.0)]
    tax_stock: Fraction
    tax_etf: Fraction
    board_lot: PositiveInt
    reference_position: PositiveFloat


# --- root ---------------------------------------------------------------------------------


class ScorePointsConfig(_Model):
    """Points of each part of the 0-100 base score (spec v2 §4.1)."""

    principles: PositiveInt  # A: investor-principle frameworks
    consistency: PositiveInt  # B: V1/V3 agreement on walk-forward-positive metrics
    valuation: PositiveInt  # C: P/E and P/B within the stock's own range
    risk: PositiveInt  # D: volatility and drawdown against the benchmark
    chips: PositiveInt  # E: institutional flows and margin use

    @model_validator(mode="after")
    def _check_total(self) -> ScorePointsConfig:
        total = self.principles + self.consistency + self.valuation + self.risk + self.chips
        if total != 100:
            raise ValueError(f"score points must add up to 100, got {total}")
        return self


class ConsistencyScoreConfig(_Model):
    walk_forward_results: Path
    full_share: Fraction  # this share of counted metrics STABLE -> full points
    partial_share: Fraction  # at least this share -> half points

    @model_validator(mode="after")
    def _check_order(self) -> ConsistencyScoreConfig:
        if self.partial_share > self.full_share:
            raise ValueError("consistency.partial_share must not exceed full_share")
        return self


class ValuationScoreConfig(_Model):
    lookback_years: PositiveInt
    min_years: PositiveFloat


class RiskScoreConfig(_Model):
    lookback_years: PositiveInt
    min_trading_days: PositiveInt
    ratio_full: PositiveFloat  # stock/benchmark at or below this: full points
    ratio_zero: PositiveFloat  # at or above this: none

    @model_validator(mode="after")
    def _check_order(self) -> RiskScoreConfig:
        if self.ratio_zero <= self.ratio_full:
            raise ValueError("risk.ratio_zero must exceed ratio_full")
        return self


class ChipsScoreConfig(_Model):
    lookback_trading_days: PositiveInt
    investors: Annotated[list[str], Field(min_length=1)]
    net_share_full: PositiveFloat  # net buying of this share of volume: full points
    margin_usage_zero: Annotated[float, Field(gt=0.0, le=1.0)]


class ScoreLabelConfig(_Model):
    min_score: Annotated[float, Field(ge=0.0, le=100.0)]
    name: str = Field(min_length=1)
    note: str | None = None


class CapitalTierConfig(_Model):
    capital: PositiveInt
    holdings: PositiveInt


class CapitalFactorConfig(_Model):
    cost_ratio_steps: Annotated[list[Fraction], Field(min_length=1)]
    cost_step_penalty: Fraction
    odd_lot_penalty: Fraction
    concentration: Fraction  # one position above this share of capital is too concentrated
    concentration_penalty: Fraction
    floor: Fraction


class ScoreCapsConfig(_Model):
    below_risk_free: Annotated[float, Field(ge=0.0, le=100.0)]
    below_etf: Annotated[float, Field(ge=0.0, le=100.0)]


class ResearchScoreConfig(_Model):
    """The 0-100 score of spec v2 §4 (base score, calibration caps, capital tiers, labels)."""

    points: ScorePointsConfig
    min_available_points: Annotated[int, Field(ge=1, le=100)]
    consistency: ConsistencyScoreConfig
    valuation: ValuationScoreConfig
    risk: RiskScoreConfig
    chips: ChipsScoreConfig
    capital_tiers: Annotated[list[CapitalTierConfig], Field(min_length=1)]
    default_capital: PositiveInt
    capital_factor: CapitalFactorConfig
    caps: ScoreCapsConfig
    labels: Annotated[list[ScoreLabelConfig], Field(min_length=1)]

    @model_validator(mode="after")
    def _check(self) -> ResearchScoreConfig:
        capitals = [t.capital for t in self.capital_tiers]
        if capitals != sorted(set(capitals)):
            raise ValueError("capital_tiers must be in increasing order of capital")
        if self.default_capital not in capitals:
            raise ValueError("default_capital must be one of the capital tiers")
        floors = [label.min_score for label in self.labels]
        if floors != sorted(floors, reverse=True) or floors[-1] != 0:
            raise ValueError("labels must go from the highest min_score down to 0")
        return self


class CalibrationConfig(_Model):
    """Full-market calibration of the base score (spec v2 §4.2)."""

    output: Path
    work_dir: Path  # per-stage parquet files (derived values only)
    cache_dir: Path
    cache_max_age_days: PositiveInt
    # The run reads every source as of this fixed date, so runs on different days share the
    # cache and give the same result.
    data_as_of: date
    start: date  # first month-end base date
    last_base_date: date
    horizon_months: PositiveInt
    out_of_sample_start: date
    liquidity_lookback_trading_days: PositiveInt
    liquidity_keep_share: Fraction  # keep this share of the most traded stocks each month
    groups: Annotated[int, Field(ge=2)]
    min_rank_correlation: Annotated[float, Field(ge=-1.0, le=1.0)]
    confidence: Fraction
    bootstrap_draws: PositiveInt
    seed: int
    shards: PositiveInt
    excluded_code_prefixes: list[str]

    @model_validator(mode="after")
    def _check_order(self) -> CalibrationConfig:
        if not self.start < self.out_of_sample_start <= self.last_base_date:
            raise ValueError("calibration needs start < out_of_sample_start <= last_base_date")
        if add_months(self.last_base_date, self.horizon_months) > self.data_as_of:
            raise ValueError("calibration.last_base_date + horizon must not pass data_as_of")
        return self


class AppConfig(_Model):
    general: GeneralConfig
    markets: MarketsConfig
    weighting: WeightingConfig
    stability: StabilityConfig
    frameworks: FrameworksConfig
    scoring: ScoringConfig
    data_sources: DataSourcesConfig
    risk: RiskConfig
    technicals: TechnicalsConfig
    llm: LlmConfig
    report: ReportConfig
    validation: ValidationConfig
    app: AppExportConfig
    costs: CostsConfig
    research_score: ResearchScoreConfig
    calibration: CalibrationConfig

    @model_validator(mode="after")
    def _check_cross_section(self) -> AppConfig:
        split = self.general.regime_split_date
        if self.general.ifrs_start_year >= split.year:
            raise ValueError("ifrs_start_year must be before the regime split year")
        crash = self.stability.crash_exclusion
        if crash.price_start < split:
            raise ValueError("crash_exclusion must fall inside V1 (on or after regime_split_date)")
        fixed = self.stability.internal_split.fixed_date
        if fixed is not None and fixed <= split:
            raise ValueError("internal_split.fixed_date must be after regime_split_date")
        groups = set(self.markets.tw.industries.groups)
        fw = self.frameworks
        used = {
            *fw.guyu.excluded_industries,
            *fw.guyu.expected_roe_by_industry,
            *fw.chen_chung_ming.preferred_industries,
            "financial",
        }
        if used - groups:
            raise ValueError(f"industry groups used by frameworks are undefined: {used - groups}")
        if not getattr(self.markets, self.general.default_market).enabled:
            raise ValueError(f"default_market '{self.general.default_market}' is not enabled")
        return self


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and validate the configuration file, raising ``ConfigError`` on any problem."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file is not valid YAML: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config file must contain a mapping at the top level: {path}")
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config {path}:\n{exc}") from exc


# --- secrets ------------------------------------------------------------------------------


class MissingSecretError(Exception):
    """Raised when a secret is needed but not set in the environment or ``.env``."""


class Secrets:
    """API keys read from the environment (optionally populated from ``.env``).

    Secrets are never part of ``AppConfig``, never printed and only checked when used,
    so commands that do not need them (e.g. ``config show``) work without a ``.env``.
    """

    FINMIND_TOKEN = "FINMIND_TOKEN"
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    NAMES = (FINMIND_TOKEN, ANTHROPIC_API_KEY)

    def __init__(self, env_file: Path | str | None = ".env") -> None:
        if env_file is not None and Path(env_file).is_file():
            load_dotenv(env_file, override=False)

    def get(self, name: str) -> str:
        if name not in self.NAMES:
            raise KeyError(f"unknown secret: {name}")
        value = os.environ.get(name, "").strip()
        if not value:
            raise MissingSecretError(f"{name} is not set; add it to .env (see .env.example)")
        return value

    def is_set(self, name: str) -> bool:
        return bool(os.environ.get(name, "").strip())

    def __repr__(self) -> str:
        status = ", ".join(f"{n}={'set' if self.is_set(n) else 'unset'}" for n in self.NAMES)
        return f"Secrets({status})"
