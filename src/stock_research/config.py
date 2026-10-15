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


class TwBenchmarkConfig(_Model):
    primary: str
    fallback_ticker: str
    fallback_uses_adjusted_prices: bool


class RiskFreeRate(_Model):
    start: date
    end: date
    rate: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def _check_range(self) -> RiskFreeRate:
        if self.end < self.start:
            raise ValueError(f"risk-free period ends before it starts: {self.start}..{self.end}")
        return self


class RiskFreeConfig(_Model):
    source: str
    fallback_rates: list[RiskFreeRate]

    @model_validator(mode="after")
    def _check_no_overlap(self) -> RiskFreeConfig:
        periods = sorted(self.fallback_rates, key=lambda r: r.start)
        for prev, cur in zip(periods, periods[1:], strict=False):
            if cur.start <= prev.end:
                raise ValueError(f"risk-free fallback periods overlap at {cur.start}")
        return self


class DataValidationConfig(_Model):
    cross_source_tolerance: Fraction
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
    temperature: Annotated[float, Field(ge=0.0, le=1.0)]
    max_retries: Annotated[int, Field(ge=0)]
    log_dir: Path


class ReportConfig(_Model):
    language: Literal["zh-TW"]
    output_dir: Path


# --- root ---------------------------------------------------------------------------------


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
