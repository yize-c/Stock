"""Daily price-limit rules for Taiwan stocks.

The limit applies to the *raw* close relative to the day's reference price: normally the
previous raw close, but on an ex-rights/ex-dividend day the exchange's ex-rights reference
price. Callers pass the correct reference price; this module never adjusts prices itself.

Exemptions: the first ``new_listing_exempt_days`` trading days after listing, and the first
trading day after resuming from a capital reduction, split or par-value change. After traded
days whose price the source lacks (``days_without_price``), each of which could have moved by
the limit, the limit compounds over those days and the checked one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from stock_research.config import PriceLimitConfig


class PriceLimitStatus(StrEnum):
    WITHIN = "WITHIN"
    EXCEEDED = "EXCEEDED"
    EXEMPT = "EXEMPT"


class ReferencePriceKind(StrEnum):
    PREVIOUS_CLOSE = "PREVIOUS_CLOSE"
    EX_RIGHTS_REFERENCE = "EX_RIGHTS_REFERENCE"


@dataclass(frozen=True, slots=True)
class PriceMove:
    """One day's raw close and the reference price the limit is measured against."""

    day: date
    reference_price: float
    close: float
    reference_kind: ReferencePriceKind = ReferencePriceKind.PREVIOUS_CLOSE
    # 1 = listing day. None when unknown/irrelevant (long-listed stock).
    trading_day_since_listing: int | None = None
    resumed_after_capital_change: bool = False
    # Traded days between the reference and this close with no price in the source.
    days_without_price: int = 0

    def __post_init__(self) -> None:
        if not self.reference_price > 0:
            raise ValueError(f"reference_price must be > 0, got {self.reference_price}")
        if not self.close > 0:
            raise ValueError(f"close must be > 0, got {self.close}")
        if self.trading_day_since_listing is not None and self.trading_day_since_listing < 1:
            raise ValueError("trading_day_since_listing starts at 1 (the listing day)")


@dataclass(frozen=True, slots=True)
class PriceLimitResult:
    status: PriceLimitStatus
    change: float
    limit: float | None
    allowed: float | None
    reason: str


def limit_on(day: date, config: PriceLimitConfig) -> float:
    """The daily limit in force on ``day`` (e.g. 0.07 before 2015-06-01, 0.10 from then)."""
    current = config.rules[0].limit
    for rule in config.rules[1:]:
        assert rule.effective_from is not None  # guaranteed by config validation
        if day >= rule.effective_from:
            current = rule.limit
        else:
            break
    return current


def check_price_move(move: PriceMove, config: PriceLimitConfig) -> PriceLimitResult:
    change = move.close / move.reference_price - 1.0

    if (
        move.trading_day_since_listing is not None
        and move.trading_day_since_listing <= config.new_listing_exempt_days
    ):
        return PriceLimitResult(
            PriceLimitStatus.EXEMPT,
            change,
            None,
            None,
            f"listing trading day {move.trading_day_since_listing} "
            f"(first {config.new_listing_exempt_days} days have no limit)",
        )
    if move.resumed_after_capital_change:
        return PriceLimitResult(
            PriceLimitStatus.EXEMPT,
            change,
            None,
            None,
            "first trading day after capital reduction, split or par-value change",
        )

    limit = limit_on(move.day, config)
    days = move.days_without_price + 1
    up = (1 + limit) ** days - 1 + config.rounding_tolerance
    down = 1 - (1 - limit) ** days + config.rounding_tolerance
    allowed = up if change > 0 else down
    status = PriceLimitStatus.EXCEEDED if abs(change) > allowed else PriceLimitStatus.WITHIN
    reason = (
        f"{change:+.2%} vs {move.reference_kind.value.lower().replace('_', ' ')} "
        f"(limit ±{limit:.0%} + {config.rounding_tolerance:.1%} tolerance"
        + (f", compounded over {days} days without a price" if days > 1 else "")
        + ")"
    )
    return PriceLimitResult(status, change, limit, allowed, reason)
