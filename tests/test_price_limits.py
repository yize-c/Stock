from __future__ import annotations

from datetime import date

import pytest

from stock_research.config import load_config
from stock_research.markets.price_limits import (
    PriceLimitStatus,
    PriceMove,
    ReferencePriceKind,
    check_price_move,
    limit_on,
)
from tests.conftest import DEFAULT_CONFIG

RULES = load_config(DEFAULT_CONFIG).markets.tw.price_limits

BEFORE = date(2015, 5, 29)  # last trading day under the 7% limit
AFTER = date(2015, 6, 1)  # first day of the 10% limit


def move(day: date, close: float, reference: float = 100.0, **kwargs) -> PriceMove:
    return PriceMove(day=day, reference_price=reference, close=close, **kwargs)


def test_limit_changes_on_2015_06_01() -> None:
    assert limit_on(date(2000, 1, 4), RULES) == pytest.approx(0.07)
    assert limit_on(BEFORE, RULES) == pytest.approx(0.07)
    assert limit_on(AFTER, RULES) == pytest.approx(0.10)
    assert limit_on(date(2026, 10, 15), RULES) == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("day", "close", "status"),
    [
        (BEFORE, 107.0, PriceLimitStatus.WITHIN),  # exactly at 7%
        (BEFORE, 107.4, PriceLimitStatus.WITHIN),  # inside 0.5% rounding tolerance
        (BEFORE, 93.0, PriceLimitStatus.WITHIN),
        (BEFORE, 109.0, PriceLimitStatus.EXCEEDED),  # 9% is too much before 2015-06-01
        (BEFORE, 92.0, PriceLimitStatus.EXCEEDED),
        (AFTER, 109.0, PriceLimitStatus.WITHIN),  # ...but fine from 2015-06-01
        (AFTER, 110.0, PriceLimitStatus.WITHIN),
        (AFTER, 110.4, PriceLimitStatus.WITHIN),
        (AFTER, 110.6, PriceLimitStatus.EXCEEDED),
        (AFTER, 89.4, PriceLimitStatus.EXCEEDED),
    ],
)
def test_limit_with_tolerance(day: date, close: float, status: PriceLimitStatus) -> None:
    assert check_price_move(move(day, close), RULES).status is status


def test_result_reports_change_and_limit() -> None:
    result = check_price_move(move(AFTER, 112.0), RULES)
    assert result.change == pytest.approx(0.12)
    assert result.limit == pytest.approx(0.10)
    assert result.allowed == pytest.approx(0.105)
    assert "+12.00%" in result.reason


@pytest.mark.parametrize("day_since_listing", [1, 2, 3, 4, 5])
def test_first_five_listing_days_are_exempt(day_since_listing: int) -> None:
    result = check_price_move(
        move(AFTER, 150.0, trading_day_since_listing=day_since_listing), RULES
    )
    assert result.status is PriceLimitStatus.EXEMPT
    assert result.limit is None


def test_sixth_listing_day_is_limited() -> None:
    result = check_price_move(move(AFTER, 150.0, trading_day_since_listing=6), RULES)
    assert result.status is PriceLimitStatus.EXCEEDED


def test_ex_dividend_day_measured_against_reference_price() -> None:
    # Previous close 100, cash dividend 15 -> ex-dividend reference price 85. A close of 84
    # is -1.2% against the reference, but would look like -16% against the previous close.
    ex_day = move(
        AFTER, 84.0, reference=85.0, reference_kind=ReferencePriceKind.EX_RIGHTS_REFERENCE
    )
    assert check_price_move(ex_day, RULES).status is PriceLimitStatus.WITHIN
    assert check_price_move(move(AFTER, 84.0, reference=100.0), RULES).status is (
        PriceLimitStatus.EXCEEDED
    )


def test_resumption_after_capital_change_is_exempt() -> None:
    result = check_price_move(move(AFTER, 180.0, resumed_after_capital_change=True), RULES)
    assert result.status is PriceLimitStatus.EXEMPT
    assert "capital reduction" in result.reason


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_price": 0.0, "close": 10.0},
        {"reference_price": 10.0, "close": 0.0},
        {"reference_price": -1.0, "close": 10.0},
        {"reference_price": 10.0, "close": 10.0, "trading_day_since_listing": 0},
    ],
)
def test_invalid_move_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        PriceMove(day=AFTER, **kwargs)
