"""Trading costs (spec v2 §3), including every row of the spec's table."""

from __future__ import annotations

import pytest

from stock_research.analysis.costs import (
    cost_ratio,
    net_return,
    position_shares,
    round_trip_cost,
)
from stock_research.config import load_config
from tests.conftest import DEFAULT_CONFIG

COSTS = load_config(DEFAULT_CONFIG).costs


@pytest.mark.parametrize(
    ("amount", "total", "ratio"),
    [
        (5_000, 55.0, 0.0110),
        (10_000, 70.0, 0.0070),
        (50_000, 292.5, 0.00585),  # the spec's table rounds to 293 and 0.59%
        (100_000, 585.0, 0.00585),
    ],
)
def test_spec_table(amount, total, ratio) -> None:
    # A NT$50 stock: the amount is 50 x shares.
    assert round_trip_cost(50.0, amount // 50, False, COSTS) == pytest.approx(total)
    assert cost_ratio(amount, False, COSTS) == pytest.approx(ratio)
    assert round(cost_ratio(amount, False, COSTS), 4) == round(ratio, 4)


def test_minimum_fee_dominates_below_about_14000() -> None:
    # 20 / (0.001425) = 14,035: below it the NT$20 minimum sets the fee.
    assert cost_ratio(14_000, False, COSTS) > cost_ratio(14_100, False, COSTS)
    assert cost_ratio(14_100, False, COSTS) == pytest.approx(cost_ratio(200_000, False, COSTS))


def test_etf_tax_and_broker_discount() -> None:
    assert round_trip_cost(100.0, 1000, True, COSTS) == pytest.approx(2 * 142.5 + 100)
    discounted = COSTS.model_copy(update={"fee_discount": 0.6})
    assert round_trip_cost(100.0, 1000, False, discounted) == pytest.approx(2 * 85.5 + 300)


def test_net_return_charges_both_sides() -> None:
    assert net_return(0.0, 100_000, False, COSTS) == pytest.approx(
        (100_000 - 142.5 - 300) / (100_000 + 142.5) - 1
    )
    gross = 0.10
    sell = 110_000
    expected = (sell - sell * 0.001425 - sell * 0.003) / (100_000 + 142.5) - 1
    assert net_return(gross, 100_000, False, COSTS) == pytest.approx(expected)
    assert net_return(gross, 100_000, False, COSTS) < gross


def test_odd_lots_when_a_board_lot_does_not_fit() -> None:
    whole = position_shares(100.0, 250_000, COSTS)
    assert (whole.shares, whole.odd_lot, whole.note) == (2000, False, None)
    odd = position_shares(1000.0, 50_000, COSTS)
    assert (odd.shares, odd.odd_lot, odd.note) == (50, True, "零股流動性較差")
