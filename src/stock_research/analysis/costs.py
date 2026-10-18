"""Taiwan trading costs (spec v2 §3): broker fee on both sides, transaction tax on the sell.

``round_trip_cost`` follows the spec's formula on one amount (the table in the spec uses it);
``net_return`` applies the same rules to an actual round trip, where the sell amount includes
the gain or loss. Every return a report presents as cost-adjusted uses ``net_return``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from stock_research.config import CostsConfig


def _fee(amount: float, costs: CostsConfig) -> float:
    return max(costs.min_fee, amount * costs.fee_rate * costs.fee_discount)


def broker_fee(amount: float, costs: CostsConfig) -> float:
    """The broker fee on one order of ``amount`` (either side)."""
    return _fee(amount, costs)


def round_trip_cost(price: float, shares: int, is_etf: bool, costs: CostsConfig) -> float:
    """Two broker fees plus the sell-side tax, all on ``price * shares``."""
    amount = price * shares
    tax = amount * (costs.tax_etf if is_etf else costs.tax_stock)
    return 2 * _fee(amount, costs) + tax


def cost_ratio(amount: float, is_etf: bool, costs: CostsConfig) -> float:
    """Round-trip cost as a fraction of ``amount``."""
    return round_trip_cost(amount, 1, is_etf, costs) / amount


def net_return(gross: float, amount: float, is_etf: bool, costs: CostsConfig) -> float:
    """Return after buying ``amount`` and selling at ``amount * (1 + gross)``."""
    tax_rate = costs.tax_etf if is_etf else costs.tax_stock
    paid = amount + _fee(amount, costs)
    proceeds = amount * (1 + gross)
    received = proceeds - _fee(proceeds, costs) - proceeds * tax_rate
    return received / paid - 1


@dataclass(frozen=True)
class Position:
    shares: int
    odd_lot: bool  # True: less than one board lot fits, bought as odd lots

    @property
    def note(self) -> str | None:
        return "零股流動性較差" if self.odd_lot else None


def position_shares(price: float, allocation: float, costs: CostsConfig) -> Position:
    """Whole board lots when one fits in ``allocation``, otherwise odd-lot shares."""
    lot_price = price * costs.board_lot
    if lot_price <= allocation:
        return Position(math.floor(allocation / lot_price) * costs.board_lot, False)
    return Position(math.floor(allocation / price), True)
