"""Fundamental metrics from point-in-time statements (spec §4.4, §6).

Pipeline, all on the long statement rows from ``MarketData.financials``:

1. ``quarterly_table``: one row per quarter end with single-quarter flows (cash-flow items are
   de-cumulated from year-to-date values) and quarter-end balances.
2. ``ttm_table`` / ``annual_table``: trailing-four-quarter and fiscal-year figures plus ratios.
   A TTM row's period runs from the first day of its first quarter to its last quarter end,
   so ``TimeWindow.filter_financials`` drops TTM rows that straddle the regime split. Balance
   averages use only the quarter ends inside the period, so nothing leaks across windows.
3. ``summarize``: weighted and unweighted mean of a metric inside one window.
4. ``valuation_snapshot``: single-date metrics (P/E, P/B, yield, latest-quarter YoY) on the
   share basis of that date.

Definitions (TTM unless noted; ratios are fractions and NaN when the denominator is not
positive, so the framework layer decides between "not applicable" and "insufficient data"):

* ROE = net income attributable to parent / mean parent equity at the period's quarter ends.
* ROIC = operating income x (1 - effective tax rate) / mean (total equity + borrowings).
  Borrowings are the short-term, long-term and bond lines FinMind reports; current portions
  folded into other liabilities are not visible, so ROIC may be slightly overstated.
* Free cash flow = operating cash flow - purchases of property, plant and equipment.
* Interest coverage = (pre-tax income + interest expense) / interest expense; interest
  expense comes from the cash-flow statement's add-back line. Zero interest gives +inf.
* Core profit ratio = operating income / pre-tax income.
* EPS is restated to one share basis with ``ShareBasis`` before any comparison over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from stock_research.analysis.results import InsufficientHistory
from stock_research.analysis.weighting import WeightedStat, weighted_mean
from stock_research.data.errors import MissingDataError
from stock_research.data.shares import ShareBasis
from stock_research.data.windows import TimeWindow

INCOME_ITEMS = {
    "Revenue": "revenue",
    "GrossProfit": "gross_profit",
    "OperatingIncome": "operating_income",
    "PreTaxIncome": "pretax_income",
    "TAX": "income_tax",
    "IncomeAfterTaxes": "net_income",
    "EquityAttributableToOwnersOfParent": "net_income_parent",
    "EPS": "eps",
}
CASH_FLOW_ITEMS = {
    "CashFlowsFromOperatingActivities": "operating_cash_flow",
    "PropertyAndPlantAndEquipment": "ppe_purchases",  # reported as a negative outflow
    "InterestExpense": "interest_expense",
}
BALANCE_ITEMS = {
    "TotalAssets": "total_assets",
    "Liabilities": "total_liabilities",
    "Equity": "total_equity",
    "EquityAttributableToOwnersOfParent": "parent_equity",
    "CurrentAssets": "current_assets",
    "CurrentLiabilities": "current_liabilities",
    "CashAndCashEquivalents": "cash",
    "OrdinaryShare": "ordinary_share_capital",
}
# Alternative names some industries use (banks and insurers). The first name present wins.
INCOME_FALLBACKS = {"net_income": ("IncomeAfterTax",)}
# Lines a company without such debt simply does not report; absent means zero.
BORROWING_ITEMS = ("ShorttermBorrowings", "LongtermBorrowings", "BondsPayable")

FLOW_COLUMNS = [*INCOME_ITEMS.values(), *CASH_FLOW_ITEMS.values()]
BALANCE_COLUMNS = [*BALANCE_ITEMS.values(), "borrowings"]


def _quarter_start(end: date) -> date:
    return date(end.year, end.month - 2, 1)


def _previous_quarter_end(end: date) -> date:
    return _quarter_start(end) - timedelta(days=1)


def quarterly_table(financials: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Single-quarter flows and quarter-end balances, indexed by quarter end."""
    rows = financials.assign(period_end=pd.to_datetime(financials["period_end"]).dt.date)
    available = rows.groupby("period_end")["available_date"].max()

    def wide(statement: str, items: dict[str, str]) -> pd.DataFrame:
        part = rows.loc[(rows["statement"] == statement) & rows["item"].isin(list(items))]
        return part.pivot(index="period_end", columns="item", values="value").rename(columns=items)

    income = wide("income", INCOME_ITEMS)
    for column, names in INCOME_FALLBACKS.items():
        alternative = wide("income", {n: column for n in names})
        if column in alternative:
            current = income[column] if column in income else pd.Series(dtype=float)
            income[column] = current.combine_first(alternative[column])
    cash = _decumulate(wide("cash_flow", CASH_FLOW_ITEMS))
    balance = wide("balance", BALANCE_ITEMS)
    borrow = rows.loc[(rows["statement"] == "balance") & rows["item"].isin(BORROWING_ITEMS)]
    if not balance.empty:
        totals = borrow.groupby("period_end")["value"].sum()
        balance["borrowings"] = totals.reindex(balance.index).fillna(0.0)
        # Banks report no parent-equity line: total equity minus non-controlling interests
        # (a missing NCI line means there are none).
        nci = wide("balance", {"NoncontrollingInterests": "nci"})
        nci = nci["nci"] if "nci" in nci else pd.Series(dtype=float)
        derived = balance["total_equity"] - nci.reindex(balance.index).fillna(0.0)
        current = balance.get("parent_equity", pd.Series(np.nan, index=balance.index))
        balance["parent_equity"] = current.combine_first(derived)
    table = pd.concat([income, cash, balance], axis=1).sort_index()
    for col in [*FLOW_COLUMNS, *BALANCE_COLUMNS]:
        if col not in table:
            table[col] = np.nan
    table["available_date"] = pd.to_datetime(available.reindex(table.index)).dt.date
    table.index.name = "period_end"
    return table


def _decumulate(ytd: pd.DataFrame) -> pd.DataFrame:
    """Year-to-date values -> single quarters. A missing earlier quarter leaves NaN."""
    out = ytd.copy()
    for end in ytd.index:
        if end.month == 3:
            continue
        prior = _previous_quarter_end(end)
        if prior in ytd.index:
            out.loc[end] = ytd.loc[end] - ytd.loc[prior]
        else:
            out.loc[end] = np.nan
    return out


def _consecutive(ends: list[date]) -> bool:
    return all(_previous_quarter_end(b) == a for a, b in zip(ends, ends[1:], strict=False))


def ttm_table(quarterly: pd.DataFrame, basis: ShareBasis | None, basis_date: date) -> pd.DataFrame:
    """Trailing-four-quarter figures and ratios, one row per quarter end.

    ``eps`` is the sum of four quarterly EPS restated to the share basis of ``basis_date``
    (unrestated when ``basis`` is None, e.g. no share-changing events).
    """
    eps = quarterly["eps"].astype(float)
    if basis is not None:
        eps = basis.to_basis(eps, pd.Series(quarterly.index, index=quarterly.index), basis_date)
    rows = []
    ends = list(quarterly.index)
    for i in range(3, len(ends)):
        span = ends[i - 3 : i + 1]
        if not _consecutive(span):
            continue
        q = quarterly.loc[span]
        flows = q[FLOW_COLUMNS].sum(min_count=4)
        averages = q[BALANCE_COLUMNS].mean()
        row = {
            "period_start": _quarter_start(span[0]),
            "period_end": span[-1],
            "available_date": max(q["available_date"]),
            **{c: flows[c] for c in FLOW_COLUMNS},
            "eps": float(eps.loc[span].sum(min_count=4)),
            **{f"avg_{c}": averages[c] for c in BALANCE_COLUMNS},
            **{c: q[c].iloc[-1] for c in BALANCE_COLUMNS},
        }
        rows.append(_ratios(row))
    return pd.DataFrame(rows)


def annual_table(
    quarterly: pd.DataFrame, basis: ShareBasis | None, basis_date: date
) -> pd.DataFrame:
    """Fiscal-year figures: TTM rows ending in December."""
    ttm = ttm_table(quarterly, basis, basis_date)
    if ttm.empty:
        return ttm
    annual = ttm.loc[[d.month == 12 for d in ttm["period_end"]]].copy()
    annual["year"] = [d.year for d in annual["period_end"]]
    return annual.reset_index(drop=True)


def _div(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return float(numerator) / float(denominator)


def _ratios(row: dict) -> dict:
    tax_rate = _div(row["income_tax"], row["pretax_income"])
    nopat = (
        row["operating_income"] * (1.0 - min(max(tax_rate, 0.0), 1.0))
        if not pd.isna(tax_rate)
        else np.nan
    )
    capex = -row["ppe_purchases"] if not pd.isna(row["ppe_purchases"]) else np.nan
    interest = row["interest_expense"]
    if pd.isna(interest) or pd.isna(row["pretax_income"]):
        coverage = np.nan
    elif interest == 0:
        coverage = np.inf
    else:
        coverage = _div(row["pretax_income"] + interest, interest)
    row.update(
        gross_margin=_div(row["gross_profit"], row["revenue"]),
        operating_margin=_div(row["operating_income"], row["revenue"]),
        net_margin=_div(row["net_income"], row["revenue"]),
        roe=_div(row["net_income_parent"], row["avg_parent_equity"]),
        roic=_div(nopat, row["avg_total_equity"] + row["avg_borrowings"]),
        free_cash_flow=row["operating_cash_flow"] - capex,
        interest_coverage=coverage,
        core_profit_ratio=_div(row["operating_income"], row["pretax_income"]),
        current_ratio=_div(row["current_assets"], row["current_liabilities"]),
        liabilities_to_equity=_div(row["total_liabilities"], row["total_equity"]),
        borrowings_to_equity=_div(row["borrowings"], row["total_equity"]),
    )
    return row


def summarize(
    table: pd.DataFrame,
    column: str,
    window: TimeWindow,
    half_life_days: float,
    min_observations: int,
) -> WeightedStat | InsufficientHistory:
    """Weighted/unweighted mean of ``column`` over the rows inside ``window``.

    Infinite values (e.g. interest coverage without interest) are left out of the mean.
    Fewer than ``min_observations`` values gives ``InsufficientHistory``.
    """
    inside = window.filter_financials(table)
    values = inside.set_index("period_end")[column].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < min_observations:
        return InsufficientHistory(
            window.name, column, f"{len(values)} value(s), need {min_observations}"
        )
    return weighted_mean(values, window, half_life_days)


@dataclass(frozen=True, slots=True)
class Valuation:
    on: date
    price: float  # raw close: the price a reader sees
    price_date: date
    eps_ttm: float
    eps_period_end: date
    book_value_per_share: float
    pe: float  # NaN when EPS <= 0
    pb: float  # NaN when equity <= 0
    dividend_yield: float
    latest_quarter: date
    latest_quarter_eps_yoy: float  # NaN when last year's EPS <= 0


def valuation_snapshot(
    ticker: str,
    on: date,
    raw_prices: pd.DataFrame,
    quarterly: pd.DataFrame,
    dividends: pd.DataFrame,
    basis: ShareBasis,
) -> Valuation:
    """Single-date metrics using only figures published by ``on``, on ``on``'s share basis."""
    traded = raw_prices.loc[raw_prices["close"].notna() & (raw_prices["date"] <= on)]
    if traded.empty:
        raise MissingDataError(ticker, "close", "none", f"a raw close on or before {on}")
    price_row = traded.iloc[-1]
    price = float(price_row["close"])

    published = quarterly.loc[[a <= on for a in quarterly["available_date"]]]
    ttm = ttm_table(published, basis, on)
    if ttm.empty or pd.isna(ttm["eps"].iloc[-1]):
        raise MissingDataError(ticker, "eps_ttm", "none", f"four published quarters by {on}")
    last = ttm.iloc[-1]

    end = last["period_end"]
    shares = last["ordinary_share_capital"] / basis.par_on(end) * basis.factor(end, on)
    bvps = _div(last["parent_equity"], shares) if last["parent_equity"] > 0 else np.nan

    year_ago = date(end.year - 1, end.month, end.day)  # quarter ends exist every year
    eps = basis.to_basis(published["eps"], pd.Series(published.index, index=published.index), on)
    prior = eps.get(year_ago, np.nan)
    eps_yoy = eps[end] / prior - 1.0 if not pd.isna(prior) and prior > 0 else np.nan

    paid = dividends.loc[dividends["cash_ex_date"].notna()]
    window = paid.loc[[on - timedelta(days=365) < d <= on for d in paid["cash_ex_date"]]]
    cash = basis.to_basis(
        window["cash_dividend"], window["cash_ex_date"], on, include_start=True
    ).sum()
    return Valuation(
        on=on,
        price=price,
        price_date=price_row["date"],
        eps_ttm=float(last["eps"]),
        eps_period_end=end,
        book_value_per_share=bvps,
        pe=_div(price, last["eps"]),
        pb=_div(price, bvps),
        dividend_yield=float(cash) / price,
        latest_quarter=end,
        latest_quarter_eps_yoy=eps_yoy,
    )
