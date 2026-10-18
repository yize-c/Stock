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
    "CostOfGoodsSold": "cost_of_revenue",
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
    "Depreciation": "depreciation",
}
BALANCE_ITEMS = {
    "TotalAssets": "total_assets",
    "Liabilities": "total_liabilities",
    "Equity": "total_equity",
    "EquityAttributableToOwnersOfParent": "parent_equity",
    "CurrentAssets": "current_assets",
    "CurrentLiabilities": "current_liabilities",
    "CashAndCashEquivalents": "cash",
    "Inventories": "inventories",
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

    Only four consecutive quarters form a TTM row; a flow missing in any of them leaves the
    sum NaN. ``eps`` is the sum of four quarterly EPS restated to the share basis of
    ``basis_date`` (unrestated when ``basis`` is None).
    """
    ends = list(quarterly.index)
    if len(ends) < 4:
        return pd.DataFrame()
    # Columns a table lacks (e.g. one built before an item was read) are NaN.
    quarterly = quarterly.reindex(
        columns=[*quarterly.columns.union(FLOW_COLUMNS + BALANCE_COLUMNS, sort=False)]
    )
    eps = quarterly["eps"].astype(float)
    if basis is not None:
        eps = basis.to_basis(eps, pd.Series(ends, index=quarterly.index), basis_date)
    flows = quarterly[FLOW_COLUMNS].astype(float).rolling(4, min_periods=4).sum()
    averages = quarterly[BALANCE_COLUMNS].astype(float).rolling(4, min_periods=1).mean()
    ordinals = pd.Series(
        [d.toordinal() for d in quarterly["available_date"]], index=quarterly.index
    )
    available = ordinals.rolling(4).max()
    consecutive = [i >= 3 and _consecutive(ends[i - 3 : i + 1]) for i in range(len(ends))]
    frame = pd.DataFrame(
        {
            "period_start": [
                _quarter_start(ends[i - 3]) if i >= 3 else None for i in range(len(ends))
            ],
            "period_end": ends,
            "available_date": [
                date.fromordinal(int(a)) if not np.isnan(a) else None for a in available
            ],
        },
        index=quarterly.index,
    )
    frame[FLOW_COLUMNS] = flows
    frame["eps"] = eps.rolling(4, min_periods=4).sum()
    for col in BALANCE_COLUMNS:
        frame[f"avg_{col}"] = averages[col]
        frame[col] = quarterly[col].astype(float)
    frame = frame.loc[consecutive].reset_index(drop=True)
    return _ratios(frame)


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


def _div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Ratio; NaN where the denominator is not positive or either side is missing."""
    return (numerator / denominator.where(denominator > 0)).astype(float)


def _ratios(frame: pd.DataFrame) -> pd.DataFrame:
    tax_rate = _div(frame["income_tax"], frame["pretax_income"]).clip(0.0, 1.0)
    nopat = frame["operating_income"] * (1.0 - tax_rate)
    interest = frame["interest_expense"]
    coverage = (frame["pretax_income"] + interest) / interest.where(interest > 0)
    coverage = coverage.mask(interest == 0, np.inf).mask(frame["pretax_income"].isna())
    return frame.assign(
        gross_margin=_div(frame["gross_profit"], frame["revenue"]),
        operating_margin=_div(frame["operating_income"], frame["revenue"]),
        net_margin=_div(frame["net_income"], frame["revenue"]),
        roe=_div(frame["net_income_parent"], frame["avg_parent_equity"]),
        roic=_div(nopat, frame["avg_total_equity"] + frame["avg_borrowings"]),
        free_cash_flow=frame["operating_cash_flow"] + frame["ppe_purchases"],
        interest_coverage=coverage,
        core_profit_ratio=_div(frame["operating_income"], frame["pretax_income"]),
        current_ratio=_div(frame["current_assets"], frame["current_liabilities"]),
        liabilities_to_equity=_div(frame["total_liabilities"], frame["total_equity"]),
        borrowings_to_equity=_div(frame["borrowings"], frame["total_equity"]),
    )


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
    *,
    unrestated_ttm: pd.DataFrame | None = None,
) -> Valuation:
    """Single-date metrics using only figures published by ``on``, on ``on``'s share basis.

    ``unrestated_ttm`` is ``ttm_table(quarterly, None, ...)`` computed once by a caller that
    needs many dates; its rows published by ``on`` are exactly the TTM rows of the quarters
    published by ``on``, and only the chosen row's EPS is restated to ``on``'s basis.
    """
    traded = raw_prices.loc[raw_prices["close"].notna() & (raw_prices["date"] <= on)]
    if traded.empty:
        raise MissingDataError(ticker, "close", "none", f"a raw close on or before {on}")
    price_row = traded.iloc[-1]
    price = float(price_row["close"])

    published = quarterly.loc[[a <= on for a in quarterly["available_date"]]]
    if unrestated_ttm is None:
        rows = ttm_table(published, None, on)
    else:
        known = [a is not None and a <= on for a in unrestated_ttm["available_date"]]
        rows = unrestated_ttm.loc[known] if len(unrestated_ttm) else unrestated_ttm
    if rows.empty:
        raise MissingDataError(ticker, "eps_ttm", "none", f"four published quarters by {on}")
    last = rows.iloc[-1]
    end = last["period_end"]
    eps = basis.to_basis(published["eps"], pd.Series(published.index, index=published.index), on)
    # The four quarters of the TTM row, restated to on's share basis.
    quarters = [q for q in published.index if last["period_start"] <= q <= end]
    eps_ttm = float(eps.loc[quarters].sum(min_count=4)) if len(quarters) == 4 else np.nan
    if pd.isna(eps_ttm):
        raise MissingDataError(ticker, "eps_ttm", "none", f"four published quarters by {on}")

    shares = last["ordinary_share_capital"] / basis.par_on(end) * basis.factor(end, on)
    bvps = last["parent_equity"] / shares if last["parent_equity"] > 0 and shares > 0 else np.nan

    year_ago = date(end.year - 1, end.month, end.day)  # quarter ends exist every year
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
        eps_ttm=eps_ttm,
        eps_period_end=end,
        book_value_per_share=bvps,
        pe=price / eps_ttm if eps_ttm > 0 else np.nan,
        pb=price / bvps if bvps > 0 else np.nan,
        dividend_yield=float(cash) / price,
        latest_quarter=end,
        latest_quarter_eps_yoy=eps_yoy,
    )


def annual_growth(annual: pd.DataFrame, column: str) -> pd.DataFrame:
    """Year-over-year growth of an annual column.

    Each row's period runs from the start of the prior year to the end of the current one,
    so a growth rate belongs to a window only if both years do. Growth from a base <= 0 is NaN.
    """
    rows = annual.sort_values("period_end").reset_index(drop=True)
    out = []
    for prev, cur in zip(rows.itertuples(), rows.iloc[1:].itertuples(), strict=False):
        if cur.year != prev.year + 1:
            continue
        base, value = getattr(prev, column), getattr(cur, column)
        out.append(
            {
                "year": cur.year,
                "period_start": prev.period_start,
                "period_end": cur.period_end,
                "available_date": cur.available_date,
                "growth": value / base - 1.0 if base > 0 else np.nan,
            }
        )
    return pd.DataFrame(
        out, columns=["year", "period_start", "period_end", "available_date", "growth"]
    )


def core_eps(quarterly: pd.DataFrame, basis: ShareBasis, basis_date: date) -> pd.Series:
    """Operating income per share for each quarter, on the share basis of ``basis_date``.

    Shares are those outstanding at the quarter end (capital / par).
    """
    pars = np.array([basis.par_on(end) for end in quarterly.index])
    shares = quarterly["ordinary_share_capital"].astype(float) / pars
    per_share = quarterly["operating_income"].astype(float) / shares.where(shares > 0)
    days = pd.Series(quarterly.index, index=quarterly.index)
    return basis.to_basis(per_share, days, basis_date)


def basis_price(raw_prices: pd.DataFrame, on: date, basis: ShareBasis, basis_date: date) -> float:
    """Raw close on the last trading day <= ``on``, restated to the share basis of
    ``basis_date`` (splits and stock dividends removed, cash dividends not added back)."""
    traded = raw_prices.loc[raw_prices["close"].notna() & (raw_prices["date"] <= on)]
    if traded.empty:
        raise MissingDataError(None, "close", "none", f"a raw close on or before {on}")
    row = traded.iloc[-1]
    return float(row["close"]) / basis.factor(row["date"], basis_date)


def fiscal_year_dividends(
    dividends: pd.DataFrame, basis: ShareBasis, basis_date: date
) -> pd.DataFrame:
    """Cash, stock and total dividends per share by fiscal year, on one share basis.

    Quarterly and half-year declarations are summed into their fiscal year. A year counts
    only up to the last fiscal year whose final period (ending 31 Dec) has been declared, so a
    year whose dividends are not all announced yet never looks like a cut. A year inside that
    range without any declaration is a year without dividends; it is known once a later year's
    declaration is published. Rights offerings (no fiscal period) are ignored.
    """
    columns = ["year", "period_start", "period_end", "available_date", "cash", "stock", "total"]
    if dividends.empty:
        return pd.DataFrame(columns=columns)
    # Fiscal years before the share basis starts (pre-IFRS) are not restated or used.
    declared = dividends.loc[
        [
            not pd.isna(end) and end is not None and end >= basis.start
            for end in dividends["period_end"]
        ]
    ]
    if declared.empty:
        return pd.DataFrame(columns=columns)
    when = [
        ex if ex is not None and not pd.isna(ex) else announced
        for ex, announced in zip(declared["cash_ex_date"], declared["available_date"], strict=True)
    ]
    stock_when = [
        ex if ex is not None and not pd.isna(ex) else announced
        for ex, announced in zip(declared["stock_ex_date"], declared["available_date"], strict=True)
    ]
    frame = pd.DataFrame(
        {
            "year": [pd.Timestamp(d).year for d in declared["period_end"]],
            "final": [pd.Timestamp(d).month == 12 for d in declared["period_end"]],
            "available_date": list(declared["available_date"]),
            "cash": basis.to_basis(
                declared["cash_dividend"].reset_index(drop=True),
                pd.Series(when),
                basis_date,
                include_start=True,
            ),
            "stock": basis.to_basis(
                declared["stock_dividend"].reset_index(drop=True),
                pd.Series(stock_when),
                basis_date,
                include_start=True,
            ),
        }
    )
    complete = frame.loc[frame["final"], "year"]
    if complete.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    first, last = int(frame["year"].min()), int(complete.max())
    for year in range(first, last + 1):
        part = frame.loc[frame["year"] == year]
        if part.empty:
            later = frame.loc[frame["year"] > year, "available_date"]
            available, cash, stock = min(later), 0.0, 0.0
        else:
            available, cash, stock = (
                max(part["available_date"]),
                part["cash"].sum(),
                part["stock"].sum(),
            )
        rows.append(
            {
                "year": year,
                "period_start": date(year, 1, 1),
                "period_end": date(year, 12, 31),
                "available_date": available,
                "cash": float(cash),
                "stock": float(stock),
                "total": float(cash + stock),
            }
        )
    return pd.DataFrame(rows, columns=columns)
