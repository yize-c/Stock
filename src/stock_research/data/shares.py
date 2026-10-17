"""Share-count adjustments for per-share figures (EPS, book value, dividends per share).

FinMind's historical EPS is not restated for later stock dividends or splits, so per-share
figures from different years are on different share bases. Each corporate event gets a share
multiplier ``m`` (shares after / shares before), and a figure is moved to the share basis of a
later date by dividing it by the product of the multipliers in between.

Multipliers come from the exchange reference prices already used for adjusted prices:

* ex-rights/ex-dividend day: ``m = (close_before - cash dividend) / reference_price``. For a
  stock dividend this equals ``1 + stock dividend / par``; for a rights offering it is the
  bonus-element factor (IAS 33). On a pure cash day (``息``/``除息``) ``m`` is 1. Where a stock
  dividend is declared, ``m`` is checked against ``1 + stock dividend / par``.
* split, reverse split, par-value change: ``m = close_before / reference_price``; the par
  value changes by ``1 / m``.
* capital reduction to offset losses: ``m = close_before / reference_price``.
* capital reduction with a cash refund at par: the reference price is
  ``(P - (1 - r) * par) / r``, so ``m = r = (P - par) / (reference_price - par)``.

Only events from ``start`` on (the IFRS start: per-share figures are never needed for earlier
periods, and FinMind's dividend declarations begin in 2005) are used; restating a figure
dated before ``start`` raises.

Par values start at ``initial_par`` (config default, or TPEx's official value) and follow the
split/par-change events. Share counts derived from capital / par are checked against
``net income attributable to parent / EPS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import cached_property

import numpy as np
import pandas as pd

from stock_research.data.errors import AnomalyError, MissingDataError, SourceMismatchError
from stock_research.data.fetchers.finmind import EventKind

_LOSS_OFFSET = "Making up losses"
_CASH_REFUND = "Cash refund"
_RIGHTS = "權"  # in 權, 權息, 除權, 除權息: the event changes the share count
_DIVIDEND = "息"


@dataclass(frozen=True)
class ShareBasis:
    """Share multipliers by effective day, and the par value in force on any day."""

    ticker: str
    multipliers: pd.DataFrame  # effective_date, kind, multiplier, par_after
    initial_par: float
    start: date

    def factor(self, after: date, through: date, *, include_start: bool = False) -> float:
        """Product of multipliers effective in ``(after, through]`` (``[after, through]``)."""
        if after < self.start:
            raise MissingDataError(
                self.ticker, "share_basis", after.isoformat(), f"a date on or after {self.start}"
            )
        days, values = self._arrays
        lower = days >= after.toordinal() if include_start else days > after.toordinal()
        return float(np.prod(values[lower & (days <= through.toordinal())]))

    @cached_property
    def _arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Effective days (ordinals) and multipliers, built once per basis."""
        days = np.array([d.toordinal() for d in self.multipliers["effective_date"]], dtype=int)
        return days, self.multipliers["multiplier"].to_numpy(dtype=float)

    def par_on(self, day: date) -> float:
        prior = self.multipliers.loc[self.multipliers["effective_date"] <= day, "par_after"]
        return float(prior.iloc[-1]) if not prior.empty else self.initial_par

    def to_basis(
        self,
        values: pd.Series,
        days: pd.Series,
        basis: date,
        *,
        include_start: bool = False,
    ) -> pd.Series:
        """Per-share ``values`` (each dated by ``days``) restated to the share basis of ``basis``.

        Use ``include_start=True`` for dividends per share: a stock dividend paid on the same
        ex-date also enlarges the share count the cash dividend was paid on.
        """
        dates = [pd.Timestamp(d).date() for d in days]
        early = [d for d in dates if d < self.start]
        if early:
            self.factor(early[0], basis)  # raises: no basis before the start
        ordinals = np.array([d.toordinal() for d in dates], dtype=int)
        effective, multipliers = self._arrays
        later = (
            effective[None, :] >= ordinals[:, None]
            if include_start
            else (effective[None, :] > ordinals[:, None])
        )
        inside = later & (effective[None, :] <= basis.toordinal())
        factors = np.prod(np.where(inside, multipliers[None, :], 1.0), axis=1)
        return values.astype(float) / factors


def share_basis(
    ticker: str,
    events: pd.DataFrame,
    dividends: pd.DataFrame,
    *,
    start: date,
    initial_par: float,
    tolerance: float,
) -> ShareBasis:
    """Build multipliers from applied price events (``AdjustedPrices.events``) from ``start``.

    ``initial_par`` is the par value in force at ``start``.
    """
    rows = []
    par = initial_par
    records = [] if events.empty else events.sort_values("effective_date").to_dict("records")
    for event in (e for e in records if e["effective_date"] >= start):
        kind = event["kind"]
        before = float(event["close_before"])
        reference = float(event["reference_price"])
        if kind == EventKind.EX_RIGHTS_DIVIDEND.value:
            multiplier = _ex_rights_multiplier(ticker, event, dividends, par, tolerance)
        elif kind == EventKind.SPLIT.value:
            multiplier = before / reference
            par = par / multiplier
        elif kind == EventKind.CAPITAL_REDUCTION.value:
            multiplier = _reduction_multiplier(ticker, event, par)
        else:
            raise AnomalyError(ticker, "event.kind", kind, "a known price-event kind")
        if not multiplier > 0:
            raise AnomalyError(ticker, f"{kind}.multiplier", multiplier, "> 0")
        rows.append(
            {
                "effective_date": event["effective_date"],
                "kind": kind,
                "multiplier": multiplier,
                "par_after": par,
            }
        )
    frame = pd.DataFrame(rows, columns=["effective_date", "kind", "multiplier", "par_after"])
    return ShareBasis(ticker, frame, initial_par, start)


def _ex_rights_multiplier(
    ticker: str, event: dict, dividends: pd.DataFrame, par: float, tolerance: float
) -> float:
    detail = str(event["detail"])
    if _RIGHTS not in detail:
        if _DIVIDEND not in detail:
            raise AnomalyError(ticker, "ex_rights.kind", detail, "a label with 權 or 息")
        return 1.0
    scheduled = event["date"]
    share_rows = dividends.loc[dividends["stock_ex_date"] == scheduled]
    if share_rows.empty:
        raise MissingDataError(
            ticker,
            "dividend.stock_ex_date",
            f"no declaration for ex-rights day {scheduled}",
            "a stock dividend or rights offering declared for that day",
        )
    cash = float(dividends.loc[dividends["cash_ex_date"] == scheduled, "cash_dividend"].sum())
    multiplier = (float(event["close_before"]) - cash) / float(event["reference_price"])
    stock = float(share_rows["stock_dividend"].sum())
    if stock > 0 and not share_rows["rights_issue_shares"].gt(0).any():
        expected = 1.0 + stock / par
        if abs(multiplier / expected - 1.0) > tolerance:
            raise SourceMismatchError(
                ticker,
                "ex_rights.share_multiplier",
                f"{scheduled}: from reference price {multiplier:.5f}, "
                f"from NT${stock:.4f} stock dividend at par {par} {expected:.5f}",
                f"within {tolerance:.2%} (check the par value)",
            )
    return multiplier


def _reduction_multiplier(ticker: str, event: dict, par: float) -> float:
    before = float(event["close_before"])
    reference = float(event["reference_price"])
    reason = event["detail"]
    if reason == _LOSS_OFFSET:
        return before / reference
    if reason == _CASH_REFUND:
        if reference == par:
            raise AnomalyError(ticker, "capital_reduction.reference_price", reference, "!= par")
        ratio = (before - par) / (reference - par)
        if not 0 < ratio < 1:
            raise AnomalyError(
                ticker,
                "capital_reduction.share_ratio",
                f"{event['date']}: {ratio:.4f} (close {before}, reference {reference}, par {par})",
                "a ratio between 0 and 1",
            )
        return ratio
    raise AnomalyError(
        ticker, "capital_reduction.reason", reason, f"{_LOSS_OFFSET!r} or {_CASH_REFUND!r}"
    )


def check_share_count(
    ticker: str,
    implied: pd.Series,
    from_capital: pd.Series,
    tolerance: float,
) -> float:
    """Median ratio of implied shares (net income / EPS) to capital / par, per quarter.

    Both series are indexed by quarter end. Single quarters are noisy (EPS is rounded to two
    decimals and some quarters are derived by subtraction), so the median is compared; a
    wrong par value shows up as a factor of 2 or more. Returns the median ratio.
    """
    both = pd.concat([implied, from_capital], axis=1, keys=["implied", "capital"]).dropna()
    both = both.loc[(both["implied"] > 0) & (both["capital"] > 0)]
    if both.empty:
        raise MissingDataError(ticker, "share_count", "no comparable quarters", "at least one")
    ratio = float((both["implied"] / both["capital"]).median())
    if abs(ratio - 1.0) > tolerance:
        raise SourceMismatchError(
            ticker,
            "share_count",
            f"median implied/capital-based shares {ratio:.3f}",
            f"within {tolerance:.0%} of 1 (check the par value)",
        )
    return ratio


def share_counts(financials: pd.DataFrame, basis: ShareBasis) -> tuple[pd.Series, pd.Series]:
    """Per quarter end: shares implied by net income attributable to parent / EPS, and
    shares from ordinary share capital / par. Inputs are long statement rows."""

    def item(statement: str, name: str) -> pd.Series:
        rows = financials.loc[(financials["statement"] == statement) & (financials["item"] == name)]
        return rows.set_index(pd.to_datetime(rows["period_end"]).dt.date)["value"].astype(float)

    eps = item("income", "EPS")
    implied = item("income", "EquityAttributableToOwnersOfParent") / eps.where(eps != 0)
    capital = item("balance", "OrdinaryShare")
    pars = np.array([basis.par_on(d) for d in capital.index])
    return implied.replace([np.inf, -np.inf], np.nan), capital / pars
