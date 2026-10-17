"""Dividend- and split-adjusted (還原權息) prices computed from exchange reference prices.

On each corporate event's effective day the exchange publishes a reference price: the
previous raw close with the dividend, stock dividend, capital reduction or split taken out.
The event's factor is ``reference_price / close_before``. A day's adjusted price is its raw
price times the product of the factors of every event that takes effect *after* that day,
so returns across an event reflect the shareholder's total return (dividends reinvested at
the reference price). This matches the exchange's own 還原權值 method.

Rules:

* Only events effective on or before ``as_of`` are applied, so a later event never changes
  what an ``--as-of`` run sees.
* An event dated on a day without trading (e.g. a typhoon closure on the scheduled ex-date)
  takes effect on the next day with a price.
* ``close_before`` must equal the last raw close before the effective day; otherwise the
  event and the price series disagree and ``SourceMismatchError`` is raised.
* Two events on the same effective day cannot be separated and raise ``AnomalyError``.

Raw and adjusted prices live in separate frames with different column names (``close`` vs
``adj_close``) so they cannot be mixed by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from stock_research.data.errors import AnomalyError, SourceMismatchError
from stock_research.data.fetchers.finmind import EventKind

# Exchange prices have at most two decimals; anything beyond rounding is a real mismatch.
_PRICE_MATCH_TOLERANCE = 0.005
ADJUSTED_COLUMNS = ("adj_open", "adj_high", "adj_low", "adj_close")
# Events after which the raw price basis changes for other sources too (Yahoo treats stock
# dividends as splits and rescales history), so raw closes before them are not comparable.
_BASIS_CHANGE_KINDS = {EventKind.CAPITAL_REDUCTION.value, EventKind.SPLIT.value}


@dataclass(frozen=True)
class AdjustedPrices:
    ticker: str
    as_of: date
    raw: pd.DataFrame  # date, open, high, low, close, volume, no_trade, ...
    adjusted: pd.DataFrame  # date, adj_open, adj_high, adj_low, adj_close, adj_factor
    events: pd.DataFrame  # applied events with effective_date and factor

    def ex_rights_reference(self) -> dict[date, float]:
        """Reference prices on ex-rights/ex-dividend days, for the price-limit check."""
        ex = self.events.loc[self.events["kind"] == EventKind.EX_RIGHTS_DIVIDEND.value]
        return dict(zip(ex["effective_date"], ex["reference_price"].astype(float), strict=True))

    def capital_change_resumptions(self) -> set[date]:
        """Days trading resumed after a capital reduction, split or par-value change."""
        mask = self.events["kind"].isin(list(_BASIS_CHANGE_KINDS))
        return set(self.events.loc[mask, "effective_date"])

    def capital_change_halts(self) -> list[tuple[date, date, str]]:
        """Trading halts before a capital reduction, split or par-value change resumes.

        Shares are re-issued during these, so the exchange suspends trading; the event data
        documents the halt (from the day after the last close to the day before resumption).
        Returns ``(start, end, kind)`` for each non-empty halt.
        """
        days = list(self.raw["date"])
        halts = []
        mask = self.events["kind"].isin(list(_BASIS_CHANGE_KINDS))
        for event in self.events.loc[mask].itertuples(index=False):
            position = days.index(event.effective_date)
            start = days[position - 1] + timedelta(days=1)
            end = event.effective_date - timedelta(days=1)
            if start <= end:
                halts.append((start, end, str(event.kind)))
        return halts

    def basis_change_dates(self) -> list[date]:
        """Effective days of events that change share count (incl. stock dividends)."""
        stock = self.events["kind"].eq(EventKind.EX_RIGHTS_DIVIDEND.value) & self.events[
            "detail"
        ].str.contains("權", regex=False)
        mask = stock | self.events["kind"].isin(list(_BASIS_CHANGE_KINDS))
        return sorted(self.events.loc[mask, "effective_date"])


def adjust_prices(
    ticker: str, raw: pd.DataFrame, events: pd.DataFrame, as_of: date
) -> AdjustedPrices:
    raw = raw.loc[pd.to_datetime(raw["date"]) <= pd.Timestamp(as_of)].reset_index(drop=True)
    if raw.empty:
        raise AnomalyError(ticker, "prices", "empty", f"raw prices on or before {as_of}")
    days = list(raw["date"])
    applied = _effective_events(ticker, raw, events, as_of)

    factors = np.ones(len(days))
    for event in applied.itertuples(index=False):
        before = np.array([d < event.effective_date for d in days])
        factors[before] *= event.factor
    adjusted = pd.DataFrame({"date": days})
    for col, adj in zip(("open", "high", "low", "close"), ADJUSTED_COLUMNS, strict=True):
        adjusted[adj] = raw[col].astype(float).to_numpy() * factors
    adjusted["adj_factor"] = factors
    return AdjustedPrices(ticker, as_of, raw, adjusted, applied)


def _effective_events(
    ticker: str, raw: pd.DataFrame, events: pd.DataFrame, as_of: date
) -> pd.DataFrame:
    columns = [*events.columns, "effective_date", "factor"]
    if events.empty:
        return pd.DataFrame(columns=columns)
    days = list(raw["date"])
    closes = raw["close"].astype(float).to_numpy()
    rows = []
    for event in events.sort_values("date").to_dict("records"):
        effective = next((d for d in days if d >= event["date"]), None)
        if effective is None or effective > as_of:
            continue  # not yet in effect in this price history
        position = days.index(effective)
        prior = closes[:position][~np.isnan(closes[:position])]
        if prior.size == 0:
            continue  # nothing before it to adjust
        before = float(event["close_before"])
        if abs(prior[-1] - before) > _PRICE_MATCH_TOLERANCE:
            raise SourceMismatchError(
                ticker,
                f"{event['kind']}.close_before",
                f"{event['date']}: event {before} vs last raw close {prior[-1]}",
                "the event's pre-event close to equal the last raw close",
            )
        rows.append(
            {
                **event,
                "effective_date": effective,
                "factor": float(event["reference_price"]) / before,
            }
        )
    out = pd.DataFrame(rows, columns=columns)
    clash = out["effective_date"].duplicated(keep=False)
    if clash.any():
        raise AnomalyError(
            ticker,
            "events.effective_date",
            sorted({d.isoformat() for d in out.loc[clash, "effective_date"]}),
            "at most one price-basis event per day",
        )
    return out.reset_index(drop=True)
