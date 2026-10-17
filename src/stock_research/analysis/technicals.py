"""Optional technical indicators (spec §9), computed from adjusted closes.

Descriptions only ("above the 240-day average"); there are no buy or sell signals, and the
report notes that the empirical value of technical analysis is disputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from stock_research.config import TechnicalsConfig

DISPUTED = "技術分析的實證效果有爭議，以下僅為描述，不構成買賣訊號。"


@dataclass(frozen=True)
class Technicals:
    on: date
    close: float
    sma: dict[int, float | None]  # window -> value (None: not enough history)
    rsi: float | None
    macd: float | None
    macd_signal: float | None
    descriptions: tuple[str, ...]


def technicals(adjusted: pd.DataFrame, config: TechnicalsConfig) -> Technicals:
    closes = adjusted.set_index("date")["adj_close"].dropna().astype(float)
    last = float(closes.iloc[-1])
    sma = {
        n: float(closes.tail(n).mean()) if len(closes) >= n else None for n in config.sma_windows
    }
    delta = closes.diff().dropna()
    n = config.rsi_window
    rsi = None
    if len(delta) >= n:
        gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
        rsi = 100.0 if loss == 0 else float(100 - 100 / (1 + gain / loss))
    m = config.macd
    macd = signal = None
    if len(closes) >= m.slow + m.signal:
        line = (
            closes.ewm(span=m.fast, adjust=False).mean()
            - closes.ewm(span=m.slow, adjust=False).mean()
        )
        macd = float(line.iloc[-1])
        signal = float(line.ewm(span=m.signal, adjust=False).mean().iloc[-1])
    names = {20: "月線", 60: "季線", 240: "年線"}
    text = [
        f"價格{'高於' if last > value else '低於或等於'}{names.get(window, f'{window} 日均線')}"
        for window, value in sma.items()
        if value is not None
    ]
    if rsi is not None:
        text.append(f"RSI({n}) 為 {rsi:.1f}")
    if macd is not None and signal is not None:
        text.append(f"MACD {'在' if macd > signal else '不在'}訊號線之上")
    return Technicals(
        on=pd.Timestamp(closes.index[-1]).date(),
        close=last,
        sma=sma,
        rsi=rsi,
        macd=macd,
        macd_signal=signal,
        descriptions=tuple(text),
    )
