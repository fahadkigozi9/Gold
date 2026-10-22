"""Causal technical indicators (value at bar t only uses data up to bar t)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100.0 - 100.0 / (1.0 + ru / rd.replace(0.0, np.nan))
    out = out.where(~((rd == 0) & (ru > 0)), 100.0)
    out = out.where(~((rd == 0) & (ru == 0)), 50.0)
    return out


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    return true_range(h, l, c).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    a = atr(h, l, c, n)
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd_hist(c: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    line = ema(c, fast) - ema(c, slow)
    return line - line.ewm(span=signal, adjust=False, min_periods=signal).mean()


def bollinger(c: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series]:
    mid = sma(c, n)
    sd = c.rolling(n, min_periods=n).std(ddof=0)
    return mid + k * sd, mid - k * sd


def donchian_high(h: pd.Series, n: int = 20) -> pd.Series:
    """Highest high of the previous n bars (excludes the current bar)."""
    return h.rolling(n, min_periods=n).max().shift(1)


def donchian_low(l: pd.Series, n: int = 20) -> pd.Series:
    """Lowest low of the previous n bars (excludes the current bar)."""
    return l.rolling(n, min_periods=n).min().shift(1)


def stoch_k(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    lo = l.rolling(n, min_periods=n).min()
    hi = h.rolling(n, min_periods=n).max()
    return 100 * (c - lo) / (hi - lo).replace(0.0, np.nan)


def roc(c: pd.Series, n: int = 10) -> pd.Series:
    return c.pct_change(n) * 100.0


def zscore(c: pd.Series, n: int = 20) -> pd.Series:
    sd = c.rolling(n, min_periods=n).std(ddof=0).replace(0.0, np.nan)
    return (c - sma(c, n)) / sd
