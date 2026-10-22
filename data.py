"""Loading, cleaning and summarising gold OHLC data."""
from __future__ import annotations

import io
from typing import IO

import numpy as np
import pandas as pd

from . import indicators as ind

OHLC = ["open", "high", "low", "close"]
TF_RULES = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}


class DataError(ValueError):
    pass


def clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate, drop bad rows and repair impossible highs/lows."""
    df = df.copy()
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in OHLC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=OHLC)
    df = df[(df[OHLC] > 0).all(axis=1)]
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    if len(df) < 200:
        raise DataError(f"Only {len(df)} usable bars - need at least 200 to backtest.")
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().strip("<>").lower().replace(" ", "_") for c in df.columns]
    rename = {"o": "open", "h": "high", "l": "low", "c": "close", "tickvol": "volume",
              "tick_volume": "volume", "vol": "volume", "datetime": "time", "timestamp": "time",
              "gmt_time": "time", "local_time": "time"}
    return df.rename(columns=rename)


def load_csv(file: IO | str) -> pd.DataFrame:
    """Parse CSV exports from MetaTrader 5, TradingView, Dukascopy, Yahoo and similar."""
    if isinstance(file, str):
        raw = open(file, "rb").read()
    else:
        raw = file.read()
        if isinstance(raw, str):
            raw = raw.encode()
    text = raw.decode("utf-8-sig", errors="replace")
    first = text.splitlines()[0] if text else ""
    sep = "\t" if first.count("\t") >= 3 else (";" if first.count(";") >= 3 else ",")
    df = _normalise_columns(pd.read_csv(io.StringIO(text), sep=sep))
    if "date" in df.columns and "time" in df.columns and "datetime" not in df.columns:
        stamp = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
    elif "time" in df.columns:
        stamp = df["time"]
    elif "date" in df.columns:
        stamp = df["date"]
    else:
        raise DataError("No time/date column found.")
    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise DataError(f"Missing columns: {missing}. Found: {list(df.columns)}")
    if pd.api.types.is_numeric_dtype(stamp):
        unit = "ms" if stamp.iloc[0] > 1e11 else "s"
        idx = pd.to_datetime(stamp, unit=unit)
    else:
        idx = pd.to_datetime(stamp, errors="coerce", utc=False)
    if getattr(idx.dt, "tz", None) is not None:
        idx = idx.dt.tz_convert("UTC").dt.tz_localize(None)
    out = df[OHLC + (["volume"] if "volume" in df.columns else [])].copy()
    out.index = pd.DatetimeIndex(idx, name="time")
    return clean_ohlc(out)


def fetch_yfinance(ticker: str = "GC=F", interval: str = "1h", period: str = "730d") -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, interval=interval, period=period, auto_adjust=False,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        raise DataError("Yahoo Finance returned no data. Intraday history is limited "
                        "(1m: 7d, 5m/15m/30m: 60d, 1h: 730d).")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = _normalise_columns(raw)
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    raw.index = idx.rename("time")
    cols = OHLC + (["volume"] if "volume" in raw.columns else [])
    return clean_ohlc(raw[cols])


def fetch_mt5(symbol: str = "XAUUSDz", timeframe: str = "M15", bars: int = 50_000) -> pd.DataFrame:
    """Pull candles from a locally running MetaTrader 5 terminal (Windows only)."""
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise DataError("MetaTrader5 package not installed (Windows only: pip install MetaTrader5).") from exc
    if not mt5.initialize():
        raise DataError(f"Could not connect to MT5 terminal: {mt5.last_error()}")
    try:
        tf = getattr(mt5, f"TIMEFRAME_{timeframe.upper()}")
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            raise DataError(f"No data for {symbol}: {mt5.last_error()}")
    finally:
        mt5.shutdown()
    df = pd.DataFrame(rates)
    df.index = pd.to_datetime(df["time"], unit="s").rename("time")
    df = df.rename(columns={"tick_volume": "volume"})
    return clean_ohlc(df[OHLC + ["volume"]])


def make_synthetic(n: int = 20_000, seed: int = 42, freq: str = "15min", start_price: float = 2000.0) -> pd.DataFrame:
    """Regime-switching synthetic gold-like series for demos and tests (NOT real prices)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n * 2, freq=freq)
    idx = idx[idx.dayofweek < 5][:n]
    bars_per_hour = max(1, int(round(60 / pd.Timedelta(freq).seconds * 60)))
    hour = idx.hour.to_numpy()
    season = 0.6 + 1.4 * np.exp(-((hour - 14) ** 2) / 18.0) + 0.5 * np.exp(-((hour - 8) ** 2) / 8.0)
    regime = np.zeros(n)
    state, drift = 0, 0.0
    for i in range(n):
        if rng.random() < 0.004:
            state = rng.integers(0, 3)
            drift = {0: 0.0, 1: 0.012, 2: -0.012}[int(state)]
        regime[i] = drift
    vol = 0.0006 * season * np.where(rng.random(n) < 0.002, 3.0, 1.0)
    ret = regime * 0.02 + rng.normal(0, 1, n) * vol
    close = start_price * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, 1, (n, 2))) * vol[:, None] * close[:, None] * 0.6
    high = np.maximum(open_, close) + wick[:, 0]
    low = np.minimum(open_, close) - wick[:, 1]
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": rng.integers(50, 2000, n)}, index=pd.DatetimeIndex(idx, name="time"))
    return clean_ohlc(df)


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = TF_RULES[tf]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule).agg(agg).dropna(subset=OHLC)
    return clean_ohlc(out)


def infer_bar_minutes(df: pd.DataFrame) -> float:
    d = df.index.to_series().diff().dropna().dt.total_seconds() / 60.0
    return float(d.median()) if len(d) else 0.0


def summarize_market(df: pd.DataFrame) -> dict:
    """Compact, LLM-friendly description of the dataset."""
    atr14 = ind.atr(df["high"], df["low"], df["close"], 14)
    adx14 = ind.adx(df["high"], df["low"], df["close"], 14)
    rng = (df["high"] - df["low"])
    hourly = rng.groupby(df.index.hour).mean()
    top_hours = hourly.sort_values(ascending=False).head(5).index.tolist()
    daily = df["close"].resample("1D").last().dropna()
    return {
        "bars": int(len(df)),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "bar_minutes": round(infer_bar_minutes(df), 1),
        "price_range": [round(float(df["low"].min()), 2), round(float(df["high"].max()), 2)],
        "median_atr14": round(float(atr14.median()), 3),
        "median_bar_range": round(float(rng.median()), 3),
        "pct_time_adx_above_25": round(float((adx14 > 25).mean() * 100), 1),
        "most_volatile_hours": [int(h) for h in top_hours],
        "daily_return_stdev_pct": round(float(daily.pct_change().std() * 100), 3) if len(daily) > 2 else None,
    }
