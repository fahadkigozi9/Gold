"""Bar-by-bar backtester for single-position gold strategies.

Execution model (deliberately conservative):
* Signal is read on the CLOSE of bar i, the trade is entered at the OPEN of bar i+1.
* Stop-loss and take-profit are ATR multiples measured at the signal bar.
* If SL and TP are both touched inside one bar, the STOP is assumed to hit first.
* Gaps through a stop fill at the (worse) open; gaps through a target fill at the open.
* Costs (spread + 2x slippage + commission) are charged on every trade, in price units.
* One position at a time. Results are expressed in R-multiples; equity compounds
  ``risk_pct`` of the current balance per trade.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class BTConfig:
    spread: float = 0.30              # USD per oz (price units)
    slippage: float = 0.05            # USD per oz, each side
    commission_per_lot: float = 0.0   # USD round-turn per 1.0 lot
    contract_size: float = 100.0      # oz per lot (XAUUSD)
    risk_pct: float = 1.0             # % of balance risked per trade
    initial_balance: float = 10_000.0
    min_stop: float = 0.8             # minimum stop distance in price units

    @property
    def cost_price(self) -> float:
        return self.spread + 2 * self.slippage + self.commission_per_lot / self.contract_size

    def to_dict(self) -> dict:
        return asdict(self)


class MarketData:
    """Numpy view of an OHLC DataFrame with a DatetimeIndex."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        self.t = df.index.to_numpy()
        self.n = len(df)


TRADE_COLS = ["entry_time", "exit_time", "dir", "entry", "exit", "reason", "r", "bars"]


def simulate(md: MarketData, sig: np.ndarray, atr_arr: np.ndarray, sl_mult: float,
             tp_mult: float, max_bars: int, cfg: BTConfig,
             lo: int = 0, hi: int | None = None) -> pd.DataFrame:
    """Simulate trades whose entry bar lies in [lo+1, hi). Returns a trades DataFrame."""
    hi = md.n if hi is None else min(hi, md.n)
    rows = []
    if hi - lo < 3:
        return pd.DataFrame(columns=TRADE_COLS)
    cand = np.flatnonzero(sig[lo:hi - 1] != 0) + lo
    cost = cfg.cost_price
    next_ok = lo
    for i in cand:
        if i < next_ok:
            continue
        a = atr_arr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        d = int(sig[i])
        s = i + 1
        entry = md.o[s]
        sl_dist = max(sl_mult * a, cfg.min_stop)
        tp_dist = max(tp_mult * a, 1e-9)
        e = min(s + max_bars, hi)
        lw, hw, ow = md.l[s:e], md.h[s:e], md.o[s:e]
        if d == 1:
            sl, tp = entry - sl_dist, entry + tp_dist
            sl_hit, tp_hit = lw <= sl, hw >= tp
        else:
            sl, tp = entry + sl_dist, entry - tp_dist
            sl_hit, tp_hit = hw >= sl, lw <= tp
        big = len(lw) + 1
        k_sl = int(np.argmax(sl_hit)) if sl_hit.any() else big
        k_tp = int(np.argmax(tp_hit)) if tp_hit.any() else big
        if k_sl == big and k_tp == big:
            k = len(lw) - 1
            exit_px, reason = md.c[s + k], "time"
        elif k_sl <= k_tp:
            k, reason = k_sl, "sl"
            exit_px = min(sl, ow[k]) if d == 1 else max(sl, ow[k])
        else:
            k, reason = k_tp, "tp"
            exit_px = max(tp, ow[k]) if d == 1 else min(tp, ow[k])
        exit_idx = s + k
        r = ((exit_px - entry) * d - cost) / sl_dist
        rows.append((md.t[s], md.t[exit_idx], d, entry, exit_px, reason, r, k + 1))
        next_ok = exit_idx
    return pd.DataFrame(rows, columns=TRADE_COLS)


def equity_curve(r: np.ndarray, cfg: BTConfig) -> np.ndarray:
    eq = np.empty(len(r) + 1)
    eq[0] = cfg.initial_balance
    for j, x in enumerate(r):
        eq[j + 1] = max(eq[j] * (1.0 + cfg.risk_pct / 100.0 * x), 0.0)
    return eq


def empty_metrics(cfg: BTConfig) -> dict:
    return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy_r": 0.0,
            "avg_win_r": 0.0, "avg_loss_r": 0.0, "total_return_pct": 0.0, "max_dd_pct": 0.0,
            "return_dd": 0.0, "sqn": 0.0, "max_consec_losses": 0, "long_trades": 0,
            "short_trades": 0, "avg_bars": 0.0, "final_balance": cfg.initial_balance}


def compute_metrics(trades: pd.DataFrame, cfg: BTConfig) -> dict:
    n = len(trades)
    if n == 0:
        return empty_metrics(cfg)
    r = trades["r"].to_numpy(float)
    wins, losses = r[r > 0], r[r <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    pf = 99.0 if gross_loss == 0 and gross_win > 0 else (gross_win / gross_loss if gross_loss > 0 else 0.0)
    eq = equity_curve(r, cfg)
    peak = np.maximum.accumulate(eq)
    dd = float(((peak - eq) / np.where(peak > 0, peak, 1.0)).max() * 100.0)
    ret = (eq[-1] / cfg.initial_balance - 1.0) * 100.0
    sd = r.std(ddof=1) if n > 1 else 0.0
    sqn = float(np.sqrt(n) * r.mean() / sd) if sd > 0 else 0.0
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    return {
        "trades": int(n),
        "win_rate": round(float((r > 0).mean() * 100.0), 2),
        "profit_factor": round(float(min(pf, 99.0)), 3),
        "expectancy_r": round(float(r.mean()), 4),
        "avg_win_r": round(float(wins.mean()), 3) if len(wins) else 0.0,
        "avg_loss_r": round(float(losses.mean()), 3) if len(losses) else 0.0,
        "total_return_pct": round(float(ret), 2),
        "max_dd_pct": round(dd, 2),
        "return_dd": round(float(ret / dd), 2) if dd > 0 else (round(float(ret), 2) if ret > 0 else 0.0),
        "sqn": round(sqn, 3),
        "max_consec_losses": int(best),
        "long_trades": int((trades["dir"] == 1).sum()),
        "short_trades": int((trades["dir"] == -1).sum()),
        "avg_bars": round(float(trades["bars"].mean()), 1),
        "final_balance": round(float(eq[-1]), 2),
    }


def run_backtest(md: MarketData, sig: np.ndarray, atr_arr: np.ndarray, spec: dict,
                 cfg: BTConfig, lo: int = 0, hi: int | None = None):
    """Run one backtest. Returns (trades_df, metrics_dict)."""
    trades = simulate(md, sig, atr_arr, spec["sl_atr"], spec["tp_atr"], spec["max_bars"], cfg, lo, hi)
    return trades, compute_metrics(trades, cfg)


def monte_carlo_drawdown(r: np.ndarray, cfg: BTConfig, runs: int = 1000, seed: int = 7) -> dict:
    """Reshuffle trade order to estimate drawdown risk the single backtest path hides."""
    if len(r) < 5:
        return {"dd_median": 0.0, "dd_p95": 0.0, "prob_loss": 0.0}
    rng = np.random.default_rng(seed)
    dds, finals = [], []
    for _ in range(runs):
        eq = equity_curve(rng.permutation(r), cfg)
        peak = np.maximum.accumulate(eq)
        dds.append(((peak - eq) / np.where(peak > 0, peak, 1.0)).max() * 100.0)
        finals.append(eq[-1])
    return {"dd_median": round(float(np.median(dds)), 2),
            "dd_p95": round(float(np.percentile(dds, 95)), 2),
            "prob_loss": round(float(np.mean(np.array(finals) < cfg.initial_balance) * 100.0), 1)}
