"""Seed strategy templates and local mutation operators."""
from __future__ import annotations

import numpy as np

from .dsl import clone, validate_spec, SpecError


def _i(name, **p):
    return {"ind": name, **p}


TEMPLATES_RAW = [
    {
        "name": "EMA trend pullback",
        "rationale": "Buy dips toward the 20 EMA while 50 EMA is above 200 EMA; mirror for shorts.",
        "long": [{"a": _i("ema", n=50), "op": ">", "b": _i("ema", n=200)},
                 {"a": _i("dist_ema_atr", n=20), "op": "<", "b": -0.3},
                 {"a": _i("rsi", n=14), "op": "crosses_above", "b": 40}],
        "short": [{"a": _i("ema", n=50), "op": "<", "b": _i("ema", n=200)},
                  {"a": _i("dist_ema_atr", n=20), "op": ">", "b": 0.3},
                  {"a": _i("rsi", n=14), "op": "crosses_below", "b": 60}],
        "sl_atr": 1.5, "tp_atr": 2.5, "atr_n": 14, "max_bars": 80,
    },
    {
        "name": "Donchian breakout with ADX",
        "rationale": "Trade 20-bar breakouts only when trend strength (ADX) is rising above 20.",
        "long": [{"a": _i("close"), "op": ">", "b": _i("donchian_high", n=20)},
                 {"a": _i("adx", n=14), "op": ">", "b": 20}],
        "short": [{"a": _i("close"), "op": "<", "b": _i("donchian_low", n=20)},
                  {"a": _i("adx", n=14), "op": ">", "b": 20}],
        "sl_atr": 1.8, "tp_atr": 3.0, "atr_n": 14, "max_bars": 100,
    },
    {
        "name": "Bollinger mean reversion",
        "rationale": "Fade closes outside the 2-sigma band when ADX shows no strong trend.",
        "long": [{"a": _i("close"), "op": "<", "b": _i("bb_lower", n=20, k=2.0)},
                 {"a": _i("rsi", n=14), "op": "<", "b": 35},
                 {"a": _i("adx", n=14), "op": "<", "b": 25}],
        "short": [{"a": _i("close"), "op": ">", "b": _i("bb_upper", n=20, k=2.0)},
                  {"a": _i("rsi", n=14), "op": ">", "b": 65},
                  {"a": _i("adx", n=14), "op": "<", "b": 25}],
        "sl_atr": 2.0, "tp_atr": 1.2, "atr_n": 14, "max_bars": 40,
    },
    {
        "name": "RSI extremes z-score fade",
        "rationale": "Fade stretched z-scores confirmed by stochastic exhaustion.",
        "long": [{"a": _i("zscore", n=30), "op": "<", "b": -2.0},
                 {"a": _i("stoch_k", n=14), "op": "crosses_above", "b": 20}],
        "short": [{"a": _i("zscore", n=30), "op": ">", "b": 2.0},
                  {"a": _i("stoch_k", n=14), "op": "crosses_below", "b": 80}],
        "sl_atr": 1.5, "tp_atr": 1.5, "atr_n": 14, "max_bars": 30,
    },
    {
        "name": "MACD momentum with trend filter",
        "rationale": "Take MACD histogram zero-crosses in the direction of the 100 EMA slope.",
        "long": [{"a": _i("macd_hist", fast=12, slow=26, signal=9), "op": "crosses_above", "b": 0},
                 {"a": _i("close"), "op": ">", "b": _i("ema", n=100)}],
        "short": [{"a": _i("macd_hist", fast=12, slow=26, signal=9), "op": "crosses_below", "b": 0},
                  {"a": _i("close"), "op": "<", "b": _i("ema", n=100)}],
        "sl_atr": 1.5, "tp_atr": 2.0, "atr_n": 14, "max_bars": 60,
    },
    {
        "name": "Session momentum candle",
        "rationale": "Strong full-body candle during the busy 12-16 hour window, with the trend.",
        "long": [{"a": _i("hour"), "op": ">=", "b": 12}, {"a": _i("hour"), "op": "<=", "b": 16},
                 {"a": _i("body_atr"), "op": ">", "b": 0.8},
                 {"a": _i("close"), "op": ">", "b": _i("ema", n=50)}],
        "short": [{"a": _i("hour"), "op": ">=", "b": 12}, {"a": _i("hour"), "op": "<=", "b": 16},
                  {"a": _i("body_atr"), "op": "<", "b": -0.8},
                  {"a": _i("close"), "op": "<", "b": _i("ema", n=50)}],
        "sl_atr": 1.2, "tp_atr": 1.8, "atr_n": 14, "max_bars": 20,
    },
]

TEMPLATES = [validate_spec(t) for t in TEMPLATES_RAW]


def _mutate_operand(op, rng: np.random.Generator):
    if isinstance(op, dict):
        out = dict(op)
        for k, v in op.items():
            if k != "ind" and rng.random() < 0.6:
                out[k] = v * rng.uniform(0.6, 1.6)
        return out
    jitter = rng.uniform(0.75, 1.25)
    return op * jitter if abs(op) > 1e-9 else rng.normal(0, 0.3)


def mutate(spec: dict, rng: np.random.Generator) -> dict | None:
    """Return a perturbed, validated copy of ``spec`` (or None if it became invalid)."""
    s = clone(spec)
    for side in ("long", "short"):
        for c in s[side]:
            if rng.random() < 0.5:
                c["a"] = _mutate_operand(c["a"], rng)
            if rng.random() < 0.5:
                c["b"] = _mutate_operand(c["b"], rng)
        if len(s[side]) > 2 and rng.random() < 0.1:
            s[side].pop(int(rng.integers(len(s[side]))))
    s["sl_atr"] *= rng.uniform(0.7, 1.4)
    s["tp_atr"] *= rng.uniform(0.7, 1.5)
    if rng.random() < 0.4:
        s["max_bars"] = int(s["max_bars"] * rng.uniform(0.5, 2.0))
    s["name"] = (spec["name"].split(" ~")[0] + " ~mut")[:60]
    try:
        return validate_spec(s)
    except SpecError:
        return None
