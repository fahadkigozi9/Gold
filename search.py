"""Strategy search: templates + Gemini proposals + local mutation, validated out-of-sample."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .backtester import BTConfig, MarketData, run_backtest
from .dsl import SeriesCache, signals, spec_hash, validate_spec
from .strategies import TEMPLATES, mutate

OBJECTIVES = {
    "balanced": "Balanced (SQN: edge x sqrt(trades))",
    "winrate": "High win rate (needs PF >= 1.2 and positive expectancy)",
    "profit_factor": "Profit factor",
    "return_dd": "Return / max drawdown",
}


def score(m: dict, objective: str, min_trades: int) -> float:
    """Higher is better. Under-sampled strategies are pushed to the bottom."""
    if m["trades"] < min_trades:
        return -1000.0 + m["trades"] / max(min_trades, 1)
    if objective == "winrate":
        if m["profit_factor"] < 1.2 or m["expectancy_r"] <= 0:
            return -100.0 + m["win_rate"] / 100.0
        return m["win_rate"] + 8.0 * min(m["profit_factor"], 3.0) - 0.3 * m["max_dd_pct"]
    if objective == "profit_factor":
        shrink = min(1.0, m["trades"] / (3.0 * min_trades))
        return (min(m["profit_factor"], 5.0) - 1.0) * shrink - 0.01 * m["max_dd_pct"]
    if objective == "return_dd":
        return float(m["return_dd"]) if m["expectancy_r"] > 0 else -10.0 + m["return_dd"]
    return m["sqn"] - 0.02 * m["max_dd_pct"]


@dataclass
class SearchConfig:
    objective: str = "balanced"
    rounds: int = 4
    per_round: int = 8
    mutations_per_round: int = 12
    min_trades: int = 40
    holdout_frac: float = 0.3
    top_k: int = 10
    seed: int = 1


@dataclass
class Candidate:
    spec: dict
    source: str
    round: int
    train: dict
    score: float
    id: str = ""
    holdout: dict | None = None
    robust: bool | None = None
    fold_profitable: str = ""

    def row(self) -> dict:
        h = self.holdout or {}
        t = self.train
        return {
            "id": self.id, "name": self.spec["name"], "source": self.source, "round": self.round,
            "score": round(self.score, 3),
            "train_trades": t["trades"], "train_win%": t["win_rate"], "train_PF": t["profit_factor"],
            "train_expR": t["expectancy_r"], "train_DD%": t["max_dd_pct"],
            "hold_trades": h.get("trades"), "hold_win%": h.get("win_rate"), "hold_PF": h.get("profit_factor"),
            "hold_expR": h.get("expectancy_r"), "hold_DD%": h.get("max_dd_pct"),
            "hold_return%": h.get("total_return_pct"),
            "folds_profitable": self.fold_profitable, "robust": self.robust,
        }


class Lab:
    """Holds one dataset and evaluates specs against it with shared indicator caching."""

    def __init__(self, df: pd.DataFrame, cfg: BTConfig, holdout_frac: float = 0.3):
        self.df, self.cfg = df, cfg
        self.md = MarketData(df)
        self.cache = SeriesCache(df)
        self.split = int(len(df) * (1.0 - holdout_frac))

    def _signals(self, spec: dict):
        sig = signals(spec, self.cache)
        atr = self.cache.get("atr", n=spec["atr_n"]).to_numpy(float)
        return sig, atr

    def evaluate(self, spec: dict, lo: int = 0, hi: int | None = None):
        sig, atr = self._signals(spec)
        return run_backtest(self.md, sig, atr, spec, self.cfg, lo, hi)

    def train(self, spec):
        return self.evaluate(spec, 0, self.split)[1]

    def holdout(self, spec):
        return self.evaluate(spec, self.split, None)[1]

    def folds(self, spec: dict, k: int = 5) -> list[dict]:
        edges = np.linspace(0, self.md.n, k + 1).astype(int)
        return [self.evaluate(spec, int(a), int(b))[1] for a, b in zip(edges[:-1], edges[1:])]


def _compact(c: Candidate) -> dict:
    t = c.train
    return {"spec": {k: c.spec[k] for k in ("name", "long", "short", "sl_atr", "tp_atr", "atr_n", "max_bars")},
            "train": {k: t[k] for k in ("trades", "win_rate", "profit_factor", "expectancy_r", "max_dd_pct")}}


def build_prompt(summary: dict, board: list[Candidate], n: int, objective: str, rnd: int,
                 cfg: BTConfig) -> str:
    parts = [
        f"Dataset: {json.dumps(summary)}",
        f"Trading costs per trade (price units): spread {cfg.spread}, slippage {cfg.slippage}/side. "
        f"Risk {cfg.risk_pct}% per trade.",
        f"Objective: {OBJECTIVES[objective]}. You only see TRAIN-period results; a hidden holdout will judge robustness.",
        f"Round {rnd}. Propose exactly {n} NEW strategies as JSON.",
    ]
    if board:
        top = [_compact(c) for c in board[:6]]
        bad = [_compact(c) for c in board[-3:]] if len(board) > 9 else []
        parts.append("Current best (refine/combine, do not duplicate):\n" + json.dumps(top))
        if bad:
            parts.append("Worst performers (avoid these ideas):\n" + json.dumps(bad))
    else:
        parts.append("No results yet: propose a diverse first batch.")
    return "\n\n".join(parts)


def run_search(lab: Lab, scfg: SearchConfig, gemini=None, summary: dict | None = None,
               progress: Callable[[float, str], None] | None = None) -> tuple[list[Candidate], list[str]]:
    """Run the search; returns (ranked candidates incl. holdout validation, log lines)."""
    from .data import summarize_market

    rng = np.random.default_rng(scfg.seed)
    summary = summary or summarize_market(lab.df.iloc[:lab.split])
    log: list[str] = []
    seen: dict[str, Candidate] = {}

    def say(frac, msg):
        log.append(msg)
        if progress:
            progress(min(max(frac, 0.0), 1.0), msg)

    def add(spec: dict, source: str, rnd: int) -> Candidate | None:
        h = spec_hash(spec)
        if h in seen:
            return None
        m = lab.train(spec)
        cand = Candidate(spec, source, rnd, m, score(m, scfg.objective, scfg.min_trades), id=h)
        seen[h] = cand
        return cand

    def board():
        return sorted(seen.values(), key=lambda c: c.score, reverse=True)

    total = scfg.rounds + 2
    say(0.02, f"Evaluating {len(TEMPLATES)} template strategies on the training window...")
    for t in TEMPLATES:
        add(t, "template", 0)

    for rnd in range(1, scfg.rounds + 1):
        new = 0
        if gemini is not None:
            say(rnd / total, f"Round {rnd}/{scfg.rounds}: asking Gemini for {scfg.per_round} strategies...")
            try:
                prompt = build_prompt(summary, board(), scfg.per_round, scfg.objective, rnd, lab.cfg)
                specs, errs = gemini.propose(prompt)
                for e in errs[:3]:
                    log.append(f"  rejected spec: {e}")
                for s in specs:
                    new += add(s, "gemini", rnd) is not None
                log.append(f"  Gemini: {len(specs)} valid, {len(errs)} rejected")
            except Exception as exc:  # keep going with local search
                log.append(f"  Gemini failed this round ({exc}); continuing with mutations only.")
        top = [c for c in board() if c.score > -500][:5] or board()[:5]
        for _ in range(scfg.mutations_per_round):
            parent = top[int(rng.integers(len(top)))]
            child = mutate(parent.spec, rng)
            if child is not None:
                new += add(child, "mutation", rnd) is not None
        b = board()
        say((rnd + 0.5) / total, f"Round {rnd}: +{new} new, best train score {b[0].score:.2f} ({b[0].spec['name']})")

    say((total - 1) / total, "Validating top candidates on the unseen holdout and 5 chronological folds...")
    finalists = board()[:scfg.top_k]
    for c in finalists:
        c.holdout = lab.holdout(c.spec)
        folds = lab.folds(c.spec, 5)
        ok = sum(1 for f in folds if f["trades"] >= 5 and f["expectancy_r"] > 0)
        c.fold_profitable = f"{ok}/5"
        h = c.holdout
        # Strict gate: searching many variants guarantees some look good by luck, so the bar is high.
        c.robust = bool(h["trades"] >= max(30, scfg.min_trades) and h["profit_factor"] >= 1.25
                        and h["expectancy_r"] > 0 and ok >= 4
                        and c.train["expectancy_r"] > 0
                        and h["sqn"] >= 2.5  # t-stat of holdout mean R; ~Bonferroni for top_k finalists
                        and h["profit_factor"] >= 0.6 * c.train["profit_factor"]
                        and h["max_dd_pct"] <= 1.5 * max(c.train["max_dd_pct"], 5.0))
    finalists.sort(key=lambda c: (c.robust is True, score(c.holdout, scfg.objective, max(10, scfg.min_trades // 3))),
                   reverse=True)
    say(1.0, f"Done. {sum(1 for c in finalists if c.robust)} of {len(finalists)} finalists passed robustness checks.")
    return finalists, log
