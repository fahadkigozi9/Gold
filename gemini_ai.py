"""Thin Gemini wrapper that turns market context into validated strategy specs."""
from __future__ import annotations

import json
import re
import time
from typing import Callable

from .dsl import INDICATOR_DOC, OPS, SpecError, validate_spec

DEFAULT_MODEL = "gemini-3.8-flash"
MODEL_CHOICES = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview", "gemini-3.6-flash"]

SYSTEM_PROMPT = f"""You are a quantitative researcher designing rule-based intraday/swing strategies for gold (XAUUSD).
You output ONLY JSON. Strategies are written in a strict JSON DSL that a backtester executes. You never write code.

DSL
- A strategy: {{"name": str, "rationale": str, "long": [condition,...], "short": [condition,...],
  "sl_atr": float 0.2-8, "tp_atr": float 0.2-15, "atr_n": int 5-50, "max_bars": int 1-1000}}
- A condition: {{"a": operand, "op": one of {list(OPS)}, "b": operand}}. Conditions on one side are AND-ed.
- An operand is a number OR {{"ind": name, ...params}}. At least one operand per condition must be an indicator.
- Signals are read on bar close and executed at the next bar open. SL/TP are ATR multiples; if both are hit in one bar the stop is assumed first.
- Allowed indicators (anything else is rejected): {json.dumps(INDICATOR_DOC)}

Design rules
1. Maximize out-of-sample robustness, not in-sample fit. Prefer 2-4 conditions per side and few, round parameters.
2. Win rate alone is a trap: a high win rate with tiny TP and wide SL usually has negative expectancy after costs.
   Aim for a high win rate AND profit factor > 1.3 AND positive expectancy in R.
3. Make the batch diverse: trend pullback, breakout, mean reversion, volatility squeeze/expansion, session/hour filters, momentum exhaustion.
4. When given a leaderboard, refine winners (nudge parameters, add a filter that removes losing conditions), combine ideas, and fix weaknesses. Do not repeat a spec already listed.
5. Costs matter: gold spread is real. Avoid strategies that trade every few bars with tiny targets.
Return: {{"strategies": [ ... ]}}"""


class GeminiError(RuntimeError):
    pass


def _extract_json(text: str):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def parse_strategies(text: str) -> tuple[list[dict], list[str]]:
    """Parse model output into validated specs. Returns (specs, error_messages)."""
    data = _extract_json(text)
    if isinstance(data, dict):
        data = data.get("strategies", data.get("results", [data] if "long" in data or "short" in data else []))
    if not isinstance(data, list):
        raise GeminiError("Model did not return a list of strategies.")
    specs, errors = [], []
    for raw in data:
        try:
            specs.append(validate_spec(raw))
        except SpecError as exc:
            errors.append(f"{(raw or {}).get('name', '?') if isinstance(raw, dict) else '?'}: {exc}")
    return specs, errors


class GeminiClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, temperature: float = 0.9,
                 client=None, sleep: Callable[[float], None] = time.sleep):
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key)
        self._client = client
        self.model = model
        self.temperature = temperature
        self._sleep = sleep

    def _generate(self, prompt: str, system: str, json_mode: bool, temperature: float) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        last = None
        for attempt in range(4):
            try:
                resp = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                text = getattr(resp, "text", None)
                if not text:
                    raise GeminiError("Empty response from Gemini (possibly blocked).")
                return text
            except Exception as exc:  # network, 429, 5xx
                last = exc
                code = getattr(exc, "code", None)
                if code in (400, 401, 403, 404):
                    break  # not retryable: bad key / bad model name / bad request
                self._sleep(2 ** attempt * 1.5)
        raise GeminiError(f"Gemini request failed: {last}")

    def propose(self, prompt: str) -> tuple[list[dict], list[str]]:
        text = self._generate(prompt, SYSTEM_PROMPT, True, self.temperature)
        try:
            return parse_strategies(text)
        except (json.JSONDecodeError, GeminiError) as exc:
            raise GeminiError(f"Could not parse Gemini output: {exc}") from exc

    def review(self, spec: dict, train: dict, holdout: dict, folds: str = "") -> str:
        prompt = (
            "Critique this gold trading strategy for overfitting and practical risk. Be blunt and specific, "
            "max 180 words, then give 3 concrete improvements.\n"
            f"Spec: {json.dumps(spec)}\nTrain metrics: {json.dumps(train)}\n"
            f"Holdout metrics (unseen): {json.dumps(holdout)}\nFold consistency: {folds}"
        )
        return self._generate(prompt, "You are a skeptical quant risk reviewer.", False, 0.4)
