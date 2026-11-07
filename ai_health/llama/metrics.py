"""Prometheus text parsing and tokens/s computation for llama-server.

Preferred source: ``/metrics`` counters (``tokens_predicted_total`` over
``tokens_predicted_seconds_total`` = true decode speed while generating).
Fallback: ``/slots`` ``next_token.n_decoded`` deltas over wall time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

PREFIX = "llamacpp:"


def parse_prometheus(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        if not name:
            continue
        try:
            v = float(value)
        except ValueError:
            continue
        if name.startswith(PREFIX):
            name = name[len(PREFIX):]
        out[name.strip()] = v
    return out


def slot_decoded(slot: dict) -> int | None:
    """Tokens generated so far in a slot. ``next_token`` is an object in older
    llama.cpp builds and a list (one entry per sequence) in newer ones."""
    nt = slot.get("next_token")
    if isinstance(nt, dict):
        nt = [nt]
    if not isinstance(nt, list):
        return None
    vals = [int(x.get("n_decoded") or 0) for x in nt if isinstance(x, dict)]
    return sum(vals) if vals else None


@dataclass
class Throughput:
    gen_tps: float | None = None
    prompt_tps: float | None = None
    busy: bool | None = None
    tokens_last_min: int | None = None
    source: str = "none"  # "metrics" | "slots" | "none"
    requests_processing: int | None = None
    requests_deferred: int | None = None


class _CounterRate:
    """Rate = d(count)/d(seconds) between updates; resets are not negative."""

    def __init__(self, count_key: str, secs_key: str):
        self.count_key, self.secs_key = count_key, secs_key
        self.prev: tuple[float, float] | None = None
        self.last: float | None = None

    def update(self, m: dict[str, float]) -> tuple[float | None, float]:
        """Returns (rate, token delta)."""
        if self.count_key not in m or self.secs_key not in m:
            return None, 0.0
        cur = (m[self.count_key], m[self.secs_key])
        prev, self.prev = self.prev, cur
        if prev is None:
            return None, 0.0
        dc, ds = cur[0] - prev[0], cur[1] - prev[1]
        if dc < 0 or ds < 0:  # server restarted: counters went back to 0
            self.last = None
            return None, 0.0
        if ds > 0:
            self.last = dc / ds
        return self.last, dc


class ThroughputTracker:
    WINDOW = 60.0

    def __init__(self):
        self._gen = _CounterRate("tokens_predicted_total", "tokens_predicted_seconds_total")
        self._prompt = _CounterRate("prompt_tokens_total", "prompt_seconds_total")
        self._slot_prev: dict[int, int] = {}
        self._slot_t: float | None = None
        self._slot_last: float | None = None
        self._window: deque[tuple[float, float]] = deque()

    def _tokens_last_min(self, t: float, delta: float) -> int:
        if delta > 0:
            self._window.append((t, delta))
        while self._window and self._window[0][0] <= t - self.WINDOW:
            self._window.popleft()
        return int(sum(d for _, d in self._window))

    def update(self, t: float, metrics: dict[str, float] | None,
               slots: list[dict] | None) -> Throughput:
        if metrics is not None and "tokens_predicted_total" in metrics:
            gen, dgen = self._gen.update(metrics)
            prompt, _ = self._prompt.update(metrics)
            proc = metrics.get("requests_processing")
            busy = proc > 0 if proc is not None else dgen > 0
            return Throughput(
                gen_tps=gen, prompt_tps=prompt, busy=bool(busy),
                tokens_last_min=self._tokens_last_min(t, dgen), source="metrics",
                requests_processing=int(proc) if proc is not None else None,
                requests_deferred=int(metrics["requests_deferred"])
                if "requests_deferred" in metrics else None)
        if slots is not None:
            delta = 0
            cur: dict[int, int] = {}
            for s in slots:
                n = slot_decoded(s) or 0
                sid = int(s.get("id", 0))
                cur[sid] = n
                prev = self._slot_prev.get(sid)
                if prev is not None:
                    delta += n - prev if n >= prev else n
            first = self._slot_t is None
            dt = None if first else t - self._slot_t
            self._slot_prev, self._slot_t = cur, t
            busy = any(bool(s.get("is_processing")) for s in slots)
            if dt and dt > 0 and delta > 0:
                self._slot_last = delta / dt
            return Throughput(
                gen_tps=None if first else self._slot_last, busy=busy,
                tokens_last_min=self._tokens_last_min(t, delta), source="slots",
                requests_processing=sum(1 for s in slots if s.get("is_processing")))
        return Throughput()
