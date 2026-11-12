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
        self.changed = False

    def update(self, m: dict[str, float]) -> tuple[float | None, float]:
        """Returns (rate, token delta); sets ``changed`` when the counters moved."""
        self.changed = False
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
            self.changed = True
        return self.last, dc


class ThroughputTracker:
    """llama-server adds a request's tokens to /metrics only when the request
    finishes, so while generating the live rate comes from /slots and the
    exact decode-time average from /metrics once the request is done."""

    WINDOW = 60.0

    def __init__(self):
        self._gen = _CounterRate("tokens_predicted_total", "tokens_predicted_seconds_total")
        self._prompt = _CounterRate("prompt_tokens_total", "prompt_seconds_total")
        self._slot_prev: dict[int, int] = {}
        self._slot_t: float | None = None
        self._slot_last: float | None = None
        self._window: deque[tuple[float, float]] = deque()
        # live slot tokens not yet seen in the completed-request counter
        self._uncommitted = 0.0

    def _tokens_last_min(self, t: float, delta: float) -> int:
        if delta > 0:
            self._window.append((t, delta))
        while self._window and self._window[0][0] <= t - self.WINDOW:
            self._window.popleft()
        return int(sum(d for _, d in self._window))

    def _slots(self, t: float, slots: list[dict]) -> tuple[float | None, int, bool, bool]:
        """(live rate this tick or None, decoded delta, busy, first sample)."""
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
        rate = None
        if dt and dt > 0 and delta > 0:
            rate = delta / dt
            self._slot_last = rate
        return rate, delta, busy, first

    def update(self, t: float, metrics: dict[str, float] | None,
               slots: list[dict] | None) -> Throughput:
        live = self._slots(t, slots) if slots is not None else None
        if metrics is not None and "tokens_predicted_total" in metrics:
            gen, dgen = self._gen.update(metrics)
            prompt, _ = self._prompt.update(metrics)
            proc = metrics.get("requests_processing")
            if proc is not None:
                busy = proc > 0
            elif live is not None:
                busy = live[2]
            else:
                busy = dgen > 0
            gen_out, source = gen, "metrics"
            if busy and not self._gen.changed and live is not None and live[0] is not None:
                gen_out, source = live[0], "slots"
            # count tokens once: live slot deltas, plus the part of the counter
            # delta (finished requests) that the slots did not show between polls
            if live is None:
                delta = dgen
            else:
                self._uncommitted += live[1]
                unseen = max(0.0, dgen - self._uncommitted)
                self._uncommitted = max(0.0, self._uncommitted - dgen)
                delta = live[1] + unseen
            return Throughput(
                gen_tps=gen_out, prompt_tps=prompt, busy=bool(busy),
                tokens_last_min=self._tokens_last_min(t, delta), source=source,
                requests_processing=int(proc) if proc is not None else None,
                requests_deferred=int(metrics["requests_deferred"])
                if "requests_deferred" in metrics else None)
        if live is not None:
            rate, delta, busy, first = live
            return Throughput(
                gen_tps=None if first else self._slot_last, busy=busy,
                tokens_last_min=self._tokens_last_min(t, delta), source="slots",
                requests_processing=sum(1 for s in slots if s.get("is_processing")))
        return Throughput()
