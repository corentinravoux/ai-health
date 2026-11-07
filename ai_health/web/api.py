"""API handlers: pure functions over the services, returning (status, payload)."""

from __future__ import annotations

from ai_health import __version__


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _int_param(query: dict, name: str, default: int, lo: int, hi: int) -> int:
    raw = query.get(name, [str(default)])[0]
    try:
        v = int(raw)
    except ValueError:
        raise ApiError(400, f"{name} must be an integer") from None
    return max(lo, min(hi, v))


def snapshot(ctx, query, body):
    return 200, ctx.sampler.latest()


def history(ctx, query, body):
    s = ctx.sampler
    return 200, s.history(_int_param(query, "seconds", s.history_s, 1, s.history_s))


def meta(ctx, query, body):
    latest = ctx.sampler.latest() or {}
    devices = (latest.get("gpu") or {}).get("devices") or []
    return 200, {
        "version": __version__,
        "history_s": ctx.sampler.history_s,
        "interval": ctx.sampler.interval,
        "gpu_name": devices[0]["name"] if devices else None,
        "vram_total": devices[0]["mem_total"] if devices else None,
        "ram_total": (latest.get("system") or {}).get("ram_total"),
        "explorer": ctx.explorer is not None,
    }


ROUTES = {
    ("GET", "/api/snapshot"): snapshot,
    ("GET", "/api/history"): history,
    ("GET", "/api/meta"): meta,
}
