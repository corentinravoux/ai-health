"""Short-timeout HTTP probes of a llama-server (/health, /props, /slots, /metrics)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .metrics import parse_prometheus

Fetch = Callable[[str, float], tuple[int, bytes]]


@dataclass
class ProbeResult:
    ok: bool
    status: str  # "ok" | "disabled" | "unreachable" | "auth" | "loading" | "error"
    data: object | None = None
    detail: str | None = None


def _urllib_fetch(url: str, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "ai-health"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(4 * 2**20)
    except urllib.error.HTTPError as e:
        return e.code, e.read(65536) if e.fp else b""


def _parse_slots(body: bytes) -> list[dict]:
    """``/slots`` must be a list of slot objects; anything else (an error
    object, ``[null]``) is a bad payload, not an empty or idle server."""
    data = json.loads(body)
    if not isinstance(data, list) or not all(isinstance(s, dict) for s in data):
        raise ValueError("expected a JSON list of slot objects")
    return data


class ServerProbe:
    def __init__(self, host: str, port: int, timeout: float = 0.3, fetch: Fetch | None = None):
        if host in ("0.0.0.0", "", "::", "[::]"):
            host = "127.0.0.1"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self._fetch = fetch or _urllib_fetch

    def _get(self, path: str, model: str | None, parse: Callable[[bytes], object]) -> ProbeResult:
        url = self.base + path
        if model:
            url += "?model=" + urllib.parse.quote(model, safe="")
        try:
            code, body = self._fetch(url, self.timeout)
        except (OSError, TimeoutError, urllib.error.URLError) as e:
            return ProbeResult(False, "unreachable", detail=str(e))
        if code in (401, 403):
            return ProbeResult(False, "auth", detail="authentication required (--api-key)")
        if code in (404, 501):
            return ProbeResult(False, "disabled", detail=f"HTTP {code}")
        if code == 503:
            return ProbeResult(False, "loading", detail=body[:200].decode(errors="replace"))
        if code != 200:
            return ProbeResult(False, "error", detail=f"HTTP {code}")
        try:
            return ProbeResult(True, "ok", data=parse(body))
        except (ValueError, UnicodeDecodeError) as e:
            return ProbeResult(False, "error", detail=f"bad payload: {e}")

    def health(self) -> ProbeResult:
        return self._get("/health", None, json.loads)

    def props(self, model: str | None = None) -> ProbeResult:
        return self._get("/props", model, json.loads)

    def slots(self) -> ProbeResult:
        return self._get("/slots", None, _parse_slots)

    def metrics(self, model: str | None = None) -> ProbeResult:
        return self._get("/metrics", model, lambda b: parse_prometheus(b.decode()))
