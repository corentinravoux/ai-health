"""API handlers: pure functions over the services, returning (status, payload)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from ai_health import __version__
from ai_health.explorer import external
from ai_health.explorer.placement import args_from_flags
from ai_health.llama.cmdline import LlamaArgs


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


def _explorer(ctx):
    if ctx.explorer is None:
        raise ApiError(503, "explorer unavailable")
    return ctx.explorer


def _need_model(ctx):
    ex = _explorer(ctx)
    if ex.current() is None:
        raise ApiError(409, "no model open")
    return ex


def model_open(ctx, query, body):
    path = (body or {}).get("path")
    if not isinstance(path, str) or not path.strip():
        raise ApiError(400, "body must contain a 'path' string")
    try:
        return 200, _explorer(ctx).open(path.strip())
    except FileNotFoundError as e:
        raise ApiError(404, str(e)) from None
    except (IsADirectoryError, ValueError) as e:
        raise ApiError(400, str(e)) from None


def model_close(ctx, query, body):
    _explorer(ctx).close()
    return 200, {"ok": True}


def model_architecture(ctx, query, body):
    return 200, _need_model(ctx).architecture()


def model_tensors(ctx, query, body):
    return 200, _need_model(ctx).tensors()


def model_tensor(ctx, query, body):
    ex = _need_model(ctx)
    name = query.get("name", [""])[0]
    reduce = query.get("reduce", ["absmax"])[0]
    if reduce not in ("absmax", "mean"):
        raise ApiError(400, "reduce must be absmax or mean")
    try:
        return 200, ex.view(name, reduce)
    except KeyError:
        raise ApiError(404, f"no tensor named {name!r}") from None


def _gpu_numbers(ctx) -> tuple[int | None, int | None]:
    devices = ((ctx.sampler.latest() or {}).get("gpu") or {}).get("devices") or []
    if not devices:
        return None, None
    d = devices[0]
    return d["mem_total"] - d["mem_used"], d["mem_total"]


def _args_from_dict(d: dict) -> LlamaArgs:
    names = {f.name for f in dataclasses.fields(LlamaArgs)}
    a = LlamaArgs(**{k: v for k, v in d.items() if k in names and k != "overrides"})
    a.overrides = [tuple(x) for x in d.get("overrides") or []]
    return a


def model_placement(ctx, query, body):
    ex = _need_model(ctx)
    body = body or {}
    free, capacity = _gpu_numbers(ctx)
    if "pid" in body:
        servers = (ctx.sampler.latest() or {}).get("servers") or []
        srv = next((s for s in servers if s.get("pid") == body["pid"]), None)
        if srv is None or not srv.get("args"):
            raise ApiError(404, f"no llama-server with pid {body['pid']}")
        args = _args_from_dict(srv["args"])
        if free is not None:
            free += srv.get("vram_bytes") or 0  # the server's own weights would be freed
        source = f"pid {body['pid']}"
    elif isinstance(body.get("flags"), str):
        args = args_from_flags(body["flags"])
        source = "flags"
    else:
        raise ApiError(400, "body must contain 'pid' or 'flags'")
    out = ex.placement(args, free, capacity)
    out.update(source=source, vram_free_for_model=free, flags=_flag_text(args))
    return 200, out


def _flag_text(a: LlamaArgs) -> str:
    from ai_health.llama.cmdline import key_flags_summary

    return " ".join(f"{k}={v}" for k, v in key_flags_summary(a))


def model_files(ctx, query, body):
    return 200, _explorer(ctx).recent_and_found()


def viewers(ctx, query, body):
    return 200, external.available_viewers()


def viewers_launch(ctx, query, body):
    tool = (body or {}).get("tool")
    if tool not in ("netron", "model_explorer"):
        raise ApiError(400, f"unknown viewer: {tool}")
    ex = _need_model(ctx)
    try:
        return 200, {"message": external.launch_viewer(tool, Path(ex.current().path))}
    except RuntimeError as e:
        raise ApiError(409, str(e)) from None


def settings_get(ctx, query, body):
    s = ctx.settings
    return 200, {"theme": s.theme if s else "dark", "history_s": s.history_s if s else 900}


def settings_set(ctx, query, body):
    """The desktop window runs in private mode (no localStorage), so UI
    preferences are kept in the settings file instead."""
    if ctx.settings is None:
        raise ApiError(503, "settings unavailable")
    theme = (body or {}).get("theme")
    if theme not in ("dark", "light"):
        raise ApiError(400, "theme must be 'dark' or 'light'")
    ctx.settings.theme = theme
    if ctx.save_settings:
        ctx.save_settings(ctx.settings)
    return settings_get(ctx, query, body)


SETTINGS_ROUTES = {
    ("GET", "/api/settings"): settings_get,
    ("POST", "/api/settings"): settings_set,
}

EXPLORER_ROUTES = {
    ("POST", "/api/model/open"): model_open,
    ("POST", "/api/model/close"): model_close,
    ("GET", "/api/model/architecture"): model_architecture,
    ("GET", "/api/model/tensors"): model_tensors,
    ("GET", "/api/model/tensor"): model_tensor,
    ("POST", "/api/model/placement"): model_placement,
    ("GET", "/api/model/files"): model_files,
    ("GET", "/api/viewers"): viewers,
    ("POST", "/api/viewers/launch"): viewers_launch,
}

ROUTES = {
    ("GET", "/api/snapshot"): snapshot,
    ("GET", "/api/history"): history,
    ("GET", "/api/meta"): meta,
}


def all_routes() -> dict:
    return {**ROUTES, **EXPLORER_ROUTES}
