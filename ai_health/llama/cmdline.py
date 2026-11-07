"""Parse the launch flags of a llama.cpp process (argv + LLAMA_ARG_* env).

Aliases and env names verified against ``~/Software/llama.cpp/common/arg.cpp``
(2026-11-07). CLI wins over env; unknown flags are ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

TRUTHY = {"1", "true", "on", "yes", "enabled", "enable"}
FALSY = {"0", "false", "off", "no", "disabled", "disable"}


@dataclass
class LlamaArgs:
    kind: str = "other"  # "server" | "cli" | "bench" | "other"
    model: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    ngl: int | str = "auto"  # int, "auto", "all"
    fit: bool = True
    cpu_moe: bool = False
    n_cpu_moe: int = 0
    n_cpu_ffn: int = 0
    overrides: list[tuple[str, str]] = field(default_factory=list)
    ctx: int | None = None
    threads: int | None = None
    batch: int | None = None
    ubatch: int | None = None
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    flash_attn: str = "auto"
    kv_offload: bool = True
    metrics: bool = False
    slots: bool = True
    router: bool = False
    api_key_set: bool = False
    alias: str | None = None
    parallel: int | None = None
    hf_repo: str | None = None
    raw: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)


# flag -> (field, kind). kinds: str int ngl ot fit fa on off secret
_FLAGS: dict[str, tuple[str, str]] = {}


def _add(names: str, fieldname: str, kind: str) -> None:
    for n in names.split():
        _FLAGS[n] = (fieldname, kind)


_add("-m --model", "model", "str")
_add("-hf -hfr --hf-repo", "hf_repo", "str")
_add("--host", "host", "str")
_add("--port", "port", "int")
_add("-ngl --gpu-layers --n-gpu-layers", "ngl", "ngl")
_add("-fit --fit", "fit", "fit")
_add("-cmoe --cpu-moe", "cpu_moe", "on")
_add("-ncmoe --n-cpu-moe", "n_cpu_moe", "int")
_add("-ncffn --n-cpu-ffn", "n_cpu_ffn", "int")
_add("-ot --override-tensor", "overrides", "ot")
_add("-c --ctx-size", "ctx", "int")
_add("-t --threads", "threads", "int")
_add("-b --batch-size", "batch", "int")
_add("-ub --ubatch-size", "ubatch", "int")
_add("-ctk --cache-type-k", "cache_type_k", "str")
_add("-ctv --cache-type-v", "cache_type_v", "str")
_add("-fa --flash-attn", "flash_attn", "fa")
_add("-kvo --kv-offload", "kv_offload", "on")
_add("-nkvo --no-kv-offload", "kv_offload", "off")
_add("--metrics", "metrics", "on")
_add("--slots", "slots", "on")
_add("--no-slots", "slots", "off")
_add("-np --parallel", "parallel", "int")
_add("-a --alias", "alias", "str")
_add("--api-key --api-key-file", "api_key_set", "secret")

_ENV: dict[str, tuple[str, str]] = {
    "LLAMA_ARG_MODEL": ("model", "str"),
    "LLAMA_ARG_HF_REPO": ("hf_repo", "str"),
    "LLAMA_ARG_HOST": ("host", "str"),
    "LLAMA_ARG_PORT": ("port", "int"),
    "LLAMA_ARG_N_GPU_LAYERS": ("ngl", "ngl"),
    "LLAMA_ARG_FIT": ("fit", "fit"),
    "LLAMA_ARG_CPU_MOE": ("cpu_moe", "bool"),
    "LLAMA_ARG_N_CPU_MOE": ("n_cpu_moe", "int"),
    "LLAMA_ARG_N_CPU_FFN": ("n_cpu_ffn", "int"),
    "LLAMA_ARG_OVERRIDE_TENSOR": ("overrides", "ot"),
    "LLAMA_ARG_CTX_SIZE": ("ctx", "int"),
    "LLAMA_ARG_THREADS": ("threads", "int"),
    "LLAMA_ARG_BATCH": ("batch", "int"),
    "LLAMA_ARG_UBATCH": ("ubatch", "int"),
    "LLAMA_ARG_CACHE_TYPE_K": ("cache_type_k", "str"),
    "LLAMA_ARG_CACHE_TYPE_V": ("cache_type_v", "str"),
    "LLAMA_ARG_FLASH_ATTN": ("flash_attn", "fa"),
    "LLAMA_ARG_KV_OFFLOAD": ("kv_offload", "bool"),
    "LLAMA_ARG_ENDPOINT_METRICS": ("metrics", "bool"),
    "LLAMA_ARG_ENDPOINT_SLOTS": ("slots", "bool"),
    "LLAMA_ARG_N_PARALLEL": ("parallel", "int"),
    "LLAMA_ARG_ALIAS": ("alias", "str"),
    "LLAMA_API_KEY": ("api_key_set", "secret"),
    "LLAMA_ARG_API_KEY_FILE": ("api_key_set", "secret"),
}

_VALUE_KINDS = {"str", "int", "ngl", "ot", "fit", "fa", "secret", "bool"}


def _kind_of(argv0: str) -> str:
    base = os.path.basename(argv0)
    if base == "llama-server":
        return "server"
    if base in ("llama-cli", "llama"):
        return "cli"
    if base == "llama-bench":
        return "bench"
    return "other"


def _convert(kind: str, value: str):
    if kind == "int":
        return int(value)
    if kind == "ngl":
        v = value.strip().lower()
        if v in ("auto", "all"):
            return v
        n = int(v)
        return "auto" if n == -1 else "all" if n <= -2 else n
    if kind in ("fit", "bool"):
        v = value.strip().lower()
        if v in TRUTHY:
            return True
        if v in FALSY:
            return False
        raise ValueError(value)
    if kind == "fa":
        v = value.strip().lower()
        return "on" if v in TRUTHY else "off" if v in FALSY else "auto"
    if kind == "ot":
        pairs = []
        for item in value.split(","):
            if "=" in item:
                pat, buf = item.rsplit("=", 1)
                pairs.append((pat, buf.strip()))
        return pairs
    if kind == "secret":
        return True
    return value


def _apply(a: LlamaArgs, fieldname: str, kind: str, value, source: str) -> None:
    try:
        if kind == "on":
            v = True
        elif kind == "off":
            v = False
        else:
            v = _convert(kind, value)
    except (ValueError, TypeError):
        return
    if kind == "ot":
        if a.sources.get("overrides") == "env" and source == "cli":
            a.overrides = []
        a.overrides.extend(v)
    else:
        setattr(a, fieldname, v)
    a.sources[fieldname] = source


def parse_llama_args(argv: list[str], environ: dict[str, str] | None = None) -> LlamaArgs:
    a = LlamaArgs(kind=_kind_of(argv[0]) if argv else "other")
    for f in ("ngl", "ctx", "fit", "metrics", "kv_offload", "cache_type_k", "cache_type_v"):
        a.sources[f] = "default"

    for name, value in (environ or {}).items():
        if name in _ENV:
            fieldname, kind = _ENV[name]
            _apply(a, fieldname, kind, value, "env")

    raw: list[str] = [argv[0]] if argv else []
    cli_seen: set[str] = set()
    i = 1
    while i < len(argv):
        tok = argv[i]
        flag, eq, inline = tok.partition("=") if tok.startswith("-") else (tok, "", "")
        spec = _FLAGS.get(flag)
        if spec is None:
            raw.append(tok)
            i += 1
            continue
        fieldname, kind = spec
        if kind in _VALUE_KINDS:
            if eq:
                value = inline
                i += 1
            elif i + 1 < len(argv):
                value = argv[i + 1]
                i += 2
            else:
                i += 1
                continue
            if fieldname == "overrides" and fieldname not in cli_seen:
                a.overrides = []  # CLI replaces env overrides
            raw.extend([flag, "***" if kind == "secret" else value])
        else:
            value = None
            raw.append(tok)
            i += 1
        cli_seen.add(fieldname)
        _apply(a, fieldname, kind, value, "cli")

    a.raw = raw
    a.router = a.kind == "server" and not a.model and not a.hf_repo
    return a


def key_flags_summary(a: LlamaArgs) -> list[tuple[str, str]]:
    """Short (label, value) pairs for UI chips, most important first."""
    out: list[tuple[str, str]] = []
    if a.ngl == "auto":
        out.append(("ngl", "auto (fit)" if a.fit else "auto"))
    else:
        out.append(("ngl", str(a.ngl)))
    out.append(("ctx", str(a.ctx) if a.ctx else "model default"))
    if a.cpu_moe:
        out.append(("cpu-moe", "all"))
    if a.n_cpu_moe:
        out.append(("n-cpu-moe", str(a.n_cpu_moe)))
    if a.n_cpu_ffn:
        out.append(("n-cpu-ffn", str(a.n_cpu_ffn)))
    for pat, buf in a.overrides:
        out.append(("ot", f"{pat}={buf}"))
    if a.cache_type_k != "f16" or a.cache_type_v != "f16":
        out.append(("kv", f"{a.cache_type_k}/{a.cache_type_v}"))
    if a.flash_attn != "auto":
        out.append(("fa", a.flash_attn))
    if not a.kv_offload:
        out.append(("kv-offload", "off"))
    if a.threads:
        out.append(("threads", str(a.threads)))
    if a.parallel:
        out.append(("parallel", str(a.parallel)))
    if a.kind == "server":
        out.append(("metrics", "on" if a.metrics else "off"))
    return out
