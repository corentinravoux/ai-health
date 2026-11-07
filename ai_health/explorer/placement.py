"""Where each tensor lives (GPU or CPU) for a given set of llama.cpp flags,
and a VRAM estimate (weights + KV cache + compute buffer).

Rules from ~/Software/llama.cpp (verified 2026-11-07):
- src/llama-model.cpp: i_gpu_start = max(n_layer + 1 - ngl, 0); repeating
  layer il is on GPU iff il >= i_gpu_start; the output layer is index n_layer;
  the input layer (token_embd) always stays on CPU; ngl < 0 means all.
- common/common.h: --cpu-moe / --n-cpu-moe N / --n-cpu-ffn N add CPU
  overrides with LLM_FFN_EXPS_REGEX / LLM_FFN_DENSE_REGEX.
- src/llama-model-loader.cpp: overrides are regex_search'ed in order, first
  match wins, and apply to any tensor.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

from ai_health.llama.cmdline import LlamaArgs, parse_llama_args

from .model import ModelInfo

GiB = 2**30
MiB = 2**20
FIT_MARGIN = 1 * GiB
EXPS_RE = r"\.ffn_(up|down|gate|gate_up)_(ch|)exps"
DENSE_RE = r"\.ffn_(up|down|gate)\."
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625,
            "q4_1": 0.625, "q5_0": 0.6875, "q5_1": 0.75, "iq4_nl": 0.5625}


@dataclass
class Placement:
    device: dict[str, str]
    layers: list[dict]
    gpu_weight_bytes: int
    cpu_weight_bytes: int
    ngl_effective: int | None
    estimated: bool
    rules: list[str] = field(default_factory=list)
    output_device: str = "GPU"  # output layer (also when tied to token_embd)


@dataclass
class VramEstimate:
    weights: int
    kv_cache: int
    compute_buffer: int
    total: int
    capacity: int | None
    fits: bool | None
    notes: list[str] = field(default_factory=list)


def args_from_flags(text: str) -> LlamaArgs:
    argv = shlex.split(text or "")
    if not argv or argv[0].startswith("-"):
        argv = ["llama-server", *argv]
    return parse_llama_args(argv)


def _overrides(args: LlamaArgs, n_cpu_moe: int) -> list[tuple[re.Pattern, str, str]]:
    out = []
    for pat, buf in args.overrides:
        try:
            out.append((re.compile(pat), "CPU" if buf.upper().startswith("CPU") else "GPU",
                        f"-ot {pat}={buf}"))
        except re.error:
            continue
    if args.cpu_moe:
        out.append((re.compile(EXPS_RE), "CPU", "--cpu-moe"))
    for i in range(n_cpu_moe):
        out.append((re.compile(rf"blk\.{i}{EXPS_RE}"), "CPU", f"--n-cpu-moe {n_cpu_moe}"))
    for i in range(args.n_cpu_ffn):
        out.append((re.compile(rf"blk\.{i}{DENSE_RE}"), "CPU", f"--n-cpu-ffn {args.n_cpu_ffn}"))
    return out


def _place(model: ModelInfo, args: LlamaArgs, ngl: int, n_cpu_moe: int) -> tuple[dict, set]:
    n = model.hparams.n_layer
    i_gpu_start = max(n + 1 - ngl, 0)
    ovr = _overrides(args, n_cpu_moe)
    device, used = {}, set()
    for t in model.tensors:
        if t.layer is not None and t.component != "vision":
            d = "GPU" if t.layer >= i_gpu_start else "CPU"
        elif t.component in ("output",) or t.role == "output_norm":
            d = "GPU" if n >= i_gpu_start else "CPU"
        else:  # input layer (token_embd...), multimodal, misc
            d = "CPU"
        for rx, target, label in ovr:
            if rx.search(t.name):
                d = target
                used.add(label)
                break
        device[t.name] = d
    return device, used


def _summarise(model: ModelInfo, device: dict) -> tuple[list[dict], int, int]:
    layers: dict[int, dict] = {}
    gpu = cpu = 0
    for t in model.tensors:
        on_gpu = device[t.name] == "GPU"
        gpu += t.n_bytes if on_gpu else 0
        cpu += 0 if on_gpu else t.n_bytes
        if t.layer is None or t.component == "vision":
            continue
        L = layers.setdefault(t.layer, {"index": t.layer, "gpu_bytes": 0, "cpu_bytes": 0})
        L["gpu_bytes" if on_gpu else "cpu_bytes"] += t.n_bytes
    out = []
    for i in sorted(layers):
        L = layers[i]
        L["device"] = ("MIXED" if L["gpu_bytes"] and L["cpu_bytes"]
                       else "GPU" if L["gpu_bytes"] else "CPU")
        out.append(L)
    return out, gpu, cpu


def _resolve_ngl(args: LlamaArgs, n: int) -> int:
    if isinstance(args.ngl, int):
        return args.ngl
    return n + 1  # "all", or "auto" before fitting


def _build(model, args, ngl, n_cpu_moe, estimated, rules) -> Placement:
    device, used = _place(model, args, ngl, n_cpu_moe)
    layers, gpu, cpu = _summarise(model, device)
    n = model.hparams.n_layer
    rules = list(rules)
    rules.insert(0, f"-ngl {ngl}: repeating layers {max(n + 1 - ngl, 0)}..{n - 1} on GPU"
                    + (", output layer on GPU" if ngl >= 1 else ", output layer on CPU")
                 if ngl <= n else f"-ngl {ngl} (≥ {n + 1}): all {n} layers + output on GPU")
    rules.insert(1, "token_embd (input layer) always on CPU")
    rules += [f"{label}: CPU/GPU override applied" for label in sorted(used)]
    out_dev = "GPU" if ngl >= 1 and n >= max(n + 1 - ngl, 0) else "CPU"
    return Placement(device, layers, gpu, cpu, ngl, estimated, rules, out_dev)


def compute_placement(model: ModelInfo, args: LlamaArgs, vram_free: int | None = None) -> Placement:
    n = model.hparams.n_layer
    ngl = _resolve_ngl(args, n)
    if args.ngl != "auto":
        return _build(model, args, ngl, args.n_cpu_moe, False, [])
    if not args.fit:
        return _build(model, args, ngl, args.n_cpu_moe, False, ["-ngl auto with --fit off: all layers"])
    if vram_free is None:
        return _build(model, args, ngl, args.n_cpu_moe, True,
                      ["auto/fit: free VRAM unknown, assumed everything fits (estimated)"])

    budget = vram_free - FIT_MARGIN
    n_cpu_moe = args.n_cpu_moe
    is_moe = model.hparams.n_expert > 0 and not args.cpu_moe

    def total(ngl_, ncm):
        p = _build(model, args, ngl_, ncm, True, [])
        return estimate_vram(model, args, p, None).total

    trace = []
    if total(ngl, n_cpu_moe) > budget:
        if is_moe:
            while n_cpu_moe < n and total(ngl, n_cpu_moe) > budget:
                n_cpu_moe += 1
            trace.append(f"auto/fit simulated: experts of the first {n_cpu_moe} layers moved to CPU")
        while ngl > 0 and total(ngl, n_cpu_moe) > budget:
            ngl -= 1
        if ngl < n + 1:
            trace.append(f"auto/fit simulated: -ngl reduced to {ngl}")
    trace.append(f"auto/fit simulated against {vram_free / GiB:.2f} GiB free − 1 GiB margin "
                 "(llama.cpp may also shrink an unset -c)")
    return _build(model, args, ngl, n_cpu_moe, True, trace)


def estimate_vram(model: ModelInfo, args: LlamaArgs, placement: Placement,
                  capacity: int | None) -> VramEstimate:
    hp = model.hparams
    notes = []
    ctx = args.ctx or hp.ctx_train or 4096
    if not args.ctx:
        notes.append(f"-c unset: using the model's training context {ctx}")
    bk = KV_BYTES.get(args.cache_type_k.lower(), 2.0)
    bv = KV_BYTES.get(args.cache_type_v.lower(), 2.0)
    per_layer = ctx * (hp.n_head_kv or 0) * (hp.head_dim or 0) * (bk + bv)
    kv_roles = ("attn_k", "attn_qkv", "attn_kv_a_mqa")
    ssm_layers = {t.layer for t in model.tensors if t.layer is not None and t.component == "ssm"}
    attn_layers = {t.layer for t in model.tensors
                   if t.layer is not None and t.role in kv_roles} - ssm_layers
    gpu_layers = {L["index"] for L in placement.layers if L["device"] != "CPU"}
    if args.kv_offload:
        kv = round(per_layer * len(attn_layers & gpu_layers))
    else:
        kv = 0
        notes.append("--no-kv-offload: KV cache in RAM")
    if hp.sliding_window:
        notes.append("sliding-window attention: KV estimate is an upper bound")
    weights = placement.gpu_weight_bytes
    compute = max(512 * MiB, int(0.06 * weights)) if weights else 0
    notes.append("compute buffer is a rough allowance")
    total = weights + kv + compute
    return VramEstimate(weights, kv, compute, total, capacity,
                        (total <= capacity) if capacity else None, notes)
