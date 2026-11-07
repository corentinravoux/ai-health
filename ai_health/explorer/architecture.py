"""Turn a ModelInfo into the diagram contract used by the explorer front end:
stem -> N x block (template + per-layer detail) -> head, plus summaries."""

from __future__ import annotations

from collections import Counter, defaultdict

from .model import ModelInfo, TensorInfo

ROLE_ORDER = ["attn_q", "attn_k", "attn_v", "attn_qkv", "attn_q_a", "attn_q_b", "attn_kv_a_mqa",
              "attn_kv_b", "attn_k_b", "attn_v_b", "attn_q_norm", "attn_k_norm", "attn_output",
              "ffn_gate", "ffn_up", "ffn_down", "ffn_gate_inp", "exp_probs_b",
              "ffn_gate_exps", "ffn_up_exps", "ffn_gate_up_exps", "ffn_down_exps",
              "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ffn_gate_inp_shexp"]
PRE_ATTN_NORMS = ("attn_norm",)
QK_NORMS = ("attn_q_norm", "attn_k_norm", "attn_kv_a_norm", "attn_q_a_norm")
POST_ATTN_NORMS = ("post_attention_norm", "attn_post_norm", "attn_output_norm")
PRE_FFN_NORMS = ("ffn_norm",)
POST_FFN_NORMS = ("post_ffw_norm", "layer_output_norm")
GROUP_SEQUENCE = ["attn_norm", "attention", "ssm", "attn_post_norm", "ffn_norm", "ffn",
                  "moe_router", "moe_experts", "moe_shared", "ffn_post_norm", "other"]
LABELS = {"attn_norm": "RMSNorm", "attention": "Self-attention", "ssm": "State-space mixer",
          "attn_post_norm": "Post-attention norm", "ffn_norm": "RMSNorm",
          "ffn": "SwiGLU FFN", "moe_router": "Router", "moe_experts": "Experts",
          "moe_shared": "Shared expert", "ffn_post_norm": "Post-FFN norm", "other": "Other"}
GROUP_COMPONENT = {"attn_norm": "norm", "attention": "attention", "ssm": "ssm",
                   "attn_post_norm": "norm", "ffn_norm": "norm", "ffn": "ffn",
                   "moe_router": "moe_router", "moe_experts": "moe_experts",
                   "moe_shared": "moe_shared", "ffn_post_norm": "norm", "other": "other"}


def _group_of(t: TensorInfo) -> str:
    r, c = t.role, t.component
    if c == "norm":
        if r in QK_NORMS:
            return "attention"
        if r in PRE_ATTN_NORMS:
            return "attn_norm"
        if r in POST_ATTN_NORMS:
            return "attn_post_norm"
        if r in PRE_FFN_NORMS:
            return "ffn_norm"
        if r in POST_FFN_NORMS:
            return "ffn_post_norm"
        if r.startswith("ssm"):
            return "ssm"
        return "other"
    if c in ("attention", "ssm", "ffn", "moe_router", "moe_experts", "moe_shared"):
        return c
    return "other"


def _role_key(role: str) -> tuple[int, str]:
    return (ROLE_ORDER.index(role) if role in ROLE_ORDER else len(ROLE_ORDER), role)


def _item(id_: str, label: str, tensors: list[TensorInfo], **extra) -> dict:
    return {"id": id_, "label": label, "tensors": [t.name for t in tensors],
            "bytes": sum(t.n_bytes for t in tensors), "params": sum(t.n_params for t in tensors),
            "qtypes": sorted({t.qtype for t in tensors}), **extra}


def build_architecture(model: ModelInfo) -> dict:
    hp = model.hparams
    by_layer: dict[int, list[TensorInfo]] = defaultdict(list)
    stem_t: list[TensorInfo] = []
    head_norm: list[TensorInfo] = []
    head_out: list[TensorInfo] = []
    extras_t: list[TensorInfo] = []
    for t in model.tensors:
        if t.component == "vision":
            extras_t.append(t)
        elif t.layer is not None:
            by_layer[t.layer].append(t)
        elif t.component == "embedding" or t.role == "token_embd_norm":
            stem_t.append(t)
        elif t.role == "output_norm":
            head_norm.append(t)
        elif t.component == "output":
            head_out.append(t)
        else:
            extras_t.append(t)

    # -- layers ------------------------------------------------------------
    layers = []
    roles_by_group: dict[str, set[str]] = defaultdict(set)
    group_presence: Counter = Counter()
    mixers = []
    for i in sorted(by_layer):
        groups: dict[str, dict] = {}
        for t in sorted(by_layer[i], key=lambda t: _role_key(t.role)):
            g = _group_of(t)
            roles_by_group[g].add(t.role)
            d = groups.setdefault(g, {"bytes": 0, "params": 0, "tensors": []})
            d["bytes"] += t.n_bytes
            d["params"] += t.n_params
            d["tensors"].append(t.name)
        group_presence.update(groups.keys())
        mixer = "attention" if "attention" in groups else "ssm" if "ssm" in groups else "none"
        ffn = "moe" if "moe_experts" in groups else "dense" if "ffn" in groups else "none"
        mixers.append(mixer)
        layers.append({
            "index": i, "mixer": mixer, "ffn": ffn,
            "kind": "ssm" if mixer == "ssm" else "moe" if ffn == "moe" else "dense",
            "bytes": sum(g["bytes"] for g in groups.values()),
            "params": sum(g["params"] for g in groups.values()),
            "groups": groups})

    # -- block template ----------------------------------------------------
    template = []
    for gid in GROUP_SEQUENCE:
        if gid not in group_presence:
            continue
        roles = sorted(roles_by_group[gid], key=_role_key)
        g = {"id": gid, "label": LABELS[gid], "component": GROUP_COMPONENT[gid], "roles": roles}
        if gid == "attention" and hp.n_head:
            kv = hp.n_head_kv or hp.n_head
            g["detail"] = {"n_head": hp.n_head, "n_head_kv": kv, "head_dim": hp.head_dim,
                           "gqa_ratio": hp.n_head // kv if kv else None}
        if gid == "ffn" and "ffn_gate" not in roles:
            g["label"] = "MLP"
        if gid == "moe_experts":
            g["detail"] = {"n_expert": hp.n_expert, "n_expert_used": hp.n_expert_used}
        if gid in ("attn_norm", "ffn_norm"):
            biases = any(t.name.endswith(".bias") for ts in by_layer.values() for t in ts
                         if _group_of(t) == gid)
            g["label"] = "LayerNorm" if biases else "RMSNorm"
        g["in_layers"] = group_presence[gid]
        template.append(g)
    hybrid = len({m for m in mixers if m != "none"}) > 1

    # -- stem / head -------------------------------------------------------
    stem = [_item(t.role, {"token_embd": "Token embedding", "position_embd": "Position embedding",
                           "token_embd_norm": "Embedding norm"}.get(t.role, t.role), [t],
                  shape_hint="vocab × d_model" if t.role == "token_embd" else None)
            for t in stem_t]
    head = []
    if head_norm:
        head.append(_item("output_norm", "Final norm", head_norm))
    if head_out:
        head.append(_item("output", "Output head", head_out, tied=False,
                          shape_hint="d_model × vocab"))
    else:
        emb = [t.name for t in stem_t if t.role == "token_embd"]
        head.append({"id": "output", "label": "Output head (tied)", "tensors": emb,
                     "bytes": 0, "params": 0, "qtypes": [], "tied": True,
                     "shape_hint": "shares the token embedding"})
    extras = [_item(t.name, t.role, [t], component=t.component) for t in extras_t]

    # -- summaries -----------------------------------------------------------
    params_total = sum(t.n_params for t in model.tensors)
    bytes_total = sum(t.n_bytes for t in model.tensors)
    comp: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    qt: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for t in model.tensors:
        comp[t.component][0] += t.n_params
        comp[t.component][1] += t.n_bytes
        qt[t.qtype][0] += t.n_bytes
        qt[t.qtype][1] += 1
    n_attn = sum(1 for m in mixers if m == "attention")
    kv_per_tok = 2 * n_attn * (hp.n_head_kv or 0) * (hp.head_dim or 0) * 2
    summary = {
        "params_total": params_total, "bytes_total": bytes_total,
        "bits_per_weight": 8 * bytes_total / params_total if params_total else None,
        "by_component": sorted(({"component": c, "params": p, "bytes": b,
                                 "share": p / params_total if params_total else 0}
                                for c, (p, b) in comp.items()), key=lambda d: -d["params"]),
        "by_qtype": sorted(({"qtype": q, "bytes": b, "tensors": n} for q, (b, n) in qt.items()),
                           key=lambda d: -d["bytes"]),
        "n_layer_attn": n_attn,
        "kv_bytes_per_token_f16": kv_per_tok,
    }
    return {
        "name": model.name, "arch": hp.arch, "path": str(model.path),
        "file_bytes": model.file_bytes, "n_tensors": len(model.tensors),
        "hparams": {k: getattr(hp, k) for k in hp.__dataclass_fields__},
        "stem": stem,
        "block": {"kind": "transformer", "hybrid": hybrid, "groups": template,
                  "residuals": [["input", "attention"], ["attention", "ffn"]]},
        "layers": layers, "head": head, "extras": extras, "summary": summary,
    }
