"""Map a GGUF tensor name to (layer, component, role).

Role names follow ``gguf.constants.TENSOR_NAMES`` (``blk.{bid}.<role>``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_BLK = re.compile(r"^blk\.(\d+)\.(.+)$")
_MM_PREFIXES = ("v.", "mm.", "a.", "enc.", "vision.", "audio.")
_SUFFIXES = (".weight", ".bias", ".scale")


@dataclass(frozen=True)
class TensorClass:
    layer: int | None
    component: str  # embedding attention ffn moe_router moe_experts moe_shared norm ssm output vision other
    role: str


def _strip(name: str) -> str:
    for suf in _SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _block_component(role: str) -> str:
    if role.startswith("nextn."):
        return "other"
    if "norm" in role or role.endswith("_ln"):
        return "norm"
    if role.startswith(("ffn_gate_inp", "exp_probs_b", "ffn_gate_tid2eid")):
        return "moe_router"
    if "shexp" in role:
        return "moe_shared"
    if re.search(r"(_exps|_chexps|_exp_embd)$", role) or role.endswith("_exps"):
        return "moe_experts"
    if role.startswith(("attn_", "indexer")):
        return "attention"
    if role.startswith(("ffn", "channel_mix")):
        return "ffn"
    if role.startswith(("ssm_", "time_mix", "shortconv")):
        return "ssm"
    return "other"


def classify_tensor(name: str) -> TensorClass:
    base = _strip(name)
    if base.startswith(_MM_PREFIXES):
        rest = base.split(".", 1)[1]
        m = _BLK.match(rest)
        if m:
            return TensorClass(int(m.group(1)), "vision", m.group(2))
        return TensorClass(None, "vision", rest)
    m = _BLK.match(base)
    if m:
        role = m.group(2)
        return TensorClass(int(m.group(1)), _block_component(role), role)
    if base in ("token_embd", "position_embd", "token_types", "per_layer_token_embd"):
        return TensorClass(None, "embedding", base)
    if "norm" in base:
        return TensorClass(None, "norm", base)
    if base in ("output", "cls", "cls.output"):
        return TensorClass(None, "output", base)
    return TensorClass(None, "other", base)
