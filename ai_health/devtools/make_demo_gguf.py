"""Build small synthetic GGUF files (tests and UI demos; no real model needed).

Tensor layout mimics llama.cpp naming: ``token_embd``, ``blk.{i}.attn_*``,
``blk.{i}.ffn_*`` (dense) or ``ffn_gate_inp`` + ``ffn_*_exps`` (MoE),
``output_norm``, ``output``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import gguf
import numpy as np

_QTYPES = {
    "F32": gguf.GGMLQuantizationType.F32,
    "F16": gguf.GGMLQuantizationType.F16,
    "Q8_0": gguf.GGMLQuantizationType.Q8_0,
    "Q4_0": gguf.GGMLQuantizationType.Q4_0,
}


def make_gguf(
    path: str | Path,
    arch: str = "llama",
    n_layer: int = 4,
    n_embd: int = 64,
    n_head: int = 4,
    n_head_kv: int = 2,
    n_ff: int = 128,
    vocab: int = 256,
    n_expert: int = 0,
    n_expert_used: int = 0,
    qtype: str = "F32",
    split: int = 1,
    tied: bool = False,
    ssm_layers: tuple[int, ...] = (),
    linear_attn_layers: tuple[int, ...] = (),
    name: str = "demo",
    kv_heads_per_layer: bool = False,
) -> Path:
    """Write a synthetic GGUF and return the path of its first shard."""
    path = Path(path)
    rng = np.random.default_rng(0)
    head_dim = n_embd // n_head
    tensors: list[tuple[str, np.ndarray, bool]] = []  # (name, data, quantizable)

    def w(*shape: int, scale: float = 0.02) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm() -> np.ndarray:
        return (1.0 + rng.standard_normal(n_embd) * 0.05).astype(np.float32)

    tensors.append(("token_embd.weight", w(vocab, n_embd), True))
    for i in range(n_layer):
        p = f"blk.{i}."
        tensors.append((p + "attn_norm.weight", norm(), False))
        if i in linear_attn_layers:  # Qwen3-Next style: fused qkv + state-space tensors
            tensors.append((p + "attn_qkv.weight", w(3 * n_head * head_dim, n_embd), True))
            tensors.append((p + "ssm_conv1d.weight", w(3 * n_head * head_dim, 4), False))
            tensors.append((p + "ssm_out.weight", w(n_embd, n_head * head_dim), True))
        elif i in ssm_layers:
            tensors.append((p + "ssm_in.weight", w(2 * n_embd, n_embd), True))
            tensors.append((p + "ssm_out.weight", w(n_embd, n_embd), True))
        else:
            tensors.append((p + "attn_q.weight", w(n_head * head_dim, n_embd), True))
            tensors.append((p + "attn_k.weight", w(n_head_kv * head_dim, n_embd), True))
            tensors.append((p + "attn_v.weight", w(n_head_kv * head_dim, n_embd), True))
            tensors.append((p + "attn_output.weight", w(n_embd, n_head * head_dim), True))
        tensors.append((p + "ffn_norm.weight", norm(), False))
        if n_expert:
            tensors.append((p + "ffn_gate_inp.weight", w(n_expert, n_embd), False))
            tensors.append((p + "ffn_gate_exps.weight", w(n_expert, n_ff, n_embd), True))
            tensors.append((p + "ffn_up_exps.weight", w(n_expert, n_ff, n_embd), True))
            tensors.append((p + "ffn_down_exps.weight", w(n_expert, n_embd, n_ff), True))
        else:
            tensors.append((p + "ffn_gate.weight", w(n_ff, n_embd), True))
            tensors.append((p + "ffn_up.weight", w(n_ff, n_embd), True))
            tensors.append((p + "ffn_down.weight", w(n_embd, n_ff), True))
    tensors.append(("output_norm.weight", norm(), False))
    if not tied:
        tensors.append(("output.weight", w(vocab, n_embd), True))

    per_shard = math.ceil(len(tensors) / split) if split > 1 else 0
    writer = gguf.GGUFWriter(path, arch, split_max_tensors=per_shard)
    writer.add_name(name)
    writer.add_block_count(n_layer)
    writer.add_uint32(f"{arch}.context_length", 2048)
    writer.add_uint32(f"{arch}.embedding_length", n_embd)
    writer.add_uint32(f"{arch}.feed_forward_length", n_ff)
    writer.add_uint32(f"{arch}.attention.head_count", n_head)
    if kv_heads_per_layer:  # one entry per layer, as hybrid/variable-GQA models store it
        writer.add_array(f"{arch}.attention.head_count_kv", [n_head_kv] * n_layer)
    else:
        writer.add_uint32(f"{arch}.attention.head_count_kv", n_head_kv)
    writer.add_float32(f"{arch}.rope.freq_base", 10000.0)
    if n_expert:
        writer.add_uint32(f"{arch}.expert_count", n_expert)
        writer.add_uint32(f"{arch}.expert_used_count", n_expert_used)
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(vocab)])

    qt = _QTYPES[qtype]
    for tname, data, quantizable in tensors:
        if quantizable and qt not in (gguf.GGMLQuantizationType.F32,):
            if qt == gguf.GGMLQuantizationType.F16:
                writer.add_tensor(tname, data.astype(np.float16))
            else:
                writer.add_tensor(tname, gguf.quants.quantize(data, qt), raw_dtype=qt)
        else:
            writer.add_tensor(tname, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if split > 1:
        return path.with_name(f"{path.stem}-00001-of-{split:05d}.gguf")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write a synthetic GGUF for demos.")
    ap.add_argument("path")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--embd", type=int, default=256)
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--qtype", default="Q8_0", choices=sorted(_QTYPES))
    a = ap.parse_args(argv)
    out = make_gguf(a.path, n_layer=a.layers, n_embd=a.embd, n_head=8, n_head_kv=2,
                    n_ff=a.embd * 3, vocab=2048, n_expert=a.experts,
                    n_expert_used=2 if a.experts else 0, qtype=a.qtype)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
