"""Open a GGUF model (all shards) without reading tensor data.

``gguf.GGUFReader`` memory-maps the file, so tensor bytes are only touched
when a view asks for them (``read_tensor_raw``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import gguf
import numpy as np

from .classify import classify_tensor

_SPLIT = re.compile(r"^(?P<stem>.+)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$")
MAX_ARRAY = 64


@dataclass
class TensorInfo:
    name: str
    shape: list[int]  # ggml order (ne0 first), as llama.cpp prints it
    qtype: str
    n_bytes: int
    n_params: int
    layer: int | None
    component: str
    role: str
    shard: int
    offset: int


@dataclass
class HParams:
    arch: str
    n_layer: int
    n_embd: int
    n_head: int | None
    n_head_kv: int | None
    head_dim: int | None
    n_ff: int | None
    n_expert: int
    n_expert_used: int
    ctx_train: int | None
    vocab: int | None
    rope_base: float | None
    sliding_window: int | None


@dataclass
class ModelInfo:
    path: Path
    shards: list[Path]
    name: str
    file_bytes: int
    hparams: HParams
    metadata: dict[str, object]
    tensors: list[TensorInfo]
    readers: list = field(default_factory=list, repr=False)
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def tensor(self, name: str) -> TensorInfo:
        if not self._index:
            self._index = {t.name: i for i, t in enumerate(self.tensors)}
        return self.tensors[self._index[name]]


def _shard_paths(path: Path) -> list[Path]:
    m = _SPLIT.match(path.name)
    if not m:
        return [path]
    n = int(m.group("n"))
    return [path.with_name(f"{m.group('stem')}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def _field_value(f) -> object:
    if not f.types:
        return None
    if f.types[0] == gguf.GGUFValueType.ARRAY:
        n = len(f.data)
        if n > MAX_ARRAY:
            return f"[array len={n}]"
        return f.contents()
    return f.contents()


def _first(meta: dict, *keys, default=None):
    for k in keys:
        v = meta.get(k)
        if v is not None:
            return v
    return default


def _as_int(v) -> int | None:
    if v is None:
        return None
    if isinstance(v, list):  # per-layer arrays (e.g. head_count_kv): take the max
        nums = [x for x in v if isinstance(x, (int, float))]
        return int(max(nums)) if nums else None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _check_magic(p: Path) -> None:
    with open(p, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError(f"{p.name} is not a GGUF file (bad magic bytes)")


def open_model(path: str | Path) -> ModelInfo:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"no such model file: {path}")
    shards = _shard_paths(path)
    for p in shards:
        if not p.is_file():
            raise FileNotFoundError(f"missing shard: {p}")
        _check_magic(p)
    readers = [gguf.GGUFReader(p, "r") for p in shards]

    meta: dict[str, object] = {}
    for r in readers[:1]:
        for key, f in r.fields.items():
            if key.startswith("GGUF."):
                continue
            meta[key] = _field_value(f)
    vocab_field = readers[0].fields.get("tokenizer.ggml.tokens")
    arch = str(meta.get("general.architecture") or "unknown")
    a = arch + "."

    tensors: list[TensorInfo] = []
    for si, r in enumerate(readers):
        for t in r.tensors:
            c = classify_tensor(t.name)
            tensors.append(TensorInfo(
                name=t.name, shape=[int(x) for x in t.shape], qtype=t.tensor_type.name,
                n_bytes=int(t.n_bytes), n_params=int(t.n_elements), layer=c.layer,
                component=c.component, role=c.role, shard=si, offset=int(t.data_offset)))

    n_embd = _as_int(meta.get(a + "embedding_length")) or 0
    n_head = _as_int(meta.get(a + "attention.head_count"))
    head_dim = _as_int(_first(meta, a + "attention.key_length"))
    if head_dim is None and n_head:
        head_dim = n_embd // n_head
    vocab = len(vocab_field.data) if vocab_field is not None else None
    if vocab is None:
        emb = next((t for t in tensors if t.name == "token_embd.weight"), None)
        vocab = emb.shape[-1] if emb else None
    hp = HParams(
        arch=arch,
        n_layer=_as_int(meta.get(a + "block_count")) or 0,
        n_embd=n_embd, n_head=n_head,
        n_head_kv=_as_int(meta.get(a + "attention.head_count_kv")) or n_head,
        head_dim=head_dim,
        n_ff=_as_int(meta.get(a + "feed_forward_length")),
        n_expert=_as_int(meta.get(a + "expert_count")) or 0,
        n_expert_used=_as_int(meta.get(a + "expert_used_count")) or 0,
        ctx_train=_as_int(meta.get(a + "context_length")),
        vocab=vocab,
        rope_base=float(meta[a + "rope.freq_base"]) if a + "rope.freq_base" in meta else None,
        sliding_window=_as_int(meta.get(a + "attention.sliding_window")),
    )
    return ModelInfo(
        path=path, shards=shards,
        name=str(meta.get("general.name") or path.stem),
        file_bytes=sum(p.stat().st_size for p in shards),
        hparams=hp, metadata=meta, tensors=tensors, readers=readers)


def read_tensor_raw(model: ModelInfo, name: str) -> tuple[np.ndarray, str]:
    """Memory-mapped raw data (quantized bytes or float array) and qtype name."""
    info = model.tensor(name)
    reader = model.readers[info.shard]
    for t in reader.tensors:
        if t.name == name:
            return t.data, info.qtype
    raise KeyError(name)
