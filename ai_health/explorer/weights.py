"""Per-tensor weight views: dequantize (row-sampled if huge), downsampled
heatmap, histogram and summary statistics."""

from __future__ import annotations

import base64
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import gguf
import numpy as np

from .model import ModelInfo, read_tensor_raw

HIST_BINS = 128


@dataclass
class TensorView:
    name: str
    qtype: str
    shape: list[int]
    available: bool
    reason: str | None = None
    heatmap_b64: str | None = None
    heat_w: int = 0
    heat_h: int = 0
    heat_min: float = 0.0
    heat_max: float = 0.0
    reduce: str = "absmax"  # "absmax" | "mean"
    sampled: bool = False
    rows_used: int = 0
    rows_total: int = 0
    hist_counts: list[int] = field(default_factory=list)
    hist_edges: list[float] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)


def _block_reduce(x: np.ndarray, factor: int, reduce: str) -> np.ndarray:
    if factor <= 1:
        return np.abs(x) if reduce == "absmax" else x
    h, w = x.shape
    hp, wp = math.ceil(h / factor) * factor, math.ceil(w / factor) * factor
    padded = np.full((hp, wp), np.nan, dtype=np.float32)
    padded[:h, :w] = x
    blocks = padded.reshape(hp // factor, factor, wp // factor, factor)
    with np.errstate(all="ignore"):
        if reduce == "absmax":
            return np.nanmax(np.abs(blocks), axis=(1, 3))
        return np.nanmean(blocks, axis=(1, 3))


def _stats(x: np.ndarray) -> dict[str, float]:
    x64 = x.astype(np.float64, copy=False)
    mean = float(x64.mean())
    std = float(x64.std())
    absx = np.abs(x64)
    return {
        "mean": mean, "std": std, "min": float(x64.min()), "max": float(x64.max()),
        "absmax": float(absx.max()), "rms": float(np.sqrt((x64 ** 2).mean())),
        "near_zero_frac": float((absx < 0.01 * std + 1e-12).mean()),
        "outlier_frac": float((np.abs(x64 - mean) > 6 * std).mean()) if std > 0 else 0.0,
    }


def tensor_view(model: ModelInfo, name: str, reduce: str = "absmax", max_side: int = 512,
                max_rows: int = 8192) -> TensorView:
    if reduce not in ("absmax", "mean"):
        raise ValueError("reduce must be 'absmax' or 'mean'")
    info = model.tensor(name)
    view = TensorView(name=name, qtype=info.qtype, shape=list(info.shape), available=False,
                      reduce=reduce)
    raw, qtype_name = read_tensor_raw(model, name)
    raw2 = raw.reshape(1, -1) if raw.ndim == 1 else raw.reshape(-1, raw.shape[-1])
    rows_total = raw2.shape[0]
    view.rows_total = rows_total
    if rows_total > max_rows:
        idx = np.unique(np.linspace(0, rows_total - 1, max_rows).round().astype(np.int64))
        raw2 = raw2[idx]
        view.sampled = True
    view.rows_used = raw2.shape[0]
    try:
        qtype = gguf.GGMLQuantizationType[qtype_name]
        data = np.asarray(gguf.quants.dequantize(np.ascontiguousarray(raw2), qtype),
                          dtype=np.float32)
    except (NotImplementedError, KeyError, ValueError) as e:
        view.reason = f"dequantization not available for {qtype_name} ({e})"
        return view
    data = data.reshape(view.rows_used, -1)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        view.reason = "tensor has no finite values"
        return view

    view.stats = _stats(finite)
    counts, edges = np.histogram(finite, bins=HIST_BINS)
    view.hist_counts = counts.astype(int).tolist()
    view.hist_edges = edges.astype(float).tolist()

    factor = max(1, math.ceil(max(data.shape) / max_side))
    red = _block_reduce(data, factor, reduce)
    red = np.nan_to_num(red, nan=0.0)
    if reduce == "absmax":
        lo, hi = 0.0, float(np.percentile(red, 99.5)) or float(red.max()) or 1.0
    else:
        m = float(max(abs(np.percentile(red, 0.5)), abs(np.percentile(red, 99.5)))) or 1.0
        lo, hi = -m, m
    scaled = np.clip((red - lo) / (hi - lo), 0.0, 1.0)
    img = (scaled * 255).round().astype(np.uint8)
    view.heat_h, view.heat_w = img.shape
    view.heat_min, view.heat_max = lo, hi
    view.heatmap_b64 = base64.b64encode(img.tobytes()).decode()
    view.available = True
    return view


class ViewCache:
    def __init__(self, maxsize: int = 64):
        self.maxsize = maxsize
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, model: ModelInfo, name: str, reduce: str) -> TensorView:
        key = (str(model.path), name, reduce)
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        v = tensor_view(model, name, reduce=reduce)
        with self._lock:
            self._d[key] = v
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)
        return v

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
