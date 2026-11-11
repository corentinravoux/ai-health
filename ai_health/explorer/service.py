"""Explorer state: the open model, its architecture, cached tensor views."""

from __future__ import annotations

import dataclasses
import os
import threading
from pathlib import Path

from ai_health.llama.cmdline import LlamaArgs

from .architecture import build_architecture
from .model import ModelInfo, open_model
from .placement import compute_placement, estimate_vram
from .weights import TensorView, ViewCache

DEFAULT_DIRS = ["~/models", "~/Models", "~/Software/models", "~/.cache/llama.cpp",
                "~/.cache/huggingface/hub"]
MAX_FOUND = 200


def _gguf_files(root: Path, max_depth: int) -> list[Path]:
    """Bounded walk (depth-limited, *.gguf only)."""
    out: list[Path] = []
    root = root.expanduser()
    if not root.is_dir():
        return out
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        depth = len(Path(dirpath).parts) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for f in filenames:
            if f.endswith(".gguf"):
                out.append(Path(dirpath) / f)
                if len(out) >= MAX_FOUND:
                    return out
    return out


def _file_entry(p: Path) -> dict:
    try:
        size = p.stat().st_size
    except OSError:
        size = None
    return {"path": str(p), "name": p.name, "bytes": size, "exists": size is not None}


class ExplorerService:
    def __init__(self, settings=None, search_dirs=None, on_change=None, recent_keep: int = 8):
        self._settings = settings
        self._recent: list[str] = list(settings.recent_models) if settings else []
        self._search_dirs = [Path(d) for d in (search_dirs if search_dirs is not None else DEFAULT_DIRS)]
        self._on_change = on_change
        self._keep = recent_keep
        self._lock = threading.RLock()
        self._model: ModelInfo | None = None
        self._arch: dict | None = None
        self._views = ViewCache()

    def open(self, path: str) -> dict:
        p = Path(path).expanduser()
        if p.is_dir():
            raise IsADirectoryError(f"{p} is a directory, not a .gguf file")
        model = open_model(p)  # FileNotFoundError / ValueError on bad input
        arch = build_architecture(model)
        with self._lock:
            self._model, self._arch = model, arch
            self._views.clear()
            self._recent = [str(p)] + [r for r in self._recent if r != str(p)]
            del self._recent[self._keep:]
            if self._settings is not None:
                self._settings.recent_models = list(self._recent)
        if self._on_change:
            self._on_change()
        return arch

    def close(self) -> None:
        with self._lock:
            self._model = self._arch = None
            self._views.clear()

    def current(self) -> ModelInfo | None:
        return self._model

    def _require(self) -> ModelInfo:
        m = self._model
        if m is None:
            raise LookupError("no model open")
        return m

    def architecture(self) -> dict:
        self._require()
        return self._arch

    def tensors(self) -> list[dict]:
        m = self._require()
        rows = [dataclasses.asdict(t) for t in m.tensors]

        def section(r):
            if r["layer"] is not None and r["component"] != "vision":
                return 1
            if r["component"] == "embedding":
                return 0
            if r["component"] == "output" or r["role"] == "output_norm":
                return 2
            return 3

        return sorted(rows, key=lambda r: (section(r), r["layer"] or 0, r["name"]))

    def view(self, name: str, reduce: str) -> TensorView:
        m = self._require()
        m.tensor(name)  # KeyError if unknown
        return self._views.get(m, name, reduce)

    def placement(self, args: LlamaArgs, vram_free: int | None, capacity: int | None) -> dict:
        m = self._require()
        p = compute_placement(m, args, vram_free=vram_free)
        e = estimate_vram(m, args, p, capacity)
        return {"placement": dataclasses.asdict(p), "vram": dataclasses.asdict(e)}

    def recent_and_found(self) -> dict:
        found: list[Path] = []
        seen = set()
        for d in self._search_dirs:
            depth = 6 if d.name == "hub" else 2
            for f in _gguf_files(d, depth):
                if str(f) not in seen:
                    seen.add(str(f))
                    found.append(f)
        return {
            "recent": [_file_entry(Path(r)) for r in self._recent],
            "found": [_file_entry(f) for f in sorted(found)],
            "current": str(self._model.path) if self._model else None,
            "search_dirs": [str(d) for d in self._search_dirs],
        }
