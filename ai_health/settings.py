"""Small JSON settings file at ``$XDG_CONFIG_HOME/ai-health/settings.json``."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    width: int = 1440
    height: int = 900
    history_s: int = 900
    theme: str = "dark"
    recent_models: list[str] = field(default_factory=list)


def default_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "ai-health" / "settings.json"


def load_settings(path: Path | None = None) -> Settings:
    path = path or default_path()
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return Settings()
    if not isinstance(raw, dict):
        return Settings()
    s = Settings()
    for f in dataclasses.fields(Settings):
        if f.name in raw:
            v = raw[f.name]
            default = getattr(s, f.name)
            if isinstance(default, list):
                ok = isinstance(v, list) and all(isinstance(x, str) for x in v)
            else:
                ok = type(v) is type(default)
            if ok:
                setattr(s, f.name, v)
    return s


_save_lock = threading.Lock()


def save_settings(s: Settings, path: Path | None = None) -> None:
    """Atomic write. HTTP handler threads can save at the same time (model
    open, theme change), so each save gets its own temporary file."""
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _save_lock:
        text = json.dumps(dataclasses.asdict(s), indent=2)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(text)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def remember_model(s: Settings, model_path: str, keep: int = 8) -> None:
    s.recent_models = [model_path] + [m for m in s.recent_models if m != model_path]
    del s.recent_models[keep:]
