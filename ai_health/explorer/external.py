"""Optional external viewers: Netron (experimental GGUF support) and Google
Model Explorer with the model-explorer-gguf adapter. Neither is required."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

INSTALL_HINT = ("~/.local/bin/uv pip install --python ~/Software/venv/bin/python "
                "netron model-explorer model-explorer-gguf")


def available_viewers() -> dict[str, bool]:
    return {
        "netron": importlib.util.find_spec("netron") is not None or shutil.which("netron") is not None,
        "model_explorer": shutil.which("model-explorer") is not None
        and importlib.util.find_spec("model_explorer_gguf") is not None,
    }


def launch_viewer(tool: str, path: Path) -> str:
    if tool not in ("netron", "model_explorer"):
        raise ValueError(f"unknown viewer: {tool}")
    if not available_viewers()[tool]:
        raise RuntimeError(f"{tool} is not installed ({INSTALL_HINT})")
    if tool == "netron":
        exe = shutil.which("netron")
        argv = [exe] if exe else [sys.executable, "-m", "netron"]
        argv.append(str(path))
    else:
        argv = ["model-explorer", "--extensions=model_explorer_gguf", str(path)]
    subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    return f"started {tool} on {path.name} (it opens its own browser tab)"
