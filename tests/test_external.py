import pytest

from ai_health.explorer import external


def test_viewers_detection(monkeypatch):
    monkeypatch.setattr(external.shutil, "which", lambda name: "/bin/" + name if name == "model-explorer" else None)
    monkeypatch.setattr(external.importlib.util, "find_spec",
                        lambda name: object() if name in ("netron", "model_explorer_gguf") else None)
    assert external.available_viewers() == {"netron": True, "model_explorer": True}
    monkeypatch.setattr(external.importlib.util, "find_spec", lambda name: None)
    assert external.available_viewers() == {"netron": False, "model_explorer": False}


def test_launch_builds_command(monkeypatch, tmp_path):
    calls = []

    class P:
        def __init__(self, argv, **kw):
            calls.append((argv, kw))

    monkeypatch.setattr(external.subprocess, "Popen", P)
    monkeypatch.setattr(external, "available_viewers", lambda: {"netron": True, "model_explorer": True})
    f = tmp_path / "m.gguf"
    external.launch_viewer("netron", f)
    assert calls[0][0][-1] == str(f) and calls[0][1]["start_new_session"] is True
    external.launch_viewer("model_explorer", f)
    assert "--extensions=model_explorer_gguf" in calls[1][0]


def test_launch_unknown_or_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(external, "available_viewers", lambda: {"netron": False, "model_explorer": False})
    with pytest.raises(ValueError):
        external.launch_viewer("photoshop", tmp_path / "m.gguf")
    with pytest.raises(RuntimeError, match="not installed"):
        external.launch_viewer("netron", tmp_path / "m.gguf")
