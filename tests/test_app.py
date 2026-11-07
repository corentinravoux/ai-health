import json
import threading
import urllib.request

from ai_health.app import build_parser, build_sampler, main, parse_hostport
from ai_health.settings import Settings, load_settings, save_settings


def test_parser_defaults():
    a = build_parser().parse_args([])
    assert a.browser is False and a.no_window is False and a.port == 0
    assert a.history == 900 and a.open is None and a.extra_server == []


def test_parse_hostport():
    assert parse_hostport("127.0.0.1:8099") == ("127.0.0.1", 8099)
    assert parse_hostport("8099") == ("127.0.0.1", 8099)


class NoGpu:
    available, reason = False, "test"

    def sample(self):
        return []


def test_extra_server_flag():
    a = build_parser().parse_args(["--extra-server", "127.0.0.1:8099", "--extra-server", "localhost:9000"])
    s = build_sampler(a, gpu=NoGpu())
    assert s._extra == [("127.0.0.1", 8099), ("localhost", 9000)]


def test_settings_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    save_settings(Settings(width=1000, recent_models=["/m/a.gguf"]), p)
    s = load_settings(p)
    assert s.width == 1000 and s.recent_models == ["/m/a.gguf"] and s.height == 900


def test_settings_corrupt_file_falls_back_to_defaults(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    assert load_settings(p) == Settings()
    p.write_text('{"width": "wide", "unknown": 1}')
    assert load_settings(p).width == 1440


def test_main_no_window_starts_and_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    stop = threading.Event()
    ready = {}
    got = threading.Event()

    def on_ready(url):
        ready["url"] = url
        got.set()

    th = threading.Thread(target=main, args=(["--no-window", "--port", "0"],),
                          kwargs={"stop_event": stop, "on_ready": on_ready, "gpu": NoGpu()}, daemon=True)
    th.start()
    assert got.wait(10)
    base, token = ready["url"].split("/?token=")
    r = urllib.request.Request(base + "/api/meta", headers={"X-AI-Health-Token": token})
    with urllib.request.urlopen(r, timeout=5) as resp:
        assert json.loads(resp.read())["version"] == "0.1.0"
    stop.set()
    th.join(10)
    assert not th.is_alive()
