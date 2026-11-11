import http.client
import json
import queue
import threading

import pytest

import ai_health
from ai_health.web.server import make_server


class FakeSampler:
    history_s = 900
    interval = 1.0

    def __init__(self):
        self.samples = [{"t": float(i), "gpu": {"devices": [{"name": "RTX", "mem_total": 12}]},
                         "system": {"ram_total": 16}} for i in range(5)]
        self.queues = []

    def latest(self):
        return self.samples[-1]

    def history(self, seconds):
        return self.samples[-seconds:]

    def subscribe(self):
        q = queue.Queue()
        q.put(self.samples[0])
        q.put(self.samples[1])
        self.queues.append(q)
        return q

    def unsubscribe(self, q):
        self.queues.remove(q)


@pytest.fixture
def saved():
    return []


@pytest.fixture
def srv(tmp_path, saved):
    (tmp_path / "index.html").write_text("<!doctype html><title>ai-health</title>")
    (tmp_path / "js").mkdir()
    (tmp_path / "js" / "main.js").write_text("export {}")
    from ai_health.settings import Settings

    server, base_url, token = make_server(FakeSampler(), None, tmp_path, port=0, token="tok",
                                          settings=Settings(), save_settings=lambda s: saved.append(s.theme))
    th = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    th.start()
    yield server, server.server_address[1], token
    server.shutdown()
    server.server_close()


def req(port, path, token="tok", host=None, method="GET", body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": host or f"127.0.0.1:{port}"}
    if token:
        headers["X-AI-Health-Token"] = token
    if body is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(body)
    c.request(method, path, body=body, headers=headers)
    r = c.getresponse()
    return r.status, r.getheader("Content-Type"), r.read()


def test_base_url_contains_token(tmp_path):
    (tmp_path / "index.html").write_text("x")
    server, base_url, token = make_server(FakeSampler(), None, tmp_path, port=0)
    try:
        assert base_url.startswith(f"http://127.0.0.1:{server.server_address[1]}/?token=")
        assert len(token) >= 24 and base_url.endswith(token)
    finally:
        server.server_close()


def test_static_index_served(srv):
    _, port, _ = srv
    status, ctype, body = req(port, "/", token=None)
    assert status == 200 and ctype.startswith("text/html") and b"ai-health" in body
    status, ctype, _ = req(port, "/js/main.js", token=None)
    assert status == 200 and "javascript" in ctype


def test_snapshot_json(srv):
    _, port, _ = srv
    status, ctype, body = req(port, "/api/snapshot")
    assert status == 200 and ctype == "application/json"
    assert json.loads(body)["t"] == 4.0


def test_history_json(srv):
    _, port, _ = srv
    status, _, body = req(port, "/api/history?seconds=2")
    assert [s["t"] for s in json.loads(body)] == [3.0, 4.0]


def test_history_bad_param_400(srv):
    _, port, _ = srv
    assert req(port, "/api/history?seconds=abc")[0] == 400


def test_meta(srv):
    _, port, _ = srv
    meta = json.loads(req(port, "/api/meta")[2])
    assert meta["version"] == ai_health.__version__ and meta["history_s"] == 900 and meta["interval"] == 1.0
    assert meta["gpu_name"] == "RTX" and meta["vram_total"] == 12 and meta["ram_total"] == 16


def test_api_rejects_missing_token(srv):
    _, port, _ = srv
    assert req(port, "/api/snapshot", token=None)[0] == 403
    assert req(port, "/api/snapshot", token="wrong")[0] == 403


def test_api_accepts_query_token(srv):
    _, port, _ = srv
    assert req(port, "/api/snapshot?token=tok", token=None)[0] == 200


def test_api_rejects_foreign_host(srv):
    _, port, _ = srv
    assert req(port, "/api/snapshot", host=f"evil.example:{port}")[0] == 403
    assert req(port, "/", token=None, host="evil.example")[0] == 403
    assert req(port, "/api/snapshot", host=f"localhost:{port}")[0] == 200


def test_unknown_api_404(srv):
    _, port, _ = srv
    assert req(port, "/api/nope")[0] == 404


def test_path_traversal_blocked(srv):
    _, port, _ = srv
    assert req(port, "/../pyproject.toml", token=None)[0] == 404
    assert req(port, "/%2e%2e/%2e%2e/etc/passwd", token=None)[0] == 404


def test_stream_emits_events(srv):
    server, port, _ = srv
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("GET", "/api/stream?token=tok", headers={"Host": f"127.0.0.1:{port}"})
    r = c.getresponse()
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    events = []
    while len(events) < 2:
        line = r.fp.readline().decode()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    assert [e["t"] for e in events] == [0.0, 1.0]
    c.close()


def test_settings_get_and_set_theme(srv, saved):
    _, port, _ = srv
    assert json.loads(req(port, "/api/settings")[2])["theme"] == "dark"
    status, _, body = req(port, "/api/settings", method="POST", body={"theme": "light"})
    assert status == 200 and json.loads(body)["theme"] == "light" and saved == ["light"]
    assert json.loads(req(port, "/api/settings")[2])["theme"] == "light"
    assert req(port, "/api/settings", method="POST", body={"theme": "blue"})[0] == 400


@pytest.mark.parametrize("length", ["-5", "abc"])
def test_bad_content_length_is_400(srv, length):
    _, port, _ = srv
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.putrequest("POST", "/api/settings", skip_host=True)
    c.putheader("Host", f"127.0.0.1:{port}")
    c.putheader("X-AI-Health-Token", "tok")
    c.putheader("Content-Length", length)
    c.endheaders()
    assert c.getresponse().status == 400
