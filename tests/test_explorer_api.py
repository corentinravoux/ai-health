import http.client
import json
import threading

import pytest

from ai_health.explorer.service import ExplorerService
from ai_health.web import api
from ai_health.web.server import make_server

GiB = 2**30


class FakeSampler:
    history_s, interval = 900, 1.0

    def __init__(self, servers=()):
        self.servers = list(servers)

    def latest(self):
        return {"t": 0.0, "servers": self.servers, "system": {"ram_total": 16 * GiB},
                "gpu": {"devices": [
                    {"name": "RTX", "mem_total": 12 * GiB, "mem_used": 7 * GiB,
                     "processes": [{"pid": 42, "used_bytes": 3 * GiB}]},
                    {"name": "RTX", "mem_total": 12 * GiB, "mem_used": 2 * GiB,
                     "processes": [{"pid": 42, "used_bytes": 2 * GiB}]}]}}

    def history(self, s):
        return []


@pytest.fixture
def setup(tmp_path, make_gguf):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("x")
    model = make_gguf(n_layer=4)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "a.gguf").write_bytes(model.read_bytes())
    svc = ExplorerService(search_dirs=[models_dir])
    server_entry = {"pid": 42, "model": str(model), "role": "server", "vram_bytes": 5 * GiB,
                    "args": {"ngl": 2, "fit": True, "ctx": 1024, "model": str(model)}}
    sampler = FakeSampler([server_entry])
    server, _, _ = make_server(sampler, svc, static, port=0, token="t", routes=api.all_routes())
    th = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    th.start()
    yield server.server_address[1], svc, model, models_dir, tmp_path
    server.shutdown()
    server.server_close()


def call(port, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Host": f"127.0.0.1:{port}", "X-AI-Health-Token": "t"}
    c.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
    r = c.getresponse()
    return r.status, json.loads(r.read())


def test_open_returns_architecture(setup):
    port, svc, model, *_ = setup
    status, arch = call(port, "POST", "/api/model/open", {"path": str(model)})
    assert status == 200 and arch["hparams"]["n_layer"] == 4 and len(arch["layers"]) == 4
    assert call(port, "GET", "/api/model/architecture")[1]["name"] == "demo"


def test_architecture_before_open_is_409(setup):
    port, *_ = setup
    assert call(port, "GET", "/api/model/architecture")[0] == 409


def test_open_rejects_non_gguf(setup):
    port, _, _, _, tmp = setup
    bad = tmp / "notes.gguf"
    bad.write_text("hello")
    status, body = call(port, "POST", "/api/model/open", {"path": str(bad)})
    assert status == 400 and "GGUF" in body["error"]


def test_open_rejects_directories_and_missing(setup):
    port, _, _, models_dir, tmp = setup
    assert call(port, "POST", "/api/model/open", {"path": str(models_dir)})[0] == 400
    assert call(port, "POST", "/api/model/open", {"path": str(tmp / "nope.gguf")})[0] == 404
    assert call(port, "POST", "/api/model/open", {})[0] == 400


def test_tensors_and_tensor_view_endpoint(setup):
    port, _, model, *_ = setup
    call(port, "POST", "/api/model/open", {"path": str(model)})
    status, tensors = call(port, "GET", "/api/model/tensors")
    assert status == 200 and tensors[0]["name"] == "token_embd.weight"
    status, view = call(port, "GET", "/api/model/tensor?name=blk.0.attn_q.weight&reduce=mean")
    assert status == 200 and view["available"] and view["reduce"] == "mean"
    assert call(port, "GET", "/api/model/tensor?name=nope")[0] == 404
    assert call(port, "GET", "/api/model/tensor?name=blk.0.attn_q.weight&reduce=x")[0] == 400


def test_placement_from_flags(setup):
    port, _, model, *_ = setup
    call(port, "POST", "/api/model/open", {"path": str(model)})
    status, p = call(port, "POST", "/api/model/placement", {"flags": "-ngl 2 -c 1024"})
    assert status == 200
    assert [layer["device"] for layer in p["placement"]["layers"]] == ["CPU", "CPU", "CPU", "GPU"]
    assert p["vram"]["capacity"] == 12 * GiB
    assert p["source"] == "flags"


def test_placement_from_pid_uses_server_args(setup):
    port, _, model, *_ = setup
    call(port, "POST", "/api/model/open", {"path": str(model)})
    status, p = call(port, "POST", "/api/model/placement", {"pid": 42})
    assert status == 200 and p["placement"]["ngl_effective"] == 2 and p["source"] == "pid 42"
    # free VRAM on GPU 0 = total - used + what the server itself holds there
    # (not its 2 GiB on GPU 1)
    assert p["vram_free_for_model"] == (12 - 7 + 3) * GiB
    assert call(port, "POST", "/api/model/placement", {"pid": 7})[0] == 404


def test_close_releases_model(setup):
    port, svc, model, *_ = setup
    call(port, "POST", "/api/model/open", {"path": str(model)})
    assert call(port, "POST", "/api/model/close", {})[0] == 200
    assert svc.current() is None


def test_files_lists_found_models(setup):
    port, _, model, models_dir, _ = setup
    call(port, "POST", "/api/model/open", {"path": str(model)})
    status, files = call(port, "GET", "/api/model/files")
    assert status == 200
    assert str(models_dir / "a.gguf") in [f["path"] for f in files["found"]]
    assert files["recent"][0]["path"] == str(model)
    assert files["current"] == str(model)


def test_viewers_endpoints(setup, monkeypatch):
    port, _, model, *_ = setup
    from ai_health.explorer import external

    monkeypatch.setattr(external, "available_viewers", lambda: {"netron": False, "model_explorer": False})
    assert call(port, "GET", "/api/viewers")[1] == {"netron": False, "model_explorer": False}
    call(port, "POST", "/api/model/open", {"path": str(model)})
    assert call(port, "POST", "/api/viewers/launch", {"tool": "photoshop"})[0] == 400
    assert call(port, "POST", "/api/viewers/launch", {"tool": "netron"})[0] == 409


def test_service_recent_list_is_capped(make_gguf):
    svc = ExplorerService(search_dirs=[])
    paths = [make_gguf(n_layer=1) for _ in range(10)]
    for p in paths:
        svc.open(str(p))
    rec = svc.recent_and_found()["recent"]
    assert len(rec) == 8 and rec[0]["path"] == str(paths[-1])
