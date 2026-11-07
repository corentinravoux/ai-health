import threading

import pytest
from fake_procs import FakeProc

from ai_health.collectors.gpu import GpuProcess, GpuSample
from ai_health.collectors.processes import find_llama_processes
from ai_health.collectors.system import SystemSample
from ai_health.llama.probe import ProbeResult
from ai_health.sampler import Sampler

GiB = 2**30
BIN = "/x/llama-server"


class FakeGpu:
    def __init__(self, used=6 * GiB, total=12 * GiB, procs=()):
        self.available, self.reason = True, None
        self.used, self.total, self.procs = used, total, list(procs)

    def sample(self):
        return [GpuSample(0, "RTX", self.total, self.used, None, 50, 10, 60, 100.0, 170.0,
                          1500, 7000, [GpuProcess(p, b) for p, b in self.procs], "nvml")]


def fake_system():
    return SystemSample(10.0, [10.0], 0.5, 16 * GiB, 8 * GiB, 8 * GiB, 18 * GiB, 0)


class FakeProbe:
    metrics_data = {"tokens_predicted_total": 0.0, "tokens_predicted_seconds_total": 0.0,
                    "requests_processing": 1}
    calls = []

    def __init__(self, host, port, **kw):
        self.port = port

    def health(self):
        FakeProbe.calls.append(("health", self.port))
        return ProbeResult(True, "ok", {"status": "ok"})

    def slots(self):
        return ProbeResult(True, "ok", [{"id": 0, "n_ctx": 4096, "is_processing": True,
                                         "next_token": {"n_decoded": 7}}])

    def metrics(self, model=None):
        return ProbeResult(True, "ok", dict(FakeProbe.metrics_data))

    def props(self, model=None):
        return ProbeResult(True, "ok", {})


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(procs=(), gpu=None, **kw):
    clock = Clock()
    plist = list(procs)
    s = Sampler(gpu or FakeGpu(), system=fake_system,
                find_procs=lambda: find_llama_processes(plist),
                probe_factory=FakeProbe, clock=clock, name_map=lambda pids: {p: f"p{p}" for p in pids},
                **kw)
    return s, clock, plist


def server(pid=10, extra=(), **kw):
    return FakeProc(pid, 1, "llama-server", [BIN, "-m", "/m/a.gguf", "--port", str(8000 + pid), *extra], **kw)


def test_tick_contract_keys():
    s, _, _ = make([server(extra=["--metrics"])])
    sample = s.tick()
    assert set(sample) >= {"t", "gpu", "system", "servers", "stopped", "processes", "vram_holders",
                           "vram_unattributed", "warnings"}
    srv = sample["servers"][0]
    assert srv["pid"] == 10 and srv["port"] == 8010 and srv["role"] == "server"
    assert srv["health"] == "ok" and srv["metrics_status"] == "ok"
    assert srv["slots"] == [{"id": 0, "n_ctx": 4096, "busy": True, "n_decoded": 7, "n_prompt": None}]
    assert srv["throughput"]["source"] == "metrics"
    assert ["ngl", "auto (fit)"] in [list(f) for f in srv["flags"]]
    assert "raw" not in srv["args"]
    assert sample["gpu"]["devices"][0]["mem_total"] == 12 * GiB


def test_history_window_trims():
    s, clock, _ = make(history_s=5)
    for i in range(10):
        clock.t = float(i)
        s.tick()
    assert len(s.history(100)) == 5
    assert [x["t"] for x in s.history(3)] == [7.0, 8.0, 9.0]
    assert s.latest()["t"] == 9.0


def test_subscribers_receive_ticks():
    s, _, _ = make()
    q = s.subscribe()
    s.tick()
    assert q.get_nowait()["t"] == 0.0
    s.unsubscribe(q)
    s.tick()
    assert q.empty()


def test_process_disappears_between_samples():
    s, clock, plist = make([server(extra=["--metrics"])])
    s.tick()
    plist.clear()
    clock.t = 1.0
    sample = s.tick()
    assert sample["servers"] == []
    assert s._trackers == {}


def test_probe_skipped_for_cli_processes():
    FakeProbe.calls.clear()
    s, _, _ = make([FakeProc(11, 1, "llama-cli", ["llama-cli", "-m", "x"])])
    sample = s.tick()
    assert FakeProbe.calls == []
    assert sample["servers"] == []
    assert sample["processes"][0]["role"] == "cli"


def test_metrics_not_probed_when_disabled_flag():
    s, _, _ = make([server()])
    srv = s.tick()["servers"][0]
    assert srv["metrics_status"] == "disabled"
    assert srv["throughput"]["source"] == "slots"


def test_extra_server_probed_without_process():
    s, _, _ = make(extra_servers=[("127.0.0.1", 8099)])
    [srv] = s.tick()["servers"]
    assert srv["role"] == "external" and srv["port"] == 8099 and srv["pid"] is None
    assert srv["metrics_status"] == "ok"


def test_vram_holders_and_unattributed():
    gpu = FakeGpu(used=6 * GiB, procs=[(10, 4 * GiB), (99, GiB)])
    s, _, _ = make([server()], gpu=gpu)
    sample = s.tick()
    holders = {h["pid"]: h for h in sample["vram_holders"]}
    assert holders[10]["is_llama"] and holders[10]["bytes"] == 4 * GiB
    assert not holders[99]["is_llama"] and holders[99]["name"] == "p99"
    assert sample["vram_unattributed"] == GiB
    assert sample["servers"][0]["vram_bytes"] == 4 * GiB


def codes(sample):
    return {(w["level"], w["code"]) for w in sample["warnings"]}


@pytest.mark.parametrize("used,expected", [(11.6, ("danger", "vram_high")),
                                           (10.5, ("warn", "vram_high"))])
def test_vram_warnings(used, expected):
    s, _, _ = make(gpu=FakeGpu(used=int(used * GiB)))
    assert expected in codes(s.tick())


def test_no_vram_warning_when_low():
    s, _, _ = make()
    assert not any(c == "vram_high" for _, c in codes(s.tick()))


def test_swap_warning_small_is_warn():
    s, _, _ = make([server(swap=2**20)])
    assert ("warn", "llm_swapping") in codes(s.tick())


def test_swap_warning_large_is_danger():
    s, _, _ = make([server(swap=600 * 2**20)])
    assert ("danger", "llm_swapping") in codes(s.tick())


def test_metrics_off_hint():
    s, _, _ = make([server()])
    w = [w for w in s.tick()["warnings"] if w["code"] == "metrics_off"][0]
    assert w["level"] == "info" and "--metrics" in w["hint"]


def test_layers_on_cpu_from_flags():
    s, _, _ = make([server(extra=["--n-cpu-moe", "4"])])
    assert ("info", "layers_on_cpu") in codes(s.tick())


def test_layers_on_cpu_from_ngl_vs_block_count(make_gguf):
    path = make_gguf(n_layer=4)
    p = FakeProc(10, 1, "llama-server", [BIN, "-m", str(path), "-ngl", "2"])
    s, _, _ = make([p])
    assert ("info", "layers_on_cpu") in codes(s.tick())
    p2 = FakeProc(10, 1, "llama-server", [BIN, "-m", str(path), "-ngl", "99"])
    s2, _, _ = make([p2])
    assert ("info", "layers_on_cpu") not in codes(s2.tick())


def test_gpu_unavailable_still_samples():
    gpu = FakeGpu()
    gpu.available, gpu.reason = False, "no driver"
    gpu.sample = lambda: []
    s, _, _ = make(gpu=gpu)
    sample = s.tick()
    assert sample["gpu"] == {"available": False, "reason": "no driver", "devices": []}
    assert sample["system"]["ram_total"] == 16 * GiB


def test_start_stop_thread():
    s, _, _ = make(interval=0.01)
    got = threading.Event()
    q = s.subscribe()
    s.start()
    try:
        q.get(timeout=2)
        got.set()
    finally:
        s.stop()
    assert got.is_set()


def test_slot_summary_handles_list_form_and_prompt_tokens(fakes_dir):
    import json as _json

    data = _json.loads((fakes_dir / "slots_list_form.json").read_text())

    class ListProbe(FakeProbe):
        def slots(self):
            return ProbeResult(True, "ok", data)

    clock = Clock()
    s = Sampler(FakeGpu(), system=fake_system, find_procs=lambda: find_llama_processes([server()]),
                probe_factory=ListProbe, clock=clock, name_map=lambda pids: {})
    [sl] = s.tick()["servers"][0]["slots"]
    assert sl == {"id": 0, "n_ctx": 32768, "busy": True, "n_decoded": 25, "n_prompt": 309}


def test_vanished_server_listed_as_stopped_for_a_while():
    s, clock, plist = make([server(extra=["--metrics"])])
    s.tick()
    plist.clear()
    clock.t = 1.0
    sample = s.tick()
    [gone] = sample["stopped"]
    assert gone["pid"] == 10 and gone["stopped_at"] == 1.0 and gone["name"] == "a.gguf"
    clock.t = 200.0
    assert s.tick()["stopped"] == []
