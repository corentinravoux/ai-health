import json
import threading
import urllib.request

from ai_health.devtools.fake_server import make_fake_server
from ai_health.llama.metrics import parse_prometheus


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as r:
        return r.read().decode()


def test_fake_server_counters_increase():
    clock = {"t": 100.0}
    srv = make_fake_server(port=0, clock=lambda: clock["t"], tps=35.0, busy_s=20, idle_s=5)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    th.start()
    try:
        port = srv.server_address[1]
        assert json.loads(get(port, "/health"))["status"] == "ok"
        m1 = parse_prometheus(get(port, "/metrics"))
        clock["t"] += 2.0
        m2 = parse_prometheus(get(port, "/metrics"))
        dt = m2["tokens_predicted_seconds_total"] - m1["tokens_predicted_seconds_total"]
        dn = m2["tokens_predicted_total"] - m1["tokens_predicted_total"]
        assert dt > 0 and abs(dn / dt - 35.0) < 1.0
        slots = json.loads(get(port, "/slots"))
        assert slots[0]["n_ctx"] == 8192 and "n_decoded" in slots[0]["next_token"]
    finally:
        srv.shutdown()
        srv.server_close()
