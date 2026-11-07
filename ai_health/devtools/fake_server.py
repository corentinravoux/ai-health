"""Dev-only stand-in for llama-server (/health, /props, /slots, /metrics) with
counters that rise like a real generation: busy for ``busy_s`` seconds at
~``tps`` tokens/s, then idle for ``idle_s``.

    python -m ai_health.devtools.fake_server --port 8099
    ai-health --browser --extra-server 127.0.0.1:8099
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _State:
    def __init__(self, clock, tps, busy_s, idle_s):
        self.clock, self.tps, self.busy_s, self.idle_s = clock, tps, busy_s, idle_s
        self.t0 = clock()

    def snapshot(self):
        t = self.clock() - self.t0
        period = self.busy_s + self.idle_s
        cycles, phase = divmod(t, period)
        busy = phase < self.busy_s
        busy_time = cycles * self.busy_s + min(phase, self.busy_s)
        # rate slows by up to 15 % within a generation, like a filling KV cache:
        # tokens = integral of tps * (1 - a * p / busy_s) dp
        a, p = 0.15, min(phase, self.busy_s)
        per_cycle = self.tps * self.busy_s * (1 - a / 2)
        predicted = int(cycles * per_cycle + self.tps * (p - a * p * p / (2 * self.busy_s)))
        n_decoded = int(min(phase, self.busy_s) * self.tps) if busy else 0
        prompt_tokens = int((cycles + 1) * 1800)
        return {"busy": busy, "busy_time": busy_time, "predicted": predicted,
                "n_decoded": n_decoded, "prompt_tokens": prompt_tokens,
                "prompt_seconds": (cycles + 1) * 1.4}


def make_fake_server(port=8099, clock=time.time, tps=35.0, busy_s=20, idle_s=5):
    state = _State(clock, tps, busy_s, idle_s)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            s = state.snapshot()
            path = self.path.split("?")[0]
            if path == "/health":
                return self._send(200, json.dumps({"status": "ok"}))
            if path == "/props":
                return self._send(200, json.dumps({"total_slots": 1, "model_path": "fake.gguf",
                                                   "default_generation_settings": {"n_ctx": 8192}}))
            if path == "/slots":
                return self._send(200, json.dumps([{
                    "id": 0, "n_ctx": 8192, "is_processing": s["busy"],
                    "next_token": {"has_next_token": s["busy"], "n_decoded": s["n_decoded"]}}]))
            if path == "/metrics":
                lines = [
                    f"llamacpp:prompt_tokens_total {s['prompt_tokens']}",
                    f"llamacpp:prompt_seconds_total {s['prompt_seconds']:.3f}",
                    f"llamacpp:tokens_predicted_total {s['predicted']}",
                    f"llamacpp:tokens_predicted_seconds_total {s['busy_time']:.3f}",
                    f"llamacpp:requests_processing {1 if s['busy'] else 0}",
                    "llamacpp:requests_deferred 0",
                ]
                return self._send(200, "\n".join(lines) + "\n", "text/plain")
            return self._send(404, json.dumps({"error": "not found"}))

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    return srv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--tps", type=float, default=35.0)
    a = ap.parse_args(argv)
    srv = make_fake_server(a.port, tps=a.tps)
    print(f"fake llama-server on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
