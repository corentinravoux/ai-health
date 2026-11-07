"""1 Hz sampler: GPU + system + llama.cpp processes + llama-server probes.

Produces one JSON-able dict per tick (the contract the front end reads),
keeps a ring buffer of history and pushes each sample to subscribers.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from ai_health.collectors.processes import find_llama_processes, process_name_map
from ai_health.collectors.system import system_sample
from ai_health.llama.cmdline import LlamaArgs, key_flags_summary
from ai_health.llama.metrics import ThroughputTracker, slot_decoded
from ai_health.llama.probe import ProbeResult, ServerProbe

MiB = 2**20
SERVER_ROLES = ("server", "router-child")
SWAP_DANGER = 512 * MiB  # a little old swap is normal; this much hurts tokens/s


@functools.lru_cache(maxsize=32)
def _block_count(path: str, mtime: float) -> int | None:
    try:
        import gguf

        r = gguf.GGUFReader(path, "r")
        arch_f = r.fields.get("general.architecture")
        arch = bytes(arch_f.parts[arch_f.data[0]]).decode()
        f = r.fields.get(f"{arch}.block_count")
        return int(f.parts[f.data[0]][0]) if f else None
    except Exception:
        return None


def block_count_of(path: str | None) -> int | None:
    """``block_count`` from a GGUF header (cached by path + mtime)."""
    if not path:
        return None
    try:
        return _block_count(path, os.path.getmtime(path))
    except OSError:
        return None


def _args_dict(a: LlamaArgs) -> dict:
    d = dataclasses.asdict(a)
    d.pop("raw", None)
    return d


def _cpu_layers_reason(a: LlamaArgs) -> str | None:
    if a.cpu_moe:
        return "--cpu-moe keeps all expert weights on CPU"
    if a.n_cpu_moe:
        return f"--n-cpu-moe {a.n_cpu_moe} keeps experts of the first {a.n_cpu_moe} layers on CPU"
    if a.n_cpu_ffn:
        return f"--n-cpu-ffn {a.n_cpu_ffn} keeps FFN of the first {a.n_cpu_ffn} layers on CPU"
    if any(buf.upper().startswith("CPU") for _, buf in a.overrides):
        return "-ot overrides place some tensors on CPU"
    if isinstance(a.ngl, int):
        n = block_count_of(a.model)
        if a.ngl == 0:
            return "-ngl 0: everything runs on CPU"
        if n is not None and a.ngl < n + 1:
            return f"-ngl {a.ngl} of {n + 1} layers: {n + 1 - a.ngl} on CPU"
    return None


class Sampler:
    def __init__(self, gpu, system=system_sample, find_procs=find_llama_processes,
                 probe_factory=ServerProbe, interval: float = 1.0, history_s: int = 900,
                 clock=time.time, extra_servers=(), name_map=process_name_map):
        self.gpu = gpu
        self._system = system
        self._find_procs = find_procs
        self._probe_factory = probe_factory
        self.interval = interval
        self.history_s = history_s
        self._clock = clock
        self._extra = list(extra_servers)
        self._name_map = name_map
        self._history: deque[dict] = deque(maxlen=max(1, int(history_s / interval)))
        self._trackers: dict[object, ThroughputTracker] = {}
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="probe")

    # -- probing -----------------------------------------------------------
    def _probe(self, host: str, port: int, args: LlamaArgs | None) -> dict:
        p = self._probe_factory(host, port)
        health = p.health()
        slots = p.slots() if args is None or args.slots else ProbeResult(False, "disabled")
        if args is None or args.metrics:
            metrics = p.metrics()
        else:
            metrics = ProbeResult(False, "disabled", detail="started without --metrics")
        return {"health": health, "slots": slots, "metrics": metrics}

    # -- one tick ----------------------------------------------------------
    def tick(self) -> dict:
        t = self._clock()
        devices = self.gpu.sample() if self.gpu.available else []
        system = self._system()
        procs = self._find_procs()

        vram_by_pid: dict[int, int] = {}
        for d in devices:
            for gp in d.processes:
                if gp.used_bytes is not None:
                    vram_by_pid[gp.pid] = vram_by_pid.get(gp.pid, 0) + gp.used_bytes

        targets = []  # (key, host, port, args, proc)
        for lp in procs:
            if lp.role in SERVER_ROLES:
                targets.append((lp.pid, lp.args.host, lp.args.port, lp.args, lp))
        for host, port in self._extra:
            targets.append((("ext", host, port), host, port, None, None))
        futures = {key: self._pool.submit(self._probe, host, port, args)
                   for key, host, port, args, _ in targets}

        servers = []
        live_keys = set()
        for key, host, port, args, lp in targets:
            live_keys.add(key)
            try:
                res = futures[key].result(timeout=max(2.0, self.interval * 2))
            except Exception as e:  # pragma: no cover - defensive
                err = ProbeResult(False, "error", detail=str(e))
                res = {"health": err, "slots": err, "metrics": err}
            tracker = self._trackers.setdefault(key, ThroughputTracker())
            m, s = res["metrics"], res["slots"]
            tp = tracker.update(t, m.data if m.ok else None, s.data if s.ok else None)
            slots = []
            if s.ok and isinstance(s.data, list):
                for sl in s.data:
                    slots.append({"id": sl.get("id"), "n_ctx": sl.get("n_ctx"),
                                  "busy": bool(sl.get("is_processing")),
                                  "n_decoded": slot_decoded(sl),
                                  "n_prompt": sl.get("n_prompt_tokens")})
            model = (args.model if args else None)
            servers.append({
                "pid": lp.pid if lp else None, "role": lp.role if lp else "external",
                "host": host, "port": port, "model": model,
                "name": (args.alias if args and args.alias else
                         os.path.basename(model) if model else f"{host}:{port}"),
                "flags": key_flags_summary(args) if args else [],
                "args": _args_dict(args) if args else None,
                "health": res["health"].status, "metrics_status": m.status,
                "slots_status": s.status, "slots": slots,
                "throughput": dataclasses.asdict(tp),
                "rss": lp.rss if lp else None, "swap": lp.swap if lp else None,
                "cpu_percent": lp.cpu_percent if lp else None,
                "threads": lp.threads if lp else None,
                "vram_bytes": vram_by_pid.get(lp.pid) if lp else None,
                "children": lp.children if lp else [],
                "cpu_layers": _cpu_layers_reason(args) if args else None,
            })
        for key in list(self._trackers):
            if key not in live_keys:
                del self._trackers[key]

        llama_pids = {lp.pid for lp in procs}
        processes = [{
            "pid": lp.pid, "ppid": lp.ppid, "name": lp.exe_name, "role": lp.role,
            "model": lp.args.model, "rss": lp.rss, "swap": lp.swap,
            "cpu_percent": lp.cpu_percent, "threads": lp.threads,
            "vram_bytes": vram_by_pid.get(lp.pid), "children": lp.children,
        } for lp in procs]

        names = self._name_map([p for p in vram_by_pid if p not in llama_pids])
        exe = {lp.pid: lp.exe_name for lp in procs}
        holders = sorted(({"pid": pid, "name": exe.get(pid) or names.get(pid) or str(pid),
                           "bytes": b, "is_llama": pid in llama_pids}
                          for pid, b in vram_by_pid.items()), key=lambda h: -h["bytes"])
        used_total = sum(d.mem_used for d in devices)
        unattributed = max(0, used_total - sum(vram_by_pid.values())) if devices else 0

        sample = {
            "t": t,
            "gpu": {"available": bool(self.gpu.available), "reason": self.gpu.reason,
                    "devices": [dataclasses.asdict(d) for d in devices]},
            "system": dataclasses.asdict(system),
            "servers": servers,
            "processes": processes,
            "vram_holders": holders,
            "vram_unattributed": unattributed,
        }
        sample["warnings"] = self._warnings(sample, procs)
        with self._lock:
            self._history.append(sample)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(sample)
            except queue.Full:
                pass
        return sample

    @staticmethod
    def _warnings(sample: dict, procs) -> list[dict]:
        out = []
        for d in sample["gpu"]["devices"]:
            frac = d["mem_used"] / d["mem_total"] if d["mem_total"] else 0
            if frac > 0.95:
                out.append({"level": "danger", "code": "vram_high",
                            "text": f"VRAM {frac:.0%} full on {d['name']}",
                            "hint": "Lower -ngl, add --n-cpu-moe, reduce -c or quantize the KV cache (-ctk q8_0 -ctv q8_0)."})
            elif frac > 0.85:
                out.append({"level": "warn", "code": "vram_high",
                            "text": f"VRAM {frac:.0%} used on {d['name']}",
                            "hint": "Little headroom left for a longer context."})
        for lp in procs:
            if lp.swap:
                out.append({"level": "danger" if lp.swap >= SWAP_DANGER else "warn",
                            "code": "llm_swapping",
                            "text": f"{lp.exe_name} (pid {lp.pid}) has {lp.swap / MiB:.0f} MiB in swap",
                            "hint": "RAM is too small for what runs on CPU: offload more to GPU or use a smaller quant."})
        for s in sample["servers"]:
            label = s["name"]
            if s["metrics_status"] == "disabled" and s["role"] != "external":
                out.append({"level": "info", "code": "metrics_off",
                            "text": f"{label}: /metrics is off, tokens/s estimated from /slots",
                            "hint": "Restart llama-server with --metrics for exact tokens/s."})
            if s["health"] == "auth" or s["metrics_status"] == "auth":
                out.append({"level": "warn", "code": "server_auth",
                            "text": f"{label}: endpoints need an API key (--api-key)",
                            "hint": "Throughput cannot be read; process and memory data are still shown."})
            if s["cpu_layers"]:
                out.append({"level": "info", "code": "layers_on_cpu",
                            "text": f"{label}: part of the model runs on CPU",
                            "hint": s["cpu_layers"]})
        order = {"danger": 0, "warn": 1, "info": 2}
        return sorted(out, key=lambda w: order[w["level"]])

    # -- access ------------------------------------------------------------
    def latest(self) -> dict | None:
        with self._lock:
            return self._history[-1] if self._history else None

    def history(self, seconds: int) -> list[dict]:
        with self._lock:
            if not self._history:
                return []
            cutoff = self._history[-1]["t"] - seconds
            return [s for s in self._history if s["t"] > cutoff]

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=32)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    # -- thread ------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.tick()
            except Exception:  # never let one bad tick kill the monitor
                import traceback

                traceback.print_exc()
            self._stop.wait(max(0.0, self.interval - (time.monotonic() - t0)))

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._pool.shutdown(wait=False, cancel_futures=True)
