"""Discover running llama.cpp processes and read their flags, memory and CPU."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import psutil

from ai_health.llama.cmdline import LlamaArgs, parse_llama_args

_ERRORS = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, ProcessLookupError)
# psutil.Process objects are kept between samples so cpu_percent() has a baseline.
_PROC_CACHE: dict[int, psutil.Process] = {}


@dataclass
class LlamaProcess:
    pid: int
    ppid: int
    exe_name: str
    args: LlamaArgs
    rss: int
    swap: int | None
    cpu_percent: float
    threads: int
    started: float
    children: list[int] = field(default_factory=list)
    role: str = "other"
    listen_ports: list[int] = field(default_factory=list)  # "server" | "router" | "router-child" | "cli" | "bench" | "other"


def _llama_name(name: str) -> bool:
    return name == "llama" or name.startswith("llama-")


def is_llama_process(name: str, argv: list[str]) -> bool:
    base = os.path.basename(argv[0]) if argv else ""
    return _llama_name(name) or _llama_name(base)


def _iter_system_procs():
    alive = set()
    for p in psutil.process_iter(["name", "create_time"]):
        alive.add(p.pid)
        cached = _PROC_CACHE.get(p.pid)
        if cached is None or cached.info["create_time"] != p.info["create_time"]:
            _PROC_CACHE[p.pid] = cached = p
        yield cached
    for pid in list(_PROC_CACHE):
        if pid not in alive:
            del _PROC_CACHE[pid]


def _listening(p) -> list[tuple[str, int]]:
    try:
        conns = p.net_connections(kind="inet")
    except (*_ERRORS, AttributeError):
        return []
    return sorted({(c.laddr.ip, c.laddr.port) for c in conns
                   if c.status == psutil.CONN_LISTEN and c.laddr}, key=lambda x: x[1])


def find_llama_processes(procs=None) -> list[LlamaProcess]:
    found: list[LlamaProcess] = []
    for p in (procs if procs is not None else _iter_system_procs()):
        try:
            info = getattr(p, "info", None)
            name = info["name"] if isinstance(info, dict) and info.get("name") else p.name()
            # comm (name) is the executable basename truncated to 15 chars, so
            # it is enough to gate on; reading every cmdline costs ~50 ms/tick.
            if not _llama_name(name):
                continue
            argv = p.cmdline()
            if not is_llama_process(name, argv):
                continue
            try:
                env = p.environ()
            except _ERRORS:
                env = {}
            args = parse_llama_args(argv or [name], env)
            listen = _listening(p) if args.kind == "server" else []
            if listen and args.sources.get("port") not in ("cli", "env"):
                # the default port differs between llama.cpp versions: trust the socket
                ip, port = listen[0]
                args.port, args.sources["port"] = port, "socket"
                if args.sources.get("host") not in ("cli", "env"):
                    args.host = "127.0.0.1" if ip in ("0.0.0.0", "::") else ip
            try:
                swap = int(p.memory_full_info().swap)
            except (*_ERRORS, AttributeError):
                swap = None
            found.append(LlamaProcess(
                pid=p.pid, ppid=p.ppid(), exe_name=os.path.basename(argv[0]) if argv else name,
                args=args, rss=int(p.memory_info().rss), swap=swap,
                cpu_percent=float(p.cpu_percent(interval=None)), threads=int(p.num_threads()),
                started=float(p.create_time()), listen_ports=[port for _, port in listen],
                role={"server": "server", "cli": "cli", "bench": "bench"}.get(args.kind, "other")))
        except _ERRORS:
            continue
    by_pid = {lp.pid: lp for lp in found}
    for lp in found:
        if lp.role == "server" and lp.args.router:
            lp.role = "router"
    for lp in found:
        parent = by_pid.get(lp.ppid)
        if parent is not None:
            parent.children.append(lp.pid)
            if parent.role == "router" and lp.role == "server":
                lp.role = "router-child"
    return found


def process_name_map(pids: list[int]) -> dict[int, str]:
    out = {}
    for pid in pids:
        try:
            out[pid] = psutil.Process(pid).name()
        except _ERRORS:
            pass
    return out
