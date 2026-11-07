"""ai-health entry point: wire collectors, sampler, explorer and the UI."""

from __future__ import annotations

import argparse
import signal
import threading
import webbrowser
from pathlib import Path

from ai_health import __version__
from ai_health.collectors.gpu import GpuCollector
from ai_health.sampler import Sampler
from ai_health.settings import load_settings, save_settings
from ai_health.web import api
from ai_health.web.server import make_server

STATIC = Path(__file__).parent / "static"


def parse_hostport(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    return (host or "127.0.0.1"), int(port)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ai-health",
                                 description="Monitor a local llama.cpp LLM and explore GGUF models.")
    ap.add_argument("--version", action="version", version=f"ai-health {__version__}")
    ap.add_argument("--browser", action="store_true", help="open the UI in the default browser instead of a window")
    ap.add_argument("--no-window", action="store_true", help="only run the local server (prints its URL)")
    ap.add_argument("--port", type=int, default=0, help="local port (default: random)")
    ap.add_argument("--history", type=int, default=900, help="seconds of history kept in memory")
    ap.add_argument("--open", metavar="MODEL.gguf", help="open this model in the explorer at start")
    ap.add_argument("--extra-server", metavar="HOST:PORT", action="append", default=[],
                    help="also probe this llama-server even if no local process is found (repeatable)")
    return ap


def build_sampler(args, gpu=None) -> Sampler:
    return Sampler(gpu if gpu is not None else GpuCollector(),
                   history_s=args.history,
                   extra_servers=[parse_hostport(x) for x in args.extra_server])


def build_explorer(settings):
    from ai_health.explorer.service import ExplorerService

    return ExplorerService(settings=settings, on_change=lambda: save_settings(settings))


def main(argv: list[str] | None = None, stop_event: threading.Event | None = None,
         on_ready=None, gpu=None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    sampler = build_sampler(args, gpu)
    explorer = build_explorer(settings)
    routes = api.all_routes()
    if args.open:
        try:
            explorer.open(args.open)
        except (OSError, ValueError) as e:
            print(f"ai-health: could not open {args.open}: {e}")
    server, url, _ = make_server(sampler, explorer, STATIC, port=args.port, routes=routes)
    sampler.start()
    th = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2},
                          name="http", daemon=True)
    th.start()
    if on_ready:
        on_ready(url)
    stop = stop_event or threading.Event()
    try:
        if args.no_window or args.browser:
            print(f"ai-health {__version__} on {url}", flush=True)
            if args.browser:
                webbrowser.open(url)
            if stop_event is None and threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGTERM, lambda *a: stop.set())
            try:
                while not stop.wait(0.5):
                    pass
            except KeyboardInterrupt:
                pass
        else:
            from ai_health.desktop import run_window

            run_window(url, settings)
    finally:
        sampler.stop()
        server.shutdown()
        server.server_close()
        explorer.close()
        if not (args.no_window or args.browser):
            save_settings(settings)
        if hasattr(sampler.gpu, "close"):
            sampler.gpu.close()
    return 0
