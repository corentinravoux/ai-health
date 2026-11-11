"""Local HTTP server: static front end, token-protected JSON API, SSE stream.

Binds to 127.0.0.1 only. Every /api/* request must carry the per-launch token
(header ``X-AI-Health-Token`` or ``?token=``) and a loopback ``Host`` header,
so other web pages open in the user's browser cannot drive the API
(CSRF / DNS rebinding).
"""

from __future__ import annotations

import dataclasses
import hmac
import json
import mimetypes
import posixpath
import queue
import secrets
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from . import api

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("font/woff2", ".woff2")
MAX_BODY = 1 * 2**20
HEARTBEAT_S = 15.0


def _json_default(o):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def dumps(obj) -> bytes:
    return json.dumps(obj, default=_json_default, separators=(",", ":")).encode()


class AppServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, ctx):
        super().__init__(addr, handler)
        self.ctx = ctx
        self.stopping = threading.Event()

    def shutdown(self):
        self.stopping.set()
        super().shutdown()


class Handler(BaseHTTPRequestHandler):
    server: AppServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    # -- helpers -----------------------------------------------------------
    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload):
        self._send(status, dumps(payload), "application/json")

    def _error(self, status: int, message: str):
        self._json(status, {"error": message})

    def _host_ok(self) -> bool:
        port = self.server.server_address[1]
        host = (self.headers.get("Host") or "").lower()
        return host in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _token_ok(self, query: dict) -> bool:
        token = self.headers.get("X-AI-Health-Token") or query.get("token", [""])[0]
        return hmac.compare_digest(token.encode(), self.server.ctx.token.encode())

    # -- dispatch ----------------------------------------------------------
    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def _dispatch(self, method: str):
        parsed = urllib.parse.urlsplit(self.path)
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.parse_qs(parsed.query)
        if not self._host_ok():
            return self._error(403, "forbidden host")
        if not path.startswith("/api/"):
            if method != "GET":
                return self._error(405, "method not allowed")
            return self._static(path)
        if not self._token_ok(query):
            return self._error(403, "missing or bad token")
        if method == "GET" and path == "/api/stream":
            return self._stream()
        handler = self.server.ctx.routes.get((method, path))
        if handler is None:
            return self._error(404, "unknown endpoint")
        body = None
        if method == "POST":
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._error(400, "bad Content-Length")
            if n < 0:
                return self._error(400, "bad Content-Length")
            if n > MAX_BODY:
                return self._error(413, "body too large")
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._error(400, "body must be JSON")
            if not isinstance(body, dict):
                return self._error(400, "body must be a JSON object")
        try:
            status, payload = handler(self.server.ctx, query, body)
        except api.ApiError as e:
            return self._error(e.status, e.message)
        except Exception as e:  # report, don't kill the server thread
            return self._error(500, f"{type(e).__name__}: {e}")
        self._json(status, payload)

    def _static(self, path: str):
        root: Path = self.server.ctx.static_dir
        rel = posixpath.normpath(path).lstrip("/")
        if rel in ("", "."):
            rel = "index.html"
        target = (root / rel).resolve()
        if root.resolve() not in target.parents or not target.is_file():
            return self._error(404, "not found")
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype == "image/svg+xml":
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    def _stream(self):
        sampler = self.server.ctx.sampler
        q = sampler.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            while not self.server.stopping.is_set():
                try:
                    sample = q.get(timeout=HEARTBEAT_S)
                    self.wfile.write(b"data: " + dumps(sample) + b"\n\n")
                except queue.Empty:
                    self.wfile.write(b": heartbeat\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            sampler.unsubscribe(q)


def make_server(sampler, explorer, static_dir: Path, port: int = 0,
                token: str | None = None, routes: dict | None = None,
                settings=None, save_settings=None):
    token = token or secrets.token_urlsafe(24)
    ctx = SimpleNamespace(sampler=sampler, explorer=explorer, static_dir=Path(static_dir),
                          token=token, routes={**api.SETTINGS_ROUTES, **(routes or api.ROUTES)},
                          settings=settings, save_settings=save_settings)
    server = AppServer(("127.0.0.1", port), Handler, ctx)
    base_url = f"http://127.0.0.1:{server.server_address[1]}/?token={token}"
    return server, base_url, token
