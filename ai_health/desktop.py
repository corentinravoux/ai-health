"""Native window (pywebview, GTK/WebKit2 on Linux) around the local server."""

from __future__ import annotations

from pathlib import Path

from ai_health.settings import Settings

ICON = Path(__file__).parent / "static" / "icon.svg"


class JsApi:
    """Exposed to the page as ``window.pywebview.api``."""

    def __init__(self):
        self._window = None

    def pick_gguf(self) -> str | None:
        import webview

        if self._window is None:
            return None
        res = self._window.create_file_dialog(
            webview.FileDialog.OPEN, file_types=("GGUF models (*.gguf)", "All files (*.*)"))
        return res[0] if res else None


def run_window(url: str, settings: Settings) -> None:
    """Blocks until the window is closed; updates ``settings`` width/height."""
    import webview

    api = JsApi()
    window = webview.create_window(
        "ai-health", url, js_api=api, width=settings.width, height=settings.height,
        min_size=(1100, 700), background_color="#0e1013", text_select=True)
    api._window = window

    def on_resized(width, height):
        settings.width, settings.height = int(width), int(height)

    def on_loaded():
        # Native drag-and-drop gives real file paths (pywebviewFullPath).
        def on_drop(e):
            files = (e.get("dataTransfer") or {}).get("files") or []
            for f in files:
                p = f.get("pywebviewFullPath")
                if p:
                    window.evaluate_js(f"window.aiHealthDropped && window.aiHealthDropped({p!r})")
                    break
        try:
            from webview.dom import DOMEventHandler

            window.dom.document.events.drop += DOMEventHandler(on_drop, True, True)
        except Exception:
            pass

    window.events.resized += on_resized
    window.events.loaded += on_loaded
    webview.start(gui="gtk", private_mode=True, icon=str(ICON) if ICON.exists() else None)
