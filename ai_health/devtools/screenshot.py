"""Dev helper: start ai-health headless and screenshot the UI with Playwright
(Firefox). Not used by the app itself.

    python -m ai_health.devtools.screenshot out.png --mode explorer --open demo.gguf
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", default="monitor", choices=["monitor", "explorer"])
    ap.add_argument("--theme", default="dark", choices=["dark", "light"])
    ap.add_argument("--wait", type=float, default=4.0, help="seconds to let data stream in")
    ap.add_argument("--size", default="1440x900")
    ap.add_argument("--open", help="model to open in the explorer")
    ap.add_argument("--extra-server", action="append", default=[])
    ap.add_argument("--click", action="append", default=[], help="CSS selector to click before the shot")
    ap.add_argument("--fill", action="append", default=[], help="selector=text to type before the shot")
    ap.add_argument("--js", action="append", default=[], help="JS to evaluate first (order: js, fill, click)")
    a = ap.parse_args(argv)

    cmd = [sys.executable, "-m", "ai_health", "--no-window", "--port", "0"]
    for x in a.extra_server:
        cmd += ["--extra-server", x]
    if a.open:
        cmd += ["--open", a.open]
    import os
    import tempfile

    env = dict(os.environ, XDG_CONFIG_HOME=tempfile.mkdtemp(prefix="ai-health-shot-"))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env)
    try:
        line = proc.stdout.readline()
        url = line.strip().split(" on ", 1)[1]
        if a.mode == "explorer":
            url += "#explorer"
        from playwright.sync_api import sync_playwright

        w, h = (int(x) for x in a.size.split("x"))
        with sync_playwright() as p:
            browser = p.firefox.launch()
            page = browser.new_page(viewport={"width": w, "height": h},
                                    color_scheme="dark" if a.theme == "dark" else "light")
            page.add_init_script(f"try{{localStorage.setItem('ai-health-theme','{a.theme}')}}catch(e){{}}")
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(url)
            time.sleep(a.wait)
            for js in a.js:
                page.evaluate(js)
                time.sleep(1.0)
            for item in a.fill:
                sel, _, text = item.partition("=")
                page.fill(sel, text)
                time.sleep(1.2)
            for sel in a.click:
                page.click(sel)
                time.sleep(0.8)
            page.screenshot(path=a.out)
            browser.close()
        for e in errors:
            print("page error:", e)
        print(a.out)
        return 1 if errors else 0
    finally:
        proc.terminate()
        proc.wait(5)


if __name__ == "__main__":
    raise SystemExit(main())
