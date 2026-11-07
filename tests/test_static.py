"""The front end has no build step: check that every reference resolves and
that nothing reaches the network (the app must work offline)."""

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "ai_health" / "static"
REF = re.compile(r'(?:src|href)="([^"#?]+)')
IMPORT = re.compile(r'''(?:import\s[^'"]*?from\s*|import\s*\(\s*|^import\s*)['"](\.{1,2}/[^'"]+)['"]''', re.M)
URL_IN_CSS = re.compile(r'url\(\s*["\']?([^"\')]+)')


def test_index_exists():
    assert (STATIC / "index.html").is_file()


def test_index_references_exist():
    html = (STATIC / "index.html").read_text()
    refs = [r for r in REF.findall(html) if not r.startswith(("data:", "mailto:"))]
    assert refs, "index.html references nothing"
    for r in refs:
        assert (STATIC / r.lstrip("/")).is_file(), f"index.html -> missing {r}"


def test_js_imports_resolve():
    files = list((STATIC / "js").rglob("*.js"))
    assert files
    for f in files:
        for rel in IMPORT.findall(f.read_text()):
            assert (f.parent / rel).resolve().is_file(), f"{f.relative_to(STATIC)} imports missing {rel}"


def test_css_urls_resolve():
    for f in (STATIC / "css").glob("*.css"):
        for rel in URL_IN_CSS.findall(f.read_text()):
            if rel.startswith(("data:", "#")):
                continue
            assert (f.parent / rel).resolve().is_file(), f"{f.name} -> missing {rel}"


def test_no_external_urls():
    for f in STATIC.rglob("*"):
        if f.suffix not in (".html", ".js", ".css", ".svg") or "vendor" in f.parts:
            continue
        text = f.read_text()
        hits = [u for u in re.findall(r'https?://[^\s"\')]+', text)
                if not u.startswith("http://www.w3.org/")]
        assert not hits, f"{f.relative_to(STATIC)} references the network: {hits[:3]}"
