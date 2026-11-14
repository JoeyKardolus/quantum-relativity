"""One self-contained results page from experiment result dicts.

Contract: `build_page(results, out_path)` writes a single HTML file: a <title>
and <style> first, no <html>/<head>/<body> tags (a host wraps it), inline CSS
with light and dark palettes, no external requests. One card per result in
the given order: title, verdict badge, one picture (result["picture"], else
the first of result["plots"], read from result["plot_dir"] and embedded as a
base64 PNG data URI) and the meaning in plain words (result["meaning"], else
result["summary"]). No tables, no settings: the full result dicts go to
results.json next to the page. A named picture that is missing raises
FileNotFoundError rather than leaving a hole. Returns out_path.
"""
from __future__ import annotations

import base64
import html
import json
import os

_TITLE = "Quantum-relativity theory: first results"
_INTRO = ("A hobby simulation of space as a network of joints and connections with no coordinates. "
          "Each card is one milestone and what its result means for the theory.")
_VIEWER = "A separate viewer page shows the gravity scenes as they move."

_STYLE = """
:root{--bg:#fbfaf7;--card:#ffffff;--ink:#1d1c1a;--muted:#6b6862;--line:#e3e0d8;
--pass:#1f7a3f;--fail:#b3261e;--unclear:#9a6700;--badge-ink:#ffffff;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--card:#1f1f1d;
--ink:#ecebe7;--muted:#a19e96;--line:#34332f;--pass:#4fbf73;--fail:#ef6b62;--unclear:#e0b44c;
--badge-ink:#111111;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#161615;--card:#1f1f1d;--ink:#ecebe7;--muted:#a19e96;--line:#34332f;
--pass:#4fbf73;--fail:#ef6b62;--unclear:#e0b44c;--badge-ink:#111111;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:46rem;margin:0 auto;padding-inline:16px;padding-block:2rem 4rem}
h1{font-size:1.7rem;line-height:1.25;margin:0 0 .5rem}
.intro{color:var(--muted);margin:0 0 .3rem}
.viewer{color:var(--muted);margin:0 0 2rem;font-size:.92rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:1.2rem 1.3rem;margin:0 0 1.6rem}
.head{display:flex;flex-wrap:wrap;align-items:center;gap:.6rem;justify-content:space-between;margin:0 0 .9rem}
h2{font-size:1.2rem;margin:0}
.num{color:var(--muted);font-weight:400}
.badge{color:var(--badge-ink);border-radius:999px;padding:.12rem .75rem;font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.pass{background:var(--pass)}.fail{background:var(--fail)}.unclear{background:var(--unclear)}
.card img{display:block;width:100%;max-width:100%;height:auto;border-radius:8px;background:#fff;margin:0 0 .9rem}
.meaning{margin:0;font-size:1.02rem}
"""


def build_page(results: list[dict], out_path: str) -> str:
    """Write the results page and results.json beside it; return the page path."""
    cards = "\n".join(_card(result) for result in results)
    page = (f"<title>{_TITLE}</title>\n<style>{_STYLE}</style>\n<main>\n<h1>{_TITLE}</h1>\n"
            f"<p class=\"intro\">{html.escape(_INTRO)}</p>\n<p class=\"viewer\">{html.escape(_VIEWER)}</p>\n"
            f"{cards}\n</main>\n")
    folder = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(folder, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(page)
    with open(os.path.join(folder, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1, default=str)
    return out_path


def _card(result: dict) -> str:
    verdict = result["verdict"]
    if verdict not in ("pass", "fail", "unclear"):
        raise ValueError(f"unknown verdict {verdict!r}")
    picture = result.get("picture") or next(iter(result.get("plots", [])), None)
    image = _image(result.get("plot_dir", ""), picture) if picture else ""
    meaning = result.get("meaning") or result.get("summary", "")
    return (f"<section class=\"card\">\n<div class=\"head\"><h2><span class=\"num\">{result['milestone']}.</span> "
            f"{html.escape(result['title'])}</h2><span class=\"badge {verdict}\">{verdict}</span></div>\n"
            f"{image}<p class=\"meaning\">{html.escape(meaning)}</p>\n</section>")


def _image(plot_dir: str, name: str) -> str:
    with open(os.path.join(plot_dir, name), "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"<img alt=\"{html.escape(name)}\" src=\"data:image/png;base64,{encoded}\">\n"
