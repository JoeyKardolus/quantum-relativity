"""One self-contained results page from experiment result dicts.

Contract: `build_page(results, out_path)` writes a single HTML file: a <title>
and <style> first, no <html>/<head>/<body> tags (a host wraps it), inline CSS
with light and dark palettes, no external requests. One card per result in
the given order: title, question, verdict badge, summary, a measured vs known
table, the plots (read from result["plot_dir"], embedded as base64 PNG data
URIs) and settings and seed in small print. A listed plot that is missing
raises FileNotFoundError rather than leaving a hole. Returns out_path.
"""
from __future__ import annotations

import base64
import html
import json
import os

_TITLE = "Connection theory: first results"
_INTRO = ("A hobby simulation of space as a network of joints and connections with no coordinates. "
          "Each card is one milestone: what was asked, what was measured, and the known value it "
          "should match.")

_STYLE = """
:root{--bg:#fbfaf7;--card:#ffffff;--ink:#1d1c1a;--muted:#6b6862;--line:#e3e0d8;
--pass:#1f7a3f;--fail:#b3261e;--unclear:#9a6700;--badge-ink:#ffffff;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--card:#1f1f1d;
--ink:#ecebe7;--muted:#a19e96;--line:#34332f;--pass:#4fbf73;--fail:#ef6b62;--unclear:#e0b44c;
--badge-ink:#111111;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#161615;--card:#1f1f1d;--ink:#ecebe7;--muted:#a19e96;--line:#34332f;
--pass:#4fbf73;--fail:#ef6b62;--unclear:#e0b44c;--badge-ink:#111111;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:58rem;margin:0 auto;padding-inline:16px;padding-block:2rem 4rem}
h1{font-size:1.7rem;margin:0 0 .4rem}
.intro{color:var(--muted);margin:0 0 2rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1.2rem 1.3rem;margin:0 0 1.5rem}
.head{display:flex;flex-wrap:wrap;align-items:baseline;gap:.6rem;justify-content:space-between}
h2{font-size:1.25rem;margin:0}
.num{color:var(--muted);font-weight:400}
.badge{color:var(--badge-ink);border-radius:999px;padding:.1rem .7rem;font-size:.8rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.pass{background:var(--pass)}.fail{background:var(--fail)}.unclear{background:var(--unclear)}
.question{font-style:italic;color:var(--muted);margin:.5rem 0}
.table{overflow-x:auto;margin:1rem 0}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{text-align:left;padding:.35rem .5rem;border-bottom:1px solid var(--line);vertical-align:top}
td.value{font-variant-numeric:tabular-nums;white-space:nowrap}
th{color:var(--muted);font-weight:600}
.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(16rem,1fr));gap:.8rem;margin:1rem 0}
.plots img{width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}
.small{font-size:.75rem;color:var(--muted);word-break:break-word}
.small summary{cursor:pointer}
pre{white-space:pre-wrap;margin:.4rem 0 0}
"""


def build_page(results: list[dict], out_path: str) -> str:
    """Write the results page to out_path and return the path."""
    cards = "\n".join(_card(result) for result in results)
    page = (f"<title>{_TITLE}</title>\n<style>{_STYLE}</style>\n<main>\n<h1>{_TITLE}</h1>\n"
            f"<p class=\"intro\">{html.escape(_INTRO)}</p>\n{cards}\n</main>\n")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(page)
    return out_path


def _card(result: dict) -> str:
    verdict = result["verdict"]
    if verdict not in ("pass", "fail", "unclear"):
        raise ValueError(f"unknown verdict {verdict!r}")
    plots = "".join(_image(result.get("plot_dir", ""), name) for name in result.get("plots", []))
    settings = html.escape(json.dumps(result.get("settings", {}), indent=1, default=str))
    return (f"<section class=\"card\">\n<div class=\"head\"><h2><span class=\"num\">{result['milestone']}.</span> "
            f"{html.escape(result['title'])}</h2><span class=\"badge {verdict}\">{verdict}</span></div>\n"
            f"<p class=\"question\">{html.escape(result['question'])}</p>\n"
            f"<p>{html.escape(result['summary'])}</p>\n{_table(result.get('numbers', []))}\n"
            f"<div class=\"plots\">{plots}</div>\n"
            f"<details class=\"small\"><summary>Settings, seed {result.get('seed')}, "
            f"{result.get('seconds', '?')} s</summary><pre>{settings}</pre></details>\n</section>")


def _table(numbers: list[dict]) -> str:
    rows = "".join(f"<tr><td>{html.escape(entry['name'])}</td><td class=\"value\">{_value(entry['measured'])}</td>"
                   f"<td class=\"value\">{_value(entry['known'])}</td><td>{html.escape(str(entry.get('note', '')))}</td></tr>"
                   for entry in numbers)
    return ("<div class=\"table\"><table><thead><tr><th>Measurement</th><th>Measured</th><th>Known</th>"
            f"<th>Note</th></tr></thead><tbody>{rows}</tbody></table></div>")


def _value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3g}" if value and (abs(value) < 1e-3 or abs(value) >= 1e5) else f"{value:.4g}"
    return html.escape(str(value))


def _image(plot_dir: str, name: str) -> str:
    with open(os.path.join(plot_dir, name), "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"<img alt=\"{html.escape(name)}\" src=\"data:image/png;base64,{encoded}\">"
