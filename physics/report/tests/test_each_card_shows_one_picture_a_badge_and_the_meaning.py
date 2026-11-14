"""Each card shows one picture, a verdict badge and the meaning; details go to results.json.

WHAT IT PINS. physics/report through `build_page`: the page opens with its
<title> and <style>, has no <html>/<head>/<body> tags, no external request,
no table, no settings and no em dash; each card embeds exactly one PNG whole
as a base64 data URI (result["picture"], else the first of "plots"), shows
its verdict badge and its meaning (result["meaning"], else "summary"); the
full result dicts are written to results.json beside the page.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. The results
are typed here by hand with two tiny PNGs drawn by matplotlib into tmp_path;
the expected data URIs are those files' bytes base64-encoded by the standard
library.

GUARDS. v2 results page, 2026-11-14.
"""
from __future__ import annotations

import base64
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from physics.report import build_page  # noqa: E402


def _png(path, heights) -> str:
    figure, axes = plt.subplots(figsize=(1, 1))
    axes.plot(heights)
    figure.savefig(path)
    plt.close(figure)
    return base64.b64encode(path.read_bytes()).decode("ascii")


def test_each_card_shows_one_picture_a_badge_and_the_meaning(tmp_path) -> None:
    """Page structure, one embedded picture per card, badges, meanings, results.json."""
    chosen = _png(tmp_path / "chosen.png", [0, 1, 4])
    other = _png(tmp_path / "other.png", [4, 1, 0])
    results = [
        {"milestone": 1, "title": "Validate", "question": "Is it 3D?", "verdict": "pass",
         "summary": "old summary", "meaning": "Space came out 3D (3.05). The network alone carries the dimension.",
         "picture": "chosen.png", "plots": ["other.png", "chosen.png"], "plot_dir": str(tmp_path),
         "numbers": [{"name": "dimension", "measured": 3.05, "known": 3}],
         "settings": {"joints": 30000}, "seed": 1, "seconds": 2.0},
        {"milestone": 3, "title": "Gravity test", "question": "Do lumps fall?", "verdict": "unclear",
         "summary": "Gap shrank, centres stayed.", "plots": ["other.png"], "plot_dir": str(tmp_path),
         "settings": {}, "seed": 1, "seconds": 1.0},
    ]
    page = tmp_path / "out" / "index.html"
    build_page(results, str(page))
    text = page.read_text(encoding="utf-8")
    assert text.startswith("<title>Quantum-relativity theory: first results</title>\n<style>")
    for banned in ("<html", "<head", "<body", "http://", "https://", "—", "<table", "30000", "old summary"):
        assert banned not in text
    assert "viewer page" in text
    first, second = text.split("<section")[1:]
    assert first.count("<img") == 1 and f"data:image/png;base64,{chosen}\"" in first
    assert second.count("<img") == 1 and f"data:image/png;base64,{other}\"" in second
    assert 'class="badge pass"' in first and 'class="badge unclear"' in second
    assert "Space came out 3D (3.05). The network alone carries the dimension." in first
    assert "Gap shrank, centres stayed." in second
    assert json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8")) == results
