"""The report page embeds every plot and shows each verdict and number.

WHAT IT PINS. physics/report through `build_page`: the page opens with its
<title> and <style> and has no <html>/<head>/<body> tags, makes no external
request, embeds each listed PNG whole as a base64 data URI, shows each
card's verdict badge, and prints every measured and known number.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. The results
are built here by hand: two milestones with a real PNG drawn by matplotlib
into tmp_path; the expected data URI is that file's bytes base64-encoded by
the standard library, and the numbers are the literals typed below.

ONE FUNCTION. This module's one test.

WHAT IT WAS PROVED AGAINST. Embedding only the first 64 bytes of each PNG,
which leaves a broken image in the page.

GUARDS. First version of the physics package, 2026-11-14.
"""
from __future__ import annotations

import base64

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from physics.report import build_page  # noqa: E402


def test_the_report_page_embeds_every_plot_and_shows_each_verdict_and_number(tmp_path) -> None:
    """Page structure, embedded PNG bytes, badges and numbers."""
    figure, axes = plt.subplots(figsize=(2, 2))
    axes.plot([0, 1, 2], [0, 1, 4])
    figure.savefig(tmp_path / "curve.png")
    plt.close(figure)
    results = [
        {"milestone": 1, "title": "Validate", "question": "Is it 3D?", "verdict": "pass",
         "summary": "It is.", "numbers": [{"name": "dimension", "measured": 3.05, "known": 3,
                                           "note": "within 10%"}],
         "plots": ["curve.png"], "plot_dir": str(tmp_path), "settings": {"joints": 30000},
         "seed": 1, "seconds": 2.0},
        {"milestone": 3, "title": "Gravity test", "question": "Do lumps fall?", "verdict": "unclear",
         "summary": "Gap shrank, centres stayed.", "numbers": [{"name": "travel", "measured": 0.0,
                                                               "known": None, "note": "fixed ruler"}],
         "plots": [], "settings": {}, "seed": 1, "seconds": 1.0},
    ]
    page = (tmp_path / "index.html")
    build_page(results, str(page))
    text = page.read_text(encoding="utf-8")
    assert text.startswith("<title>Quantum-relativity theory: first results</title>")
    assert text.index("<style>") < text.index("<main>")
    for tag in ("<html", "<head", "<body", "http://", "https://", "—"):
        assert tag not in text
    encoded = base64.b64encode((tmp_path / "curve.png").read_bytes()).decode("ascii")
    assert f"data:image/png;base64,{encoded}\"" in text
    assert 'class="badge pass"' in text and 'class="badge unclear"' in text
    assert text.index("Validate") < text.index("Gravity test")
    for shown in ("3.05", ">3<", "n/a", "within 10%", "Do lumps fall?"):
        assert shown in text
