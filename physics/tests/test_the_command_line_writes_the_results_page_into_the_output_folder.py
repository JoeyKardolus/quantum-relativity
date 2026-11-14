"""The command line writes the results page into the output folder.

WHAT IT PINS. physics through `python -m physics` (`physics.__main__.main`):
it runs the experiments with the given seed and output folder, prints one line
per milestone with its verdict, and writes OUT/index.html.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. The experiments
are replaced by one hand-made result with no plots, so the expected verdict
line, seed and page path are the literals typed here.

ONE FUNCTION. This module's one test.

WHAT IT WAS PROVED AGAINST. The page written under another name than
index.html.

GUARDS. First version of the physics package, 2026-11-14.
"""
from __future__ import annotations

import sys

import physics.__main__ as command


def test_the_command_line_writes_the_results_page_into_the_output_folder(tmp_path, monkeypatch, capsys) -> None:
    """--seed and --out reach the experiments; the page lands at OUT/index.html."""
    seen = {}

    def fake_run_all(seed, out_dir):
        seen.update(seed=seed, out_dir=out_dir)
        return [{"milestone": 2, "title": "Grid dynamics", "question": "Do the rules work?",
                 "verdict": "pass", "summary": "They do.", "numbers": [], "plots": [],
                 "settings": {}, "seed": seed, "seconds": 0.1}]

    monkeypatch.setattr(command, "run_all", fake_run_all)
    monkeypatch.setattr(sys, "argv", ["physics", "--seed", "9", "--out", str(tmp_path)])
    command.main()
    assert seen == {"seed": 9, "out_dir": str(tmp_path)}
    assert "2 Grid dynamics: pass" in capsys.readouterr().out
    assert "Grid dynamics" in (tmp_path / "index.html").read_text(encoding="utf-8")
