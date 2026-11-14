"""Report layer: turns experiment result dicts into one self-contained HTML page.

Module map:
  build_page.py  build_page(results, out_path): one card per milestone (badge, one picture, meaning), results.json beside it
"""
from physics.report.build_page import build_page

__all__ = ["build_page"]
