"""Report layer: turns experiment result dicts into one self-contained HTML page.

Module map:
  build_page.py  build_page(results, out_path): cards per milestone, plots embedded as data URIs
"""
from physics.report.build_page import build_page

__all__ = ["build_page"]
