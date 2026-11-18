"""Load records: every record a search has written so far.

Contract: `load_records(results)` returns the records of the JSON-lines file `results`
in the order written; an empty list when the file does not exist. Blank lines are
skipped; a line that is not JSON raises.
"""
from __future__ import annotations

import json
import pathlib


def load_records(results) -> list:
    """All records in a results file."""
    results = pathlib.Path(results)
    if not results.exists():
        return []
    return [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
