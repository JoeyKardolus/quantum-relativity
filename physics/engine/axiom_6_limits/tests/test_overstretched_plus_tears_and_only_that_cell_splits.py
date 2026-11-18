"""Every tear of the uneven run, record by record: the volume was too long, and each piece is longer still."""
from __future__ import annotations

from physics.engine import Settings, size_law_host
from physics.engine.tests.conftest import RUNS, needs_gpu, run_records

pytestmark = needs_gpu

SETTINGS = Settings(**RUNS["uneven"][0])


def _tears() -> list:
    return [event for record in run_records("uneven") for event in record["axiom_6_limits"]["events"]
            if event["what"] == "tear"]


def test_every_tear_was_of_a_volume_that_was_too_long():
    tears = _tears()
    assert tears, "the uneven run had no tear"
    for event in tears:
        whole = sum(event["quanta"])
        assert whole >= 3 and size_law_host(whole, SETTINGS.size_constant) > SETTINGS.tear_length


def test_each_piece_is_longer_than_the_volume_was_which_is_the_open_tear_chain():
    tears = _tears()
    assert tears, "the uneven run had no tear"
    for event in tears:
        before = size_law_host(sum(event["quanta"]), SETTINGS.size_constant)
        assert all(length > before for length in event["lengths"])
