"""Shared by the tests: one patch grown three layers out by its leak, made once a run (2 s on the CPU)."""
import pytest

from physics.engine.axiom_5_packing.logic.lumps import grown
from physics.engine.axiom_5_packing.logic.staging import copy_of


@pytest.fixture(scope="session")
def grown_patch():
    """The patch as it grew: 62 volumes at 13.8 turns, the first holding 332 calm volumes. Never staged in."""
    return grown(3, 7)


@pytest.fixture
def patch(grown_patch):
    """The patch again, its arrays its own, for a check to stage in."""
    return copy_of(grown_patch)
