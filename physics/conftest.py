"""Shared by the tests: the ball grown three layers out by its leak, grown once (two minutes on the CPU) and kept in the tmp folder."""
import pytest

from physics.engine.axiom_5_packing.logic.ball import ball
from physics.engine.axiom_5_packing.logic.staging import copy_of


@pytest.fixture(scope="session")
def grown_patch():
    """The ball as it grew: 111 volumes at 108 turns, the first holding 1,640 calm volumes. Never staged in."""
    return ball()


@pytest.fixture
def patch(grown_patch):
    """The patch again, its arrays its own, for a check to stage in."""
    return copy_of(grown_patch)
