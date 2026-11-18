"""Shared by every test of the engine: the real runs, made once, with every axiom checked after every step.

Contract:
  `needs_gpu`           a marker that skips a test cleanly on a machine without CUDA.
  `RUNS`                the real runs, fixed here before any run: name -> (settings that differ
                        from the defaults, cells a side, seed, steps).
      default   `Settings()`, 16 a side, seed 1, 40 steps. Since the start is laid at rest
                in balance (2026-11-18) the default universe stands: no collapse and no
                tear in these steps.
      collapsing  the defaults with `collapse_thin = 1.01`, 40 steps: a joint then
                collapses once it holds a hundredth more than it was made with, which
                the default run's joints do within eight steps (largest squeeze 1.05).
                Picked after the default run had shown no collapse, for one reason: it
                is the run in which collapses happen (495 by step 8, 1,874 by step 24).
      uneven    `unevenness = 0.3` and `push_strength = 40`, 16 a side, seed 1, 8 steps:
                a start with more spread in its volumes' energies, so some start longer
                than `tear_length`, and joints that hold, so those volumes tear before
                anything collapses. Picked, like `held`, after the first default run: it is
                the run in which tears happen. The tears double every step (the open
                tear chain), which is why it is 8 steps.
      held      the defaults with `push_strength = 40` and `tear_length = 200`, 16 a
                side, seed 1, 24 steps. Picked after the two runs above had shown the
                collapse and the tear chain, for one reason: with these two numbers the
                start's joints hold (largest squeeze 0.07) and no volume tears in the
                first 24 steps, so the fluid, the swirls, the crossing, the heat and
                the pull can be checked on a grid that stands, at the default pull. It
                is a test run, not a default and not a result.
      gentle    `held` with `pull_strength = 1e-9`, a hundred times weaker, 24 steps: the
                pull then turns a cell's fluid by a hundredth of a radian a step
                instead of one radian, so what the pull does to the swirls and to the
                crossing can be told from what the fluid does by itself.
  `run_records(name)`   one record per step of that run, cached for the test session. After
                        every step it calls every axiom's `check_step`, `read_books`,
                        `book_law` and `check_pattern`. A record holds:
      "step"; "axiom_1_energy" ... "axiom_7_quantum" (that axiom's `check_step` dict);
      "books"; "book_law"; "pattern"; "version" (tears and collapses so far);
      "relations" (a digest of every cell's neighbour entries and of the way table).
  `state_digest(state)` a digest of every box's quanta and phase and of the relations.
  `first_bad(counts)`   plain words for a count that must stay 0: the first step it is
                        not, and how large it gets.
Nobody re-runs a run: clause tests assert on the records.
"""
from __future__ import annotations

import functools
import hashlib

import numpy as np
import pytest
import warp as wp

from physics.engine import Settings, book_law, check_pattern, new_universe, read_books, step
from physics.engine.axiom_1_energy.tests.check_step import check_step as check_energy
from physics.engine.axiom_2_swirls.tests.check_step import check_step as check_swirls
from physics.engine.axiom_3_structure.tests.check_step import check_step as check_structure
from physics.engine.axiom_4_crossing.tests.check_step import check_step as check_crossing
from physics.engine.axiom_5_packing.tests.check_step import check_step as check_packing
from physics.engine.axiom_6_limits.tests.check_step import check_step as check_limits
from physics.engine.axiom_7_quantum.tests.check_step import check_step as check_quantum

wp.config.quiet = True

HAS_GPU = wp.is_cuda_available()
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="needs CUDA: the real runs are made on the GPU")

RUNS = {
    "default": ({}, 16, 1, 40),
    "collapsing": ({"collapse_thin": 1.01}, 16, 1, 40),
    "uneven": ({"unevenness": 0.3, "push_strength": 40.0}, 16, 1, 8),
    "held": ({"push_strength": 40.0, "tear_length": 200.0}, 16, 1, 24),
    "gentle": ({"pull_strength": 1e-9, "push_strength": 40.0, "tear_length": 200.0}, 16, 1, 24),
}
CHECKS = {
    "axiom_1_energy": check_energy,
    "axiom_2_swirls": check_swirls,
    "axiom_3_structure": check_structure,
    "axiom_4_crossing": check_crossing,
    "axiom_5_packing": check_packing,
    "axiom_6_limits": check_limits,
    "axiom_7_quantum": check_quantum,
}
_RELATION_ARRAYS = ("adj_at", "adj_n", "way_a", "way_b", "way_mouth_a", "way_mouth_b", "way_joint", "way_pass")


def _relations_digest(state) -> str:
    """One digest of every cell's neighbour entries and of the whole way table."""
    digest = hashlib.sha256()
    for name in _RELATION_ARRAYS:
        digest.update(np.ascontiguousarray(state.host(name)).tobytes())
    digest.update(np.ascontiguousarray(state.adj_id.numpy()).tobytes())
    digest.update(np.ascontiguousarray(state.adj_code.numpy()).tobytes())
    return digest.hexdigest()


def state_digest(state) -> str:
    """One digest of the whole state that matters: every box's quanta and phase, and the relations."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(state.quanta.numpy()[:state.count]).tobytes())
    digest.update(np.ascontiguousarray(state.phase.numpy()[:state.count]).tobytes())
    digest.update(_relations_digest(state).encode())
    return digest.hexdigest()


def _record(state, settings) -> dict:
    """Everything checked after one step: every axiom's counts, the books and the pattern."""
    record = {"step": int(state.step)}
    for name, check in CHECKS.items():
        record[name] = check(state, settings)
    books = read_books(state)
    record["books"] = books
    record["book_law"] = book_law(state.start_books, books)
    record["pattern"] = check_pattern(state)
    record["version"] = int(state.version)
    record["relations"] = _relations_digest(state)
    return record


@functools.lru_cache(maxsize=None)
def run_records(name: str) -> list:
    """The per-step records of one real run, run once per test session."""
    if not HAS_GPU:
        pytest.skip("needs CUDA: the real runs are made on the GPU")
    changed, side, seed, steps = RUNS[name]
    settings = Settings(**changed)
    state = new_universe(settings, cells_per_side=side, seed=seed, device="cuda")
    records = []
    for _ in range(steps):
        step(state, settings)
        records.append(_record(state, settings))
    return records


def first_bad(counts: list) -> str:
    """Plain words for a count that must stay 0: the first step it is not, and how large it gets."""
    bad = [step for step, count in enumerate(counts, start=1) if count]
    if not bad:
        return "0 at every step"
    return (f"first not 0 at step {bad[0]} ({counts[bad[0] - 1]}), not 0 in {len(bad)} of {len(counts)} steps, "
            f"largest size {max(abs(count) for count in counts)}, last {counts[-1]}")
