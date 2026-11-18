"""The engine's front: its public names, the one Settings record, the merged state, and nothing loose on disk."""
from __future__ import annotations

import dataclasses
import pathlib

import pytest

import physics.engine as engine

ROOT = pathlib.Path(engine.__file__).parent
AXIOMS = ("axiom_1_energy", "axiom_2_swirls", "axiom_3_structure", "axiom_4_crossing", "axiom_5_packing",
          "axiom_6_limits", "axiom_7_quantum")
LOOSE_ALLOWED = {"CLAUDE.md", "__init__.py", "conftest.py"}
CODE_DIRECTORIES = {"definitions", "logic", "tests"}


def test_every_public_name_is_there():
    assert all(hasattr(engine, name) for name in engine.__all__)


def test_settings_holds_every_axiom_s_numbers_and_refuses_zero_and_unknown_names():
    names = {field.name for field in dataclasses.fields(engine.Settings)}
    assert {"step_time", "size_constant", "friction", "light_speed", "collapse_thin", "fastest_speed"} <= names
    for name in ("step_time", "size_constant", "friction", "pull_strength", "push_strength", "tear_length",
                 "slowest_speed"):
        with pytest.raises(ValueError):
            engine.Settings(**{name: 0.0})
    with pytest.raises(TypeError):
        engine.Settings(crossing_rate=1.0)
    with pytest.raises(TypeError):
        engine.Settings(fluid_pressure=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        engine.Settings().friction = 0.1


def test_no_state_name_is_claimed_twice():
    assert not set(engine.ARRAYS) & set(engine.HOST)
    assert {"quanta", "phase", "kind", "length", "pull", "way_moved", "fresh_quanta", "swirl_axis", "heat_quanta"} <= set(engine.ARRAYS)


def test_the_engine_holds_seven_axiom_directories_and_nothing_loose():
    assert sorted(path.name for path in ROOT.iterdir() if path.is_dir() and path.name != "__pycache__") \
        == sorted(AXIOMS + ("tests",))
    for axiom in AXIOMS:
        inside = {path.name for path in (ROOT / axiom).iterdir() if path.is_dir() and path.name != "__pycache__"}
        assert inside == CODE_DIRECTORIES, axiom
    for folder in [ROOT] + [ROOT / axiom for axiom in AXIOMS]:
        loose = {path.name for path in folder.iterdir() if path.is_file()}
        assert loose <= LOOSE_ALLOWED and {"CLAUDE.md", "__init__.py"} <= loose, folder


def test_every_definitions_and_logic_directory_has_its_overlay_and_its_tests():
    for axiom in AXIOMS:
        for part in ("definitions", "logic"):
            folder = ROOT / axiom / part
            assert (folder / "CLAUDE.md").is_file() and (folder / "__init__.py").is_file(), folder
            assert (folder / "tests" / "__init__.py").is_file(), folder
            inside = {path.name for path in folder.iterdir() if path.is_dir() and path.name != "__pycache__"}
            assert inside == {"tests"}, folder


def test_definitions_never_import_logic():
    for axiom in AXIOMS:
        for source in (ROOT / axiom / "definitions").glob("*.py"):
            assert ".logic." not in source.read_text(), source
