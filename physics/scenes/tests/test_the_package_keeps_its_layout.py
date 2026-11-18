"""The package keeps the engine's layout: a purpose directory holds CLAUDE.md, __init__.py, tests/, definitions/ and logic/, nothing else is loose, and definitions never import logic."""
import re
from pathlib import Path

import physics.scenes as scenes

ROOT = Path(scenes.__file__).parent
SCENE_NAMES = {"orbit", "bound_lump", "black_hole", "hubble_gap", "timescape", "expansion", "structure", "dimension",
               "isotropy", "spin", "quantum_split", "two_lumps", "gravity"}
SHARED = {"run", "placed_start", "volumes", "along_grid"}
LOOSE = {"CLAUDE.md", "__init__.py"}
OVERLAY_WORDS = 500


def _children(directory: Path) -> tuple[set, set]:
    """The names of a directory's subdirectories and of its files, caches left out."""
    entries = [entry for entry in directory.iterdir() if entry.name != "__pycache__"]
    return {entry.name for entry in entries if entry.is_dir()}, {entry.name for entry in entries if entry.is_file()}


def test_the_package_root_holds_the_scenes_the_shared_pieces_and_its_tests():
    folders, files = _children(ROOT)
    assert folders == SCENE_NAMES | SHARED | {"tests"} and files == LOOSE
    assert set(scenes.SCENES) == SCENE_NAMES


def test_every_purpose_directory_has_definitions_logic_and_tests_and_nothing_loose():
    for purpose in sorted(SCENE_NAMES | SHARED):
        folders, files = _children(ROOT / purpose)
        assert folders == {"definitions", "logic", "tests"} and files == LOOSE, purpose
        for part in ("definitions", "logic"):
            inner_folders, inner_files = _children(ROOT / purpose / part)
            assert inner_folders == {"tests"} and LOOSE < inner_files, f"{purpose}/{part}"
            assert all(name.endswith(".py") for name in inner_files - LOOSE), f"{purpose}/{part}"
            listed = (ROOT / purpose / part / "__init__.py").read_text()
            assert all(name in listed for name in inner_files - LOOSE), f"{purpose}/{part}/__init__.py lists every file"


def test_every_tests_directory_holds_a_real_test_file():
    tests = [path for path in ROOT.rglob("tests") if path.is_dir()]
    assert len(tests) == 1 + 3 * len(SCENE_NAMES | SHARED)
    for directory in tests:
        folders, files = _children(directory)
        assert not folders and any(name.startswith("test_") for name in files), directory
        assert all(name.endswith(".py") for name in files), directory


def test_every_scene_has_a_pass_mark_and_no_definition_imports_logic():
    for scene in sorted(SCENE_NAMES):
        mark = (ROOT / scene / "definitions" / "pass_mark.py").read_text()
        assert "def passes(" in mark and "before" in mark, f"{scene}: a mark says it was written before a run"
    for module in ROOT.glob("*/definitions/*.py"):
        assert not re.search(r"^\s*(from|import)\s+\S*\.logic", module.read_text(), re.MULTILINE), module


def test_every_overlay_is_under_500_words():
    overlays = list(ROOT.rglob("CLAUDE.md"))
    assert len(overlays) == 1 + 3 * len(SCENE_NAMES | SHARED)
    for overlay in overlays:
        assert len(overlay.read_text().split()) <= OVERLAY_WORDS, overlay
