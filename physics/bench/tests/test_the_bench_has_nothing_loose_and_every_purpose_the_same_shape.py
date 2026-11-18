"""The bench's layout: its public names, two purpose directories of one shape, nothing loose."""
import pathlib

import physics.bench as bench

ROOT = pathlib.Path(bench.__file__).parent
PURPOSES = {"throughput", "largest_universe"}
LOOSE_ALLOWED = {"CLAUDE.md", "__init__.py"}


def _names(directory: pathlib.Path) -> tuple:
    entries = [entry for entry in directory.iterdir() if entry.name != "__pycache__"]
    return ({entry.name for entry in entries if entry.is_dir()}, {entry.name for entry in entries if entry.is_file()})


def test_every_public_name_is_there():
    assert all(hasattr(bench, name) for name in bench.__all__)


def test_the_bench_holds_its_two_purposes_its_tests_and_nothing_else():
    directories, files = _names(ROOT)
    assert directories == PURPOSES | {"tests"}
    assert files == LOOSE_ALLOWED | {"__main__.py"}


def test_every_purpose_holds_definitions_logic_and_tests_each_with_an_overlay_and_tests():
    for purpose in PURPOSES:
        directories, files = _names(ROOT / purpose)
        assert (directories, files) == ({"definitions", "logic", "tests"}, LOOSE_ALLOWED), purpose
        for part in ("definitions", "logic"):
            directories, files = _names(ROOT / purpose / part)
            assert directories == {"tests"}, (purpose, part)
            assert LOOSE_ALLOWED <= files and all(name.endswith(".py") for name in files - LOOSE_ALLOWED)


def test_every_tests_directory_holds_only_tests_and_at_least_one():
    for tests in ROOT.rglob("tests"):
        directories, files = _names(tests)
        assert directories == set(), tests
        assert files - {"__init__.py"} and all(name.startswith("test_") for name in files - {"__init__.py"}), tests


def test_definitions_never_import_logic_and_only_one_file_imports_the_engine():
    for source in ROOT.rglob("definitions/*.py"):
        assert ".logic" not in source.read_text(), source
    importers = {source.name for source in ROOT.rglob("*.py")
                 if "tests" not in source.parts and "from physics.engine" in source.read_text()}
    assert importers == {"engine_run.py"}
