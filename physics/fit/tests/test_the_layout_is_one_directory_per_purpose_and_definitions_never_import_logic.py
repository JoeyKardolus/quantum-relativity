"""The fit package is one directory per purpose, each with definitions, logic and tests, nothing loose, and no definition imports logic."""
import pathlib

FIT = pathlib.Path(__file__).parents[1]
PURPOSES = ("fitted", "measures", "score", "search")
LOOSE = {"CLAUDE.md", "__init__.py"}


def _names(directory):
    return {entry.name for entry in directory.iterdir() if entry.name != "__pycache__"}


def test_the_package_holds_its_purposes_and_nothing_loose():
    assert _names(FIT) == LOOSE | {"__main__.py", "tests"} | set(PURPOSES)
    for purpose in PURPOSES:
        assert _names(FIT / purpose) == LOOSE | {"definitions", "logic", "tests"}, purpose
        for part in ("definitions", "logic"):
            inside = FIT / purpose / part
            assert LOOSE | {"tests"} <= _names(inside), inside
            assert all(entry.is_dir() or entry.suffix in (".py", ".md") for entry in inside.iterdir()), inside


def test_every_tests_directory_holds_a_test_and_every_overlay_is_under_500_words():
    for tests in [FIT / "tests"] + [path for path in FIT.rglob("tests") if path.is_dir()]:
        assert list(tests.glob("test_*.py")), f"{tests} holds no test"
    overlays = list(FIT.rglob("CLAUDE.md"))
    assert len(overlays) == 1 + 3 * len(PURPOSES)
    for overlay in overlays:
        assert len(overlay.read_text().split()) <= 500, overlay


def test_no_definition_imports_logic_and_the_score_and_measures_import_no_search():
    for source in FIT.rglob("*.py"):
        parts = source.relative_to(FIT).parts
        imports = [line for line in source.read_text().splitlines() if line.startswith(("from ", "import "))]
        if "definitions" in parts and "tests" not in parts:
            assert not any(".logic" in line for line in imports), source
        if parts[0] in ("measures", "score") and "tests" not in parts:
            assert not any("physics.fit.search" in line or "physics.fit.fitted" in line for line in imports), source
