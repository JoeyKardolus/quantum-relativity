"""The command line on the real engine: one JSON line per result, with the numbers asked for."""
import json

import pytest

wp = pytest.importorskip("warp")
if not wp.is_cuda_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from physics.bench.__main__ import main  # noqa: E402


def _lines(capsys) -> list:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]


def test_throughput_prints_the_rates_and_the_run_s_facts(capsys):
    main(["throughput", "--cells", "8", "--steps", "2"])
    (line,) = _lines(capsys)
    assert (line["cells_per_side"], line["cells"], line["steps"], line["steps_done"]) == (8, 512, 2, 3)
    assert line["steps_per_second"] == pytest.approx(2 / line["seconds"])


def test_size_with_named_sizes_prints_one_search_a_size(capsys):
    main(["size", "--sizes", "8", "4", "--steps", "1"])
    first, second = _lines(capsys)
    assert [probe["size"] for probe in first["probes"]] == [4]
    assert [probe["size"] for probe in second["probes"]] == [8]
    assert second["probes"][0]["cells"] == 512
