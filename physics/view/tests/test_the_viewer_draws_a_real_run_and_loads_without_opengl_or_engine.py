"""The whole viewer: the command line on a real engine run, and what loading the package pulls in."""
import subprocess
import sys

from physics.view.tests.gpu_cases import need_cuda


def _run(*arguments, seconds=300):
    return subprocess.run([sys.executable, "-m", "physics.view", *arguments], capture_output=True, text=True,
                          timeout=seconds)


def test_loading_the_package_loads_no_opengl_and_no_engine():
    code = ("import sys, physics.view; "
            "print(sorted(name for name in sys.modules if name.split('.')[0] in ('pyglet', 'warp') "
            "or name.startswith('physics.engine')))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.strip() == "[]", done.stdout + done.stderr


def test_every_public_name_is_there_and_is_the_one_its_file_defines():
    import physics.view as view
    assert sorted(view.__all__) == view.__all__ and len(view.__all__) == 20
    for name in view.__all__:
        assert getattr(view, name).__name__ == name


def test_the_command_line_saves_slice_pictures_of_a_real_run_with_collapses(tmp_path):
    need_cuda()
    done = _run("--cells", "8", "--pictures", str(tmp_path), "--steps", "4", "--every", "2", "--slice")
    assert done.returncode == 0, done.stderr[-2000:]
    lines = [line for line in done.stdout.splitlines() if line.startswith("step ")]
    assert [line.split(":")[0] for line in lines] == ["step 0", "step 2", "step 4"]
    assert "collapses 0 tears 0" in lines[0] and "volumes 64" in lines[0]
    assert int(lines[-1].split("collapses ")[1].split()[0]) > 0, "the default run collapses joints"
    files = sorted(path.name for path in tmp_path.iterdir())
    assert files == ["step_00000.png", "step_00002.png", "step_00004.png"]
    first, last = (tmp_path / files[0]).read_bytes(), (tmp_path / files[-1]).read_bytes()
    assert first[:8] == b"\x89PNG\r\n\x1a\n" and first != last, "the picture follows the run"


def test_a_setting_that_does_not_exist_or_is_no_number_exits_with_a_message():
    need_cuda()
    unknown = _run("--set", "no_such_setting=1", "--pictures", "/nonexistent", "--slice")
    assert unknown.returncode == 2 and "no_such_setting" in unknown.stderr
    words = _run("--set", "friction=much")
    assert words.returncode == 2 and "expected name=number" in words.stderr
