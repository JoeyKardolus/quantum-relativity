"""Window: the viewer's door. `PYTHONPATH=. .venv/bin/python -m physics.view`

Opens a window on one energy volume, drawn as its swirl, or with `--pictures DIR` saves
pictures of it. One process on the card: this one, engine and drawing. `BOXES`, `SPEED`
and `FRAMES` only draw: fine enough to look smooth, a round in a few seconds.
"""
import argparse
import os
from pathlib import Path

import warp as wp

from physics.engine import one_cell
from physics.view.definitions.camera import DISTANCE, PITCH, YAW
from physics.view.logic.save_png import save_png
from physics.view.logic.streaks import streaks
from physics.view.logic.title_line import title_line
from physics.view.logic.tracers import COUNT, MARK_EVERY, MARKS, lay_tracers, step_carrying

BOXES, SPEED = 24, 8
FRAMES = 120  # the window is drawn at most this often a second


def save_pictures(folder: Path, across: int, steps: int, every: int, device: str) -> None:
    """Run `steps` steps and save a picture every `every` steps, the start included."""
    import pyglet
    from physics.view.logic.picture import picture
    from physics.view.logic.swirl_painter import SwirlPainter
    hidden = pyglet.window.Window(visible=False)  # the card draws into a picture, not this
    folder.mkdir(parents=True, exist_ok=True)
    store, painter = one_cell(across, device=device), SwirlPainter(MARKS, COUNT)
    tracers = lay_tracers(store, 0)
    for done in range(steps + 1):
        if done % every == 0:
            file = folder / f"step_{done:05d}.png"
            painter.load(streaks(tracers))
            save_png(file, picture(painter, across, YAW, PITCH, DISTANCE, 800, 800))
            print(f"{file}: {title_line(store)}", flush=True)
        step_carrying(store, tracers, 0, mark=done % MARK_EVERY == 0)
    hidden.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="One red energy volume with its swirl.")
    parser.add_argument("--boxes", type=int, default=BOXES, help="boxes along an edge (even)")
    parser.add_argument("--speed", type=int, default=SPEED, help="steps a tick, a drawing speed")
    parser.add_argument("--pictures", type=Path, metavar="DIR", help="save pictures, no window")
    parser.add_argument("--steps", type=int, default=80, help="steps to run for the pictures")
    parser.add_argument("--every", type=int, default=20, help="steps between two pictures")
    chosen = parser.parse_args()
    wp.config.quiet = True
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")  # under WSL2 Mesa draws on the CPU without
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")  # these two
    if chosen.pictures:
        return save_pictures(chosen.pictures, chosen.boxes, chosen.steps, chosen.every, device)
    import pyglet
    from physics.view.logic.cell_window import CellWindow
    CellWindow(chosen.boxes, chosen.speed, device)
    pyglet.app.run(1 / FRAMES)


if __name__ == "__main__":
    main()
