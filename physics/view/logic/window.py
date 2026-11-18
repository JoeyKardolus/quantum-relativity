"""Window: the viewer's door. `PYTHONPATH=. .venv/bin/python -m physics.view`

Opens a window on one energy volume, drawn as its swirl; `--scene crossing` shows a
volume, a joint and a volume, `--scene grid` four volumes and their joints; `--pictures
DIR` saves pictures instead. One process on the card: this one, engine and drawing.
`BOXES`, `SPEED` and `FRAMES` only draw: fine enough to look smooth.
"""
import argparse
import os
from pathlib import Path

import warp as wp

from physics.view.definitions.camera import PITCH, YAW
from physics.view.logic.save_png import save_png
from physics.view.logic.scene import (
    SCENES, drawn, eye_distance, lay_scene, title, tracer_count)
from physics.view.logic.tracers import MARK_EVERY, MARKS, lay_tracers, step_carrying

BOXES, SPEED = 24, 8
LINE_BOXES = 12  # the line's course is the same at any boxes, and a swirl's turn is quicker here
FRAMES = 120  # the window is drawn at most this often a second


def save_pictures(folder: Path, across: int, steps: int, every: int, device: str,
                  cells: int, yaw: float = YAW, pitch: float = PITCH) -> None:
    """Run `steps` steps and save a picture every `every` steps, the start included."""
    import pyglet
    from physics.view.logic.picture import picture
    from physics.view.logic.swirl_painter import SwirlPainter
    hidden = pyglet.window.Window(visible=False)  # the card draws into a picture, not this
    folder.mkdir(parents=True, exist_ok=True)
    store, painter = lay_scene(cells, across, device), SwirlPainter(MARKS, tracer_count(cells))
    tracers = lay_tracers(store, 0, tracer_count(cells), cells=cells)
    for done in range(steps + 1):
        if done % every == 0:
            file = folder / f"step_{done:05d}.png"
            painter.load(drawn(store, tracers, cells))
            wide = 800 if cells == 1 else 1000  # the line is seen as the window shows it
            seen = picture(painter, across, yaw, pitch, eye_distance(cells), wide, 800)
            save_png(file, seen)
            print(f"{file}: {title(store, cells)}", flush=True)
        step_carrying(store, tracers, 0, mark=done % MARK_EVERY == 0, cells=cells)
    hidden.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Energy volumes as red swirls, joints blue.")
    parser.add_argument("--scene", choices=list(SCENES), default="one", help="what to show")
    parser.add_argument("--boxes", type=int, help="boxes along an edge (even)")
    parser.add_argument("--speed", type=int, default=SPEED, help="steps a tick, a drawing speed")
    parser.add_argument("--pictures", type=Path, metavar="DIR", help="save pictures, no window")
    parser.add_argument("--steps", type=int, default=80, help="steps to run for the pictures")
    parser.add_argument("--every", type=int, default=20, help="steps between two pictures")
    parser.add_argument("--turned", type=float, nargs=2, default=(YAW, PITCH), metavar="ANGLE",
                        help="the pictures' view: turned about the upright, and upward")
    chosen = parser.parse_args()
    chosen.boxes = chosen.boxes or (BOXES if chosen.scene == "one" else LINE_BOXES)
    wp.config.quiet = True
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")  # under WSL2 Mesa draws on the CPU without
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")  # these two
    if chosen.pictures:
        return save_pictures(chosen.pictures, chosen.boxes, chosen.steps, chosen.every, device,
                             SCENES[chosen.scene], *chosen.turned)
    import pyglet
    from physics.view.logic.cell_window import CellWindow
    CellWindow(chosen.boxes, chosen.speed, device, SCENES[chosen.scene])
    pyglet.app.run(1 / FRAMES)


if __name__ == "__main__":
    main()
