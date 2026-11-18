"""Window: the viewer's door. `PYTHONPATH=. .venv/bin/python -m physics.view`

Opens a window on one energy volume, drawn as its swirl; `--scene crossing` shows a
volume, a joint and a volume, `--scene calm` a small grid that wraps round, as a flat cut
through one layer (`--cut`, `--slab` layers thick); `--pictures DIR` saves pictures. One process on the
card: this one. `BOXES`, `SPEED` and `FRAMES` only draw: fine enough to look smooth.
"""
import argparse
import os
from pathlib import Path

import warp as wp

from physics.view.definitions.camera import PITCH, YAW
from physics.view.logic.save_png import save_png
from physics.view.logic.scene import SCENES, drawn, eye_distance, layers, title, trace
from physics.view.logic.tracers import MARK_EVERY, MARKS, step_carrying

BOXES, SPEED = 24, 8
LINE_BOXES = 12  # the line's course is the same at any boxes, and a swirl's turn is quicker here
FRAMES = 120  # the window is drawn at most this often a second


def save_pictures(folder: Path, across: int, steps: int, every: int, device: str,
                  scene: str, cut: int, slab: int, turned: tuple | None) -> None:
    """Run `steps` steps and save a picture every `every` steps, the start included."""
    import pyglet
    from physics.view.logic.picture import picture
    from physics.view.logic.swirl_painter import SwirlPainter
    hidden = pyglet.window.Window(visible=False)  # the card draws into a picture, not this
    folder.mkdir(parents=True, exist_ok=True)
    store = SCENES[scene](across, device=device)
    layer = cut if layers(store) else None  # a grid's layer; a line shows every cell
    tracers = trace(store, layer, slab)
    painter = SwirlPainter(MARKS, tracers.tail.shape[1])
    painter.thick, painter.dots = max(layers(store), 1.0), scene != "one"
    yaw, pitch = turned or ((0.0, 0.0) if layers(store) and not slab else (YAW, PITCH))
    for done in range(steps + 1):
        if done % every == 0:
            file = folder / f"step_{done:05d}.png"
            painter.load(drawn(store, tracers, layer, slab))
            wide = 800 if store.grid == (1, 1, 1) else 1000  # several: as the window shows
            seen = picture(painter, across, yaw, pitch, eye_distance(store), wide, 800)
            save_png(file, seen)
            print(f"{file}: {title(store)}", flush=True)
        step_carrying(store, tracers, 0, mark=done % MARK_EVERY == 0, cells=store.count)
    hidden.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Energy volumes as red swirls, joints blue.")
    parser.add_argument("--scene", choices=list(SCENES), default="one", help="what to show")
    parser.add_argument("--boxes", type=int, help="boxes along an edge (even)")
    parser.add_argument("--speed", type=int, default=SPEED, help="steps a tick, a drawing speed")
    parser.add_argument("--pictures", type=Path, metavar="DIR", help="save pictures, no window")
    parser.add_argument("--steps", type=int, default=80, help="steps to run for the pictures")
    parser.add_argument("--every", type=int, default=20, help="steps between two pictures")
    parser.add_argument("--turned", type=float, nargs=2, metavar="ANGLE",
                        help="the pictures' view: turned about the upright, and upward")
    parser.add_argument("--cut", type=int, default=0, help="the pictures' layer of a grid")
    parser.add_argument("--slab", type=int, default=0, help="layers thick, in 3D; 0: a flat cut")
    chosen = parser.parse_args()
    chosen.boxes = chosen.boxes or (BOXES if chosen.scene == "one" else LINE_BOXES)
    wp.config.quiet = True
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")  # under WSL2 Mesa draws on the CPU without
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")  # these two
    if chosen.pictures:
        return save_pictures(chosen.pictures, chosen.boxes, chosen.steps, chosen.every, device,
                             chosen.scene, chosen.cut, chosen.slab, chosen.turned)
    import pyglet
    from physics.view.logic.cell_window import CellWindow
    CellWindow(chosen.boxes, chosen.speed, device, chosen.scene)
    pyglet.app.run(1 / FRAMES)


if __name__ == "__main__":
    main()
