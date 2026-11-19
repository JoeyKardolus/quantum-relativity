"""Window: the viewer's door. `PYTHONPATH=. .venv/bin/python -m physics.view`

A window on one energy volume; `--scene crossing` a volume, a joint and a volume,
`--scene grow` one volume whose leak tears off layer after
layer, `--scene lumps` two fuller regions in a small grown patch, `--scene hole` a packed
middle that collapses; `--pictures DIR` saves pictures (`--along`, `--turned`, `--cut`, `--ring`). One process on the card. `BOXES`, `SPEED`, `FRAMES` only draw.
"""
import argparse
import os
from pathlib import Path

import warp as wp

from physics.view.definitions.camera import ALONG, PITCH, YAW, eye_distance
from physics.view.logic.save_png import save_png
from physics.view.logic.scene import SCENES, Seen, advance, drawn, title, trace
from physics.view.logic.tracers import MARKS

BOXES, SPEED = 24, 8
LINE_BOXES = 12  # the line's course is the same at any boxes, and a swirl's turn is quicker here
GROW_BOXES, GROW_SPEED = 6, 1  # the least boxes, for many cells; a layer about every second
FRAMES = 120  # the window is drawn at most this often a second


def save_pictures(folder: Path, across: int, steps: int, every: int, device: str,
                  scene: str, flat: bool, turned: tuple | None, along: int | None, ring: int = -1) -> None:
    """Run `steps` steps and save a picture every `every` steps, the start included."""
    import pyglet
    from physics.view.logic.picture import picture
    from physics.view.logic.swirl_painter import SwirlPainter
    hidden = pyglet.window.Window(visible=False)  # the card draws into a picture, not this
    folder.mkdir(parents=True, exist_ok=True)
    store = SCENES[scene](across, device=device)
    seen = Seen(trace(store), whole=False if flat else None, ring=ring)
    painter = SwirlPainter(MARKS, seen.tracers.tail.shape[1])
    alone = store.side == 0
    yaw, pitch = turned or ((YAW, PITCH) if along is None and alone else ALONG[2 if along is None else along])
    for done in range(steps + 1):
        marks = drawn(store, seen)  # every step, as the window does: the middles settle
        if done % every == 0:
            file = folder / f"step_{done:05d}.png"
            painter.load(marks)
            seen_now = picture(painter, across, yaw, pitch, eye_distance(store), 800 if alone else 1000, 800)
            save_png(file, seen_now)
            print(f"{file}: {title(store, seen)}", flush=True)
        advance(store, seen)
    hidden.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Energy volumes as red swirls, joints blue.")
    parser.add_argument("--scene", choices=list(SCENES), default="one", help="what to show")
    parser.add_argument("--boxes", type=int, help="boxes along an edge (even)")
    parser.add_argument("--speed", type=int, help="steps a tick, a drawing speed")
    parser.add_argument("--pictures", type=Path, metavar="DIR", help="save pictures, no window")
    parser.add_argument("--steps", type=int, default=80, help="steps to run for the pictures")
    parser.add_argument("--every", type=int, default=20, help="steps between two pictures")
    parser.add_argument("--turned", type=float, nargs=2, help="the view: about the upright, upward")
    parser.add_argument("--along", type=int, choices=(0, 1, 2), help="the direction looked along")
    parser.add_argument("--cut", action="store_true", help="the pictures show the flat cut")
    parser.add_argument("--ring", type=int, choices=(0, 1, 2), default=-1, help="one ring's fluid alone")
    chosen = parser.parse_args()
    grows = chosen.scene in ("grow", "lumps", "hole")
    chosen.boxes = chosen.boxes or (GROW_BOXES if grows else BOXES if chosen.scene == "one" else LINE_BOXES)
    chosen.speed = chosen.speed or (GROW_SPEED if grows else SPEED)
    wp.config.quiet = True
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")  # under WSL2 Mesa draws on the CPU without
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")  # these two
    if chosen.pictures:
        return save_pictures(chosen.pictures, chosen.boxes, chosen.steps, chosen.every, device,
                             chosen.scene, chosen.cut, chosen.turned, chosen.along, chosen.ring)
    import pyglet
    from physics.view.logic.cell_window import CellWindow
    CellWindow(chosen.boxes, chosen.speed, device, chosen.scene)
    pyglet.app.run(1 / FRAMES)


if __name__ == "__main__":
    main()
