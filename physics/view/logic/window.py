"""Window: the viewer's door. `PYTHONPATH=. .venv/bin/python -m physics.view`

A window on one energy volume; `--scene crossing` a volume, a joint and a volume,
`--scene grow` one volume whose leak tears off layer after
layer, `--scene hole` a packed middle that collapses, `--scene fall`, `dense` and `coast` a
grown ball with a region made fuller (grown once, two minutes, then kept); `--pictures DIR`
saves pictures (`--along`, `--turned`, `--cut`, `--whole`, `--ring`). `--scene grow --far DIR` runs the
universe far forward with no window, a log and kept states in DIR (`far.py`); `--from FILE`
shows a kept state and steps on from it. One process on the card. `BOXES`, `SPEED`, `FRAMES` only draw.
"""
import argparse
import os
from pathlib import Path

import warp as wp

from physics.engine.axiom_5_packing.logic.far import MORE, far, resumed
from physics.view.logic.pictures import save_pictures
from physics.view.logic.scene import SCENES

BOXES, SPEED = 24, 8
LINE_BOXES = 12  # the line's course is the same at any boxes, and a swirl's turn is quicker here
GROW_BOXES, GROW_SPEED = 6, 1  # the least boxes, for many cells; a layer about every second
FRAMES = 120  # the window is drawn at most this often a second


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
    parser.add_argument("--whole", action="store_true", help="the pictures show every cell, however many")
    parser.add_argument("--ring", type=int, choices=(0, 1, 2), default=-1, help="one ring's fluid alone")
    parser.add_argument("--from", dest="start", type=Path, metavar="FILE", help="grow: start from a kept state")
    parser.add_argument("--far", type=Path, metavar="DIR", help="grow: run far forward, no window")
    parser.add_argument("--minutes", type=float, default=40, help="how long the far run goes")
    chosen = parser.parse_args()
    grows = chosen.scene not in ("one", "crossing")  # many cells: the least boxes
    chosen.boxes = chosen.boxes or (GROW_BOXES if grows else BOXES if chosen.scene == "one" else LINE_BOXES)
    chosen.speed = chosen.speed or (GROW_SPEED if grows else SPEED)
    wp.config.quiet = True
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")  # under WSL2 Mesa draws on the CPU without
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")  # these two
    if chosen.start:
        SCENES["grow"] = lambda _across, device: resumed(chosen.start, device, MORE)  # room to grow on
    if chosen.far:
        far(chosen.far, chosen.minutes, device, start=chosen.start and resumed(chosen.start, device))
        return
    if chosen.pictures:
        return save_pictures(chosen.pictures, chosen.boxes, chosen.steps, chosen.every, device,
                             chosen.scene, True if chosen.cut else False if chosen.whole else None,
                             chosen.turned, chosen.along, chosen.ring)
    import pyglet
    from physics.view.logic.cell_window import CellWindow
    CellWindow(chosen.boxes, chosen.speed, device, chosen.scene)
    pyglet.app.run(1 / FRAMES)


if __name__ == "__main__":
    main()
