"""Window: the viewer's door. `PYTHONPATH=. .venv/bin/python -m physics.view.logic.window`

Opens a window on one red energy volume with its swirl, or with `--pictures DIR` saves
pictures without a window. One process on the card: this one, engine and painting.
"""
import argparse
import os
from pathlib import Path

import warp as wp

from physics.engine import one_cell, step
from physics.engine.axiom_3_structure.definitions.kinds import ACROSS
from physics.view.definitions.camera import DISTANCE, PITCH, YAW
from physics.view.logic.paint import paint
from physics.view.logic.save_png import save_png
from physics.view.logic.title_line import title_line


def save_pictures(folder: Path, across: int, steps: int, every: int, device: str) -> None:
    """Run `steps` steps and save a picture every `every` steps, the start included."""
    folder.mkdir(parents=True, exist_ok=True)
    store = one_cell(across, device=device)
    for done in range(steps + 1):
        if done % every == 0:
            file = folder / f"step_{done:05d}.png"
            save_png(file, paint(store, 0, YAW, PITCH, DISTANCE, 800, 800))
            print(f"{file}: {title_line(store)}", flush=True)
        step(store)


def main() -> None:
    parser = argparse.ArgumentParser(description="One red energy volume with its swirl.")
    parser.add_argument("--boxes", type=int, default=ACROSS, help="boxes along an edge (even)")
    parser.add_argument("--speed", type=int, default=1, help="steps a frame, a drawing speed")
    parser.add_argument("--pictures", type=Path, metavar="DIR", help="save pictures, no window")
    parser.add_argument("--steps", type=int, default=80, help="steps to run for the pictures")
    parser.add_argument("--every", type=int, default=20, help="steps between two pictures")
    chosen = parser.parse_args()
    wp.config.quiet = True
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    if chosen.pictures:
        return save_pictures(chosen.pictures, chosen.boxes, chosen.steps, chosen.every, device)
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")  # under WSL2 Mesa draws on the CPU without
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")  # these two
    import pyglet
    from physics.view.logic.cell_window import CellWindow
    CellWindow(chosen.boxes, chosen.speed, device)
    pyglet.app.run(1 / 120)


if __name__ == "__main__":
    main()
