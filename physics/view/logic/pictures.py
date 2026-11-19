"""Pictures: a scene stepped with no window, a picture saved every so many steps.

The same painter that draws the window draws each picture (`picture.py`); the scene's
numbers go to the terminal beside each file's name, as the window's title has them.
"""
from pathlib import Path

from physics.view.definitions.camera import ALONG, PITCH, YAW, eye_distance
from physics.view.logic.save_png import save_png
from physics.view.logic.scene import SCENES, Seen, advance, drawn, title, trace
from physics.view.logic.tracers import MARKS


def save_pictures(folder: Path, across: int, steps: int, every: int, device: str,
                  scene: str, flat: bool | None, turned: tuple | None, along: int | None, ring: int = -1) -> None:
    """Run `steps` steps and save a picture every `every` steps, the start included.

    `flat`: the flat cut (True), every cell (False), or as the window chooses (None)."""
    import pyglet
    from physics.view.logic.picture import picture
    from physics.view.logic.swirl_painter import SwirlPainter
    hidden = pyglet.window.Window(visible=False)  # the card draws into a picture, not this
    folder.mkdir(parents=True, exist_ok=True)
    store = SCENES[scene](across, device=device)
    seen = Seen(trace(store), whole=None if flat is None else not flat, ring=ring)
    painter = SwirlPainter(MARKS, seen.tracers.tail.shape[1])
    alone = store.side == 0
    yaw, pitch = turned or ((YAW, PITCH) if along is None and alone else ALONG[2 if along is None else along])
    for done in range(steps + 1):
        marks = drawn(store, seen)  # every step, as the window does: the middles settle
        if done % every == 0:
            file = folder / f"step_{store.step:07d}.png"
            painter.load(marks)
            seen_now = picture(painter, across, yaw, pitch, eye_distance(store), 800 if alone else 1000, 800)
            save_png(file, seen_now)
            print(f"{file}: {title(store, seen)}", flush=True)
        advance(store, seen)
    hidden.close()
