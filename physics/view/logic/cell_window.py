"""Cell window: the window on one energy volume and its keys.

Space starts and stops (it starts stopped), R lays a fresh start, G sends the cell to
calm if it may go or takes it own again, drag turns the view, the wheel zooms, Esc quits.
"""
import pyglet
from pyglet.window import key

from physics.engine import one_cell, step
from physics.engine.axiom_3_structure.definitions.store import slot_of
from physics.engine.axiom_3_structure.logic.give_back import give_back
from physics.engine.axiom_3_structure.logic.take_own import take_own
from physics.view.definitions.camera import DISTANCE, PITCH, YAW
from physics.view.logic.paint import paint
from physics.view.logic.title_line import title_line


class CellWindow(pyglet.window.Window):
    def __init__(self, across: int, speed: int, device: str):
        super().__init__(1000, 800, resizable=True, vsync=False, caption="one energy volume")
        self.fresh = lambda: one_cell(across, device=device)
        self.store, self.speed, self.running = self.fresh(), speed, False
        self.yaw, self.pitch, self.distance = YAW, PITCH, DISTANCE
        pyglet.clock.schedule_interval(self.tick, 1 / 30)
        pyglet.clock.schedule_interval(self.retitle, 0.25)

    def tick(self, _elapsed: float) -> None:
        for _ in range(self.speed if self.running else 0):
            step(self.store)

    def retitle(self, _elapsed: float) -> None:
        self.set_caption(title_line(self.store) + ("" if self.running else "  (stopped: Space)"))

    def on_draw(self) -> None:
        width, height = self.width, self.height
        pixels = paint(self.store, 0, self.yaw, self.pitch, self.distance, width, height)
        picture = pyglet.image.ImageData(width, height, "RGB", pixels.tobytes(), pitch=-3 * width)
        picture.blit(0, 0)

    def on_key_press(self, symbol: int, _modifiers: int) -> None:
        if symbol == key.SPACE:
            self.running = not self.running
        elif symbol == key.R:
            self.store = self.fresh()
        elif symbol == key.G and slot_of(self.store, 0) < 0:
            take_own(self.store, 0)
        elif symbol == key.G:
            give_back(self.store, 0)
        elif symbol == key.ESCAPE:
            self.close()

    def on_mouse_drag(self, _x, _y, moved_x: int, moved_y: int, _buttons, _modifiers) -> None:
        self.yaw -= 0.01 * moved_x
        self.pitch = min(1.55, max(-1.55, self.pitch - 0.01 * moved_y))

    def on_mouse_scroll(self, _x, _y, _scroll_x, scroll_y: float) -> None:
        self.distance *= 0.9 ** scroll_y
