"""Cell window: the window on one energy volume and its keys.

Space stops and starts, R lays a fresh start, G sends the cell to calm if it may go or
takes it own again, drag turns the view, the wheel zooms, Esc quits. The fluid steps
`TICKS` times a second; the window is drawn at most as often as the door's `FRAMES`.
"""
import pyglet
from pyglet.window import key

from physics.engine import one_cell
from physics.engine.axiom_3_structure.definitions.store import slot_of
from physics.engine.axiom_3_structure.logic.boxes_of import boxes_of
from physics.engine.axiom_3_structure.logic.give_back import give_back
from physics.engine.axiom_3_structure.logic.take_own import take_own
from physics.view.definitions.camera import DISTANCE, PITCH, YAW, zoomed
from physics.view.logic.streaks import streaks
from physics.view.logic.swirl_painter import SwirlPainter
from physics.view.logic.title_line import title_line
from physics.view.logic.tracers import COUNT, MARK_EVERY, MARKS, lay_tracers, step_carrying

TICKS = 30


class CellWindow(pyglet.window.Window):
    def __init__(self, across: int, speed: int, device: str):
        super().__init__(1000, 800, resizable=True, vsync=False, caption="one energy volume")
        self.fresh = lambda: one_cell(across, device=device)
        self.speed, self.running = speed, True
        self.painter = SwirlPainter(MARKS, COUNT)
        self.lay()
        self.yaw, self.pitch, self.distance = YAW, PITCH, DISTANCE
        pyglet.clock.schedule_interval(self.tick, 1 / TICKS)
        pyglet.clock.schedule_interval(self.retitle, 0.5)

    def lay(self) -> None:
        self.store = self.fresh()
        self.tracers = lay_tracers(self.store, 0)
        self.boxes = boxes_of(self.store, 0)  # asked once: dealing a calm cell's costs a frame
        self.painter.load(streaks(self.tracers))

    def tick(self, _elapsed: float) -> None:
        if not self.running:
            return
        for _ in range(self.speed):
            step_carrying(self.store, self.tracers, 0, self.store.step % MARK_EVERY == 0,
                          self.boxes)
        self.painter.load(streaks(self.tracers))

    def retitle(self, _elapsed: float) -> None:
        self.set_caption(title_line(self.store) + ("" if self.running else "  (stopped: Space)"))

    def on_draw(self) -> None:
        self.painter.draw(self.store.across, self.yaw, self.pitch, self.distance,
                          *self.get_framebuffer_size())

    def on_key_press(self, symbol: int, _modifiers: int) -> None:
        if symbol == key.SPACE:
            self.running = not self.running
        elif symbol == key.R:
            self.lay()
        elif symbol == key.G and slot_of(self.store, 0) < 0:
            take_own(self.store, 0)
            self.boxes = boxes_of(self.store, 0)
        elif symbol == key.G:
            give_back(self.store, 0)
            self.boxes = boxes_of(self.store, 0)
        elif symbol == key.ESCAPE:
            self.close()

    def on_mouse_drag(self, _x, _y, moved_x: int, moved_y: int, _buttons, _modifiers) -> None:
        self.yaw -= 0.01 * moved_x
        self.pitch = min(1.55, max(-1.55, self.pitch - 0.01 * moved_y))

    def on_mouse_scroll(self, _x, _y, _scroll_x, scroll_y: float) -> None:
        self.distance = zoomed(self.distance, scroll_y)
