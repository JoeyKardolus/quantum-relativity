"""Cell window: the window on a scene (one energy volume, or a line of cells) and its keys.

Space stops and starts, R lays a fresh start, G (one cell only) sends the cell to calm if
it may go or takes it own again, drag turns the view, the wheel zooms, Esc quits. The fluid steps
`TICKS` times a second; the window is drawn at most as often as the door's `FRAMES`.
"""
import pyglet
from pyglet.window import key

from physics.engine.axiom_3_structure.definitions.store import slot_of
from physics.engine.axiom_3_structure.logic.boxes_of import boxes_of
from physics.engine.axiom_3_structure.logic.give_back import give_back
from physics.engine.axiom_3_structure.logic.take_own import take_own
from physics.view.definitions.camera import PITCH, YAW, zoomed
from physics.view.logic.scene import drawn, eye_distance, lay_scene, title, tracer_count
from physics.view.logic.swirl_painter import SwirlPainter
from physics.view.logic.tracers import MARK_EVERY, MARKS, lay_tracers, step_carrying

TICKS = 30


class CellWindow(pyglet.window.Window):
    def __init__(self, across: int, speed: int, device: str, cells: int = 1):
        super().__init__(1000, 800, resizable=True, vsync=False, caption="energy")
        self.fresh = lambda: lay_scene(cells, across, device)
        self.speed, self.running, self.cells = speed, True, cells
        self.painter = SwirlPainter(MARKS, tracer_count(cells))
        self.lay()
        self.yaw, self.pitch, self.distance = YAW, PITCH, eye_distance(cells)
        pyglet.clock.schedule_interval(self.tick, 1 / TICKS)
        pyglet.clock.schedule_interval(self.retitle, 0.5)

    def lay(self) -> None:
        self.store = self.fresh()
        self.tracers = lay_tracers(self.store, 0, tracer_count(self.cells), cells=self.cells)
        self.boxes = boxes_of(self.store, 0)  # asked once: dealing a calm cell's costs a frame
        self.painter.load(drawn(self.store, self.tracers, self.cells))

    def tick(self, _elapsed: float) -> None:
        if not self.running:
            return
        for _ in range(self.speed):
            step_carrying(self.store, self.tracers, 0, self.store.step % MARK_EVERY == 0,
                          self.boxes, self.cells)
        self.painter.load(drawn(self.store, self.tracers, self.cells))

    def retitle(self, _elapsed: float) -> None:
        stopped = "" if self.running else "  (stopped: Space)"
        self.set_caption(title(self.store, self.cells) + stopped)

    def on_draw(self) -> None:
        self.painter.draw(self.store.across, self.yaw, self.pitch, self.distance,
                          *self.get_framebuffer_size())

    def on_key_press(self, symbol: int, _modifiers: int) -> None:
        if symbol == key.SPACE:
            self.running = not self.running
        elif symbol == key.R:
            self.lay()
        elif symbol == key.G and self.cells > 1:
            pass  # calm and own are one cell's keys
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
