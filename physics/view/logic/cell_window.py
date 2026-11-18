"""Cell window: the window on a scene and its keys.

Space stops and starts, R lays a fresh start, drag turns the view, the wheel zooms, Esc
quits. A grid: L steps the flat cut to the next layer, C shows all layers at once and the
cut again. One cell: G sends the cell to calm if it may go or takes it own again. The fluid
steps `TICKS` times a second; the window is drawn at most as often as the door's `FRAMES`.
"""
import pyglet
from pyglet.window import key

from physics.engine.axiom_3_structure.definitions.store import slot_of
from physics.engine.axiom_3_structure.logic.boxes_of import boxes_of
from physics.engine.axiom_3_structure.logic.give_back import give_back
from physics.engine.axiom_3_structure.logic.take_own import take_own
from physics.view.definitions.camera import PITCH, YAW, zoomed
from physics.view.logic.scene import SCENES, drawn, eye_distance, layers, title, trace
from physics.view.logic.swirl_painter import SwirlPainter
from physics.view.logic.tracers import MARK_EVERY, MARKS, step_carrying

TICKS = 30


class CellWindow(pyglet.window.Window):
    def __init__(self, across: int, speed: int, device: str, scene: str = "one"):
        super().__init__(1000, 800, resizable=True, vsync=False, caption="energy")
        self.fresh = lambda: SCENES[scene](across, device=device)
        self.speed, self.running, self.store = speed, True, self.fresh()
        self.layer = 0 if layers(self.store) else None  # a grid opens on a flat cut
        self.tracers, self.boxes = trace(self.store, self.layer), boxes_of(self.store, 0)
        self.painter = SwirlPainter(MARKS, self.tracers.tail.shape[1])
        self.painter.thick = max(layers(self.store), 1.0)
        self.yaw, self.pitch = (0.0, 0.0) if layers(self.store) else (YAW, PITCH)
        self.distance = eye_distance(self.store)
        pyglet.clock.schedule_interval(self.tick, 1 / TICKS)
        pyglet.clock.schedule_interval(self.retitle, 0.5)

    def tick(self, _elapsed: float) -> None:
        for _ in range(self.speed if self.running else 0):
            step_carrying(self.store, self.tracers, 0, self.store.step % MARK_EVERY == 0,
                          self.boxes, max(self.store.count, 1))
        self.painter.load(drawn(self.store, self.tracers, self.layer))

    def retitle(self, _elapsed: float) -> None:
        stopped = "" if self.running else "  (stopped: Space)"
        self.set_caption(title(self.store) + stopped)

    def on_draw(self) -> None:
        self.painter.draw(self.store.across, self.yaw, self.pitch, self.distance,
                          *self.get_framebuffer_size())

    def on_key_press(self, symbol: int, _modifiers: int) -> None:
        cut, alone = layers(self.store), self.store.grid == (1, 1, 1)
        if symbol == key.SPACE:
            self.running = not self.running
        elif symbol == key.ESCAPE:
            return self.close()
        elif symbol == key.G and alone and slot_of(self.store, 0) < 0:
            take_own(self.store, 0)
        elif symbol == key.G and alone:
            give_back(self.store, 0)
        elif symbol == key.L and cut:
            self.layer = ((self.layer or 0) + 1) % cut
        elif symbol == key.C and cut:
            self.layer = 0 if self.layer is None else None
            self.yaw, self.pitch = (YAW, PITCH) if self.layer is None else (0.0, 0.0)
        if symbol == key.R:
            self.store = self.fresh()
        if symbol in (key.R, key.L, key.C):
            self.tracers = trace(self.store, self.layer)
        self.boxes = boxes_of(self.store, 0)  # asked once a key: a calm cell's cost a frame

    def on_mouse_drag(self, _x, _y, moved_x: int, moved_y: int, _buttons, _modifiers) -> None:
        self.yaw -= 0.01 * moved_x
        self.pitch = min(1.55, max(-1.55, self.pitch - 0.01 * moved_y))

    def on_mouse_scroll(self, _x, _y, _scroll_x, scroll_y: float) -> None:
        self.distance = zoomed(self.distance, scroll_y)
