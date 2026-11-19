"""Cell window: the window on a scene and its keys.

Space stops and starts, R lays a fresh start, drag turns, the wheel zooms, Esc quits; X, Y
and Z look along a direction; 1, 2, 3 show one ring's fluid alone, 0 all three. A growing
cluster: C, whole or flat cut. One cell: Up and Down double and halve its energy, F flips
its sign, G sends it to calm or takes it own. `TICKS` steps a second.
"""
import pyglet
from pyglet.window import key

from physics.engine.axiom_3_structure.definitions.store import slot_of
from physics.engine.axiom_3_structure.logic.boxes_of import boxes_of
from physics.engine.axiom_3_structure.logic.give_back import give_back
from physics.engine.axiom_3_structure.logic.take_own import take_own
from physics.view.definitions.camera import ALONG, PITCH, YAW, eye_distance, zoomed
from physics.view.logic.cell_keys import flipped, scaled
from physics.view.logic.scene import SCENES, Seen, advance, cut, drawn, title, trace
from physics.view.logic.swirl_painter import SwirlPainter
from physics.view.logic.tracers import MARKS

TICKS = 30


class CellWindow(pyglet.window.Window):
    def __init__(self, across: int, speed: int, device: str, scene: str = "one"):
        super().__init__(1000, 800, resizable=True, vsync=False, caption="energy")
        self.fresh = lambda: SCENES[scene](across, device=device)
        self.speed, self.running = speed, True
        self.lay()
        self.painter = SwirlPainter(MARKS, self.seen.tracers.tail.shape[1])
        self.alone, (self.yaw, self.pitch) = self.store.side == 0, (YAW, PITCH) if self.store.side == 0 else ALONG[2]
        self.distance = eye_distance(self.store)
        pyglet.clock.schedule_interval(self.tick, 1 / TICKS)
        pyglet.clock.schedule_interval(self.retitle, 0.5)

    def lay(self) -> None:
        """A fresh start of the scene, with fresh tracers."""
        self.store = self.fresh()
        self.seen, self.boxes = Seen(trace(self.store)), boxes_of(self.store, 0)

    def tick(self, _elapsed: float) -> None:
        try:
            for _ in range(self.speed if self.running else 0):
                advance(self.store, self.seen, self.boxes)
        except RuntimeError as full:  # the store's room is full
            self.running = print(full) or False
        self.painter.load(drawn(self.store, self.seen))

    def retitle(self, _elapsed: float) -> None:
        stopped = "" if self.running else "  (stopped: Space)"
        self.set_caption(title(self.store, self.seen) + stopped)

    def on_draw(self) -> None:
        self.painter.draw(self.store.across, self.yaw, self.pitch, self.distance,
                          *self.get_framebuffer_size())

    def on_key_press(self, symbol: int, _modifiers: int) -> None:
        self.running ^= symbol == key.SPACE
        if symbol in (key.X, key.Y, key.Z):
            self.yaw, self.pitch = ALONG[symbol - key.X]
        elif symbol == key.ESCAPE:
            return self.close()
        elif symbol == key.G and self.alone:
            (take_own if slot_of(self.store, 0) < 0 else give_back)(self.store, 0)
        elif symbol in (key.UP, key.DOWN, key.F) and self.alone:
            flipped(self.store, 0) if symbol == key.F else scaled(self.store, 0, symbol == key.UP)
        elif symbol in (key._0, key._1, key._2, key._3):
            self.seen.ring = symbol - key._1
        elif symbol == key.C:
            self.seen.whole = cut(self.store, self.seen)
        elif symbol == key.R:
            self.lay()
        self.boxes = boxes_of(self.store, 0)  # asked once a key: a calm cell's cost a frame

    def on_mouse_drag(self, _x, _y, moved_x: int, moved_y: int, _buttons, _modifiers) -> None:
        self.yaw -= 0.01 * moved_x
        self.pitch = min(1.55, max(-1.55, self.pitch - 0.01 * moved_y))

    def on_mouse_scroll(self, _x, _y, _scroll_x, scroll_y: float) -> None:
        self.distance = zoomed(self.distance, scroll_y)
