# physics/view/window/definitions/

What the window and its thread are made of.

- `snapshot.py`: `Snapshot`, a frozen record: one moment of the source, ready to draw.
- `key_actions.py`: `KEY_ACTIONS`, key name to `(action, argument)`.
- `window_marks.py`: the window's first size, turn and zoom speeds, refresh times.

## Must not get wrong

- Nothing here imports `logic/`, OpenGL or the engine. Key names are the names of
  pyglet's key constants as text, so this loads without pyglet; `tests/` checks every
  name is a real key and no two are the same key.
- A new key is one line in `KEY_ACTIONS`. A new action also needs its branch in
  `CellWindow.on_key_press`.
- A snapshot always carries the slice; `blocks` is `None` unless the whole grid shows.
