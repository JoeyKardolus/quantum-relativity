# physics/

Built in layers. Rules: seven axioms in `docs/bible.md#physicsquantum-relativity`. Built: a cell as three ring whirls, a joint the other way (1), the leak that always happens and the crossing that carries the fluid's own flow (2), growth by the leak (3c), the pull, pair by pair and late (4), collapse past one limit (5). A step runs wholly on the card; far volumes may be read as groups (`near`), and `far.py` runs a universe far forward and keeps it. Nothing lays a grid: the window's gravity scenes stage in a ball the engine grew (kept in tmp). `engine/`: a directory per axiom; `view/`: the window.

```
PYTHONPATH=. .venv/bin/python -m physics.view                   # one cell: Up Down energy, F sign, 1 2 3 0 rings, X Y Z, R
PYTHONPATH=. .venv/bin/python -m physics.view --scene crossing  # volume, joint, volume; M: motion
PYTHONPATH=. .venv/bin/python -m physics.view --scene grow      # the leak tears off layers: X Y Z, C
PYTHONPATH=. .venv/bin/python -m physics.view --scene grow --far DIR --minutes 40   # run far, no window: DIR/far.log, kept states
PYTHONPATH=. .venv/bin/python -m physics.view --scene grow --from DIR/last.npz      # watch a kept state and step on
PYTHONPATH=. .venv/bin/python -m physics.view --scene hole      # a packed middle collapses
PYTHONPATH=. .venv/bin/python -m physics.view --scene fall     # the grown ball, two regions made fuller
PYTHONPATH=. .venv/bin/python -m physics.view --scene dense    # the grown ball, one region made fuller
PYTHONPATH=. .venv/bin/python -m physics.view --scene coast    # the same region set going
PYTHONPATH=. .venv/bin/pytest -q physics                        # two minutes, on the CPU; a minute more when the ball is grown
```

- No file over 80 lines; `definitions/` and `logic/`, each with `tests/`; no loose files.
- One process on the card, runs of seconds; the far run is the one long run, never beside another.
- Never tune a physics number to make a picture or test look right.
- Boxes are not space: state things a cell and a swirl's turn.
- Tests here have no register rows.
