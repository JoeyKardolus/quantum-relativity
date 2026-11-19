# physics/

A fresh start (2026-11-18), built in layers, each something to look at. Rules: the seven axioms in `docs/bible.md#physicsquantum-relativity`. Built: one ball whirl (1), crossing and the squeeze (2), the universe growing from one volume by its leak (3c). Nothing lays a grid. `engine/`: a directory per axiom; `view/`: the window.

```
PYTHONPATH=. .venv/bin/python -m physics.view                   # one ball: X, Y, Z, Space, R, G
PYTHONPATH=. .venv/bin/python -m physics.view --scene crossing  # volume, joint, volume
PYTHONPATH=. .venv/bin/python -m physics.view --scene grow      # the leak tears off layers: X Y Z, C
PYTHONPATH=. .venv/bin/pytest -q physics                        # 11 seconds, on the CPU
```

- No file over 80 lines; `definitions/` and `logic/`, each with `tests/`; no loose files.
- One process on the card, runs of seconds.
- Never tune a physics number to make a picture or test look right.
- Boxes are not space: state things a cell and a swirl's turn.
- Tests here have no register rows.
