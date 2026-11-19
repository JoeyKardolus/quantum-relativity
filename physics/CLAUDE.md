# physics/

Built in layers. Rules: seven axioms in `docs/bible.md#physicsquantum-relativity`. Built: one ball whirl (1), crossing and the squeeze (2), growth by the leak (3c), the pull from cell to cell (4), collapse (5). Nothing lays a grid. `engine/`: a directory per axiom; `view/`: the window.

```
PYTHONPATH=. .venv/bin/python -m physics.view                   # one ball: X, Y, Z, Space, R, G
PYTHONPATH=. .venv/bin/python -m physics.view --scene crossing  # volume, joint, volume
PYTHONPATH=. .venv/bin/python -m physics.view --scene grow      # the leak tears off layers: X Y Z, C
PYTHONPATH=. .venv/bin/python -m physics.view --scene lumps
PYTHONPATH=. .venv/bin/pytest -q physics                        # 30 seconds, on the CPU
```

- No file over 80 lines; `definitions/` and `logic/`, each with `tests/`; no loose files.
- One process on the card, runs of seconds.
- Never tune a physics number to make a picture or test look right.
- Boxes are not space: state things a cell and a swirl's turn.
- Tests here have no register rows.
