# physics/

Built in layers. Rules: seven axioms in `docs/bible.md#physicsquantum-relativity`. Built: a cell as three ring whirls, one for each direction, turning in step with its energy, a joint the other way (1), crossing through the joint as through a gear and the squeeze (2), growth by the leak (3c), the pull from cell to cell (4), collapse (5). Nothing lays a grid. `engine/`: a directory per axiom; `view/`: the window.

```
PYTHONPATH=. .venv/bin/python -m physics.view                   # one cell: Up Down energy, F sign, 1 2 3 0 rings, X Y Z, R
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
