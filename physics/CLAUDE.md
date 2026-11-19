# physics/

Built in layers. Rules: seven axioms in `docs/bible.md#physicsquantum-relativity`. Built: a cell as three ring whirls, a joint the other way (1), the leak that always happens and the crossing that carries the fluid's own flow (2), growth by the leak (3c), the pull, pair by pair and late (4), collapse past one limit (5). Nothing lays a grid. `engine/`: a directory per axiom; `view/`: the window.

```
PYTHONPATH=. .venv/bin/python -m physics.view                   # one cell: Up Down energy, F sign, 1 2 3 0 rings, X Y Z, R
PYTHONPATH=. .venv/bin/python -m physics.view --scene crossing  # volume, joint, volume; M: motion
PYTHONPATH=. .venv/bin/python -m physics.view --scene grow      # the leak tears off layers: X Y Z, C
PYTHONPATH=. .venv/bin/python -m physics.view --scene hole      # a packed middle collapses
PYTHONPATH=. .venv/bin/python -m physics.view --scene lumps
PYTHONPATH=. .venv/bin/python -m physics.view --scene gather   # a line: the fuller volume gains
PYTHONPATH=. .venv/bin/python -m physics.view --scene coast    # a line: one set going (stops: not right)
PYTHONPATH=. .venv/bin/python -m physics.view --scene fall     # a line: two fuller ones draw together
PYTHONPATH=. .venv/bin/python -m physics.view --scene dense    # a sheet: the middle gathers; X: face on
PYTHONPATH=. .venv/bin/pytest -q physics                        # 45 seconds, on the CPU
```

- No file over 80 lines; `definitions/` and `logic/`, each with `tests/`; no loose files.
- One process on the card, runs of seconds.
- Never tune a physics number to make a picture or test look right.
- Boxes are not space: state things a cell and a swirl's turn.
- Tests here have no register rows.
