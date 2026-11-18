# physics/

A fresh start (2026-11-18), built in layers, each something to look at. Rules: the seven axioms in `docs/bible.md#physicsquantum-relativity`. Built: one swirl (1), crossing and the squeeze (2), a small grid that wraps round and stands calm (3a). `engine/`: a directory per axiom; `view/`: the window.

```
PYTHONPATH=. .venv/bin/python -m physics.view                   # one volume: Space, R, G, Esc
PYTHONPATH=. .venv/bin/python -m physics.view --scene crossing  # volume, joint, volume
PYTHONPATH=. .venv/bin/python -m physics.view --scene calm      # 4 x 4 x 4: L next layer, C in 3D
PYTHONPATH=. .venv/bin/pytest -q physics                        # 20 seconds, on the CPU
```

- No file over 80 lines; `definitions/` and `logic/`, each with `tests/`; no loose files.
- One process on the card, runs of seconds.
- Never tune a physics number to make a picture or test look right.
- Boxes are not space: state things a cell and a swirl's turn.
- Tests here have no register rows.
