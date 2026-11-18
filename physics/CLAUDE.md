# physics/

A fresh start (2026-11-18), built in layers, each one something to look at. Rules and layers: the seven axioms in `docs/bible.md#physicsquantum-relativity`. Layer 1 is built: one energy volume with a swirl. Layer 2 is built: energy crosses from a volume through a joint into the next, still moving, and the two volumes squeeze the joint, which makes energy, shrinks and pushes back. `engine/` holds one directory per axiom, `view/` the window.

```
PYTHONPATH=. .venv/bin/python -m physics.view   # Space stops, R, G, Esc; --boxes 24
PYTHONPATH=. .venv/bin/python -m physics.view --scene crossing   # volume, joint, volume; 12 boxes
PYTHONPATH=. .venv/bin/pytest -q physics                     # seconds, on the CPU
```

- **Small and plain:** no file over 80 lines.
- **Layout:** every built part holds `definitions/` and `logic/`, each with `tests/`. No loose files.
- **Light on compute:** one process on the card, runs of seconds.
- **Honest:** never tune a physics number until a picture or a test looks right.
- **Boxes are not space:** anything between cells is stated a cell and a turn of its swirl, never a box or a step.
- Tests here have no register rows.
