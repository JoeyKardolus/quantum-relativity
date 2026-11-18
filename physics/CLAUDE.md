# physics/

A fresh start (2026-11-18), built in layers, each one something to look at. Rules and layers: the seven axioms in `docs/bible.md#physicsquantum-relativity`. Layer 1 is built: one energy volume with a swirl. `engine/` holds one directory per axiom, `view/` the window.

```
PYTHONPATH=. .venv/bin/python -m physics.view   # Space stops, R, G, Esc; --boxes 24
PYTHONPATH=. .venv/bin/pytest -q physics                     # seconds, on the CPU
```

- **Small and plain:** no file over 80 lines.
- **Layout:** every built part holds `definitions/` and `logic/`, each with `tests/`. No loose files.
- **Light on compute:** one process on the card, runs of seconds.
- **Honest:** never tune a physics number until a picture or a test looks right.
- Tests here have no register rows.
