# physics/

Connection theory simulation. Rules: `docs/bible.md#physicsconnection-theory`; words: `CONTEXT.md`
(joint, connection, drift, reference spacetime; never node, edge or vertex).

## Must not get wrong

- **Joints store no coordinates.** `Network` holds joint count, connection pairs, energy, length,
  charge, an optional causal order, the merge map and `held` (energy an isolated joint keeps
  after a whole cluster collapses; count it with `total_energy()`), nothing else. A builder may scatter points to decide which
  joints connect, then must drop them; known answers go back separately, never on the Network.
- **Every measurement reads only the network**: length-weighted paths, neighbour counts, random
  walks, the causal order. If a measure needs a coordinate, it is measuring the wrong thing.
- **Energy and length are one quantity**: `energy * length = LENGTH_SCALE`
  (`network/energy_length.py`), restored each step by the energy-length rule. Rules that push or
  pull change length; the rule moves energy to match.
- **Honesty.** Never tune a setting, seed or fit window until a result passes. Fit windows in
  `ball_dimension` and `spectral_dimension` are fixed defaults. Report every number beside its
  known value with seed and settings; a miss is reported as a miss.
- **Speed.** Vectorised numpy and scipy.sparse; an experiment runs in under ~3 minutes. Only
  `rules/rewire.py` loops in Python, one attempt per iteration.
- **Layout.** One public function (or type) per file, helpers `_prefixed`, each subpackage
  `__init__.py` holds the module map and `__all__`.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics          # all tests, seconds
PYTHONPATH=. .venv/bin/python -m physics          # every milestone, results page, minutes
PYTHONPATH=. .venv/bin/python -m physics.viewer   # gravity scenes as a 3D page (layout drawn, never stored)
```

`physics/` is exempt from the test register (owner, 2026-11-14; `UNREGISTERED` in
`tools/test_register/_modules.py`): write plain pytest files under `<module>/tests/`, no rows.
