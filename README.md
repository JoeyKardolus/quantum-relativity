# quantum-relativity

A hobby simulation of one person's theory of how space, matter and forces could come from a single
kind of stuff: energy. The goal is not to replace quantum physics or relativity, but to see whether
these rules, run on a computer, produce things we know from nature: gravity, expansion, black holes,
orbits.

## The idea in short

- **Space is a foam of two kinds of cell**: energy volumes that **pull** (A) and **joints** that **push**
  (B). Every A is wrapped in joints on all sides, edges and corners included; joints touch each other
  and form one web, like mortar between bricks.
- **Energy sets size**: the more energy packed into a region, the smaller and denser it is.
- **A pulls** on other A's through the joints between them, fading with distance, like gravity.
  **A joint pushes back**, harder the more it is squeezed, until it gives way past a limit and the
  region collapses, like a star becoming a black hole.
- **Energy is a quantum fluid**, one complex number per Planck box: frictionless inside a volume, whirls
  turning only in whole steps, friction only in passing a joint.
  Squeezed joints let less through and past a seal point nothing, so packed lumps seal themselves.
- **Time is local**: where energy is more packed, everything happens more slowly.

The full rules, open problems and milestones are in the bible chapter
[`physics/quantum-relativity`](docs/bible.md#physicsquantum-relativity); the words used are defined in
[`CONTEXT.md`](CONTEXT.md).

## Status

Early. The engine was rebuilt axiom by axiom (2026-11-18): it books every quantum exactly and draws live
on the GPU, but it does not yet behave like a universe. With untuned settings more than half of the
joints collapse in the first step and a single volume is left; cells do not stay swirls and the pull
outruns light. Next: keep each axiom's clauses, then fit the few settings against a scorecard of scenes (expansion
history, structure), and test, unfitted, on others (orbits, bound lumps, black hole size, the Hubble
tension). Progress is tracked on [issue #1](https://github.com/JoeyKardolus/quantum-relativity/issues/1).
Misses are reported as misses; nothing is tuned to look good.

## Run it

```bash
uv sync --all-packages --all-extras --all-groups
PYTHONPATH=. .venv/bin/python -m physics.view      # opens a window; Space starts, R resets, Esc quits
PYTHONPATH=. .venv/bin/pytest -q physics           # the tests
```

Needs Linux or WSL2, Python 3.12+, [`uv`](https://docs.astral.sh/uv/) and an NVIDIA GPU (the engine
runs on [NVIDIA Warp](https://github.com/NVIDIA/warp)). On WSL2 the viewer switches the window's drawing
to the GPU by itself.

| Path | What it is |
|---|---|
| `physics/engine/` | the engine, one directory per axiom (`axiom_1_energy` to `axiom_7_quantum`), each with its definitions, its logic and its tests; the step only calls them in order |
| `physics/bench/` | the bench: the engine's speed, and the largest universe that runs a step a second |
| `physics/view/` | the window that shows the whole grid in 3D, every cell at its own size, or saves pictures |
| `physics/scenes/` | the scorecard: each scene runs the engine and returns measured numbers |
| `physics/fit/` | the fit: searches the settings on expansion shape and structure, by random search (nothing searched on the rebuilt engine yet) |
| `physics/measure/` | the measures the scenes share: the dimension of the volume graph, and a cell's whole turns |
| `docs/bible.md` | how everything works; the theory is the chapter `physics/quantum-relativity` |

## Built with

Written with Claude Code on [babon-harness](https://github.com/Babon-Innovations-b-v/babon-harness), a
Claude Code setup whose skills, hooks and code index live in `.claude/`. The other folders (`apps/`,
`infra/`, `business/` and so on) come from that harness template and are not part of the simulation.

## Licence

MIT. See [`LICENSE`](LICENSE).
