# quantum-relativity

A hobby simulation of one person's theory of how space, matter and forces could come from a single
kind of stuff: energy. The goal is not to replace quantum physics or relativity, but to see whether
these rules, run on a computer, produce things we know from nature: gravity, expansion, black holes,
orbits.

## The idea in short

- **Space is a foam of two kinds of energy volume**, alternating like a 3D checkerboard: volumes that
  **pull** (A) and **joints** that **push** (B). Every A touches only B's, and every B only A's.
- **Energy sets size**: the more energy packed into a region, the smaller and denser it is.
- **A pulls** on other A's through the joints between them, fading with distance, like gravity.
  **A joint pushes back**, harder the more it is squeezed, until it gives way past a limit and the
  region collapses, like a star becoming a black hole.
- **Energy behaves like a fluid**: frictionless inside a volume, with friction only in passing a joint.
  Squeezed joints let less through and past a seal point nothing, so packed lumps seal themselves.
- **Time is local**: where energy is more packed, everything happens more slowly.

The full rules, open problems and milestones are in the bible chapter
[`physics/quantum-relativity`](docs/bible.md#physicsquantum-relativity); the words used are defined in
[`CONTEXT.md`](CONTEXT.md).

## Status

Early. The engine holds total energy exactly and draws live on the GPU, but it does not yet behave like
a universe: with untuned settings everything falls into a few sealed cores and nothing expands. Next:
check every rule is built in, then fit the few settings against a scorecard of scenes (expansion
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
| `physics/fluid/` | the engine: energy as a fluid on a fine grid of boxes on the GPU |
| `physics/view/` | the GPU window that shows it |
| `docs/bible.md` | how everything works; the theory is the chapter `physics/quantum-relativity` |

## Built with

Written with Claude Code on [babon-harness](https://github.com/Babon-Innovations-b-v/babon-harness), a
Claude Code setup whose skills, hooks and code index live in `.claude/`. The other folders (`apps/`,
`infra/`, `business/` and so on) come from that harness template and are not part of the simulation.

## Licence

MIT. See [`LICENSE`](LICENSE).
