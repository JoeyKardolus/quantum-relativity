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
- **Energy is a quantum fluid**, whole quanta and a phase per Planck box: frictionless inside a volume, whirls
  turning only in whole steps, friction only in passing a joint.
  Squeezed joints let less through and past a seal point nothing, so packed lumps seal themselves.
- **Time is local**: where energy is more packed, everything happens more slowly.

The full rules, open problems and milestones are in the bible chapter
[`physics/quantum-relativity`](docs/bible.md#physicsquantum-relativity); the words used are defined in
[`CONTEXT.md`](CONTEXT.md).

## Status

Early, and started afresh on 2026-11-18: the engine before had grown too big and basic gravity still did
not work. The new one is small, built for a billion cells from the start, and grows in layers, each one
something to look at. Layer 1 is built: one energy volume whose fluid turns as one whole swirl with an
open core, every quantum counted exactly. Next: layer 2, many cells and the first gravity test; then
layer 3, a billion cells on one graphics card. Progress is tracked on
[issue #1](https://github.com/JoeyKardolus/quantum-relativity/issues/1).
Misses are reported as misses; nothing is tuned to look good.

## Run it

```bash
uv sync --all-packages --all-extras --all-groups
PYTHONPATH=. .venv/bin/python -m physics.view   # a window; Space starts, R resets, G calm/own, Esc quits
PYTHONPATH=. .venv/bin/python -m physics.view --scene crossing   # energy crossing from a volume through a joint into the next
PYTHONPATH=. .venv/bin/pytest -q physics                     # the tests, 5 seconds
```

Needs Linux or WSL2, Python 3.12+, [`uv`](https://docs.astral.sh/uv/) . The engine runs on
[NVIDIA Warp](https://github.com/NVIDIA/warp): on an NVIDIA GPU when there is one, else on the CPU. On WSL2 the viewer switches the window's drawing
to the GPU by itself.

| Path | What it is |
|---|---|
| `physics/engine/` | the engine, one directory per axiom (`axiom_1_energy` to `axiom_7_quantum`), each with its definitions, its logic and its tests; layers 1 and 2 fill the first five |
| `physics/view/` | the window on one red energy volume with its swirl, or on a volume, a blue joint and a volume with energy crossing; it can also save pictures |
| `docs/bible.md` | how everything works; the theory is the chapter `physics/quantum-relativity` |

## Built with

Written with Claude Code on [babon-harness](https://github.com/Babon-Innovations-b-v/babon-harness), a
Claude Code setup whose skills, hooks and code index live in `.claude/`. The other folders (`apps/`,
`infra/`, `business/` and so on) come from that harness template and are not part of the simulation.

## Licence

MIT. See [`LICENSE`](LICENSE).
