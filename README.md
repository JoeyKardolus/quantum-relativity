# quantum-relativity

A hobby simulation of one person's theory of how space, matter and forces could come from a single
kind of stuff: energy. The goal is not to replace quantum physics or relativity, but to see whether
these rules, run on a computer, produce things we know from nature: gravity, expansion, black holes,
orbits.

## The seven axioms

Everything in the simulation has to follow from these seven sentences, and from nothing else.

1. **Energy**: an always-moving quantum fluid, one whole signed number of quanta per Planck box at each
   moment of local time. The total is not fixed: a squeezed joint makes more minus energy out of itself,
   and when a joint collapses its minus energy flips to plus and joins the volumes beside it. Zero is empty space, which does nothing.
2. **Cells are swirls**: plus only pulls, minus only pushes, both oscillate in whole steps.
3. **Structure**: plus never touches plus; minus wraps every plus and forms one web. The cells are the
   grid, a perfect A B A B pattern. A cell has no geometry of its own, only relations to its neighbours
   and a size set by its energy alone: more energy, smaller, for energy volumes and joints alike. Lumps
   warp the grid by making their cells small. Planck boxes are not space: they sit inside each cell only
   to compute its fluid.
4. **Crossing**: energy diffuses between plus swirls only through minus; that is the only friction, and
   what friction holds back stays as energy in the volume it came from.
5. **Packing**: more packed is smaller, slower and pulls harder; pull spreads at a finite speed, weaker
   with distance; squeezed minus pushes back steeply.
6. **Limits**: a joint's energy rises only while it is squeezed, so a fresh joint is the fattest it will
   ever be and shrinks as it is squeezed; it collapses only past a limit, where the pull across it is so deep that
   nothing moving at the top speed could climb out (a black hole opening): its minus energy flips to plus
   and it merges with the pulling volumes beside it into one, so the collapse can spread; a volume's leaking outside tears off as a layer of new volumes.
7. **Quantum**: each box's fluid also carries a phase, so waves add and cancel. Moving energy can split
   over several routes at once. Every split gets a direct pull between its halves, private to them, and
   its own oscillation. When something reacts with a half, or the energy loses its momentum, it settles
   whole in one cell.

In pictures: an **energy volume** (plus, drawn red) is a round swirl of energy, like the inside of a living
cell; the **joints** (minus, drawn blue) are volumes squeezed between the energy volumes that pull across
them, thin between two and fat where several meet.

The full wording, the open problems and the plan are in the bible chapter
[`physics/quantum-relativity`](docs/bible.md#physicsquantum-relativity); the words used are defined in
[`CONTEXT.md`](CONTEXT.md).

## Status

Early, and started afresh on 2026-11-18: the engine before had grown too big and basic gravity still did
not work. The new one is small, built for a billion cells from the start, and grows in layers, each one
something to look at. Built so far: one energy volume whose fluid turns as one whole swirl with an
open core, every quantum counted exactly; energy crossing from volume to volume through a joint, and
the squeeze; a universe that grows from one energy volume, whose whirl leaks: what leaks tears off as a
layer of new volumes all round it, with joints between, and those shed the next layer; and one pull,
a number a cell that every energy volume feeds and every cell hands on to its neighbours a step at a
time, which squeezes the joints and bends crossing energy toward where it is deeper. Measured: the
grid no longer stays even. Energy gathers at the middle of the cluster (its fullest volume holds 3.4
calm volumes' energy at 16 turns against 0.3 with the pull off), the joints there are squeezed 200
times thinner than at the rim, and the red volumes there touch from 14 turns on; the cluster still
grows layer by layer, slower. Not yet: a region made fuller by hand spreads out within half a turn,
two such regions do not fall toward each other, and one set going does not keep going; so no
gravity between bodies yet. Next: joint collapse where the reds touch, then why lumps do not hold. Progress is tracked on
[issue #1](https://github.com/JoeyKardolus/quantum-relativity/issues/1).
Misses are reported as misses; nothing is tuned to look good.

## Run it

```bash
uv sync --all-packages --all-extras --all-groups
PYTHONPATH=. .venv/bin/python -m physics.view   # a window; Space starts, R resets, G calm/own, Esc quits
PYTHONPATH=. .venv/bin/python -m physics.view --scene crossing   # energy crossing from a volume through a joint into the next
PYTHONPATH=. .venv/bin/python -m physics.view --scene grow       # one volume leaks and sheds layer after layer of new volumes; X Y Z look along, C whole or flat cut
PYTHONPATH=. .venv/bin/python -m physics.view --scene lumps      # two regions made fuller by hand in a small grown patch
PYTHONPATH=. .venv/bin/pytest -q physics                     # the tests, 27 seconds
```

Needs Linux or WSL2, Python 3.12+, [`uv`](https://docs.astral.sh/uv/) . The engine runs on
[NVIDIA Warp](https://github.com/NVIDIA/warp): on an NVIDIA GPU when there is one, else on the CPU. On WSL2 the viewer switches the window's drawing
to the GPU by itself.

| Path | What it is |
|---|---|
| `physics/engine/` | the engine, one directory per axiom (`axiom_1_energy` to `axiom_7_quantum`), each with its definitions, its logic and its tests; the layers so far fill the first six |
| `physics/view/` | the window on one red energy volume with its swirl, on a volume, a blue joint and a volume with energy crossing, or on a ball of cells growing layer by layer from one volume's leak; it can also save pictures |
| `docs/bible.md` | how everything works; the theory is the chapter `physics/quantum-relativity` |

## Built with

Written with Claude Code on [babon-harness](https://github.com/Babon-Innovations-b-v/babon-harness), a
Claude Code setup whose skills, hooks and code index live in `.claude/`. The other folders (`apps/`,
`infra/`, `business/` and so on) come from that harness template and are not part of the simulation.

## Licence

MIT. See [`LICENSE`](LICENSE).
