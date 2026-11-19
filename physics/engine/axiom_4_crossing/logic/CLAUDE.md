# Axiom 4, logic

`cross.py`: one launch an axis over the own slots; only a joint with both volumes own acts; the pull bends it. `leak.py`: one launch over every box of the own volumes. `meshing.py`: do two cells' facing caps agree.

- Each axis has its own ring and caps, and its own launch.
- The cell totals are added with `atomic_add`: whole numbers, so order does not matter.
- The leak reads its shares off a copy of the totals made before the launch, so a run repeats.
- Leaked quanta are a free place's total, with no boxes: nothing acts on them until they tear off.
- A store with `leaks` off (the bench) does not leak.
