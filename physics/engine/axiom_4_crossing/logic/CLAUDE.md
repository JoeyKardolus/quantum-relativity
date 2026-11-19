# Axiom 4, logic

`cross.py`: one launch an axis over the own slots; only a joint with both its volumes own does anything. `leak.py`: one launch over every box of the own volumes.

- A volume's corner box faces three joints, one an axis: that is why each axis has its own launch.
- The cell totals are added with `atomic_add`: whole numbers, so the order does not matter.
- The leak reads its shares off a copy of the totals made before the launch, so a run repeats.
- Leaked quanta are a free place's total, with no boxes: nothing acts on them until they tear off.
- A store whose `leaks` is off (the bench) does not leak.
