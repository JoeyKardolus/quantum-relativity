# Axiom 6, logic

`make.py`: one launch over every box of the own joints, after `squeeze`.

- A joint's quanta never fall. Nothing here may take quanta away.
- What is made is booked to the quantum: the joint's boxes, its `cell_quanta` and `cell_made` change by the same whole number.
- The volumes are read, never written: they lose nothing to the making.
- The joint's total changes with `atomic_add` while the launch runs, so the kernel reads the squeeze from `own_squeeze`, never from the joint's total.
