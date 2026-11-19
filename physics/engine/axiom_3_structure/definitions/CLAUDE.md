# Axiom 3, definitions

`store.py`: the arrays, `slot_of`, `slot_at`. `size_law.py`. `places.py`: `stride_of`, `beside`, `joint_axis`, `joint_ends`, `steps_of`. `kinds.py`: boxes across a cell, quanta a box.

- A box belongs to its cell and has no place in space. A place is not space either: it says which cells are neighbours, never how far.
- Warp arrays cannot grow: a full own set, full tile rows or a full room stop with a plain message.
- `side` 0: the store keeps no places. Ask `joint_ends` and `beside` only where `side` is above 0.
- Plus at a place with no slot: leak gathering.
- A joint's volumes: `joint_ends`. A plus cell at a joint's place is a collapsed joint.
