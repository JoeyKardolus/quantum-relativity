# Axiom 4, definitions

`passage.py`: `stride_of` and `row_of` (a joint's rows along its axis), `row_minus` (the joint's minus quanta between two facing boxes), `rub_share`, `passing_turn`, `less_step` (friction on the phase step), `layer_part`.

- No number is set here. The joint's quanta decide how freely energy passes.
- `layer_part` (6 over the boxes across) cuts one step's turn and friction: the facing boxes are a smaller part of the cell the more boxes it has. Without it the course depends on the boxes.
- A joint lies along any of the three directions. A swirl is mirrored by its other sense; along the direction after its own the mirror also needs half a turn of phase, which `take_own` lays by place.
