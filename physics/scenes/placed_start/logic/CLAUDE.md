# physics/scenes/placed_start/logic/

`packed_start` builds the universe of a placed scene.

- The grid is made even first: `unevenness` and `joint_unevenness` are set to 0 whatever
  the start says; a placed scene never runs on an uneven start. The box-to-box jitter
  (`inner_unevenness`) stays, so two starts with the same seed are the same but for the
  packed volumes: that is what a twin run subtracts.
- After writing a block it recounts, resizes and repacks every cell with the engine's
  own functions, in the engine's order. Never write `cell_quanta`, `length` or `packing`
  by hand.
- It does not call `start_pull`: see the directory above.
