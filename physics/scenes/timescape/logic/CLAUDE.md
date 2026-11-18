# physics/scenes/timescape/logic/

`timescape` runs a natural start and keeps the size and every start cell's clock.

- Clocks are kept for the start lattice's ids only: a volume a tear made has not lived
  all run and cannot be an observer.
- With fewer living volumes than `observers` asked, the dense and the empty side
  overlap; `observers` in the result says how many were used.
- If no start volume lives at the end the three clock numbers are NaN.
