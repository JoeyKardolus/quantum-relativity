# physics/scenes/quantum_split/logic/

- The scene steps `wait` times, copies the store's open splits, puts a detector
  (`react`) in the cell of every split's largest half, takes one more step and reads
  that step's settlings from the store.
- A detector in a cell settles every split with a half there, not only the one it was
  put there for. That is the engine's rule (something reacts with a half) and is fine:
  every open split gets at least one detector.
- On the default settings the universe's collapses settle every split in the step it
  opens, so there may be none open at the reading; the result then says 0 splits.
- `wait` is 20 by default, chosen before any run.
