# physics/bench/throughput/

How many steps a second, and cell-steps a second, a universe runs. File map in
`__init__.py`.

- `definitions/` says what is timed and what a timing is.
- `logic/` does the timing and builds the engine's universe.
- `tests/` checks the whole thing on the real engine: the steps asked for are the steps
  the universe took. They skip without CUDA.

## Must not get wrong

- A rate without its size, step count and seed means nothing. Keep them together.
- The first step compiles kernels and sets things up; it is warm-up and never timed.
- A universe changes while it runs (tears add cells). A timing of the first 20 steps
  is not the speed of step 10,000. Say which steps were timed.
