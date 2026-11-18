# physics/scenes/run/logic/

`sampled_run` is the one loop the natural-start scenes share: measure, step, measure.

- It measures before the first step and after the last, so a result always holds the start.
- The universe is stepped in place; a caller that needs the start again builds a new one.
- An engine error is not caught: a scene that cannot run says so by failing.
