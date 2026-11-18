"""Engine start: which universe the bench times, fixed before any run.

`SEED` is the seed of the start and `SPARE` the share of cell ids kept free for tears,
both handed to the engine's `new_universe`; the settings are the engine's defaults,
every rule on. `DEVICE` is the one GPU. Nothing here is tuned to a result.
"""
SEED = 1
SPARE = 0.25
DEVICE = "cuda:0"
