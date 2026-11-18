# physics/scenes/orbit/definitions/

Formulas on host arrays; no universe.

- `blob_centre` works on the volumes' own lattice and wraps: offsets from the big
  volume run from minus half the lattice to plus half.
- `went_round` stops counting the angle at the first sample where under half the blob's
  energy is still at its own volume: a spread-out smear has a centre but is no orbit.
- `SIDEWAYS` is a start number chosen before any run, for the fit to search. Do not
  adjust it after seeing a run.
