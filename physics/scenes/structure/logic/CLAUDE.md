# physics/scenes/structure/logic/

`structure` samples the volumes' field and reads its spectrum, contrast and lumps.

- The grid needs 8 places a side or more: the spectrum wants 4 volumes a side.
- `power_growth` is 0 at a wavenumber whose start power is 0; it never divides by 0.
- If every start volume is gone the spectrum raises (a field holding no energy): the
  scene fails loudly instead of returning a spectrum of nothing.
