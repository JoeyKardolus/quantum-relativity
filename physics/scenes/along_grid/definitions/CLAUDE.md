# physics/scenes/along_grid/definitions/

Formulas only.

- `box_depth` and `box_field` are two forms of one law: `box_depth` in lengths along a
  row (a point and 4 copies each way), `box_field` in volumes on the whole cube
  (Fourier). Both take the even level out; a test holds them equal along an axis
  within 5%. Until 2026-11-18 `box_depth` and `newton_tilt` summed the copies bare,
  which read 30% too deep a quarter of the box out and did not vanish at half the box.
- `log_slope` returns NaN rather than a slope when a value is 0 or below or there are
  fewer than two points. A scene passes the NaN on; it never drops points to get a slope.
- `row_cells` uses lattice indices, which exist only at the start.
