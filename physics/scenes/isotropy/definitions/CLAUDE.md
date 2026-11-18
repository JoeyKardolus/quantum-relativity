# physics/scenes/isotropy/definitions/

- Each pair in `PAIRS` is two places at exactly the same straight-line distance in
  volumes (3, 5, 6 and 7), the first on an axis and the second off it. A new pair must
  keep that; the test checks the distances.
- `DIAGONALS` are the same count of cells along an axis, a face diagonal and a body
  diagonal. Their straight-line distances differ, so no law is read beside them and
  they carry no pass mark.
- `mirror_mean` averages over all 48 mirrors and swaps (fewer distinct for a place with
  equal or zero entries), so a lattice's own handedness cannot show as a direction.
