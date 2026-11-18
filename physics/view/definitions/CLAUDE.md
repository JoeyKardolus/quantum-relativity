# View, definitions

`camera.py`: the eye, the wheel's range, `ALONG`. `carried.py`: a tracer's place along two boxes after fluid moved. `shape.py`: where a place in a cell's block lies in the picture. `shaders.py`: the programs that draw a streak and a dot.

- A volume's block is drawn as a ball as wide as its size; `even_share` makes a mark in its pressed corners count less, so the red is round.
- A joint's block is two fans, each from the rim of the volume it faces to the square midway: together the joints fill all the room.
- Where the grid wraps, a joint's second half is drawn before the first volume.
- Blue is minus fluid: its marks count below 0.
- Many marks go to solid red, never white.
- `THICK`, `DOT` and `STILL` only draw; do not move them.
