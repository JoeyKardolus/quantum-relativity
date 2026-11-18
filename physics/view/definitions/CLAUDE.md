# View, definitions

`camera.py`: the eye, the wheel's range. `carried.py`: a tracer's place along two boxes after fluid moved. `shape.py`: where a place in a cell's block lies in the picture. `shaders.py`: the programs that draw a streak.

- A volume's block is drawn as a ball as wide as its size. A joint's block is two fans, each from the rim of the volume it faces to the square midway: together the joints fill all the room.
- Where the grid wraps, a joint's second half is drawn before the first volume.
- Blue is minus fluid: its marks carry their way gone below 0.
- Many streaks go to solid red, never white.
- `FULL` and `THICK` only draw; do not move them. A grid's eye stands further off by its volumes along, and its streaks are that much thicker.
