# View, definitions

`camera.py`: the eye, the wheel's range, `ALONG`. `carried.py`: a tracer's place along two boxes after fluid moved. `shape.py`: where a place in a cell's block lies in the picture. `shaders.py`: the programs that draw a streak and a dot.

- A volume's block is drawn as a ball as wide as its size; `even_share` keeps the red round.
- A joint's block is two fans, each from the rim of the volume it faces to the square midway: its room as the engine's distance law counts it.
- Blue is minus fluid: its marks count below 0.
- Many marks go to solid red, never white.
- `THICK`, `DOT` and `STILL` only draw; do not move them.
