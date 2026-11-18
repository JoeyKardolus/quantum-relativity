# physics/view/pictures/logic/

What happens when pictures are saved.

- `write_png.py`: an RGB image to a PNG file; returns the bytes written.
- `fill_triangle.py`: paints one triangle: every pixel whose centre lies in it.
- `draw_lines.py`: blends thin lines into an image (the cells' outlines).
- `slice_picture.py`: one slice as an image, straight down z, no perspective.
- `slice_of_the_middle.py`: the middle slice, in the shape `saved_frames` calls.
- `saved_frames.py`: steps a source and saves a picture every few steps: the middle
  slice, or with `whole` the packed cells' blocks over it.
- `whole_picture.py`: `WholePicture`: the whole grid drawn by the GPU into an
  off-screen image, exactly as the window draws it. Needs OpenGL.

## Must not get wrong

- A picture is uint8 (height, width, 3), row 0 at the top. `write_png` raises on any
  other shape.
- A slice cell is filled with its hue times its opacity over black, hard-edged, and
  then outlined in its plain hue: the pattern shows where every fill is clear.
- `saved_frames` hands every picture the grid's own frame plus a margin.
- `tests/` reads a written PNG back with zlib and gets the same pixels.
