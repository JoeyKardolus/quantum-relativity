# physics/view/pictures/logic/

What happens when pictures are saved.

- `write_png.py`: an RGB image to a PNG file; returns the bytes written.
- `fill_triangle.py`: paints one triangle: every pixel whose centre lies in it.
- `slice_picture.py`: one slice as an image, straight down z, no perspective.
- `slice_of_the_middle.py`: the middle slice, in the shape `saved_frames` calls.
- `saved_frames.py`: steps a source and saves a picture every few steps.
- `whole_picture.py`: `WholePicture`: the whole grid drawn by the GPU into an
  off-screen image, exactly as the window draws it. Needs OpenGL.

## Must not get wrong

- A picture is uint8 (height, width, 3), row 0 at the top. `write_png` raises on any
  other shape.
- A slice cell is filled with its hue times its opacity over black, hard-edged.
- `saved_frames` hands every picture the same `low` and `high`.
- `tests/` reads a written PNG back with zlib and gets the same pixels.
