# physics/view/pictures/definitions/

What a saved picture is.

- `picture_marks.py`: `WIDTH`, `HEIGHT`, `MARGIN`, `PNG_SIGNATURE`.
- `png_chunk.py`: one PNG chunk: length, name, body, CRC-32.

## Must not get wrong

- Nothing here imports `logic/`, OpenGL or the engine.
- `tests/` checks the chunk against the twelve bytes every PNG file ends with.
