"""The PNG chunk and the picture's fixed numbers, against bytes every PNG file has."""
from physics.view.pictures.definitions.picture_marks import HEIGHT, MARGIN, PNG_SIGNATURE, WIDTH
from physics.view.pictures.definitions.png_chunk import png_chunk


def test_the_end_chunk_is_the_twelve_bytes_every_png_ends_with():
    assert png_chunk(b"IEND", b"") == bytes.fromhex("0000000049454e44ae426082")


def test_a_chunk_carries_its_bodys_length_first_and_the_body_unchanged():
    chunk = png_chunk(b"tEXt", b"hello")
    assert chunk[:4] == bytes([0, 0, 0, 5]) and chunk[4:8] == b"tEXt" and chunk[8:13] == b"hello" and len(chunk) == 17


def test_the_signature_and_the_picture_size():
    assert PNG_SIGNATURE == bytes([137, 80, 78, 71, 13, 10, 26, 10])
    assert (WIDTH, HEIGHT) == (1100, 825) and 0 < MARGIN < 0.2
