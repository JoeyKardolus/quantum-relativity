"""The word for what friction was once booked as is nowhere in the code: there is no such thing."""


def test_the_word_for_what_friction_was_once_booked_as_is_gone_from_the_code():
    from pathlib import Path
    code = Path(__file__).parents[2]  # physics/
    named = [str(file) for file in code.rglob("*.py")
             if "he" + "at" in file.read_text().lower() and "site-packages" not in str(file)]
    assert named == []
