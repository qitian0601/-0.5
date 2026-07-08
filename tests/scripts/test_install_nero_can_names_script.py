from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install_nero_can_names.sh"


def test_install_nero_can_names_script_pins_serials_to_semantic_names():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'LEFT_SERIAL="003100414148570C20343133"' in text
    assert 'RIGHT_SERIAL="003400464148570A20343133"' in text
    assert 'LEFT_NAME="nero_left"' in text
    assert 'RIGHT_NAME="nero_right"' in text
    assert "Property=ID_SERIAL_SHORT=${LEFT_SERIAL}" in text
    assert "Property=ID_SERIAL_SHORT=${RIGHT_SERIAL}" in text
    assert "Name=${LEFT_NAME}" in text
    assert "Name=${RIGHT_NAME}" in text
