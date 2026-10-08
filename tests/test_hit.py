from model.hit import Hit


def test_hit_defaults_to_auto_source():
    hit = Hit(x=1.0, y=2.0)
    assert hit.source == "auto"


def test_hit_can_be_marked_as_manual():
    hit = Hit(x=1.0, y=2.0, source="manual")
    assert hit.source == "manual"
