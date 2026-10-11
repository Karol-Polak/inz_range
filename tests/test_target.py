from model.target import Target


def test_target_aim_offset_defaults_to_none():
    target = Target(center_x=100.0, center_y=100.0, radius=50.0)

    assert target.aim_offset_x is None
    assert target.aim_offset_y is None


def test_target_aim_offset_can_be_set():
    target = Target(center_x=100.0, center_y=100.0, radius=50.0, aim_offset_x=5.0, aim_offset_y=-3.0)

    assert target.aim_offset_x == 5.0
    assert target.aim_offset_y == -3.0
