from dataclasses import replace

from model.hit import Hit
from model.target import Target
from services.manual_correction import map_display_point_to_image, set_aim_point_at, toggle_hit_at

TARGET = Target(center_x=50.0, center_y=50.0, radius=40.0, type="circular")


def test_map_display_point_scales_up_when_preview_is_smaller_than_image():
    # Preview pixmap is half the size of the original image, no centering offset.
    point = map_display_point_to_image(
        display_point=(25.0, 10.0),
        label_size=(200, 100),
        pixmap_size=(200, 100),
        image_size=(400, 200),
    )

    assert point == (50.0, 20.0)


def test_map_display_point_accounts_for_centering_offset():
    # Label is wider than the scaled pixmap -> pixmap is horizontally centered.
    point = map_display_point_to_image(
        display_point=(60.0, 10.0),  # 10px into the pixmap horizontally (offset=50)
        label_size=(200, 100),
        pixmap_size=(100, 100),
        image_size=(100, 100),
    )

    assert point == (10.0, 10.0)


def test_map_display_point_returns_none_outside_pixmap_bounds():
    point = map_display_point_to_image(
        display_point=(5.0, 5.0),  # falls in the letterboxed margin, not on the pixmap
        label_size=(200, 100),
        pixmap_size=(100, 100),
        image_size=(100, 100),
    )

    assert point is None


def test_toggle_hit_at_adds_manual_hit_when_nothing_nearby():
    hits = []

    updated = toggle_hit_at(hits, TARGET, image_point=(60.0, 50.0))

    assert len(updated) == 1
    added = updated[0]
    assert added.source == "manual"
    assert added.x == 10.0  # 60 - target.center_x
    assert added.y == 0.0


def test_toggle_hit_at_removes_existing_hit_when_clicked_near_it():
    existing = Hit(x=10.0, y=0.0, distance_from_center=10.0, source="auto")
    hits = [existing]

    updated = toggle_hit_at(hits, TARGET, image_point=(61.0, 51.0))

    assert updated == []


def test_toggle_hit_at_does_not_remove_distant_hit():
    existing = Hit(x=10.0, y=0.0, distance_from_center=10.0, source="auto")
    hits = [existing]

    updated = toggle_hit_at(hits, TARGET, image_point=(90.0, 90.0))

    assert len(updated) == 2


def test_set_aim_point_at_stores_offset_from_center():
    updated = set_aim_point_at(TARGET, image_point=(60.0, 45.0))

    assert updated.aim_offset_x == 10.0  # 60 - 50
    assert updated.aim_offset_y == -5.0  # 45 - 50


def test_set_aim_point_at_does_not_mutate_the_original_target():
    original = replace(TARGET)  # a fresh copy, aim_offset_x/y still None

    set_aim_point_at(original, image_point=(60.0, 45.0))

    assert original.aim_offset_x is None
    assert original.aim_offset_y is None


def test_set_aim_point_at_preserves_other_target_fields():
    updated = set_aim_point_at(TARGET, image_point=(60.0, 45.0))

    assert updated.center_x == TARGET.center_x
    assert updated.center_y == TARGET.center_y
    assert updated.radius == TARGET.radius
    assert updated.type == TARGET.type
