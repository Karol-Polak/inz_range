"""
Refactor self-check, NOT a measure of detection quality.

Moving the magic numbers in services/target_detection.py and
services/hit_detection.py into utils/config.py must not change detection
results. This test captures the exact output of detect_target()/detect_hit()
on a few real assets/ images from BEFORE that refactor (hard-coded below)
and asserts the current code still reproduces it.

Hit count and target type must match exactly. Coordinates/center/radius are
compared with ~1px tolerance rather than exact float equality, because
Hough circle detection and the watershed split can differ by a pixel or two
across OpenCV versions/platforms even with byte-identical input and
parameters - that is not a behaviour change worth failing this check over.
"""

import math

import pytest

from services.hit_detection import detect_hit
from services.image_loader import load_image
from services.preprocessing import preprocess_image
from services.target_detection import detect_target

_TOLERANCE_PX = 1.0

# Captured from the pre-refactor code (utils/config.py constants not yet
# extracted) - see the git history of this file / the session that added it
# for exactly how these were produced.
_GOLDEN = [
    {
        "path": "assets/1134047_przestrzeliny.jpg",
        "target": {"center_x": 608.0, "center_y": 613.0, "radius": 431.0, "type": "foreground"},
        "hits": [
            (23.4229, 24.3526), (25.4874, 260.4828), (147.4814, -221.5077),
            (287.4493, 128.4817), (-238.4946, -293.5116), (0.7778, -391.9556),
            (-31.9444, -411.1111), (-408.2143, -246.7857), (-421.3, -308.0),
            (-364.5119, 378.5043), (341.65, -414.65),
        ],
    },
    {
        "path": "assets/idpa_przestrzeliny.jpg",
        "target": {"center_x": 984.5, "center_y": 1378.0, "radius": 1378.0, "type": "foreground"},
        "hits": [
            (87.0, 285.5), (-385.0, -286.5), (303.0073, -510.4897), (-8.7276, 617.0742),
            (278.9942, -562.4887), (711.0, 133.5), (-676.9775, -702.5),
            (131.0, -1070.5), (-585.0, 1125.5),
        ],
    },
    {
        "path": "assets/tarcza_zolta_przestrzeliny.png",
        "target": {"center_x": 865.0, "center_y": 842.0, "radius": 809.0, "type": "circular"},
        "hits": [
            (-38.6325, -15.9382), (131.6587, 10.0079), (139.7597, 118.3411),
            (-36.5, -189.4583), (-156.5333, 115.4902), (85.2344, 175.2578),
            (158.8659, 118.0), (149.7717, -187.2559), (-215.0714, -199.7222),
        ],
    },
]


def _closest_distance(point, candidates):
    return min(math.hypot(point[0] - c[0], point[1] - c[1]) for c in candidates)


@pytest.mark.parametrize("golden", _GOLDEN, ids=[g["path"] for g in _GOLDEN])
def test_detection_output_unchanged_after_moving_constants_to_config(golden):
    image = load_image(golden["path"])
    image = preprocess_image(image)
    target = detect_target(image)

    assert target is not None
    assert target.type == golden["target"]["type"]
    assert abs(target.center_x - golden["target"]["center_x"]) <= _TOLERANCE_PX
    assert abs(target.center_y - golden["target"]["center_y"]) <= _TOLERANCE_PX
    assert abs(target.radius - golden["target"]["radius"]) <= _TOLERANCE_PX

    hits = detect_hit(image, target)
    assert len(hits) == len(golden["hits"])

    detected_points = [(hit.x, hit.y) for hit in hits]
    for expected_point in golden["hits"]:
        assert _closest_distance(expected_point, detected_points) <= _TOLERANCE_PX
