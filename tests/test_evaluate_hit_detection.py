import cv2
import numpy as np

from evaluation.evaluate_hit_detection import aggregate, aggregate_by_group, evaluate_image


def _result(target_type, true_positives=1, false_positives=0, false_negatives=0, time_total_s=0.1):
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "mean_error_px": 1.0,
        "mean_error_mm": None,
        "target_type": target_type,
        "time_total_s": time_total_s,
    }


def test_aggregate_by_group_splits_results_by_target_type():
    results = [
        _result("circular", true_positives=5, false_positives=1),
        _result("circular", true_positives=3, false_positives=0),
        _result("foreground", true_positives=2, false_positives=2),
    ]

    grouped = aggregate_by_group(results)

    assert set(grouped.keys()) == {"circular", "foreground"}
    assert grouped["circular"]["true_positives"] == 8
    assert grouped["circular"]["false_positives"] == 1
    assert grouped["foreground"]["true_positives"] == 2
    assert grouped["foreground"]["false_positives"] == 2


def test_aggregate_by_group_matches_plain_aggregate_within_each_group():
    circular_results = [_result("circular", true_positives=5, false_positives=1)]
    all_results = circular_results + [_result("foreground", true_positives=2, false_positives=2)]

    grouped = aggregate_by_group(all_results)

    assert grouped["circular"] == aggregate(circular_results)


def test_aggregate_by_group_handles_missing_target_type():
    results = [{"true_positives": 1, "false_positives": 0, "false_negatives": 0,
                "mean_error_px": 0.0, "time_total_s": 0.1}]

    grouped = aggregate_by_group(results)

    assert "nieznany" in grouped


def test_aggregate_reports_mean_median_max_time():
    results = [
        _result("circular", time_total_s=0.1),
        _result("circular", time_total_s=0.2),
        _result("circular", time_total_s=0.3),
    ]

    summary = aggregate(results)

    assert round(summary["mean_time_s"], 3) == 0.2
    assert summary["median_time_s"] == 0.2
    assert summary["max_time_s"] == 0.3


def _build_target_image(path, size=400, face_value=50, bg_value=230, radius=150, center=(200, 200)):
    image_array = np.full((size, size, 3), bg_value, dtype=np.uint8)
    cv2.circle(image_array, center, radius, (face_value,) * 3, -1)
    cv2.circle(image_array, (180, 180), 6, (250, 250, 250), -1)
    cv2.imwrite(str(path), image_array)


def test_evaluate_image_reports_target_type_and_timing(tmp_path):
    path = tmp_path / "synthetic_target.png"
    _build_target_image(path)

    metrics = evaluate_image(str(path), ground_truth_hits=[(180.0, 180.0)])

    assert metrics is not None
    assert metrics["target_type"] == "foreground"
    assert metrics["time_total_s"] >= 0.0
    assert "detect_hit" in metrics["time_breakdown_s"]


def test_evaluate_image_returns_none_when_no_target_detected(tmp_path):
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), np.full((300, 300, 3), 200, dtype=np.uint8))

    assert evaluate_image(str(path), ground_truth_hits=[]) is None
