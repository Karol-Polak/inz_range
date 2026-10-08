from evaluation.metrics import compute_metrics, match_points


def test_match_points_pairs_up_identical_points():
    detected = [(10.0, 10.0), (50.0, 50.0)]
    ground_truth = [(10.0, 10.0), (50.0, 50.0)]

    result = match_points(detected, ground_truth, max_distance=5.0)

    assert len(result["matches"]) == 2
    assert result["false_positives"] == []
    assert result["false_negatives"] == []


def test_match_points_reports_false_positive_for_extra_detection():
    detected = [(10.0, 10.0), (200.0, 200.0)]
    ground_truth = [(10.0, 10.0)]

    result = match_points(detected, ground_truth, max_distance=5.0)

    assert len(result["matches"]) == 1
    assert result["false_positives"] == [1]
    assert result["false_negatives"] == []


def test_match_points_reports_false_negative_for_missed_hit():
    detected = [(10.0, 10.0)]
    ground_truth = [(10.0, 10.0), (200.0, 200.0)]

    result = match_points(detected, ground_truth, max_distance=5.0)

    assert len(result["matches"]) == 1
    assert result["false_positives"] == []
    assert result["false_negatives"] == [1]


def test_match_points_respects_distance_threshold():
    detected = [(10.0, 10.0)]
    ground_truth = [(20.0, 10.0)]  # 10px away

    too_strict = match_points(detected, ground_truth, max_distance=5.0)
    lenient = match_points(detected, ground_truth, max_distance=15.0)

    assert too_strict["matches"] == []
    assert len(lenient["matches"]) == 1


def test_match_points_picks_closest_ground_truth_when_ambiguous():
    detected = [(10.0, 10.0)]
    ground_truth = [(12.0, 10.0), (15.0, 10.0)]

    result = match_points(detected, ground_truth, max_distance=10.0)

    assert len(result["matches"]) == 1
    _, matched_gt_index, _ = result["matches"][0]
    assert matched_gt_index == 0  # the nearer ground-truth point


def test_compute_metrics_perfect_detection():
    detected = [(10.0, 10.0), (50.0, 50.0)]
    ground_truth = [(10.0, 10.0), (50.0, 50.0)]

    metrics = compute_metrics(detected, ground_truth, max_distance=5.0)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["mean_error_px"] == 0.0
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_compute_metrics_with_one_fp_and_one_fn():
    detected = [(10.0, 10.0), (200.0, 200.0)]
    ground_truth = [(10.0, 10.0), (300.0, 300.0)]

    metrics = compute_metrics(detected, ground_truth, max_distance=5.0)

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert round(metrics["precision"], 2) == 0.5
    assert round(metrics["recall"], 2) == 0.5


def test_compute_metrics_converts_mean_error_to_mm_when_scale_given():
    detected = [(13.0, 10.0)]
    ground_truth = [(10.0, 10.0)]  # 3px off

    metrics = compute_metrics(detected, ground_truth, max_distance=5.0, scale_mm_per_px=2.0)

    assert metrics["mean_error_px"] == 3.0
    assert metrics["mean_error_mm"] == 6.0


def test_compute_metrics_handles_no_ground_truth_hits():
    metrics = compute_metrics([], [], max_distance=5.0)

    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["true_positives"] == 0
