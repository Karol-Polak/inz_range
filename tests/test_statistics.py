from services.statistics import calculate_statistics
from model.hit import Hit


def test_empty_hit():
    stats = calculate_statistics([])
    assert stats["count"] == 0
    assert stats["accuracy_radius"] == 0.0
    assert stats["precision_radius"] == 0.0
    assert stats["cep_50"] == 0.0


def test_centered_group_has_zero_accuracy_offset():
    hits = [
        Hit(x=1.0, y=0.0),
        Hit(x=0.0, y=1.0),
        Hit(x=-1.0, y=0.0),
        Hit(x=0.0, y=-1.0),
    ]

    stats = calculate_statistics(hits)

    assert stats["count"] == 4
    assert round(stats["accuracy_radius"], 2) == 0.0
    assert round(stats["precision_radius"], 2) == 1.0
    assert round(stats["precision_std"], 2) == 0.0
    assert round(stats["cep_50"], 2) == 1.0


def test_accuracy_and_precision_are_independent():
    # Tight group, but its center (MPI) sits far from the point of aim.
    hits = [
        Hit(x=10.0, y=10.0),
        Hit(x=10.5, y=10.0),
        Hit(x=9.5, y=10.0),
        Hit(x=10.0, y=9.5),
        Hit(x=10.0, y=10.5),
    ]

    stats = calculate_statistics(hits)

    assert stats["accuracy_radius"] > 10.0  # MPI is far from the target center
    assert stats["precision_radius"] < 1.0  # but the group itself is tight


def test_cep_50_uses_true_median_distance_from_mpi():
    # MPI = (2.5, 0). Distances from MPI: |1-2.5|=1.5, |2-2.5|=0.5, |3-2.5|=0.5, |4-2.5|=1.5.
    # True median of [0.5, 0.5, 1.5, 1.5] is 1.0.
    # The old nearest-rank implementation (sorted[int(0.5*4)]) would have returned 1.5.
    hits = [
        Hit(x=1.0, y=0.0),
        Hit(x=2.0, y=0.0),
        Hit(x=3.0, y=0.0),
        Hit(x=4.0, y=0.0),
    ]

    stats = calculate_statistics(hits)

    assert round(stats["cep_50"], 2) == 1.0


def test_outlier_is_detected_via_mad_regression():
    # Regression test: the previous mean+2*std implementation failed to flag
    # this outlier because the outlier itself inflated the std deviation it
    # was being compared against.
    hits = [
        Hit(x=1.0, y=0.0),
        Hit(x=-1.0, y=0.0),
        Hit(x=0.0, y=1.0),
        Hit(x=0.0, y=-1.0),
        Hit(x=5.0, y=5.0),  # clear flier relative to the rest of the group
    ]

    stats = calculate_statistics(hits)

    assert stats["count"] == 5
    assert stats["outliers_count"] >= 1
    assert stats["extreme_spread"] > 2.0
    assert stats["std_x"] > 0.0
    assert stats["std_y"] > 0.0


def test_no_false_positive_outliers_in_tight_uniform_group():
    hits = [
        Hit(x=1.0, y=0.0),
        Hit(x=0.0, y=1.0),
        Hit(x=-1.0, y=0.0),
        Hit(x=0.0, y=-1.0),
    ]

    stats = calculate_statistics(hits)

    assert stats["outliers_count"] == 0
