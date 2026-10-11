from services.statistics import calculate_statistics, recompute_statistics
from model.hit import Hit
from model.image import Image
from model.session import Session
from model.target import Target
from services.manual_correction import toggle_hit_at


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


def test_aim_point_defaults_to_target_center():
    # No aim_point given -> identical to accuracy measured from (0, 0),
    # i.e. today's behaviour for every existing caller.
    hits = [Hit(x=10.0, y=0.0), Hit(x=10.0, y=0.0)]

    stats = calculate_statistics(hits)

    assert round(stats["accuracy_radius"], 2) == 10.0


def test_accuracy_radius_is_measured_from_the_given_aim_point():
    # MPI is at (10, 0). With the aim point also at (10, 0) - e.g. a
    # silhouette target where the shooter aims at a point other than the
    # contour centroid - accuracy should be 0, not 10.
    hits = [Hit(x=10.0, y=0.0), Hit(x=10.0, y=0.0)]

    stats = calculate_statistics(hits, aim_point=(10.0, 0.0))

    assert round(stats["accuracy_radius"], 2) == 0.0


def test_precision_metrics_are_unaffected_by_aim_point():
    # Precision is about spread around MPI, not about the aim point -
    # moving the aim point must not change precision_radius/cep_50/etc.
    hits = [
        Hit(x=1.0, y=0.0),
        Hit(x=0.0, y=1.0),
        Hit(x=-1.0, y=0.0),
        Hit(x=0.0, y=-1.0),
    ]

    default_stats = calculate_statistics(hits)
    shifted_stats = calculate_statistics(hits, aim_point=(50.0, -30.0))

    assert shifted_stats["precision_radius"] == default_stats["precision_radius"]
    assert shifted_stats["precision_std"] == default_stats["precision_std"]
    assert shifted_stats["cep_50"] == default_stats["cep_50"]
    assert shifted_stats["outliers_count"] == default_stats["outliers_count"]


def test_recompute_statistics_uses_the_sessions_own_aim_point():
    target = Target(center_x=100.0, center_y=100.0, radius=80.0, type="foreground",
                     aim_offset_x=10.0, aim_offset_y=0.0)
    session = Session(id=None, image=Image(path="x.jpg"), target=target, hits=[],
                       statistics={}, metadata={})

    stats = recompute_statistics(session)

    assert stats["accuracy_radius"] == 0.0  # no hits yet, trivially 0 either way
    assert stats == calculate_statistics(session.hits, aim_point=(10.0, 0.0))


def test_recompute_statistics_keeps_aim_point_after_toggling_a_hit():
    # Regression test: manual hit correction must not silently fall back to
    # center-relative accuracy once an aim point has been set.
    target = Target(center_x=100.0, center_y=100.0, radius=80.0, type="foreground",
                     aim_offset_x=10.0, aim_offset_y=0.0)
    session = Session(id=None, image=Image(path="x.jpg"), target=target, hits=[],
                       statistics={}, metadata={})

    # Same helper view/main_window.py's hit-correction handler uses: add a
    # hit exactly at the aim point.
    session.hits = toggle_hit_at(session.hits, target, image_point=(110.0, 100.0))
    session.statistics = recompute_statistics(session)

    aim_relative = calculate_statistics(session.hits, aim_point=(10.0, 0.0))
    center_relative = calculate_statistics(session.hits)  # the buggy behaviour

    assert session.statistics["accuracy_radius"] == aim_relative["accuracy_radius"] == 0.0
    assert session.statistics["accuracy_radius"] != center_relative["accuracy_radius"]
