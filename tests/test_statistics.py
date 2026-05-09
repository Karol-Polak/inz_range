from services.statistics import calculate_statistics
from model.hit import Hit

def test_empty_hit():
    stats = calculate_statistics([])
    assert stats["count"] == 0
    assert stats["mean_radius"] == 0.0

def test_simple_hits():
    hits = [
        Hit(x=1.0, y=0.0),
        Hit(x=0.0, y=1.0),
        Hit(x=-1.0, y=0.0),
        Hit(x=0.0, y=-1.0)
    ]

    stats = calculate_statistics(hits)

    assert stats["count"] == 4
    assert round(stats["mean_radius"], 2) == 1.0
    assert round(stats["std_radius"], 2) == 0.0
    assert round(stats["cep_50"], 2) == 1.0

def test_extended_statistics():
    hits = [
        Hit(x=1.0, y=0.0),
        Hit(x=-1.0, y=0.0),
        Hit(x=0.0, y=1.0),
        Hit(x=0.0, y=-1.0),
        Hit(x=5.0, y=5.0)  # outlier
    ]

    stats = calculate_statistics(hits)

    assert stats["count"] == 5
    assert stats["outliers_count"] >= 1
    assert stats["extreme_spread"] > 2.0
    assert stats["std_x"] > 0.0
    assert stats["std_y"] > 0.0