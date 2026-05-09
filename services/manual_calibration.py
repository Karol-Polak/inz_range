from model.target import Target


def create_target_from_manual_input(
    center_x: float,
    center_y: float,
    radius: float
) -> Target:
    """
    Tworzy obiekt Target na podstawie ręcznie podanych danych.
    """

    if radius <= 0:
        raise ValueError("Promień tarczy musi być dodatni")

    return Target(
        center_x=float(center_x),
        center_y=float(center_y),
        radius=float(radius),
        rings=None,
        type="manual",
        confidence=0.5
    )
