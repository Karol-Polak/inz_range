import cv2

from model.session import Session


def draw_analysis_overlay(session: Session):
    """
    Draws detected target and hits on top of the original target image.
    Returns an OpenCV BGR image ready for display conversion.
    """

    if session.image.original_data is None:
        raise ValueError("Brak oryginalnych danych obrazu do wizualizacji")

    output = session.image.original_data.copy()
    target = session.target

    center = (int(round(target.center_x)), int(round(target.center_y)))

    if target.type == "circular":
        radius = int(round(target.radius))
        cv2.circle(output, center, radius, (46, 111, 115), 3)

    cv2.circle(output, center, 5, (255, 255, 255), -1)
    cv2.circle(output, center, 5, (20, 32, 43), 2)

    for index, hit in enumerate(session.hits, start=1):
        if not hit.valid:
            continue

        x = int(round(target.center_x + hit.x))
        y = int(round(target.center_y + hit.y))

        cv2.circle(output, (x, y), 13, (0, 215, 255), 3)
        cv2.circle(output, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(
            output,
            str(index),
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output
