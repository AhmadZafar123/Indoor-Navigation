"""Field-of-view (FOV) spatial filter for indoor navigation.

Only objects whose center falls inside a centered, configurable rectangle are
considered relevant for spoken feedback. Everything else is still drawn by
YOLO but is not announced, so the user only hears about obstacles that are
actually in their walking path.

Usage:
    from src.decision.fov import FieldOfView

    fov = FieldOfView(width_ratio=0.4, height_ratio=0.8)
    for box in result.boxes:
        if fov.is_inside_fov(box, img_width, img_height):
            ...  # object is in the walking path
    annotated_frame = fov.draw_fov(annotated_frame)
"""

import cv2


class FieldOfView:
    """A centered rectangular region used to filter relevant detections."""

    def __init__(self, width_ratio: float = 0.4, height_ratio: float = 0.8) -> None:
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio

    def get_fov_coordinates(
        self, img_width: int, img_height: int
    ) -> tuple[int, int, int, int]:
        """Return ``(x_min, y_min, x_max, y_max)`` for the centered FOV box."""
        fov_width = img_width * self.width_ratio
        fov_height = img_height * self.height_ratio

        x_min = int((img_width - fov_width) / 2)
        x_max = int((img_width + fov_width) / 2)
        y_min = int((img_height - fov_height) / 2)
        y_max = int((img_height + fov_height) / 2)

        x_min = max(0, min(x_min, img_width))
        x_max = max(0, min(x_max, img_width))
        y_min = max(0, min(y_min, img_height))
        y_max = max(0, min(y_max, img_height))

        return x_min, y_min, x_max, y_max

    def is_inside_fov(
        self, box, img_width: int, img_height: int
    ) -> bool:
        """Return ``True`` if the center of *box* lies inside the FOV box.

        *box* is a single Ultralytics bounding box (``box.xyxy[0]`` is
        ``[x1, y1, x2, y2]``).
        """
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        x_min, y_min, x_max, y_max = self.get_fov_coordinates(img_width, img_height)

        return x_min <= cx <= x_max and y_min <= cy <= y_max

    def draw_fov(self, frame):
        """Draw the FOV region on *frame* and return it.

        The area is tinted with a semi-transparent green overlay, framed with a
        solid green border, and labelled so it is easy to debug on screen.
        """
        img_height, img_width = frame.shape[:2]
        x_min, y_min, x_max, y_max = self.get_fov_coordinates(img_width, img_height)

        green = (0, 255, 0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), green, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), green, 2)
        cv2.putText(
            frame,
            "Active Navigation FOV",
            (x_min + 10, y_min + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            green,
            2,
            cv2.LINE_AA,
        )
        return frame
