"""Eye Aspect Ratio (EAR) utilities.

EAR is the standard metric used in facial-landmark-based drowsiness
detection, introduced by Soukupova & Cech (2016). It compares the vertical
distance between the eyelids to the horizontal width of the eye. The ratio
stays roughly constant while eyes are open and drops sharply when they close.

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

where p1..p6 are six points traced around the eye (left corner, two top
points, right corner, two bottom points).

This module maps that classic 6-point layout onto MediaPipe FaceMesh's
468-point landmark model.
"""

from typing import List, Tuple

import numpy as np

Point = Tuple[float, float]

# MediaPipe FaceMesh indices, ordered as [p1, p2, p3, p4, p5, p6]
# to match the standard 6-point EAR layout.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def _distance(a: Point, b: Point) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def eye_aspect_ratio(landmarks: List[Point], eye_indices: List[int]) -> float:
    """EAR for a single eye given its 6 (x, y) landmark points."""
    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in eye_indices)

    vertical_1 = _distance(p2, p6)
    vertical_2 = _distance(p3, p5)
    horizontal = _distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def average_ear(landmarks: List[Point]) -> float:
    """Average EAR across both eyes - more stable than using a single eye,
    since it's less sensitive to head tilt or one eye being partly occluded."""
    left = eye_aspect_ratio(landmarks, LEFT_EYE)
    right = eye_aspect_ratio(landmarks, RIGHT_EYE)
    return (left + right) / 2.0
