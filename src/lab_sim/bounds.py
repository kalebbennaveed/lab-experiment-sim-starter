"""Signed distance to an extruded lab polygon; diagnostic, not a controller."""

import numpy as np

from .config import LabConfig


def lab_clearance(position: np.ndarray, lab: LabConfig, radius: float = 0.0) -> float:
    polygon = np.asarray(lab.floor, dtype=float)
    point = np.asarray(position)[:2]
    inside = False
    distance = float("inf")
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b - a
        length_squared = float(edge @ edge)
        fraction = np.clip(float((point - a) @ edge) / length_squared, 0, 1) if length_squared else 0.0
        distance = min(distance, float(np.linalg.norm(point - (a + fraction * edge))))
        if (a[1] > point[1]) != (b[1] > point[1]):
            crossing_x = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if point[0] < crossing_x:
                inside = not inside
    horizontal = distance if inside or distance < 1e-10 else -distance
    return float(min(horizontal, position[2] - lab.z_min, lab.z_max - position[2]) - radius)
