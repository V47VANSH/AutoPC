"""
Curved cursor paths for the hands.

`core/hands.py` calls `generate_path(start, target)` and walks the returned
points with the Interception driver, so the pointer travels along a short curve
instead of teleporting. Moving through intermediate points matters in practice:
apps that react to hover (menus, tooltips, drag targets) see the pointer arrive
the way they expect, and a drag passes over the intermediate drop zones.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

Point = Tuple[float, float]


def generate_path(start, target) -> List[Point]:
    """Randomized cubic Bezier from start to target with a slight sideways bow."""
    x0, y0 = float(start[0]), float(start[1])
    x3, y3 = float(target[0]), float(target[1])
    dx, dy = x3 - x0, y3 - y0
    dist = math.hypot(dx, dy)
    if dist < 2:
        return [(x0, y0), (x3, y3)]

    steps = max(12, min(90, int(dist / 10)))

    # Unit vector perpendicular to the travel direction -- the bow direction.
    px, py = -dy / dist, dx / dist
    bow = random.uniform(-0.16, 0.16) * dist

    # Two control points at ~1/3 and ~2/3 along, pushed out along the normal.
    c1 = (x0 + dx * 0.33 + px * bow * random.uniform(0.6, 1.0),
          y0 + dy * 0.33 + py * bow * random.uniform(0.6, 1.0))
    c2 = (x0 + dx * 0.66 + px * bow * random.uniform(0.6, 1.0),
          y0 + dy * 0.66 + py * bow * random.uniform(0.6, 1.0))

    pts: List[Point] = []
    for i in range(steps + 1):
        t = i / steps
        mt = 1 - t
        x = mt ** 3 * x0 + 3 * mt ** 2 * t * c1[0] + 3 * mt * t ** 2 * c2[0] + t ** 3 * x3
        y = mt ** 3 * y0 + 3 * mt ** 2 * t * c1[1] + 3 * mt * t ** 2 * c2[1] + t ** 3 * y3
        pts.append((x, y))
    return pts
