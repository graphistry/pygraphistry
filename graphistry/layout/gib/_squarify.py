"""
Squarified treemap layout — built-in implementation.

Implements the algorithm from:
  Bruls, Huizing, van Wijk. "Squarified Treemaps." (2000)

Public API:
  normalize_sizes(sizes, dx, dy) -> List[float]
  squarify(sizes, x, y, dx, dy) -> List[dict]

Internal helpers mirror the reference algorithm structure for
ease of verification. Pure Python — no external dependencies.
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _layoutrow(sizes: List[float], x: float, y: float, dx: float, dy: float) -> List[dict]:
    """Place a strip of rects filling the full height dy (dx >= dy case)."""
    covered_area = sum(sizes)
    width = covered_area / dy
    rects: List[dict] = []
    y_cursor = y
    for size in sizes:
        cell_dy = size / width
        rects.append({"x": x, "y": y_cursor, "dx": width, "dy": cell_dy})
        y_cursor += cell_dy
    return rects


def _layoutcol(sizes: List[float], x: float, y: float, dx: float, dy: float) -> List[dict]:
    """Place a strip of rects filling the full width dx (dx < dy case)."""
    covered_area = sum(sizes)
    height = covered_area / dx
    rects: List[dict] = []
    x_cursor = x
    for size in sizes:
        cell_dx = size / height
        rects.append({"x": x_cursor, "y": y, "dx": cell_dx, "dy": height})
        x_cursor += cell_dx
    return rects


def _layout(sizes: List[float], x: float, y: float, dx: float, dy: float) -> List[dict]:
    if dx >= dy:
        return _layoutrow(sizes, x, y, dx, dy)
    return _layoutcol(sizes, x, y, dx, dy)


def _leftoverrow(sizes: List[float], x: float, y: float, dx: float, dy: float):
    covered_area = sum(sizes)
    width = covered_area / dy
    return x + width, y, dx - width, dy


def _leftovercol(sizes: List[float], x: float, y: float, dx: float, dy: float):
    covered_area = sum(sizes)
    height = covered_area / dx
    return x, y + height, dx, dy - height


def _leftover(sizes: List[float], x: float, y: float, dx: float, dy: float):
    if dx >= dy:
        return _leftoverrow(sizes, x, y, dx, dy)
    return _leftovercol(sizes, x, y, dx, dy)


def _worst_ratio(sizes: List[float], x: float, y: float, dx: float, dy: float) -> float:
    rects = _layout(sizes, x, y, dx, dy)
    ratios = [max(r["dx"] / r["dy"], r["dy"] / r["dx"]) for r in rects]
    return max(ratios)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_sizes(sizes, dx: float, dy: float) -> List[float]:
    """Normalize sizes so that sum(result) == dx * dy.

    Parameters
    ----------
    sizes : list-like of numeric
        Input values (must be non-empty and sum > 0 for meaningful output).
    dx, dy : float
        Canvas dimensions.

    Returns
    -------
    List[float]
        Scaled values whose sum equals dx * dy.
    """
    arr = list(map(float, sizes))
    if len(arr) == 0:
        return arr
    total_size = sum(arr)
    total_area = float(dx) * float(dy)
    scale = total_area / total_size  # raises ZeroDivisionError if total_size == 0
    return [v * scale for v in arr]


def squarify(sizes, x: float, y: float, dx: float, dy: float) -> List[dict]:
    """Compute squarified treemap rectangles.

    Parameters
    ----------
    sizes : list-like of float
        Pre-normalized values (sum == dx * dy), sorted descending.
    x, y : float
        Top-left corner of the layout area.
    dx, dy : float
        Width and height of the layout area.

    Returns
    -------
    List[dict]
        Each dict has keys 'x', 'y', 'dx', 'dy'.
        Order corresponds to input order.
    """
    sizes = list(map(float, sizes))

    if len(sizes) == 0:
        return []

    if len(sizes) == 1:
        return _layout(sizes, x, y, dx, dy)

    # Find the split point: greedily add items to the current strip while
    # the worst aspect ratio keeps improving.
    i = 1
    while i < len(sizes) and _worst_ratio(sizes[:i], x, y, dx, dy) >= _worst_ratio(
        sizes[: i + 1], x, y, dx, dy
    ):
        i += 1

    current = sizes[:i]
    remaining = sizes[i:]

    lx, ly, ldx, ldy = _leftover(current, x, y, dx, dy)
    return _layout(current, x, y, dx, dy) + squarify(remaining, lx, ly, ldx, ldy)
