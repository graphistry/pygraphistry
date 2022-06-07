# DEPRECRATED: Non-vector operators over non-vectorized data

from .geometry import rectangle_point_intersection, angle_between_vectors, sqrt


class EdgeViewer(object):
    def setpath(self, pts):
        self._pts = pts


def route_with_lines(e, pts):
    """
     Basic edge routing with lines. The layout pass has already provided to list of points through which
     the edge shall be drawn. We just compute the position where to adjust the tail and head.

    """
    assert hasattr(e, "view")
    tail_pos = rectangle_point_intersection(e.v[0].view, p = pts[1])
    head_pos = rectangle_point_intersection(e.v[1].view, p = pts[-2])
    pts[0] = tail_pos
    pts[-1] = head_pos
    e.view.head_angle = angle_between_vectors(pts[-2], pts[-1])


def route_with_splines(e, pts):
    """
     Enhanced edge routing where 'corners' of the above polyline route are rounded with a Bezier curve.
    """
    from .geometry import set_round_corner

    route_with_lines(e, pts)
    splines = set_round_corner(e, pts)
    e.view.splines = splines


def _gen_point(p1, p2, new_distance):
    from .geometry import new_point_at_distance

    initial_distance = distance = sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    if initial_distance < 1e-10:
        return None
    if distance > new_distance:
        distance = distance - new_distance
    else:
        return None
    angle = angle_between_vectors(p1, p2)
    new = new_point_at_distance(p1, distance, angle)
    return new


def _gen_smoother_middle_points_from_3_points(pts, initial):
    p1 = pts[0]
    p2 = pts[1]
    p3 = pts[2]
    distance1 = sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    distance2 = sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2)
    if distance1 < 1e-10 or distance2 < 1e-10:
        yield p2
    else:
        if distance1 < initial or distance2 < initial:
            yield p2
        else:
            p2a = _gen_point(p1, p2, initial)
            p2b = _gen_point(p3, p2, initial)
            if p2a is None or p2b is None:
                yield p2
            else:
                yield p2a
                yield p2b


# Future work: possibly work better when we already have 4 points?
# maybe: http://stackoverflow.com/questions/1251438/catmull-rom-splines-in-python
def _round_corners(pts, round_at_distance):
    if len(pts) > 2:
        calc_with_distance = round_at_distance
        while calc_with_distance > 0.5:
            new_lst = [pts[0]]
            for i, curr in enumerate(pts[1:-1]):
                i += 1
                p1 = pts[i - 1]
                p2 = curr
                p3 = pts[i + 1]
                if len(pts) > 3:
                    # i.e.: at least 4 points
                    if sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2) < (
                            2 * calc_with_distance
                    ):
                        # prevent from crossing over.
                        new_lst.append(p2)
                        continue
                generated = _gen_smoother_middle_points_from_3_points(
                    [p1, p2, p3], calc_with_distance
                )
                for j in generated:
                    new_lst.append(j)
            new_lst.append(pts[-1])
            pts = new_lst
            calc_with_distance /= 2.0
    return pts


# ------------------------------------------------------------------------------
# Routing with a custom algorithm to round corners
# It works by generating new points up to a distance from where an edge is
# found (and then iteratively refining based on that).
# This is a custom implementation as this interpolation method worked
# well for me where others weren't so great.

# This is the point where it'll start rounding from an edge.
# (can be changed to decide up to which distance it starts
# rounding from an edge).
ROUND_AT_DISTANCE = 40


def route_with_rounded_corners(e, pts):
    route_with_lines(e, pts)
    new_pts = _round_corners(pts, round_at_distance = ROUND_AT_DISTANCE)
    pts[:] = new_pts[:]
