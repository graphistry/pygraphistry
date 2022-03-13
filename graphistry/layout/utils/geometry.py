#!/usr/bin/env python
# DEPRECRATED: Non-vector operators over non-vectorized data

from numpy import array, cos, sin
from math import atan2, sqrt


def lines_intersection(xy1, xy2, xy3, xy4):
    """
        Returns the intersection of two lines.

    """
    (x1, y1) = xy1
    (x2, y2) = xy2
    (x3, y3) = xy3
    (x4, y4) = xy4
    b = (x2 - x1, y2 - y1)
    d = (x4 - x3, y4 - y3)
    det = b[0] * d[1] - b[1] * d[0]
    if det == 0:
        return None
    c = (x3 - x1, y3 - y1)
    t = float(c[0] * b[1] - c[1] * b[0]) / (det * 1.0)
    if t < 0.0 or t > 1.0:
        return None
    t = float(c[0] * d[1] - c[1] * d[0]) / (det * 1.0)
    if t < 0.0 or t > 1.0:
        return None
    x = x1 + t * b[0]
    y = y1 + t * b[1]
    return (x, y)


def rectangle_point_intersection(rec, p):
    """
        Returns the intersection point between the Rectangle
        (w,h) that characterize the rec object and the line that goes
        from the recs' object center to the 'p' point.

    """
    x2, y2 = p[0] - rec.xy[0], p[1] - rec.xy[1]
    x1, y1 = 0, 0
    # bounding box:
    bbx2 = rec.w // 2
    bbx1 = -bbx2
    bby2 = rec.h // 2
    bby1 = -bby2

    segments = [
        ((x1, y1), (x2, y2), (bbx1, bby1), (bbx2, bby1)),
        ((x1, y1), (x2, y2), (bbx2, bby1), (bbx2, bby2)),
        ((x1, y1), (x2, y2), (bbx1, bby2), (bbx2, bby2)),
        ((x1, y1), (x2, y2), (bbx1, bby2), (bbx1, bby1)),
    ]
    for segs in segments:
        xy = lines_intersection(*segs)
        if xy is not None:
            x, y = xy
            x += rec.xy[0]
            y += rec.xy[1]
            return x, y
    # can't be an intersection unless the endpoint was inside the bb
    raise ValueError(
        "no intersection found (point inside ?!). rec: %s p: %s" % (rec, p)
    )


def angle_between_vectors(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    theta = atan2(y2 - y1, x2 - x1)
    return theta


def size_median(recs):
    mw = [v.w for v in recs]
    mh = [v.h for v in recs]
    mw.sort()
    mh.sort()
    return (mw[len(mw) // 2], mh[len(mh) // 2])


def setcurve(e, pts, tgs = None):
    """
         Returns the spline curve that path through the list of points P.
         The spline curve is a list of cubic bezier curves (nurbs) that have
         matching tangents at their extreme points.
         The method considered here is taken from "The NURBS book" (Les A. Piegl,
         Wayne Tiller, Springer, 1997) and implements a local interpolation rather
         than a global interpolation.

    Args:
        e:
        pts:
        tgs:

    Returns:

    """
    P = list(map(array, pts))
    n = len(P)
    # tangent estimation
    if tgs:
        assert len(tgs) == n
        T = list(map(array, tgs))
        Q = [P[k + 1] - P[k] for k in range(0, n - 1)]
    else:
        Q, T = tangents(P, n)
    splines = []
    for k in range(n - 1):
        t = T[k] + T[k + 1]
        a = 16.0 - (t.dot(t))
        b = 12.0 * (Q[k].dot(t))
        c = -36.0 * Q[k].dot(Q[k])
        D = (b * b) - 4.0 * a * c
        assert D >= 0
        sd = sqrt(D)
        s1, s2 = (-b - sd) / (2.0 * a), (-b + sd) / (2.0 * a)
        s = s2
        if s1 >= 0:
            s = s1
        C0 = tuple(P[k])
        C1 = tuple(P[k] + (s / 3.0) * T[k])
        C2 = tuple(P[k + 1] - (s / 3.0) * T[k + 1])
        C3 = tuple(P[k + 1])
        splines.append([C0, C1, C2, C3])
    return splines


def tangents(P, n):
    assert n >= 2
    Q = []
    T = []
    for k in range(0, n - 1):
        q = P[k + 1] - P[k]
        t = q / sqrt(q.dot(q))
        Q.append(q)
        T.append(t)
    T.append(t)
    return Q, T


def set_round_corner(e, pts):
    P = list(map(array, pts))
    n = len(P)
    Q, T = tangents(P, n)
    c0 = P[0]
    t0 = T[0]
    k0 = 0
    splines = []
    k = 1
    while k < n:
        z = abs(t0[0] * T[k][1] - (t0[1] * T[k][0]))
        if z < 1.0e-6:
            k += 1
            continue
        if (k - 1) > k0:
            splines.append([c0, P[k - 1]])
        if (k + 1) < n:
            splines.extend(setcurve(e, [P[k - 1], P[k + 1]], tgs = [T[k - 1], T[k + 1]]))
        else:
            splines.extend(setcurve(e, [P[k - 1], P[k]], tgs = [T[k - 1], T[k]]))
            break
        if (k + 2) < n:
            c0 = P[k + 1]
            t0 = T[k + 1]
            k0 = k + 1
            k += 2
        else:
            break
    return splines or [[P[0], P[-1]]]


def new_point_at_distance(pt, distance, angle):
    # in rad
    distance = float(distance)
    x, y = pt[0], pt[1]
    x += distance * cos(angle)
    y += distance * sin(angle)
    return x, y
