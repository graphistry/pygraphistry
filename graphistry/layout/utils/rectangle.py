# DEPRECRATED: Non-vector operators over non-vectorized data

class Rectangle(object):
    """
        Rectangular region.
    """

    def __init__(self, w = 1, h = 1):
        self.w = w
        self.h = h
        self.xy = [0., 0.]

    def __str__(self, *args, **kwargs):
        return "Rectangle (xy: %s) w: %s h: %s" % (self.xy, self.w, self.h)
