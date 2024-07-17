from typing import Any, Tuple
import numpy as np
import pandas as pd

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable


def polar_to_xy(g: Plottable, r: Any, angle: Any, engine_concrete: Engine) -> Tuple[Any, Any]:
    """
    Using cudf or pandas, convert polar coordinates series to x, y series  
    """
    if engine_concrete == Engine.CUDF:
        import cudf
        import cupy as cp
        if not isinstance(angle, cudf.Series):
            if isinstance(angle, pd.Series):
                angle = cudf.Series(angle)
            else:
                raise ValueError(f'Expected cudf or pd dataframe, received {type(g._nodes)}')
        angle_cp = angle.to_cupy()
        r_cp = r.to_cupy()
        x = cudf.Series(r_cp * cp.cos(angle_cp))
        y = cudf.Series(r_cp * cp.sin(angle_cp))
    elif engine_concrete == Engine.PANDAS:
        assert isinstance(angle, pd.Series)
        x = r * pd.Series(np.cos(angle))
        y = r * pd.Series(np.sin(angle))
    else:
        raise ValueError(f'polar_to_xy only supports cudf/pandas, but selected engine: {engine_concrete}')

    return x, y
