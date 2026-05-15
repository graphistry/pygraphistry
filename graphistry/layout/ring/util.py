from typing import Any, Tuple
import numpy as np
import pandas as pd

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable


def _cudf_version_at_most(cudf_module: Any, major: int, minor: int) -> bool:
    raw_version = getattr(cudf_module, "__version__", "")
    parts = []
    for part in raw_version.split(".")[:2]:
        try:
            parts.append(int(part))
        except ValueError:
            return False
    if len(parts) < 2:
        return False
    return tuple(parts) <= (major, minor)


def _series_to_numpy(values: Any) -> Any:
    if hasattr(values, "to_arrow"):
        return np.asarray(values.to_arrow().to_pylist(), dtype=float)
    if hasattr(values, "to_pandas"):
        return values.to_pandas().to_numpy()
    return np.asarray(values, dtype=float)


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
        if _cudf_version_at_most(cudf, 25, 2):
            angle_np = _series_to_numpy(angle)
            r_np = _series_to_numpy(r)
            x = cudf.Series(r_np * np.cos(angle_np))
            y = cudf.Series(r_np * np.sin(angle_np))
        else:
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
