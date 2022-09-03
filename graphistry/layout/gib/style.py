from typing import Any, Dict, List, Optional
from functools import lru_cache

from graphistry.Engine import Engine, df_concat, df_to_pdf, df_cons
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
logger = setup_logger(__name__)


@lru_cache(maxsize=1)
def lazy_paired_12():
    from palettable.colorbrewer.qualitative import Paired_12
    return Paired_12

def categorical_color_by_col(
    self: Plottable,
    col: str,
    colors: Optional[List[str]],
    engine: Engine = Engine.PANDAS
) -> 'Plottable':
    g = self
    colors = colors or lazy_paired_12().hex_colors
    palette = g._nodes[[col]].drop_duplicates().reset_index(drop=True).reset_index()
    palette['color'] = (
        (palette['index'] % len(colors))
        .map({i: colors[i % len(colors)] for i in range(len(palette))})
    )
    partition_to_color: Dict[Any, int] = (
        df_to_pdf(
            palette[[col, 'color']].set_index(col),
            engine
        )
        .to_dict()
    )['color']
    g2 = g.encode_point_color(col, categorical_mapping=partition_to_color)  # type: ignore
    return g2


def style_layout(
    self: Plottable,
    encode_color = True,
    colors: Optional[List[str]] = None,
    partition_key = 'partition',
    engine: Engine = Engine.PANDAS
) -> 'Plottable':

    g = self.layout_settings(play=0)

    if not encode_color:
        return g
    
    return categorical_color_by_col(g, partition_key, colors, engine)
