from typing import Any, Callable, Iterable, List, Optional, Set, Union, TYPE_CHECKING
import logging
from .Plottable import Plottable
logger = logging.getLogger('layouts')

if TYPE_CHECKING:
    MIXIN_BASE = Plottable
else:
    MIXIN_BASE = object

class LayoutsMixin(MIXIN_BASE):

    def __init__(self, *args, **kwargs):
        pass

    def tree_layout(
        self,
        level_col: Optional[str] = None,
        descending=True,
        level_sort_values_by: Optional[Union[str, List[str]]] = None,
        level_sort_values_by_ascending: bool = True,
        level_align: str = 'left',
        aspect_ratio=True,
        width: Optional[float] = None,
        height: Optional[float] = None,
        preserve_aspect_ratio: bool = True,

        vertical: bool = True,
        ascending: bool = True,

        *args,
        **kwargs
    ):
        """
        If level_col is None, compute via get_topological_levels() - see its as optional parameters

        Supports cudf + pandas
        """

        g : Plottable = self

        if (g._edges is None) or (len(g._edges) == 0):
            return g

        x_col = g._point_x if g._point_x is not None else 'x'
        if self._point_x is None:
            g = g.bind(point_x=x_col)

        y_col = g._point_y if g._point_y is not None else 'y'
        if g._point_y is None:
            g = g.bind(point_y=y_col)

        #y
        if level_col is None:
            level_col = 'level'
            g2 = g.get_topological_levels(level_col, *args, **kwargs)
            g2._nodes[y_col] = g2._nodes[level_col]
        else:
            if (g._nodes is None) or (not (level_col in g._nodes)):
                raise ValueError('tree_layout() with explicit level_col requires ._nodes with that as a column; see .nodes()')
            g2 = g.nodes(g._nodes.assign(**{y_col: g._nodes[level_col]}))
        if descending:
            g2._nodes[y_col] = -1 * g2._nodes[y_col]

        if (g2._nodes is None) or (len(g2._nodes) == 0):
            return g

        #x
        if level_sort_values_by is not None:
            g2 = g2.nodes(g2._nodes.sort_values(
                by=level_sort_values_by,
                ascending=level_sort_values_by_ascending))

        grouped = g2._nodes.groupby(level_col, sort=level_sort_values_by is not None)
        if hasattr(grouped, 'cumcount'):
            g2 = g2.nodes(g2._nodes.assign(**{x_col: grouped.cumcount()}))
        else:
            try:
                #TODO remove
                #cudf 0.19 fallback
                logger.info('Tree x positions using Pandas fallback for RAPIDS < 0.21')
                import cudf
                assert isinstance(g2._nodes, cudf.DataFrame)
                xs_ps = (g2
                        ._nodes[[level_col]].to_pandas()
                        .groupby(level_col, sort=level_sort_values_by is not None)
                        .cumcount())
                xs_gs = cudf.from_pandas(xs_ps)
                g2 = g2.nodes(g2._nodes.assign(**{x_col: xs_gs}))
            except:
                raise ValueError('Requires RAPIDS 0.21+ or Pandas 0.22+')

        if level_align == 'left':
            1
        elif level_align == 'right':
            g2 = g2.nodes(g2._nodes.assign(**{x_col: -g2._nodes[x_col]}))
        elif level_align == 'center':  # FIXME emitting wrong range
            mx_group = grouped.size().max()
            #TODO switch to grouped when above rapids 0.19 fallback gone
            nodes2_df = (g2
                        ._nodes.groupby(level_col, sort=True)
                        .apply(lambda df: df.assign(
                            **{x_col: (df['x'] + 0.5) / (0.0 + len(df))})))
            nodes2_df[x_col] = mx_group * nodes2_df[x_col]
            g2 = g2.nodes(nodes2_df)
        else:
            raise ValueError(f'level_align must be "left", "center", or "right", got: {level_align}')

        if width is not None or height is not None:
            computed_width = 0.0 + g2._nodes[x_col].max() - g2._nodes[x_col].min()
            computed_height = 0.0 + g2._nodes[y_col].max() - g2._nodes[y_col].min()
            aspect_ratio = computed_width / computed_height

            if width is not None:
                g2._nodes[x_col] = (width / computed_width) * g2._nodes[x_col]
            elif preserve_aspect_ratio:
                g2._nodes[x_col] = (height * aspect_ratio / computed_width) * g2._nodes[x_col]

            if height is not None:
                g2._nodes[y_col] = (height / computed_height) * g2._nodes[y_col]
            elif preserve_aspect_ratio:
                g2._nodes[y_col] = (width / (aspect_ratio * computed_height)) * g2._nodes[y_col]

        if not ascending:
            g2._nodes[y_col] = -g2._nodes[y_col]

        if not vertical:
            g2 = g2.nodes(g2._nodes.rename(columns={
                x_col: y_col,
                y_col: x_col
            }))
            g2._nodes[x_col] = -g2._nodes[x_col]

        return g2
