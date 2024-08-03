from typing import cast, List, Optional, Union, TYPE_CHECKING
import math, pandas as pd
from .Plottable import Plottable
from .layout import (
    SugiyamaLayout,
    group_in_a_box_layout as group_in_a_box_layout_base,
    modularity_weighted_layout as modularity_weighted_layout_base,
    ring_categorical as ring_categorical_base,
    ring_continuous as ring_continuous_base,
    time_ring as time_ring_base
)
from .layout.graph import Graph
from .util import deprecated, setup_logger
logger = setup_logger(__name__)

if TYPE_CHECKING:
    MIXIN_BASE = Plottable
else:
    MIXIN_BASE = object


class LayoutsMixin(MIXIN_BASE):

    def __init__(self, *args, **kwargs):
        pass

    def group_in_a_box_layout(self, *args, **kwargs):
        return group_in_a_box_layout_base(self, *args, **kwargs)
    group_in_a_box_layout.__doc__ = group_in_a_box_layout_base.__doc__

    def modularity_weighted_layout(self, *args, **kwargs):
        return modularity_weighted_layout_base(self, *args, **kwargs)
    modularity_weighted_layout.__doc__ = modularity_weighted_layout_base.__doc__

    def time_ring_layout(self, *args, **kwargs):
        return time_ring_base(self, *args, **kwargs)
    time_ring_layout.__doc__ = time_ring_base.__doc__

    def ring_categorical_layout(self, *args, **kwargs):
        return ring_categorical_base(self, *args, **kwargs)
    ring_categorical_layout.__doc__ = ring_categorical_base.__doc__

    def ring_continuous_layout(self, *args, **kwargs):
        return ring_continuous_base(self, *args, **kwargs)
    ring_continuous_layout.__doc__ = ring_continuous_base.__doc__

    def tree_layout(self,
                    level_col: Optional[str] = None,
                    level_sort_values_by: Optional[Union[str, List[str]]] = None,
                    level_sort_values_by_ascending: bool = True,
                    width: Optional[float] = None,
                    height: Optional[float] = None,
                    rotate: Optional[float] = None,
                    allow_cycles = True,
                    root=None,
                    *args,
                    **kwargs):
        """
            Improved tree layout based on the Sugiyama algorithm.
            * rotate: rotates the layout by the given angle (in degrees)
        """
        g: Plottable = self

        if (g._edges is None) or (len(g._edges) == 0):
            return g

        x_col = g._point_x if g._point_x is not None else 'x'
        if self._point_x is None:
            g = g.bind(point_x = x_col).layout_settings(
                play=0
            )

        y_col = g._point_y if g._point_y is not None else 'y'
        if g._point_y is None:
            g = g.bind(point_y = y_col).layout_settings(
                play=0
            )

        # since the coordinates are topological
        if width is None:
            width = 1
        if height is None:
            height = 1

        # ============================================================
        # level and y-values
        # ============================================================
        if level_col is None:
            level_col = 'level'
        g2 = g.materialize_nodes()
        # check cycles
        if not allow_cycles:
            if SugiyamaLayout.has_cycles(g._edges, source_column = g2._source, target_column = g2._destination):
                raise ValueError

        triples = SugiyamaLayout.arrange(g2._edges, topological_coordinates = True, source_column = g2._source, target_column = g2._destination, include_levels = True, root = root)
        g2._nodes[level_col] = [triples[id][2] for id in g2._nodes[g2._node]]
        g2._nodes[y_col] = [triples[id][1] * height for id in g2._nodes[g2._node]]
        
        if (g2._nodes is None) or (len(g2._nodes) == 0):
            return g2
        # ============================================================
        #  y-values
        # ============================================================
        if level_sort_values_by is not None:
            g2 = g2.nodes(g2._nodes.sort_values(
                by = level_sort_values_by,
                ascending = level_sort_values_by_ascending))

        g2._nodes[x_col] = [triples[id][0] * width for id in g2._nodes[g2._node]]

        if rotate is not None:
            g2 = cast(LayoutsMixin, g2).rotate(rotate)
    
        return g2


    def rotate(self, degree: float = 0):
        """
            Rotates the layout by the given angle.
        """
        g = self

        angle = math.radians(degree)

        if g._point_x is None or g._point_y is None:
            raise ValueError("No point bindings set yet for x/y")
        if g._nodes is None:
            raise ValueError("No points set yet")
        if g._point_x not in g._nodes or g._point_y not in g._nodes:
            raise ValueError(f'Did not find position columns {g._point_x} or {g._point_y} in nodes')
        
        g2 = g.nodes(
            g._nodes.assign(
                **{
                    g._point_x: g._nodes[g._point_x] * math.cos(angle) + g._nodes[g._point_y] * math.sin(angle),
                    g._point_y: -g._nodes[g._point_x] * math.sin(angle) + g._nodes[g._point_y] * math.cos(angle)
                }
            ))
        return g2


    def label_components(self):
        """
            Adds two columns with the connected component id in 'component_id' and the size of this component in 'component_size'.

        """
        g = self
        g2 = self.materialize_nodes()
        if isinstance(g._edges, pd.DataFrame):
            gg = SugiyamaLayout.graph_from_pandas(g._edges, source_column = g2._source, target_column = g2._destination)
        elif isinstance(g._edges, Graph):
            gg = g._edges
        else:
            raise TypeError
        # the component index used internally is not contingent because of the elimination algorithm used, so we renum here
        comps = {v.component: i for i, v in enumerate(gg.vertices())}
        comps_map = {u: i for i, u in enumerate(comps.values())}
        component_ids = [comps_map[comps[gg.get_vertex_from_data(id).component]] for id in g2._nodes[g2._node]]
        component_sizes = [len(list(gg.get_vertex_from_data(id).component.vertices())) for id in g2._nodes[g2._node]]
        g2._nodes['component_id'] = component_ids
        g2._nodes['component_size'] = component_sizes
        g2 = g2.nodes(g2._nodes.sort_values(
            by = "component_id",
            ascending = True))
        return g2

    @deprecated("Superseded by the layered layout implementation.")
    def deprecated_tree_layout(
            self,
            level_col: Optional[str] = None,
            descending = True,
            level_sort_values_by: Optional[Union[str, List[str]]] = None,
            level_sort_values_by_ascending: bool = True,
            level_align: str = 'left',
            aspect_ratio = True,
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

        g: Plottable = self

        if (g._edges is None) or (len(g._edges) == 0):
            return g

        x_col = g._point_x if g._point_x is not None else 'x'
        if self._point_x is None:
            g = g.bind(point_x = x_col)

        y_col = g._point_y if g._point_y is not None else 'y'
        if g._point_y is None:
            g = g.bind(point_y = y_col)

        # y
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

        # x
        if level_sort_values_by is not None:
            g2 = g2.nodes(g2._nodes.sort_values(
                by = level_sort_values_by,
                ascending = level_sort_values_by_ascending))

        grouped = g2._nodes.groupby(level_col, sort = level_sort_values_by is not None)
        if hasattr(grouped, 'cumcount'):
            g2 = g2.nodes(g2._nodes.assign(**{x_col: grouped.cumcount()}))
        else:
            try:
                # TODO remove
                # cudf 0.19 fallback
                logger.info('Tree x positions using Pandas fallback for RAPIDS < 0.21')
                import cudf
                assert isinstance(g2._nodes, cudf.DataFrame)
                xs_ps = (g2
                         ._nodes[[level_col]].to_pandas()
                         .groupby(level_col, sort = level_sort_values_by is not None)
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
            # TODO switch to grouped when above rapids 0.19 fallback gone
            nodes2_df = (g2
                    ._nodes.groupby(level_col, sort = True)
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
            g2 = g2.nodes(g2._nodes.rename(columns = {
                x_col: y_col,
                y_col: x_col
            }))
            g2._nodes[x_col] = -g2._nodes[x_col]

        return g2
