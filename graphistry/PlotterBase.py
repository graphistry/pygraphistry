from graphistry.Plottable import Plottable
from typing import Any, Callable, List, Optional, Union, TYPE_CHECKING
import copy, hashlib, logging, numpy as np, pandas as pd, pyarrow as pa, sys, uuid
from functools import lru_cache
from weakref import WeakValueDictionary

from .util import (error, in_ipython, in_databricks, make_iframe, random_string, warn)

from .bolt_util import (
    bolt_graph_to_edges_dataframe,
    bolt_graph_to_nodes_dataframe,
    node_id_key,
    start_node_id_key,
    end_node_id_key,
    to_bolt_driver)

from .arrow_uploader import ArrowUploader
from .nodexlistry import NodeXLGraphistry
from .tigeristry import Tigeristry

maybe_cudf = None
try:
    import cudf
    maybe_cudf = cudf
except ImportError:
    1

maybe_dask_dataframe = None
try:
    import dask.dataframe
    maybe_dask_dataframe = dask.dataframe
except ImportError:
    1

maybe_dask_cudf = None
try:
    import dask_cudf
    maybe_dask_cudf = dask_cudf
except ImportError:
    1

maybe_spark = None
try:
    import pyspark
    maybe_spark = pyspark
except ImportError:
    1

logger = logging.getLogger('Plotter')

CACHE_COERCION_SIZE = 100


_cache_coercion_val = None
@lru_cache(maxsize=CACHE_COERCION_SIZE)
def cache_coercion_helper(k):
    return _cache_coercion_val

def cache_coercion(k, v):
    """
        Holds references to last 100 used coercions
        Use with weak key/value dictionaries for actual lookups
    """
    global _cache_coercion_val
    _cache_coercion_val = v

    return cache_coercion_helper(k)

class WeakValueWrapper:
    def __init__(self, v):
        self.v = v

class PlotterBase(Plottable):
    """Graph plotting class.

    Created using ``Graphistry.bind()``.

    Chained calls successively add data and visual encodings, and end with a plot call.

    To streamline reuse and replayable notebooks, Plotter manipulations are immutable. Each chained call returns a new instance that derives from the previous one. The old plotter or the new one can then be used to create different graphs.

    When using memoization, for .register(api=3) sessions with .plot(memoize=True), Pandas/cudf arrow coercions are memoized, and file uploads are skipped on same-hash dataframes.

    The class supports convenience methods for mixing calls across Pandas, NetworkX, and IGraph.
    """

    _defaultNodeId = '__nodeid__'
    _pd_hash_to_arrow : WeakValueDictionary = WeakValueDictionary()
    _cudf_hash_to_arrow : WeakValueDictionary = WeakValueDictionary()

    def __init__(self, *args, **kwargs):
        super(PlotterBase, self).__init__()

        # Bindings
        self._edges : Any = None
        self._nodes : Any = None
        self._source : Optional[str] = None
        self._destination : Optional[str] = None
        self._node : Optional[str] = None
        self._edge_title : Optional[str] = None
        self._edge_label : Optional[str] = None
        self._edge_color : Optional[str] = None
        self._edge_source_color : Optional[str] = None
        self._edge_destination_color : Optional[str] = None
        self._edge_size : Optional[str] = None
        self._edge_weight : Optional[str] = None
        self._edge_icon : Optional[str] = None
        self._edge_opacity : Optional[str] = None
        self._point_title : Optional[str] = None
        self._point_label : Optional[str] = None
        self._point_color : Optional[str] = None
        self._point_size : Optional[str] = None
        self._point_weight : Optional[str] = None
        self._point_icon : Optional[str] = None
        self._point_opacity : Optional[str] = None
        self._point_x : Optional[str] = None
        self._point_y : Optional[str] = None
        # Settings
        self._height : int = 500
        self._render : bool = True
        self._url_params : dict = {'info': 'true'}
        self._privacy = None
        # Metadata
        self._name : Optional[str] = None
        self._description : Optional[str] = None
        self._style : Optional[dict] = None
        self._complex_encodings : dict = {
            'node_encodings': {'current': {}, 'default': {} },
            'edge_encodings': {'current': {}, 'default': {} }
        }
        # Integrations
        self._bolt_driver : any = None
        self._tigergraph : any = None


    def __repr__(self):
        bindings = ['edges', 'nodes', 'source', 'destination', 'node', 
                    'edge_label', 'edge_color', 'edge_size', 'edge_weight', 'edge_title', 'edge_icon', 'edge_opacity',
                    'edge_source_color', 'edge_destination_color',
                    'point_label', 'point_color', 'point_size', 'point_weight', 'point_title', 'point_icon', 'point_opacity',
                    'point_x', 'point_y']
        settings = ['height', 'url_params']

        rep = {'bindings': dict([(f, getattr(self, '_' + f)) for f in bindings]),
               'settings': dict([(f, getattr(self, '_' + f)) for f in settings])}
        if in_ipython():
            from IPython.lib.pretty import pretty
            return pretty(rep)
        else:
            return str(rep)

    def addStyle(self, fg=None, bg=None, page=None, logo=None):
        """Set general visual styles
        
        See .bind() and .settings(url_params={}) for additional styling options, and style() for another way to set the same attributes.

        To facilitate reuse and replayable notebooks, the addStyle() call is chainable. Invocation does not effect the old style: it instead returns a new Plotter instance with the new styles added to the existing ones. Both the old and new styles can then be used for different graphs.

        addStyle() will extend the existing style settings, while style() will replace any in the same group

        :param fg: Dictionary {'blendMode': str} of any valid CSS blend mode
        :type fg: dict

        :param bg: Nested dictionary of page background properties. {'color': str, 'gradient': {'kind': str, 'position': str, 'stops': list }, 'image': { 'url': str, 'width': int, 'height': int, 'blendMode': str }
        :type bg: dict

        :param logo: Nested dictionary of logo properties. { 'url': str, 'autoInvert': bool, 'position': str, 'dimensions': { 'maxWidth': int, 'maxHeight': int }, 'crop': { 'top': int, 'left': int, 'bottom': int, 'right': int }, 'padding': { 'top': int, 'left': int, 'bottom': int, 'right': int}, 'style': str}        
        :type logo: dict

        :param page: Dictionary of page metadata settings. { 'favicon': str, 'title': str } 
        :type page: dict

        :returns: Plotter
        :rtype: Plotter

        **Example: Chained merge - results in color, blendMode, and url being set**
            ::

                g2 =  g.addStyle(bg={'color': 'black'}, fg={'blendMode': 'screen'})
                g3 = g2.addStyle(bg={'image': {'url': 'http://site.com/watermark.png'}})
                
        **Example: Overwrite - results in blendMode multiply**
            ::

                g2 =  g.addStyle(fg={'blendMode': 'screen'})
                g3 = g2.addStyle(fg={'blendMode': 'multiply'})

        **Example: Gradient background**
            ::

              g.addStyle(bg={'gradient': {'kind': 'linear', 'position': 45, 'stops': [['rgb(0,0,0)', '0%'], ['rgb(255,255,255)', '100%']]}})
              
        **Example: Page settings**
            ::

              g.addStyle(page={'title': 'Site - {{ name }}', 'favicon': 'http://site.com/logo.ico'})

        """
        style = copy.deepcopy(self._style or {})
        o = {'fg': fg, 'bg': bg, 'page': page, 'logo': logo}
        for k, v in o.items():
            if not (v is None):
                if isinstance(v, dict):
                    if not (k in style) or (style[k] is None):
                        style[k] = {}
                    for k2, v2 in v.items():
                        style[k][k2] = v2
                else:
                    style[k] = v
        res = self.bind()
        res._style = style
        return res
        


    def style(self, fg=None, bg=None, page=None, logo=None):
        """Set general visual styles
        
        See .bind() and .settings(url_params={}) for additional styling options, and addStyle() for another way to set the same attributes.

        To facilitate reuse and replayable notebooks, the style() call is chainable. Invocation does not effect the old style: it instead returns a new Plotter instance with the new styles added to the existing ones. Both the old and new styles can then be used for different graphs.

        style() will fully replace any defined parameter in the existing style settings, while addStyle() will merge over previous values

        :param fg: Dictionary {'blendMode': str} of any valid CSS blend mode
        :type fg: dict

        :param bg: Nested dictionary of page background properties. {'color': str, 'gradient': {'kind': str, 'position': str, 'stops': list }, 'image': { 'url': str, 'width': int, 'height': int, 'blendMode': str }
        :type bg: dict

        :param logo: Nested dictionary of logo properties. { 'url': str, 'autoInvert': bool, 'position': str, 'dimensions': { 'maxWidth': int, 'maxHeight': int }, 'crop': { 'top': int, 'left': int, 'bottom': int, 'right': int }, 'padding': { 'top': int, 'left': int, 'bottom': int, 'right': int}, 'style': str}        
        :type logo: dict

        :param page: Dictionary of page metadata settings. { 'favicon': str, 'title': str } 
        :type page: dict

        :returns: Plotter
        :rtype: Plotter

        **Example: Chained merge - results in url and blendMode being set, while color is dropped**
            ::
                g2 =  g.style(bg={'color': 'black'}, fg={'blendMode': 'screen'})
                g3 = g2.style(bg={'image': {'url': 'http://site.com/watermark.png'}})
                
        **Example: Gradient background**
            ::
              g.style(bg={'gradient': {'kind': 'linear', 'position': 45, 'stops': [['rgb(0,0,0)', '0%'], ['rgb(255,255,255)', '100%']]}})
              
        **Example: Page settings**
            ::
              g.style(page={'title': 'Site - {{ name }}', 'favicon': 'http://site.com/logo.ico'})

        """        
        style = copy.deepcopy(self._style or {})
        o = {'fg': fg, 'bg': bg, 'page': page, 'logo': logo}
        for k, v in o.items():
            if not (v is None):
                style[k] = v
        res = self.bind()
        res._style = style
        return res


    def encode_point_color(self, column,
            palette=None, as_categorical=None, as_continuous=None, categorical_mapping=None, default_mapping=None,
            for_default=True, for_current=False):
        """Set point color with more control than bind()

        :param column: Data column name
        :type column: str

        :param palette: Optional list of color-like strings. Ex: ["black, "#FF0", "rgb(255,255,255)" ]. Used as a gradient for continuous and round-robin for categorical.
        :type palette: Optional[list]

        :param as_categorical: Interpret column values as categorical. Ex: Uses palette via round-robin when more values than palette entries.
        :type as_categorical: Optional[bool]

        :param as_continuous: Interpret column values as continuous. Ex: Uses palette for an interpolation gradient when more values than palette entries.
        :type as_continuous: Optional[bool]

        :param categorical_mapping: Mapping from column values to color-like strings. Ex: {"car": "red", "truck": #000"}
        :type categorical_mapping: Optional[dict]

        :param default_mapping: Augment categorical_mapping with mapping for values not in categorical_mapping. Ex: default_mapping="gray".
        :type default_mapping: Optional[str]

        :param for_default: Use encoding for when no user override is set. Default on.
        :type for_default: Optional[bool]

        :param for_current: Use encoding as currently active. Clearing the active encoding resets it to default, which may be different. Default on.
        :type for_current: Optional[bool]

        :returns: Plotter
        :rtype: Plotter

        **Example: Set a palette-valued column for the color, same as bind(point_color='my_column')**
            ::

                g2a = g.encode_point_color('my_int32_palette_column')
                g2b = g.encode_point_color('my_int64_rgb_column')

        **Example: Set a cold-to-hot gradient of along the spectrum blue, yellow, red**
            ::

                g2 = g.encode_point_color('my_numeric_col', palette=["blue", "yellow", "red"], as_continuous=True)

        **Example: Round-robin sample from 5 colors in hex format**
            ::

                g2 = g.encode_point_color('my_distinctly_valued_col', palette=["#000", "#00F", "#0F0", "#0FF", "#FFF"], as_categorical=True)

        **Example: Map specific values to specific colors, including with a default**
            ::

                g2a = g.encode_point_color('brands', categorical_mapping={'toyota': 'red', 'ford': 'blue'})
                g2a = g.encode_point_color('brands', categorical_mapping={'toyota': 'red', 'ford': 'blue'}, default_mapping='gray')

        """
        return self.__encode('point', 'color', 'pointColorEncoding',
            column=column, palette=palette, as_categorical=as_categorical, as_continuous=as_continuous,
            categorical_mapping=categorical_mapping, default_mapping=default_mapping,
            for_default=for_default, for_current=for_current)


    def encode_edge_color(self, column,
            palette=None, as_categorical=None, as_continuous=None, categorical_mapping=None, default_mapping=None,
            for_default=True, for_current=False):
        """Set edge color with more control than bind()

        :param column: Data column name
        :type column: str

        :param palette: Optional list of color-like strings. Ex: ["black, "#FF0", "rgb(255,255,255)" ]. Used as a gradient for continuous and round-robin for categorical.
        :type palette: Optional[list]

        :param as_categorical: Interpret column values as categorical. Ex: Uses palette via round-robin when more values than palette entries.
        :type as_categorical: Optional[bool]

        :param as_continuous: Interpret column values as continuous. Ex: Uses palette for an interpolation gradient when more values than palette entries.
        :type as_continuous: Optional[bool]

        :param categorical_mapping: Mapping from column values to color-like strings. Ex: {"car": "red", "truck": #000"}
        :type categorical_mapping: Optional[dict]

        :param default_mapping: Augment categorical_mapping with mapping for values not in categorical_mapping. Ex: default_mapping="gray".
        :type default_mapping: Optional[str]

        :param for_default: Use encoding for when no user override is set. Default on.
        :type for_default: Optional[bool]

        :param for_current: Use encoding as currently active. Clearing the active encoding resets it to default, which may be different. Default on.
        :type for_current: Optional[bool]

        :returns: Plotter
        :rtype: Plotter

        **Example: See encode_point_color**
        """

        return self.__encode('edge', 'color', 'edgeColorEncoding',
            column=column, palette=palette, as_categorical=as_categorical, as_continuous=as_continuous,
            categorical_mapping=categorical_mapping, default_mapping=default_mapping,
            for_default=for_default, for_current=for_current)

    def encode_point_size(self, column,
            categorical_mapping=None, default_mapping=None,
            for_default=True, for_current=False):
        """Set point size with more control than bind()

        :param column: Data column name
        :type column: str

        :param categorical_mapping: Mapping from column values to numbers. Ex: {"car": 100, "truck": 200}
        :type categorical_mapping: Optional[dict]

        :param default_mapping: Augment categorical_mapping with mapping for values not in categorical_mapping. Ex: default_mapping=50.
        :type default_mapping: Optional[Union[int,float]]

        :param for_default: Use encoding for when no user override is set. Default on.
        :type for_default: Optional[bool]

        :param for_current: Use encoding as currently active. Clearing the active encoding resets it to default, which may be different. Default on.
        :type for_current: Optional[bool]

        :returns: Plotter
        :rtype: Plotter

        **Example: Set a numerically-valued column for the size, same as bind(point_size='my_column')**
            ::

                g2a = g.encode_point_size('my_numeric_column')

        **Example: Map specific values to specific colors, including with a default**
            ::

                g2a = g.encode_point_size('brands', categorical_mapping={'toyota': 100, 'ford': 200})
                g2b = g.encode_point_size('brands', categorical_mapping={'toyota': 100, 'ford': 200}, default_mapping=50)

        """
        return self.__encode('point', 'size', 'pointSizeEncoding', column=column,
            categorical_mapping=categorical_mapping, default_mapping=default_mapping,
            for_default=for_default, for_current=for_current)


    def encode_point_icon(self, column,
            categorical_mapping=None, continuous_binning=None, default_mapping=None,
            comparator=None,
            for_default=True, for_current=False,
            as_text=False, blend_mode=None, style=None, border=None, shape=None):
        """Set node icon with more control than bind().
        Values from Font Awesome 4 such as "laptop": https://fontawesome.com/v4.7.0/icons/ , image URLs (http://...), and data URIs (data:...).
        When as_text=True is enabled, values are instead interpreted as raw strings.

        :param column: Data column name
        :type column: str

        :param categorical_mapping: Mapping from column values to icon name strings. Ex: {"toyota": 'car', "ford": 'truck'}
        :type categorical_mapping: Optional[dict]

        :param default_mapping: Augment categorical_mapping with mapping for values not in categorical_mapping. Ex: default_mapping=50.
        :type default_mapping: Optional[Union[int,float]]

        :param for_default: Use encoding for when no user override is set. Default on.
        :type for_default: Optional[bool]

        :param for_current: Use encoding as currently active. Clearing the active encoding resets it to default, which may be different. Default on.
        :type for_current: Optional[bool]

        :param as_text: Values should instead be treated as raw strings, instead of icons and images. (Default False.)
        :type as_text: Optional[bool]

        :param blend_mode: CSS blend mode
        :type blend_mode: Optional[str]

        :param style: CSS filter properties - opacity, saturation, luminosity, grayscale, and more
        :type style: Optional[dict]

        :param border: Border properties - 'width', 'color', and 'storke'
        :type border: Optional[dict]

        :returns: Plotter
        :rtype: Plotter

        **Example: Set a string column of icons for the point icons, same as bind(point_icon='my_column')**
            ::

                g2a = g.encode_point_icon('my_icons_column')

        **Example: Map specific values to specific icons, including with a default**
            ::

                g2a = g.encode_point_icon('brands', categorical_mapping={'toyota': 'car', 'ford': 'truck'})
                g2b = g.encode_point_icon('brands', categorical_mapping={'toyota': 'car', 'ford': 'truck'}, default_mapping='question')

        **Example: Map countries to abbreviations**
            ::

                g2b = g.encode_point_icon('country_abbrev', as_text=True)
                g2b = g.encode_point_icon('country', as_text=True, categorical_mapping={'England': 'UK', 'America': 'US'}, default_mapping='')

        **Example: Border**
            ::

                g2b = g.encode_point_icon('country', border={'width': 3, color: 'black', 'stroke': 'dashed'}, 'categorical_mapping={'England': 'UK', 'America': 'US'})

        """

        return self.__encode('point', 'icon', 'pointIconEncoding', column=column,
            categorical_mapping=categorical_mapping, continuous_binning=continuous_binning, default_mapping=default_mapping,
            comparator=comparator,
            for_default=for_default, for_current=for_current,
            as_text=as_text, blend_mode=blend_mode, style=style, border=border, shape=shape)

    def encode_edge_icon(self, column,
            categorical_mapping=None, continuous_binning=None, default_mapping=None,
            comparator=None,
            for_default=True, for_current=False,
            as_text=False, blend_mode=None, style=None, border=None, shape=None):
        """Set edge icon with more control than bind()
        Values from Font Awesome 4 such as "laptop": https://fontawesome.com/v4.7.0/icons/ , image URLs (http://...), and data URIs (data:...).
        When as_text=True is enabled, values are instead interpreted as raw strings.

        :param column: Data column name
        :type column: str

        :param categorical_mapping: Mapping from column values to icon name strings. Ex: {"toyota": 'car', "ford": 'truck'}
        :type categorical_mapping: Optional[dict]

        :param default_mapping: Augment categorical_mapping with mapping for values not in categorical_mapping. Ex: default_mapping=50.
        :type default_mapping: Optional[Union[int,float]]

        :param for_default: Use encoding for when no user override is set. Default on.
        :type for_default: Optional[bool]

        :param for_current: Use encoding as currently active. Clearing the active encoding resets it to default, which may be different. Default on.
        :type for_current: Optional[bool]

        :param as_text: Values should instead be treated as raw strings, instead of icons and images. (Default False.)
        :type as_text: Optional[bool]

        :returns: Plotter
        :rtype: Plotter

        **Example: Set a string column of icons for the edge icons, same as bind(edge_icon='my_column')**
            ::

                g2a = g.encode_edge_icon('my_icons_column')

        **Example: Map specific values to specific icons, including with a default**
            ::

                g2a = g.encode_edge_icon('brands', categorical_mapping={'toyota': 'car', 'ford': 'truck'})
                g2b = g.encode_edge_icon('brands', categorical_mapping={'toyota': 'car', 'ford': 'truck'}, default_mapping='question')

        **Example: Map countries to abbreviations**
            ::

                g2a = g.encode_edge_icon('country_abbrev', as_text=True)
                g2b = g.encode_edge_icon('country', as_text=True, categorical_mapping={'England': 'UK', 'America': 'US'}, default_mapping='')

        **Example: Border**
            ::

                g2b = g.encode_edge_icon('country', border={'width': 3, color: 'black', 'stroke': 'dashed'}, 'categorical_mapping={'England': 'UK', 'America': 'US'})

        """
        return self.__encode('edge', 'icon', 'edgeIconEncoding', column=column,
            categorical_mapping=categorical_mapping, continuous_binning=continuous_binning, default_mapping=default_mapping,
            comparator=comparator,
            for_default=for_default, for_current=for_current,
            as_text=as_text, blend_mode=blend_mode, style=style, border=border, shape=shape)


    def encode_point_badge(self, column, position='TopRight',
            categorical_mapping=None, continuous_binning=None, default_mapping=None, comparator=None,
            color=None, bg=None, fg=None,
            for_current=False, for_default=True,
            as_text=None, blend_mode=None, style=None, border=None, shape=None):

        return self.__encode_badge('point', column, position,
            categorical_mapping=categorical_mapping, continuous_binning=continuous_binning, default_mapping=default_mapping, comparator=comparator,
            color=color, bg=bg, fg=fg,
            for_current=for_current, for_default=for_default,
            as_text=as_text, blend_mode=blend_mode, style=style, border=border, shape=shape)


    def encode_edge_badge(self, column, position='TopRight',
            categorical_mapping=None, continuous_binning=None, default_mapping=None, comparator=None,
            color=None, bg=None, fg=None,
            for_current=False, for_default=True,
            as_text=None, blend_mode=None, style=None, border=None, shape=None):

        return self.__encode_badge('edge', column, position,
            categorical_mapping=categorical_mapping, continuous_binning=continuous_binning, default_mapping=default_mapping, comparator=comparator,
            color=color, bg=bg, fg=fg,
            for_current=for_current, for_default=for_default,
            as_text=as_text, blend_mode=blend_mode, style=style, border=border, shape=shape)

    def __encode_badge(self, graph_type, column, position='TopRight',
            categorical_mapping=None, continuous_binning=None, default_mapping=None, comparator=None,
            color=None, bg=None, fg=None,
            for_current=False, for_default=True,
            as_text=None, blend_mode=None, style=None, border=None, shape=None):

        return self.__encode(graph_type, f'badge{position}', f'{graph_type}Badge{position}Encoding',
            column,
            as_categorical=not (categorical_mapping is None),
            as_continuous=not (continuous_binning is None),
            categorical_mapping=categorical_mapping,
            default_mapping=default_mapping,
            for_current=for_current, for_default=for_default,
            as_text=as_text, blend_mode=blend_mode, style=style, border=border,
            continuous_binning=continuous_binning,
            comparator=comparator,
            color=color, bg=bg, fg=fg, shape=shape)


    def __encode(self, graph_type, feature, feature_binding,  # noqa: C901
            column,
            palette=None,
            as_categorical=None, as_continuous=None,
            categorical_mapping=None, default_mapping=None,
            for_default=True, for_current=False,
            as_text=None, blend_mode=None, style=None, border=None,
            continuous_binning=None, comparator=None,
            color=None, bg=None, fg=None, dimensions=None, shape=None):

        if for_default is None:
            for_default = True
        if for_current is None:
            for_current = False

        #TODO check set to api=3?

        if not (graph_type in ['point', 'edge']):
            raise ValueError({
                'message': 'graph_type must be "point" or "edge"',
                'data': {'graph_type': graph_type } })

        if (categorical_mapping is None) and (palette is None) and (continuous_binning is None) and not feature.startswith('badge'):
            return self.bind(**{f'{graph_type}_{feature}': column})

        transform = None
        if not (categorical_mapping is None):
            if not (isinstance(categorical_mapping, dict)):
                raise ValueError({
                    'message': 'categorical mapping should be a dict mapping column names to values',
                    'data': { 'categorical_mapping': categorical_mapping, 'type': str(type(categorical_mapping)) }})
            transform = {
                'variation': 'categorical',
                'mapping': {
                    'categorical': {
                        'fixed': categorical_mapping,
                        **({} if default_mapping is None else {'other': default_mapping})
                    }
                }
            }
        elif not (palette is None):

            #TODO ensure that it is a color? Unclear behavior for sizes, weights, etc.

            if not (isinstance(palette, list)) or not all([isinstance(x, str) for x in palette]):
                raise ValueError({
                    'message': 'palette should be a list of color-like strings: ["#FFFFFF", "white", ...]',
                    'data': { 'palette': palette, 'type': str(type(palette)) }})

            is_categorical = None
            if not (as_categorical is None):
                is_categorical = as_categorical
            elif not (as_continuous is None):
                is_categorical = not as_continuous
            else:
                raise ValueError({'message': 'Must pass in at least one of as_categorical, as_continuous, or categorical_mapping'})

            transform = {
                'variation': 'categorical' if is_categorical else 'continuous',
                'colors': palette
            }
        elif not (continuous_binning is None):
            if not (isinstance(continuous_binning, list)):
                raise ValueError({
                    'message': 'continous_binning should be a list of [comparable or None, mapped_value]',
                    'data': { 'continuous_binning': continuous_binning, 'type': str(type(continuous_binning)) }})

            if as_categorical:
                raise ValueError({'message': 'as_categorical cannot be True when continuous_binning is provided'})
            if as_continuous is False:
                raise ValueError({'message': 'as_continuous cannot be False when continuous_binning is set'})

            transform = {
                'variation': 'continuous',
                'mapping': {
                    'continuous': {
                        'bins': continuous_binning,
                        **({} if comparator is None else {'comparator': comparator}),
                        **({} if default_mapping is None else {'other': default_mapping})
                    }
                }
            }
        elif feature.startswith('badge'):
            transform = {'variation': 'categorical'}
        else:
            raise ValueError({'message': 'Must pass one of parameters palette or categorical_mapping'})

        encoding = {
            'graphType': graph_type,
            'encodingType': feature,
            'attribute': column,
            **transform,
            **({'bg':        bg} if not         (bg is None) else {}),  # noqa: E241,E271
            **({'color':     color} if not      (color is None) else {}),  # noqa: E241,E271
            **({'fg':        fg} if not         (fg is None) else {}),  # noqa: E241,E271
            **({'asText':    as_text} if not    (as_text is None) else {}),  # noqa: E241,E271
            **({'blendMode': blend_mode} if not (blend_mode is None) else {}),  # noqa: E241,E271
            **({'style':     style} if not      (style is None) else {}),  # noqa: E241,E271
            **({'border':    border} if not     (border is None) else {}),  # noqa: E241,E271
            **({'shape':     shape} if not      (shape is None) else {})  # noqa: E241,E271
        }

        complex_encodings = copy.deepcopy(self._complex_encodings)

        #point -> node
        graph_type_2 = 'node' if graph_type == 'point' else graph_type

        #NOTE: parameter feature_binding for cases like Legend
        if for_current:
            complex_encodings[f'{graph_type_2}_encodings']['current'][feature_binding] = encoding
        if for_default:
            complex_encodings[f'{graph_type_2}_encodings']['default'][feature_binding] = encoding

        res = copy.copy(self)
        res._complex_encodings = complex_encodings
        return res


    def bind(self, source=None, destination=None, node=None,
             edge_title=None, edge_label=None, edge_color=None, edge_weight=None, edge_size=None, edge_opacity=None, edge_icon=None,
             edge_source_color=None, edge_destination_color=None,
             point_title=None, point_label=None, point_color=None, point_weight=None, point_size=None, point_opacity=None, point_icon=None,
             point_x=None, point_y=None):
        """Relate data attributes to graph structure and visual representation.

        To facilitate reuse and replayable notebooks, the binding call is chainable. Invocation does not effect the old binding: it instead returns a new Plotter instance with the new bindings added to the existing ones. Both the old and new bindings can then be used for different graphs.


        :param source: Attribute containing an edge's source ID
        :type source: str

        :param destination: Attribute containing an edge's destination ID
        :type destination: str

        :param node: Attribute containing a node's ID
        :type node: str

        :param edge_title: Attribute overriding edge's minimized label text. By default, the edge source and destination is used.
        :type edge_title: str

        :param edge_label: Attribute overriding edge's expanded label text. By default, scrollable list of attribute/value mappings.
        :type edge_label: str

        :param edge_color: Attribute overriding edge's color. rgba (int64) or int32 palette index, see palette definitions <https://graphistry.github.io/docs/legacy/api/0.9.2/api.html#extendedpalette>`_ for values. Based on Color Brewer.
        :type edge_color: str

        :param edge_source_color: Attribute overriding edge's source color if no edge_color, as an rgba int64 value.
        :type edge_source_color: str

        :param edge_destination_color: Attribute overriding edge's destination color if no edge_color, as an rgba int64 value.
        :type edge_destination_color: str

        :param edge_weight: Attribute overriding edge weight. Default is 1. Advanced layout controls will relayout edges based on this value.
        :type edge_weight: str

        :param point_title: Attribute overriding node's minimized label text. By default, the node ID is used.
        :type point_title: str

        :param point_label: Attribute overriding node's expanded label text. By default, scrollable list of attribute/value mappings.
        :type point_label: str

        :param point_color: Attribute overriding node's color.rgba (int64) or int32 palette index, see palette definitions <https://graphistry.github.io/docs/legacy/api/0.9.2/api.html#extendedpalette>`_ for values. Based on Color Brewer.
        :type point_color: str

        :param point_size: Attribute overriding node's size. By default, uses the node degree. The visualization will normalize point sizes and adjust dynamically using semantic zoom.
        :type point_size: str

        :param point_x: Attribute overriding node's initial x position. Combine with ".settings(url_params={'play': 0}))" to create a custom layout
        :type point_x: str

        :param point_y: Attribute overriding node's initial y position. Combine with ".settings(url_params={'play': 0}))" to create a custom layout
        :type point_y: str

        :returns: Plotter
        :rtype: Plotter

        **Example: Minimal**
            ::

                import graphistry
                g = graphistry.bind()
                g = g.bind(source='src', destination='dst')

        **Example: Node colors**
            ::

                import graphistry
                g = graphistry.bind()
                g = g.bind(source='src', destination='dst',
                           node='id', point_color='color')

        **Example: Chaining**
            ::

                import graphistry
                g = graphistry.bind(source='src', destination='dst', node='id')

                g1 = g.bind(point_color='color1', point_size='size1')

                g.bind(point_color='color1b')

                g2a = g1.bind(point_color='color2a')
                g2b = g1.bind(point_color='color2b', point_size='size2b')

                g3a = g2a.bind(point_size='size3a')
                g3b = g2b.bind(point_size='size3b')

        In the above **Chaining** example, all bindings use src/dst/id. Colors and sizes bind to:
            ::

                g: default/default
                g1: color1/size1
                g2a: color2a/size1
                g2b: color2b/size2b
                g3a: color2a/size3a
                g3b: color2b/size3b


        """
        res = copy.copy(self)
        res._source = source or self._source
        res._destination = destination or self._destination
        res._node = node or self._node

        res._edge_title = edge_title or self._edge_title
        res._edge_label = edge_label or self._edge_label
        res._edge_color = edge_color or self._edge_color
        res._edge_source_color = edge_source_color or self._edge_source_color
        res._edge_destination_color = edge_destination_color or self._edge_destination_color
        res._edge_size = edge_size or self._edge_size
        res._edge_weight = edge_weight or self._edge_weight
        res._edge_icon = edge_icon or self._edge_icon
        res._edge_opacity = edge_opacity or self._edge_opacity

        res._point_title = point_title or self._point_title
        res._point_label = point_label or self._point_label
        res._point_color = point_color or self._point_color
        res._point_size = point_size or self._point_size
        res._point_weight = point_weight or self._point_weight
        res._point_opacity = point_opacity or self._point_opacity
        res._point_icon = point_icon or self._point_icon
        res._point_x = point_x or self._point_x
        res._point_y = point_y or self._point_y
        
        return res


    def nodes(self, nodes: Union[Callable, Any], node=None, *args, **kwargs) -> Plottable:
        """Specify the set of nodes and associated data.
        If a callable, will be called with current Plotter and whatever positional+named arguments

        Must include any nodes referenced in the edge list.

        :param nodes: Nodes and their attributes.
        :type nodes: Pandas dataframe or Callable

        :returns: Plotter
        :rtype: Plotter

        **Example**
            ::

                import graphistry

                es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                g = graphistry
                    .bind(source='src', destination='dst')
                    .edges(es)

                vs = pandas.DataFrame({'v': [0,1,2], 'lbl': ['a', 'b', 'c']})
                g = g.bind(node='v').nodes(vs)

                g.plot()

        **Example**
            ::

                import graphistry

                es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                g = graphistry.edges(es, 'src', 'dst')

                vs = pandas.DataFrame({'v': [0,1,2], 'lbl': ['a', 'b', 'c']})
                g = g.nodes(vs, 'v)

                g.plot()


        **Example**
            ::
                import graphistry

                def sample_nodes(g, n):
                    return g._nodes.sample(n)

                df = pandas.DataFrame({'id': [0,1,2], 'v': [1,2,0]})

                graphistry
                    .nodes(df, 'id')
                    ..nodes(sample_nodes, n=2)
                    ..nodes(sample_nodes, None, 2)  # equivalent
                    .plot()

        """

        base = self.bind(node=node) if node is not None else self
        if callable(nodes):
            nodes2 = nodes(base, *args, **kwargs)
            res = base.nodes(nodes2)
        else:
            res = copy.copy(base)
            res._nodes = nodes
        return res

    def name(self, name):
        """Upload name

        :param name: Upload name
        :type name: str"""

        res = copy.copy(self)
        res._name = name
        return res

    def description(self, description):
        """Upload description

        :param description: Upload description
        :type description: str"""

        res = copy.copy(self)
        res._description = description
        return res


    def edges(self, edges: Union[Callable, Any], source=None, destination=None, *args, **kwargs) -> Plottable:
        """Specify edge list data and associated edge attribute values.
        If a callable, will be called with current Plotter and whatever positional+named arguments

        :param edges: Edges and their attributes, or transform from Plotter to edges
        :type edges: Pandas dataframe, NetworkX graph, or IGraph graph.

        :returns: Plotter
        :rtype: Plotter

        **Example**
            ::

                import graphistry
                df = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                graphistry
                    .bind(source='src', destination='dst')
                    .edges(df)
                    .plot()

        **Example**
            ::

                import graphistry
                df = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                graphistry
                    .edges(df, 'src', 'dst')
                    .plot()

        **Example**
            ::
                import graphistry

                def sample_edges(g, n):
                    return g._edges.sample(n)

                df = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})

                graphistry
                    .edges(df, 'src', 'dst')
                    .edges(sample_edges, n=2)
                    .edges(sample_edges, None, None, 2)  # equivalent
                    .plot()

        """

        base = self

        if not (source is None):
            base = base.bind(source=source)
        if not (destination is None):
            base = base.bind(destination=destination)

        if callable(edges):
            edges2 = edges(base, *args, **kwargs)
            res = base.edges(edges2)
        else:
            res = copy.copy(base)
            res._edges = edges
        return res

    def pipe(self, graph_transform: Callable, *args, **kwargs) -> Plottable:
        """Create new Plotter derived from current

        :param graph_transform:
        :type graph_transform: Callable

        **Example: Simple**
            ::

                import graphistry

                def fill_missing_bindings(g, source='src', destination='dst):
                    return g.bind(source=source, destination=destination)

                graphistry
                    .edges(pandas.DataFrame({'src': [0,1,2], 'd': [1,2,0]}))
                    .pipe(fill_missing_bindings, destination='d')  # binds 'src'
                    .plot()
        """

        return graph_transform(self, *args, **kwargs)

    def graph(self, ig):
        """Specify the node and edge data.

        :param ig: NetworkX graph or an IGraph graph with node and edge attributes.
        :type ig: Any

        :returns: Plotter
        :rtype: Plotter
        """

        res = copy.copy(self)
        res._edges = ig
        res._nodes = None
        return res


    def settings(self, height=None, url_params={}, render=None):
        """Specify iframe height and add URL parameter dictionary.

        The library takes care of URI component encoding for the dictionary.

        :param height: Height in pixels.
        :type height: int

        :param url_params: Dictionary of querystring parameters to append to the URL.
        :type url_params: dict

        :param render: Whether to render the visualization using the native notebook environment (default True), or return the visualization URL
        :type render: bool

        """

        res = copy.copy(self)
        res._height = height or self._height
        res._url_params = dict(self._url_params, **url_params)
        res._render = self._render if render is None else render
        return res


    def privacy(self, mode: Optional[str] = None, notify: Optional[bool] = None, invited_users: Optional[List] = None, message: Optional[str] = None):
        """Set local sharing mode

        :param mode: Either "private", "public", or inherit from global privacy()
        :type mode: Optional[str]
        :param notify: Whether to email the recipient(s) upon upload, defaults to global privacy()
        :type notify: Optional[bool]
        :param invited_users: List of recipients, where each is {"email": str, "action": str} and action is "10" (view) or "20" (edit), defaults to global privacy()
        :type invited_users: Optional[List]
        :param message: Email to send when notify=True
        :type message': Optioanl[str]

        Requires an account with sharing capabilities.

        Shared datasets will appear in recipients' galleries.

        If mode is set to "private", only accounts in invited_users list can access. Mode "public" permits viewing by any user with the URL.

        Action "10" (view) gives read access, while action "20" (edit) gives edit access, like changing the sharing mode.

        When notify is true, uploads will trigger notification emails to invitees. Email will use visualization's ".name()"

        When settings are not specified, they are inherited from the global graphistry.privacy() defaults

        **Example: Limit visualizations to current user**

            ::

                import graphistry
                graphistry.register(api=3, username='myuser', password='mypassword')
                
                #Subsequent uploads default to using .privacy() settings
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, direct=True)
                g = h['graph']
                g = g.privacy()  # default uploads to mode="private"
                g.plot()


        **Example: Default to publicly viewable visualizations**

            ::

                import graphistry
                graphistry.register(api=3, username='myuser', password='mypassword')
                
                #Subsequent uploads default to using .privacy() settings
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, direct=True)
                g = h['graph']
                #g = g.privacy(mode="public")  # can skip calling .privacy() for this default
                g.plot()


        **Example: Default to sharing with select teammates, and keep notifications opt-in**

            ::

                import graphistry
                graphistry.register(api=3, username='myuser', password='mypassword')
                
                #Subsequent uploads default to using .privacy() settings
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, direct=True)
                g = h['graph']
                g = g.privacy(
                    mode="private",
                    invited_users=[
                        {"email": "friend1@acme.org", "action": "10"}, # view
                        {"email": "friend2@acme.org", "action": "20"}, # edit
                    ],
                    notify=False)
                g.plot()


        **Example: Keep visualizations public and email notifications upon upload**

            ::

                import graphistry
                graphistry.register(api=3, username='myuser', password='mypassword')
                
                #Subsequent uploads default to using .privacy() settings
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, direct=True)
                g = h['graph']
                g = g.name('my cool viz')  # For friendlier invitations
                g = g.privacy(
                    mode="public",
                    invited_users=[
                        {"email": "friend1@acme.org", "action": "10"}, # view
                        {"email": "friend2@acme.org", "action": "20"}, # edit
                    ],
                    notify=True)
                g.plot()
        """

        res = copy.copy(self)
        res._privacy = copy.copy(self._privacy or {})
        if mode is not None:
            res._privacy['mode'] = mode
        if notify is not None:
            res._privacy['notify'] = notify
        if invited_users is not None:
            res._privacy['invited_users'] = invited_users
        if message is not None:
            res._privacy['message'] = message
        return res


    def plot(
        self, graph=None, nodes=None, name=None, description=None, render=None, skip_upload=False, as_files=False, memoize=True,
        extra_html="", override_html_style=None
    ):  # noqa: C901
        """Upload data to the Graphistry server and show as an iframe of it.

        Uses the currently bound schema structure and visual encodings.
        Optional parameters override the current bindings.

        When used in a notebook environment, will also show an iframe of the visualization.

        :param graph: Edge table (pandas, arrow, cudf) or graph (NetworkX, IGraph).
        :type graph: Any

        :param nodes: Nodes table (pandas, arrow, cudf)
        :type nodes: Any

        :param name: Upload name.
        :type name: str

        :param description: Upload description.
        :type description: str

        :param render: Whether to render the visualization using the native notebook environment (default True), or return the visualization URL
        :type render: bool

        :param skip_upload: Return node/edge/bindings that would have been uploaded. By default, upload happens.
        :type skip_upload: bool

        :param as_files: Upload distinct node/edge files under the managed Files PI. Default off, will switch to default-on when stable.
        :type as_files: bool

        :param memoize: Tries to memoize pandas/cudf->arrow conversion, including skipping upload. Default on.
        :type memoize: bool

        :param extra_html: Allow injecting arbitrary HTML into the visualization iframe.
        :type extra_html: Optional[str]

        :param override_html_style: Set fully custom style tag.
        :type override_html_style: Optional[str]

        **Example: Simple**
            ::

                import graphistry
                es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                graphistry
                    .bind(source='src', destination='dst')
                    .edges(es)
                    .plot()

        **Example: Shorthand**
            ::

                import graphistry
                es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                graphistry
                    .bind(source='src', destination='dst')
                    .plot(es)

        """

        if graph is None:
            if self._edges is None:
                error('Graph/edges must be specified.')
            g = self._edges
        else:
            g = graph
        n = self._nodes if nodes is None else nodes
        name = name or self._name or ("Untitled " + random_string(10))
        description = description or self._description or ("")

        self._check_mandatory_bindings(not isinstance(n, type(None)))

        from .pygraphistry import PyGraphistry
        api_version = PyGraphistry.api_version()
        if api_version == 1:
            dataset = self._plot_dispatch(g, n, name, description, 'json', self._style, memoize)
            if skip_upload:
                return dataset
            info = PyGraphistry._etl1(dataset)
        elif api_version == 2:
            dataset = self._plot_dispatch(g, n, name, description, 'vgraph', self._style, memoize)
            if skip_upload:
                return dataset
            info = PyGraphistry._etl2(dataset)
        elif api_version == 3:
            PyGraphistry.refresh()
            dataset = self._plot_dispatch(g, n, name, description, 'arrow', self._style, memoize)
            if skip_upload:
                return dataset
            dataset.token = PyGraphistry.api_token()
            dataset.post(as_files=as_files, memoize=memoize)
            dataset.maybe_post_share_link(self)
            info = {
                'name': dataset.dataset_id,
                'type': 'arrow',
                'viztoken': str(uuid.uuid4())
            }

        viz_url = PyGraphistry._viz_url(info, self._url_params)
        cfg_client_protocol_hostname = PyGraphistry._config['client_protocol_hostname']
        full_url = ('%s:%s' % (PyGraphistry._config['protocol'], viz_url)) if cfg_client_protocol_hostname is None else viz_url

        if (render is False) or ((render is None) and not self._render):
            return full_url
        elif (render is True) or in_ipython():
            from IPython.core.display import HTML
            return HTML(make_iframe(full_url, self._height, extra_html=extra_html, override_html_style=override_html_style))
        elif in_databricks():
            return make_iframe(full_url, self._height, extra_html=extra_html, override_html_style=override_html_style)
        else:
            import webbrowser
            webbrowser.open(full_url)
            return full_url


    def pandas2igraph(self, edges, directed=True):
        """Convert a pandas edge dataframe to an IGraph graph.

        Uses current bindings. Defaults to treating edges as directed.

        **Example**
            ::

                import graphistry
                g = graphistry.bind()

                es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                g = g.bind(source='src', destination='dst')

                ig = g.pandas2igraph(es)
                ig.vs['community'] = ig.community_infomap().membership
                g.bind(point_color='community').plot(ig)
        """


        import igraph
        self._check_mandatory_bindings(False)
        self._check_bound_attribs(edges, ['source', 'destination'], 'Edge')
        
        self._node = self._node or PlotterBase._defaultNodeId
        eattribs = edges.columns.values.tolist()
        eattribs.remove(self._source)
        eattribs.remove(self._destination)
        cols = [self._source, self._destination] + eattribs
        etuples = [tuple(x) for x in edges[cols].values]
        return igraph.Graph.TupleList(etuples, directed=directed, edge_attrs=eattribs,
                                      vertex_name_attr=self._node)


    def igraph2pandas(self, ig):
        """Under current bindings, transform an IGraph into a pandas edges dataframe and a nodes dataframe.

        **Example**
            ::

                import graphistry
                g = graphistry.bind()

                es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                g = g.bind(source='src', destination='dst').edges(es)

                ig = g.pandas2igraph(es)
                ig.vs['community'] = ig.community_infomap().membership

                (es2, vs2) = g.igraph2pandas(ig)
                g.nodes(vs2).bind(point_color='community').plot()
        """

        def get_edgelist(ig):
            idmap = dict(enumerate(ig.vs[self._node]))
            for e in ig.es:
                t = e.tuple
                yield dict(
                    {self._source: idmap[t[0]], self._destination: idmap[t[1]]},
                    **e.attributes())

        self._check_mandatory_bindings(False)
        if self._node is None:
            ig.vs[PlotterBase._defaultNodeId] = [v.index for v in ig.vs]
            self._node = PlotterBase._defaultNodeId
        elif self._node not in ig.vs.attributes():
            error('Vertex attribute "%s" bound to "node" does not exist.' % self._node)

        edata = get_edgelist(ig)
        ndata = [v.attributes() for v in ig.vs]
        nodes = pd.DataFrame(ndata, columns=ig.vs.attributes())
        cols = [self._source, self._destination] + ig.es.attributes()
        edges = pd.DataFrame(edata, columns=cols)
        return (edges, nodes)


    def networkx_checkoverlap(self, g):

        import networkx as nx
        [_major, _minor] = nx.__version__.split('.', 1)

        vattribs = None
        if _major == '1':
            vattribs = g.nodes(data=True)[0][1] if g.number_of_nodes() > 0 else []
        else:
            vattribs = g.nodes(data=True) if g.number_of_nodes() > 0 else []
        if not (self._node is None) and self._node in vattribs:
            error('Vertex attribute "%s" already exists.' % self._node)

    def networkx2pandas(self, g):

        def get_nodelist(g):
            for n in g.nodes(data=True):
                yield dict({self._node: n[0]}, **n[1])

        def get_edgelist(g):
            for e in g.edges(data=True):
                yield dict({self._source: e[0], self._destination: e[1]}, **e[2])

        self._check_mandatory_bindings(False)
        self.networkx_checkoverlap(g)
        
        self._node = self._node or PlotterBase._defaultNodeId
        nodes = pd.DataFrame(get_nodelist(g))
        edges = pd.DataFrame(get_edgelist(g))
        return (edges, nodes)


    def _check_mandatory_bindings(self, node_required):
        if self._source is None or self._destination is None:
            error('Both "source" and "destination" must be bound before plotting.')
        if node_required and self._node is None:
            error('Node identifier must be bound when using node dataframe.')


    def _check_bound_attribs(self, df, attribs, typ):
        cols = df.columns.values.tolist()
        for a in attribs:
            b = getattr(self, '_' + a)
            if b not in cols:
                error('%s attribute "%s" bound to "%s" does not exist.' % (typ, a, b))


    def _plot_dispatch(self, graph, nodes, name, description, mode='json', metadata=None, memoize=True):

        g = self
        if self._point_title is None and self._point_label is None and g._nodes is not None:
            try:
                g = self.infer_labels()
            except:
                1

        if isinstance(graph, pd.core.frame.DataFrame) \
                or isinstance(graph, pa.Table) \
                or ( not (maybe_cudf is None) and isinstance(graph, maybe_cudf.DataFrame) ) \
                or ( not (maybe_dask_cudf is None) and isinstance(graph, maybe_dask_cudf.DataFrame) ) \
                or ( not (maybe_dask_dataframe is None) and isinstance(graph, maybe_dask_dataframe.DataFrame) ) \
                or ( not (maybe_spark is None) and isinstance(graph, pyspark.sql.dataframe.DataFrame) ):
            return g._make_dataset(graph, nodes, name, description, mode, metadata, memoize)

        try:
            import igraph
            if isinstance(graph, igraph.Graph):
                (e, n) = g.igraph2pandas(graph)
                return g._make_dataset(e, n, name, description, mode, metadata, memoize)
        except ImportError:
            pass

        try:
            import networkx
            if isinstance(graph, networkx.classes.graph.Graph) or \
               isinstance(graph, networkx.classes.digraph.DiGraph) or \
               isinstance(graph, networkx.classes.multigraph.MultiGraph) or \
               isinstance(graph, networkx.classes.multidigraph.MultiDiGraph):
                (e, n) = g.networkx2pandas(graph)
                return g._make_dataset(e, n, name, description, mode, metadata, memoize)
        except ImportError:
            pass

        error('Expected Pandas/Arrow/cuDF/Spark dataframe(s) or igraph/NetworkX graph.')


    # Sanitize node/edge dataframe by
    # - dropping indices
    # - dropping edges with NAs in source or destination
    # - dropping nodes with NAs in nodeid
    # - creating a default node table if none was provided.
    # - inferring numeric types of all columns containing numpy objects
    def _sanitize_dataset(self, edges, nodes, nodeid):
        self._check_bound_attribs(edges, ['source', 'destination'], 'Edge')
        elist = edges.reset_index(drop=True) \
                     .dropna(subset=[self._source, self._destination])

        obj_df = elist.select_dtypes(include=[np.object_])
        elist[obj_df.columns] = obj_df.apply(pd.to_numeric, errors='ignore')

        if nodes is None:
            nodes = pd.DataFrame()
            nodes[nodeid] = pd.concat(
                [edges[self._source], edges[self._destination]],
                ignore_index=True).drop_duplicates()
        else:
            self._check_bound_attribs(nodes, ['node'], 'Vertex')

        nlist = nodes.reset_index(drop=True) \
                     .dropna(subset=[nodeid]) \
                     .drop_duplicates(subset=[nodeid])

        obj_df = nlist.select_dtypes(include=[np.object_])
        nlist[obj_df.columns] = obj_df.apply(pd.to_numeric, errors='ignore')

        return (elist, nlist)


    def _check_dataset_size(self, elist, nlist):
        edge_count = len(elist.index)
        node_count = len(nlist.index)
        graph_size = edge_count + node_count
        if edge_count > 8e6:
            error('Maximum number of edges (8M) exceeded: %d.' % edge_count)
        if node_count > 8e6:
            error('Maximum number of nodes (8M) exceeded: %d.' % node_count)
        if graph_size > 1e6:
            warn('Large graph: |nodes| + |edges| = %d. Layout/rendering might be slow.' % graph_size)


    # Bind attributes for ETL1 by creating a copy of the designated column renamed
    # with magic names understood by ETL1 (eg. pointColor, etc)
    def _bind_attributes_v1(self, edges, nodes):
        def bind(df, pbname, attrib, default=None):
            bound = getattr(self, attrib)
            if bound:
                if bound in df.columns.tolist():
                    df[pbname] = df[bound]
                else:
                    warn('Attribute "%s" bound to %s does not exist.' % (bound, attrib))
            elif default:
                df[pbname] = df[default]

        nodeid = self._node or PlotterBase._defaultNodeId
        (elist, nlist) = self._sanitize_dataset(edges, nodes, nodeid)
        self._check_dataset_size(elist, nlist)

        bind(elist, 'edgeColor', '_edge_color')
        bind(elist, 'edgeSourceColor', '_edge_source_color')
        bind(elist, 'edgeDestinationColor', '_edge_destination_color')
        bind(elist, 'edgeLabel', '_edge_label')
        bind(elist, 'edgeTitle', '_edge_title')
        bind(elist, 'edgeSize', '_edge_size')
        bind(elist, 'edgeWeight', '_edge_weight')
        bind(elist, 'edgeOpacity', '_edge_opacity')
        bind(elist, 'edgeIcon', '_edge_icon')
        bind(nlist, 'pointColor', '_point_color')
        bind(nlist, 'pointLabel', '_point_label')
        bind(nlist, 'pointTitle', '_point_title', nodeid)
        bind(nlist, 'pointSize', '_point_size')
        bind(nlist, 'pointWeight', '_point_weight')
        bind(nlist, 'pointOpacity', '_point_opacity')
        bind(nlist, 'pointIcon', '_point_icon')
        bind(nlist, 'pointX', '_point_x')
        bind(nlist, 'pointY', '_point_y')
        return (elist, nlist)

    # Bind attributes for ETL2 by an encodings map storing the visual semantic of
    # each bound column.
    def _bind_attributes_v2(self, edges, nodes):
        def bind(enc, df, pbname, attrib, default=None):
            bound = getattr(self, attrib)
            if bound:
                if bound in df.columns.tolist():
                    enc[pbname] = {'attributes' : [bound]}
                else:
                    warn('Attribute "%s" bound to %s does not exist.' % (bound, attrib))
            elif default:
                enc[pbname] = {'attributes': [default]}

        nodeid = self._node or PlotterBase._defaultNodeId
        (elist, nlist) = self._sanitize_dataset(edges, nodes, nodeid)
        self._check_dataset_size(elist, nlist)

        edge_encodings = {
            'source': {'attributes' : [self._source]},
            'destination': {'attributes': [self._destination]},
        }
        node_encodings = {
            'nodeId': {'attributes': [nodeid]}
        }
        bind(edge_encodings, elist, 'edgeColor', '_edge_color')
        bind(edge_encodings, elist, 'edgeSourceColor', '_edge_source_color')
        bind(edge_encodings, elist, 'edgeDestinationColor', '_edge_destination_color')
        bind(edge_encodings, elist, 'edgeLabel', '_edge_label')
        bind(edge_encodings, elist, 'edgeTitle', '_edge_title')
        bind(edge_encodings, elist, 'edgeSize', '_edge_size')
        bind(edge_encodings, elist, 'edgeWeight', '_edge_weight')
        bind(edge_encodings, elist, 'edgeOpacity', '_edge_opacity')
        bind(edge_encodings, elist, 'edgeIcon', '_edge_icon')
        bind(node_encodings, nlist, 'pointColor', '_point_color')
        bind(node_encodings, nlist, 'pointLabel', '_point_label')
        bind(node_encodings, nlist, 'pointTitle', '_point_title', nodeid)
        bind(node_encodings, nlist, 'pointSize', '_point_size')
        bind(node_encodings, nlist, 'pointWeight', '_point_weight')
        bind(node_encodings, nlist, 'pointOpacity', '_point_opacity')
        bind(node_encodings, nlist, 'pointIcon', '_point_icon')
        bind(node_encodings, nlist, 'pointX', '_point_x')
        bind(node_encodings, nlist, 'pointY', '_point_y')

        encodings = {
            'nodes': node_encodings,
            'edges': edge_encodings
        }
        return (elist, nlist, encodings)

    def _table_to_pandas(self, table) -> Optional[pd.DataFrame]:
        """
            pandas | arrow | dask | cudf | dask_cudf => pandas
        """

        if table is None:
            return table

        if isinstance(table, pd.DataFrame):
            return table

        if isinstance(table, pa.Table):
            return table.to_pandas()
        
        if not (maybe_cudf is None) and isinstance(table, maybe_cudf.DataFrame):
            return table.to_pandas()

        if not (maybe_dask_cudf is None) and isinstance(table, maybe_dask_cudf.DataFrame):
            return self._table_to_pandas(table.compute())

        if not (maybe_dask_dataframe is None) and isinstance(table, maybe_dask_dataframe.DataFrame):
            return self._table_to_pandas(table.compute())

        raise Exception('Unknown type %s: Could not convert data to Pandas dataframe' % str(type(table)))

    def _table_to_arrow(self, table: Any, memoize: bool = True) -> pa.Table:  # noqa: C901
        """
            pandas | arrow | dask | cudf | dask_cudf => arrow

            dask/dask_cudf convert to pandas/cudf
        """

        logger.debug('_table_to_arrow of %s (memoize: %s)', type(table), memoize)

        if table is None:
            return table

        if isinstance(table, pa.Table):
            #TODO: should we hash just in case there's an older-identity hash match?
            return table
        
        if isinstance(table, pd.DataFrame):
            hashed = None
            if memoize:
                
                try:
                    #https://stackoverflow.com/questions/31567401/get-the-same-hash-value-for-a-pandas-dataframe-each-time
                    hashed = hashlib.sha256(pd.util.hash_pandas_object(table, index=True).values).hexdigest()
                except TypeError:
                    logger.warn('Failed memoization speedup attempt due to Pandas internal hash function failing. Continuing without memoization speedups.'
                                'This is fine, but for speedups around skipping re-uploads of previously seen tables, '
                                'try identifying which columns have types that Pandas cannot hash, and convert them '
                                'to hashable types like strings.')

                try:
                    if hashed in PlotterBase._pd_hash_to_arrow:
                        logger.debug('pd->arrow memoization hit: %s', hashed)
                        return PlotterBase._pd_hash_to_arrow[hashed].v
                    else:
                        logger.debug('pd->arrow memoization miss for id (of %s): %s', len(PlotterBase._pd_hash_to_arrow), hashed)
                except:
                    logger.debug('Failed to hash pdf', exc_info=True)
                    1

            out = pa.Table.from_pandas(table, preserve_index=False).replace_schema_metadata({})

            if memoize and (hashed is not None):
                w = WeakValueWrapper(out)
                cache_coercion(hashed, w)
                PlotterBase._pd_hash_to_arrow[hashed] = w

            return out

        if not (maybe_cudf is None) and isinstance(table, maybe_cudf.DataFrame):

            hashed = None
            if memoize:
                #https://stackoverflow.com/questions/31567401/get-the-same-hash-value-for-a-pandas-dataframe-each-time
                hashed = hashlib.sha256(table.hash_columns().tobytes()).hexdigest()
                try:
                    if hashed in PlotterBase._cudf_hash_to_arrow:
                        logger.debug('cudf->arrow memoization hit: %s', hashed)
                        return PlotterBase._cudf_hash_to_arrow[hashed].v
                    else:
                        logger.debug('cudf->arrow memoization miss for id (of %s): %s', len(PlotterBase._cudf_hash_to_arrow), hashed)
                except:
                    logger.debug('Failed to hash cudf', exc_info=True)
                    1

            out = table.to_arrow()

            if memoize:
                w = WeakValueWrapper(out)
                cache_coercion(hashed, w)
                PlotterBase._cudf_hash_to_arrow[hashed] = w

            return out
        
        # TODO: per-gdf hashing? 
        if not (maybe_dask_cudf is None) and isinstance(table, maybe_dask_cudf.DataFrame):
            logger.debug('dgdf->arrow via gdf hash check')
            dgdf = table.persist()
            gdf = dgdf.compute()
            return self._table_to_arrow(gdf, memoize)

        if not (maybe_dask_dataframe is None) and isinstance(table, maybe_dask_dataframe.DataFrame):
            logger.debug('ddf->arrow via df hash check')
            ddf = table.persist()
            df = ddf.compute()
            return self._table_to_arrow(df, memoize)

        if not (maybe_spark is None) and isinstance(table, maybe_spark.sql.dataframe.DataFrame):
            logger.debug('spark->arrow via df')
            df = table.toPandas()
            #TODO push the hash check to Spark
            return self._table_to_arrow(df, memoize)

        raise Exception('Unknown type %s: Could not convert data to Arrow' % str(type(table)))


    def _make_dataset(self, edges, nodes, name, description, mode, metadata=None, memoize: bool = True):  # noqa: C901

        logger.debug('_make_dataset (mode %s, memoize %s) name:[%s] des:[%s] (e::%s, n::%s) ',
            mode, memoize, name, description, type(edges), type(nodes))

        try:
            if len(edges) == 0:
                warn('Graph has no edges, may have rendering issues')
        except:
            1

        #compatibility checks
        if (mode == 'json') or (mode == 'vgraph'):
            if not (metadata is None):
                if ('bg' in metadata) or ('fg' in metadata) or ('logo' in metadata) or ('page' in metadata):
                    raise ValueError('Cannot set bg/fg/logo/page in api=1, api=2; try using api=3')
            if not (self._complex_encodings is None
                or self._complex_encodings == {  # noqa: W503
                    'node_encodings': {'current': {}, 'default': {} },
                    'edge_encodings': {'current': {}, 'default': {} }}):
                raise ValueError('Cannot set complex encodings ".encode_[point/edge]_[feature]()" in api=1, api=2; try using api=3 or .bind()')

        if mode == 'json':
            edges_df = self._table_to_pandas(edges)
            nodes_df = self._table_to_pandas(nodes)
            return self._make_json_dataset(edges_df, nodes_df, name)
        elif mode == 'vgraph':
            edges_df = self._table_to_pandas(edges)
            nodes_df = self._table_to_pandas(nodes)
            return self._make_vgraph_dataset(edges_df, nodes_df, name)
        elif mode == 'arrow':
            edges_arr = self._table_to_arrow(edges, memoize)
            nodes_arr = self._table_to_arrow(nodes, memoize)
            return self._make_arrow_dataset(edges=edges_arr, nodes=nodes_arr, name=name, description=description, metadata=metadata)
            #token=None, dataset_id=None, url_params = None)
        else:
            raise ValueError('Unknown mode: ' + mode)


    # Main helper for creating ETL1 payload
    def _make_json_dataset(self, edges, nodes, name):

        from .pygraphistry import PyGraphistry

        (elist, nlist) = self._bind_attributes_v1(edges, nodes)
        edict = elist.where((pd.notnull(elist)), None).to_dict(orient='records')

        bindings = {'idField': self._node or PlotterBase._defaultNodeId,
                    'destinationField': self._destination, 'sourceField': self._source}
        dataset = {'name': PyGraphistry._config['dataset_prefix'] + name,
                   'bindings': bindings, 'type': 'edgelist', 'graph': edict}

        if nlist is not None:
            ndict = nlist.where((pd.notnull(nlist)), None).to_dict(orient='records')
            dataset['labels'] = ndict
        return dataset


    # Main helper for creating ETL2 payload
    def _make_vgraph_dataset(self, edges, nodes, name):
        from . import vgraph

        (elist, nlist, encodings) = self._bind_attributes_v2(edges, nodes)
        nodeid = self._node or PlotterBase._defaultNodeId

        sources = elist[self._source]
        dests = elist[self._destination]
        elist.drop([self._source, self._destination], axis=1, inplace=True)

        # Filter out nodes which have no edges
        lnodes = pd.concat([sources, dests], ignore_index=True).unique()
        lnodes_df = pd.DataFrame(lnodes, columns=[nodeid])
        filtered_nlist = pd.merge(lnodes_df, nlist, on=nodeid, how='left')

        # Create a map from nodeId to a continuous range of integer [0, #nodes-1].
        # The vgraph protobuf format uses the continous integer ranger as internal nodeIds.
        node_map = dict([(v, i) for i, v in enumerate(lnodes.tolist())])

        dataset = vgraph.create(elist, filtered_nlist, sources, dests, nodeid, node_map, name)
        dataset['encodings'] = encodings
        return dataset

    def _make_arrow_dataset(self, edges: pa.Table, nodes: pa.Table, name: str, description: str, metadata) -> ArrowUploader:

        from .pygraphistry import PyGraphistry

        au : ArrowUploader = ArrowUploader(
            server_base_path=PyGraphistry.protocol() + '://' + PyGraphistry.server(),
            edges=edges, nodes=nodes,
            name=name, description=description,
            metadata={
                'usertag': PyGraphistry._tag,
                'key': PyGraphistry.api_key(),
                'agent': 'pygraphistry',
                'apiversion' : '3',
                'agentversion': sys.modules['graphistry'].__version__,  # type: ignore
                **(metadata or {})
            },
            certificate_validation=PyGraphistry.certificate_validation())
        au.edge_encodings = au.g_to_edge_encodings(self)
        au.node_encodings = au.g_to_node_encodings(self)
        return au

    def bolt(self, driver):
        res = copy.copy(self)
        res._bolt_driver = to_bolt_driver(driver)
        return res


    def infer_labels(self):
        """

        :return: Plotter w/neo4j

        * Prefers point_title/point_label if available
        * Fallback to node id
        * Raises exception if no nodes available, no likely candidates, and no matching node id fallback

        **Example**

                ::

                    import graphistry
                    g = graphistry.nodes(pd.read_csv('nodes.csv'), 'id_col').infer_labels()
                    g.plot()

        """

        if self._point_title is not None or self._point_label is not None:
            return self

        nodes_df = self._nodes
        if nodes_df is None:
            raise ValueError('node label inference requires ._nodes to be set')

        # full term (case insensitive)
        opts = ['nodetitle', 'nodelabel', 'label', 'title', 'name']
        if nodes_df is not None:
            for c in opts:
                for c2 in nodes_df:
                    if c == c2.lower():
                        return self.bind(point_title=c2)

        # substring
        for opt in ['title', 'label', 'name']:
            for c in nodes_df:
                if opt in c.lower():
                    return self.bind(point_title=c)

        # fallback
        if self._node is not None:
            return self.bind(point_title=self._node)

        raise ValueError('Could not find a label-like node column and no g._node id fallback set')


    def cypher(self, query, params={}):

        from .pygraphistry import PyGraphistry

        res = copy.copy(self)
        driver = self._bolt_driver or PyGraphistry._config['bolt_driver']
        if driver is None:
            raise ValueError("BOLT connection information not provided. Must first call graphistry.register(bolt=...) or g.bolt(...).")
        with driver.session() as session:
            bolt_statement = session.run(query, **params)
            graph = bolt_statement.graph()
            edges = bolt_graph_to_edges_dataframe(graph)
            nodes = bolt_graph_to_nodes_dataframe(graph)
        return res\
            .bind(
                node=node_id_key,
                source=start_node_id_key,
                destination=end_node_id_key
            )\
            .nodes(nodes)\
            .edges(edges)

    def nodexl(self, xls_or_url, source='default', engine=None, verbose=False):
        
        if not (engine is None):
            print('WARNING: Engine currently ignored, please contact if critical')
        
        return NodeXLGraphistry(self, engine).xls(xls_or_url, source, verbose)


    def tigergraph(self,
            protocol = 'http',
            server = 'localhost',
            web_port = 14240,
            api_port = 9000,
            db = None,
            user = 'tigergraph',
            pwd = 'tigergraph',
            verbose = False):
        """Register Tigergraph connection setting defaults
    
        :param protocol: Protocol used to contact the database.
        :type protocol: Optional[str]
        :param server: Domain of the database
        :type server: Optional[str]
        :param web_port: 
        :type web_port: Optional[int]
        :param api_port: 
        :type api_port: Optional[int]
        :param db: Name of the database
        :type db: Optional[str]    
        :param user:
        :type user: Optional[str]    
        :param pwd: 
        :type pwd: Optional[str]
        :param verbose: Whether to print operations
        :type verbose: Optional[bool]         
        :returns: Plotter
        :rtype: Plotter


        **Example: Standard**
                ::

                    import graphistry
                    tg = graphistry.tigergraph(protocol='https', server='acme.com', db='my_db', user='alice', pwd='tigergraph2')                    

        """
        res = copy.copy(self)
        res._tigergraph = Tigeristry(self, protocol, server, web_port, api_port, db, user, pwd, verbose)
        return res


    def gsql_endpoint(self, method_name, args = {}, bindings = {}, db = None, dry_run = False):
        """Invoke Tigergraph stored procedure at a user-definend endpoint and return transformed Plottable
    
        :param method_name: Stored procedure name
        :type method_name: str
        :param args: Named endpoint arguments
        :type args: Optional[dict]
        :param bindings: Mapping defining names of returned 'edges' and/or 'nodes', defaults to @@nodeList and @@edgeList
        :type bindings: Optional[dict]
        :param db: Name of the database, defaults to value set in .tigergraph(...)
        :type db: Optional[str]
        :param dry_run: Return target URL without running
        :type dry_run: bool
        :returns: Plotter
        :rtype: Plotter

        **Example: Minimal**
                ::

                    import graphistry
                    tg = graphistry.tigergraph(db='my_db')
                    tg.gsql_endpoint('neighbors').plot()

        **Example: Full**
                ::

                    import graphistry
                    tg = graphistry.tigergraph()
                    tg.gsql_endpoint('neighbors', {'k': 2}, {'edges': 'my_edge_list'}, 'my_db').plot()

        **Example: Read data**
                ::

                    import graphistry
                    tg = graphistry.tigergraph()
                    out = tg.gsql_endpoint('neighbors')
                    (nodes_df, edges_df) = (out._nodes, out._edges)

        """
        return self._tigergraph.gsql_endpoint(self, method_name, args, bindings, db, dry_run)

    def gsql(self, query, bindings = {}, dry_run = False):
        """Run Tigergraph query in interpreted mode and return transformed Plottable
    
        :param query: Code to run
        :type query: str
        :param bindings: Mapping defining names of returned 'edges' and/or 'nodes', defaults to @@nodeList and @@edgeList
        :type bindings: Optional[dict]
        :param dry_run: Return target URL without running
        :type dry_run: bool
        :returns: Plotter
        :rtype: Plotter

        **Example: Minimal**
                ::

                    import graphistry
                    tg = graphistry.tigergraph()
                    tg.gsql(\"\"\"
                    INTERPRET QUERY () FOR GRAPH Storage { 
                        
                        OrAccum<BOOL> @@stop;
                        ListAccum<EDGE> @@edgeList;
                        SetAccum<vertex> @@set;
                        
                        @@set += to_vertex("61921", "Pool");

                        Start = @@set;

                        while Start.size() > 0 and @@stop == false do

                        Start = select t from Start:s-(:e)-:t
                        where e.goUpper == TRUE
                        accum @@edgeList += e
                        having t.type != "Service";
                        end;

                        print @@edgeList;
                    }
                    \"\"\").plot()

       **Example: Full**
                ::

                    import graphistry
                    tg = graphistry.tigergraph()
                    tg.gsql(\"\"\"
                    INTERPRET QUERY () FOR GRAPH Storage { 
                        
                        OrAccum<BOOL> @@stop;
                        ListAccum<EDGE> @@edgeList;
                        SetAccum<vertex> @@set;
                        
                        @@set += to_vertex("61921", "Pool");

                        Start = @@set;

                        while Start.size() > 0 and @@stop == false do

                        Start = select t from Start:s-(:e)-:t
                        where e.goUpper == TRUE
                        accum @@edgeList += e
                        having t.type != "Service";
                        end;

                        print @@my_edge_list;
                    }
                    \"\"\", {'edges': 'my_edge_list'}).plot()
        """        
        return self._tigergraph.gsql(self, query, bindings, dry_run)

    def hypergraph(
        self,
        raw_events, entity_types: Optional[List[str]] = None, opts: dict = {},
        drop_na: bool = True, drop_edge_attrs: bool = False, verbose: bool = True, direct: bool = False,
        engine: str = 'pandas', npartitions: Optional[int] = None, chunksize: Optional[int] = None

    ):
        """Transform a dataframe into a hypergraph.

        :param raw_events: Dataframe to transform (pandas or cudf). 
        :type raw_events: pandas.DataFrame
        :param Optional[list] entity_types: Columns (strings) to turn into nodes, None signifies all
        :param dict opts: See below
        :param bool drop_edge_attrs: Whether to include each row's attributes on its edges, defaults to False (include)
        :param bool verbose: Whether to print size information
        :param bool direct: Omit hypernode and instead strongly connect nodes in an event
        :param bool engine: String (pandas, cudf, ...) for engine to use
        :param Optional[int] npartitions: For distributed engines, how many coarse-grained pieces to split events into
        :param Optional[int] chunksize: For distributed engines, split events after chunksize rows

        Create a graph out of the dataframe, and return the graph components as dataframes, 
        and the renderable result Plotter. Hypergraphs reveal relationships between rows and between column values.
        This transform is useful for lists of events, samples, relationships, and other structured high-dimensional data.

        Specify local compute engine by passing `engine='pandas'`, 'cudf', 'dask', 'dask_cudf' (default: 'pandas').
        If events are not in that engine's format, they will be converted into it.

        The transform creates a node for every unique value in the entity_types columns (default: all columns). 
        If direct=False (default), every row is also turned into a node. 
        Edges are added to connect every table cell to its originating row's node, or if direct=True, to the other nodes from the same row.
        Nodes are given the attribute 'type' corresponding to the originating column name, or in the case of a row, 'EventID'.
        Options further control the transform, such column category definitions for controlling whether values
        reocurring in different columns should be treated as one node,
        or whether to only draw edges between certain column type pairs. 

        Consider a list of events. Each row represents a distinct event, and each column some metadata about an event. 
        If multiple events have common metadata, they will be transitively connected through those metadata values. 
        The layout algorithm will try to cluster the events together. 
        Conversely, if an event has unique metadata, the unique metadata will turn into nodes that only have connections to the event node, and the clustering algorithm will cause them to form a ring around the event node.

        Best practice is to set EVENTID to a row's unique ID,
        SKIP to all non-categorical columns (or entity_types to all categorical columns),
        and CATEGORY to group columns with the same kinds of values.

        To prevent creating nodes for null values, set drop_na=True.
        Some dataframe engines may have undesirable null handling,
        and recommend replacing None values with np.nan .

        The optional ``opts={...}`` configuration options are:

        * 'EVENTID': Column name to inspect for a row ID. By default, uses the row index.
        * 'CATEGORIES': Dictionary mapping a category name to inhabiting columns. E.g., {'IP': ['srcAddress', 'dstAddress']}.  If the same IP appears in both columns, this makes the transform generate one node for it, instead of one for each column.
        * 'DELIM': When creating node IDs, defines the separator used between the column name and node value
        * 'SKIP': List of column names to not turn into nodes. For example, dates and numbers are often skipped.
        * 'EDGES': For direct=True, instead of making all edges, pick column pairs. E.g., {'a': ['b', 'd'], 'd': ['d']} creates edges between columns a->b and a->d, and self-edges d->d.


        :returns: {'entities': DF, 'events': DF, 'edges': DF, 'nodes': DF, 'graph': Plotter}
        :rtype: dict

        **Example: Connect user<-row->boss**

            ::

                import graphistry
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df)
                g = h['graph'].plot()

        **Example: Connect user->boss**

            ::

                import graphistry
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, direct=True)
                g = h['graph'].plot()

        **Example: Connect user<->boss**

            ::

                import graphistry
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, direct=True, opts={'EDGES': {'user': ['boss'], 'boss': ['user']}})
                g = h['graph'].plot()

        **Example: Only consider some columns for nodes**

            ::

                import graphistry
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, entity_types=['boss'])
                g = h['graph'].plot()

        **Example: Collapse matching user::<id> and boss::<id> nodes into one person::<id> node**

            ::

                import graphistry
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, opts={'CATEGORIES': {'person': ['user', 'boss']}})
                g = h['graph'].plot()

        **Example: Use cudf engine instead of pandas**

            ::

                import cudf, graphistry
                users_gdf = cudf.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_gdf, engine='cudf')
                g = h['graph'].plot()

        """
        from . import hyper
        return hyper.Hypergraph().hypergraph(
            self, raw_events, entity_types, opts, drop_na, drop_edge_attrs, verbose, direct,
            engine=engine, npartitions=npartitions, chunksize=chunksize)


    def layout_settings(
        self,

        play: Optional[int] = None,

        locked_x: Optional[bool] = None,
        locked_y: Optional[bool] = None,
        locked_r: Optional[bool] = None,

        left: Optional[float] = None,
        top: Optional[float] = None,
        right: Optional[float] = None,
        bottom: Optional[float] = None,

        lin_log: Optional[bool] = None,
        strong_gravity: Optional[bool] = None,
        dissuade_hubs: Optional[bool] = None,

        edge_influence: Optional[float] = None,
        precision_vs_speed: Optional[float] = None,
        gravity: Optional[float] = None,
        scaling_ratio: Optional[float] = None
    ):
        """Set layout options. Additive over previous settings.

        Corresponds to options at https://hub.graphistry.com/docs/api/1/rest/url/#urloptions

        **Example: Animated radial layout**

            ::

                import graphistry, pandas as pd
                edges = pd.DataFrame({'s': ['a','b','c','d'], 'boss': ['c','c','e','e']})
                nodes = pd.DataFrame({
                    'n': ['a', 'b', 'c', 'd', 'e'],
                    'y': [1,   1,   2,   3,   4],
                    'x': [1,   1,   0,   0,   0],
                })
                g = (graphistry
                    .edges(edges, 's', 'd')
                    .nodes(nodes, 'n')
                    .layout_settings(locked_r=True, play=2000)
                g.plot()
        """

        settings : dict = {
            **({} if play is None else {'play': play}),
            **({} if locked_x is None else {'lockedX': locked_x}),
            **({} if locked_y is None else {'lockedY': locked_y}),
            **({} if locked_r is None else {'lockedR': locked_r}),

            **({} if left is None else {'left': left}),
            **({} if top is None else {'top': top}),
            **({} if right is None else {'right': right}),
            **({} if bottom is None else {'bottom': bottom}),

            **({} if lin_log is None else {'linLog': lin_log}),
            **({} if strong_gravity is None else {'strongGravity': strong_gravity}),
            **({} if dissuade_hubs is None else {'dissuadeHubs': dissuade_hubs}),

            **({} if edge_influence is None else {'edgeInfluence': edge_influence}),
            **({} if precision_vs_speed is None else {'precisionVsSpeed': precision_vs_speed}),
            **({} if gravity is None else {'gravity': gravity}),
            **({} if scaling_ratio is None else {'scalingRatio': scaling_ratio}),
        }

        if len(settings.keys()) > 0:
            return self.settings(url_params={**self._url_params, **settings})
        else:
            return self
