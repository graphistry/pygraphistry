"""
Kepler.gl encoding classes for managing datasets and layers.

This module provides classes for building Kepler.gl visualizations using an immutable pattern.
Type specifications are documented in kepler_types.py (specification only, not used at runtime).
"""

from typing import Optional, List, Dict, Any, overload, Literal
import uuid
import re


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


class KeplerDataset:
    """Configure a Kepler.gl dataset for visualization.

    Creates a dataset configuration that makes Graphistry data (nodes/edges) or geographic
    data (countries/states) available to Kepler.gl for visualization.

    **Common parameters (all dataset types):**

    :param raw_dict: Native Kepler.gl dataset dictionary (if provided, all other params ignored)
    :type raw_dict: Optional[Dict[str, Any]]
    :param id: Dataset identifier (auto-generated if None)
    :type id: Optional[str]
    :param type: Dataset type - 'nodes', 'edges', 'countries', 'states', etc.
    :type type: Optional[str]
    :param label: Display label (defaults to id)
    :type label: Optional[str]
    :param include: Columns to include (whitelist)
    :type include: Optional[List[str]]
    :param exclude: Columns to exclude (blacklist)
    :type exclude: Optional[List[str]]
    :param computed_columns: Computed/aggregated columns for data enrichment
    :type computed_columns: Optional[Dict[str, Any]]

    **For nodes type:**

    No additional parameters beyond common ones.

    **For edges type:**

    :param map_node_coords: Auto-map source/target node coordinates to edges (adds columns: edgeSourceLatitude, edgeSourceLongitude, edgeTargetLatitude, edgeTargetLongitude)
    :type map_node_coords: Optional[bool]
    :param map_node_coords_mapping: Custom column names for mapped coordinates. Dict mapping default names to custom names, e.g., {"edgeSourceLongitude": "src_lng", "edgeSourceLatitude": "src_lat", "edgeTargetLongitude": "dst_lng", "edgeTargetLatitude": "dst_lat"}
    :type map_node_coords_mapping: Optional[Dict[str, str]]

    **For countries/zeroOrderAdminRegions type:**

    :param resolution: Map resolution (10=high, 50=medium, 110=low)
    :type resolution: Optional[Literal[10, 50, 110]]
    :param boundary_lakes: Include lake boundaries (default: True)
    :type boundary_lakes: Optional[bool]
    :param filter_countries_by_col: Column to filter countries
    :type filter_countries_by_col: Optional[str]
    :param include_countries: Countries to include
    :type include_countries: Optional[List[str]]
    :param exclude_countries: Countries to exclude
    :type exclude_countries: Optional[List[str]]

    **For states/provinces/firstOrderAdminRegions type:**

    :param boundary_lakes: Include lake boundaries (default: True)
    :type boundary_lakes: Optional[bool]
    :param filter_countries_by_col: Column to filter countries
    :type filter_countries_by_col: Optional[str]
    :param include_countries: Countries to include
    :type include_countries: Optional[List[str]]
    :param exclude_countries: Countries to exclude
    :type exclude_countries: Optional[List[str]]
    :param filter_1st_order_regions_by_col: Column to filter regions
    :type filter_1st_order_regions_by_col: Optional[str]
    :param include_1st_order_regions: Regions to include
    :type include_1st_order_regions: Optional[List[str]]
    :param exclude_1st_order_regions: Regions to exclude
    :type exclude_1st_order_regions: Optional[List[str]]

    **Example: Node dataset**
        ::

            from graphistry import KeplerDataset

            # Basic node dataset
            ds = KeplerDataset(id="companies", type="nodes", label="Companies")

            # With column filtering
            ds = KeplerDataset(
                type="nodes",
                include=["name", "latitude", "longitude", "revenue"]
            )

    **Example: Edge dataset with coordinate mapping**
        ::

            # Auto-map source/target node coordinates to edges
            ds = KeplerDataset(
                type="edges",
                map_node_coords=True
            )

    **Example: Countries with computed columns**
        ::

            # High-resolution countries with aggregated metrics
            ds = KeplerDataset(
                type="countries",
                resolution=10,
                computed_columns={
                    "avg_revenue": {
                        "type": "aggregate",
                        "computeFromDataset": "companies",
                        "sourceKey": "country",
                        "targetKey": "name",
                        "aggregate": "mean",
                        "aggregateCol": "revenue"
                    }
                }
            )

    **Example: Using raw_dict**
        ::

            # Pass through native Kepler.gl dataset dict
            ds = KeplerDataset({
                "info": {"id": "my-dataset", "label": "My Data"},
                "data": {...}
            })
    """

    _raw_dict: Optional[Dict[str, Any]]
    id: Optional[str]
    type: Optional[str]
    label: Optional[str]
    _kwargs: Dict[str, Any]

    # Overload for raw_dict mode (native Kepler dataset)
    @overload
    def __init__(
        self,
        raw_dict: Dict[str, Any]
    ) -> None:
        ...

    # Overload for nodes dataset
    @overload
    def __init__(
        self,
        raw_dict: None = None,
        *,
        id: Optional[str] = None,
        type: Literal["nodes"],
        label: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        computed_columns: Optional[Dict[str, Any]] = None
    ) -> None:
        ...

    # Overload for edges dataset
    @overload
    def __init__(
        self,
        raw_dict: None = None,
        *,
        id: Optional[str] = None,
        type: Literal["edges"],
        label: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        computed_columns: Optional[Dict[str, Any]] = None,
        map_node_coords: Optional[bool] = None,
        map_node_coords_mapping: Optional[Dict[str, str]] = None
    ) -> None:
        ...

    # Overload for countries dataset
    @overload
    def __init__(
        self,
        raw_dict: None = None,
        *,
        id: Optional[str] = None,
        type: Literal["countries", "zeroOrderAdminRegions"],
        label: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        computed_columns: Optional[Dict[str, Any]] = None,
        resolution: Optional[Literal[10, 50, 110]] = None,
        boundary_lakes: Optional[bool] = None,
        filter_countries_by_col: Optional[str] = None,
        include_countries: Optional[List[str]] = None,
        exclude_countries: Optional[List[str]] = None
    ) -> None:
        ...

    # Overload for states/provinces/firstOrderAdminRegions dataset
    @overload
    def __init__(
        self,
        raw_dict: None = None,
        *,
        id: Optional[str] = None,
        type: Literal["states", "provinces", "firstOrderAdminRegions"],
        label: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        computed_columns: Optional[Dict[str, Any]] = None,
        boundary_lakes: Optional[bool] = None,
        filter_countries_by_col: Optional[str] = None,
        include_countries: Optional[List[str]] = None,
        exclude_countries: Optional[List[str]] = None,
        filter_1st_order_regions_by_col: Optional[str] = None,
        include_1st_order_regions: Optional[List[str]] = None,
        exclude_1st_order_regions: Optional[List[str]] = None
    ) -> None:
        ...

    # Actual implementation
    def __init__(
        self,
        raw_dict: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        type: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs: Any
    ):
        # If raw_dict is provided, store it and bypass normal processing
        if raw_dict is not None:
            if not isinstance(raw_dict, dict):
                raise TypeError(f"raw_dict must be a dict, got {type(raw_dict)}")
            self._raw_dict = raw_dict
            self.id = None
            self.type = None
            self.label = None
            self._kwargs = {}
        else:
            self._raw_dict = None
            # Auto-generate ID if not provided
            self.id = id if id is not None else f"dataset-{uuid.uuid4().hex[:8]}"
            self.type = type
            self.label = label if label is not None else self.id
            self._kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary format for Kepler.gl."""
        # If raw_dict was provided, return it directly without modification
        if self._raw_dict is not None:
            return self._raw_dict

        # Build from structured parameters
        result: Dict[str, Any] = {
            'info': {
                'id': self.id,
                'label': self.label
            }
        }

        if self.type is not None:
            result['type'] = self.type

        # Spread remaining kwargs (include, exclude, and type-specific params)
        result.update(self._kwargs)

        return result

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        if self._raw_dict is not None:
            id_val = self._raw_dict.get('info', {}).get('id')
            type_val = self._raw_dict.get('type')
            label_val = self._raw_dict.get('info', {}).get('label')
        else:
            id_val = self.id
            type_val = self.type
            label_val = self.label

        parts = [f"id={id_val!r}"]
        type_display = type_val if type_val else "raw"
        parts.append(f"type={type_display!r}")
        if label_val:
            parts.append(f"label={label_val!r}")
        return f"KeplerDataset({', '.join(parts)})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self._raw_dict is not None:
            id_val = self._raw_dict.get('info', {}).get('id')
            type_val = self._raw_dict.get('type')
        else:
            id_val = self.id
            type_val = self.type

        type_display = type_val if type_val else "raw"
        return f"KeplerDataset(id={id_val}:{type_display})"

    def __eq__(self, other) -> bool:
        """Check equality based on all properties."""
        if not isinstance(other, KeplerDataset):
            return False
        # If both have raw_dict, compare them
        if self._raw_dict is not None and other._raw_dict is not None:
            return self._raw_dict == other._raw_dict
        # If only one has raw_dict, not equal
        if self._raw_dict is not None or other._raw_dict is not None:
            return False
        # Normal comparison
        return (
            self.id == other.id
            and self.type == other.type
            and self.label == other.label
            and self._kwargs == other._kwargs
        )


class KeplerLayer:
    """Configure a Kepler.gl visualization layer.

    Creates a layer configuration using native Kepler.gl format. Layers define how datasets
    are visualized on the map (points, arcs, hexbins, etc.). This class accepts only raw
    dictionary format for now.

    :param raw_dict: Native Kepler.gl layer configuration dictionary
    :type raw_dict: Dict[str, Any]

    **Example: Point layer**
        ::

            from graphistry import KeplerLayer

            layer = KeplerLayer({
                "id": "cities",
                "type": "point",
                "config": {
                    "dataId": "companies",
                    "label": "Company Locations",
                    "columns": {"lat": "latitude", "lng": "longitude"},
                    "color": [255, 140, 0],
                    "visConfig": {"radius": 10, "opacity": 0.8}
                }
            })

    **Example: Arc layer for connections**
        ::

            layer = KeplerLayer({
                "id": "connections",
                "type": "arc",
                "config": {
                    "dataId": "relationships",
                    "columns": {
                        "lat0": "edgeSourceLatitude",
                        "lng0": "edgeSourceLongitude",
                        "lat1": "edgeTargetLatitude",
                        "lng1": "edgeTargetLongitude"
                    },
                    "color": [0, 200, 255],
                    "visConfig": {"opacity": 0.3, "thickness": 2}
                }
            })

    **Example: Hexagon aggregation**
        ::

            layer = KeplerLayer({
                "id": "density",
                "type": "hexagon",
                "config": {
                    "dataId": "events",
                    "columns": {"lat": "latitude", "lng": "longitude"},
                    "visConfig": {
                        "worldUnitSize": 1,
                        "elevationScale": 5,
                        "enable3d": True
                    }
                }
            })
    """

    _raw_dict: Dict[str, Any]

    def __init__(
        self,
        raw_dict: Dict[str, Any]
    ):
        if not isinstance(raw_dict, dict):
            raise TypeError(f"raw_dict must be a dict, got {type(raw_dict)}")
        self._raw_dict = raw_dict

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary format for Kepler.gl."""
        return self._raw_dict

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        id_val = self._raw_dict.get('id')
        type_val = self._raw_dict.get('type')
        label_val = self._raw_dict.get('config', {}).get('label')

        parts = [f"id={id_val!r}"]
        type_display = type_val if type_val else "raw"
        parts.append(f"type={type_display!r}")
        if label_val:
            parts.append(f"label={label_val!r}")
        return f"KeplerLayer({', '.join(parts)})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        id_val = self._raw_dict.get('id')
        type_val = self._raw_dict.get('type')
        type_display = type_val if type_val else "raw"
        return f"KeplerLayer(id={id_val}:{type_display})"

    def __eq__(self, other) -> bool:
        """Check equality based on raw_dict."""
        if not isinstance(other, KeplerLayer):
            return False
        return self._raw_dict == other._raw_dict


class KeplerOptions:
    """Configure Kepler.gl visualization options.

    Controls map behavior and interaction settings such as auto-centering and read-only mode.

    :param raw_dict: Native Kepler.gl options dictionary (if provided, kwargs ignored)
    :type raw_dict: Optional[Dict[str, Any]]
    :param center_map: Auto-center map on data (default: True)
    :type center_map: Optional[bool]
    :param read_only: Disable map interactions (default: False)
    :type read_only: Optional[bool]

    **Example: Structured parameters**
        ::

            from graphistry import KeplerOptions

            # Auto-center with interactions enabled
            opts = KeplerOptions(center_map=True, read_only=False)

            # Read-only mode for presentations
            opts = KeplerOptions(read_only=True)

    **Example: Using raw_dict**
        ::

            # Pass native format
            opts = KeplerOptions({"centerMap": True, "readOnly": False})
    """

    _raw_dict: Optional[Dict[str, Any]]
    _kwargs: Dict[str, Any]

    # Overload for raw_dict mode
    @overload
    def __init__(
        self,
        raw_dict: Dict[str, Any]
    ) -> None:
        ...

    # Overload for structured mode
    @overload
    def __init__(
        self,
        raw_dict: None = None,
        *,
        center_map: Optional[bool] = None,
        read_only: Optional[bool] = None,
    ) -> None:
        ...

    # Actual implementation
    def __init__(
        self,
        raw_dict: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        if raw_dict is not None:
            if not isinstance(raw_dict, dict):
                raise TypeError(f"raw_dict must be a dict, got {type(raw_dict)}")
            self._raw_dict = raw_dict
            self._kwargs = {}
        else:
            self._raw_dict = None
            self._kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary format for Kepler.gl."""
        if self._raw_dict is not None:
            return self._raw_dict
        # Convert snake_case keys to camelCase
        return {snake_to_camel(k): v for k, v in self._kwargs.items()}

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        if self._raw_dict is not None:
            return f"KeplerOptions(raw_dict={self._raw_dict!r})"
        return f"KeplerOptions({self._kwargs!r})"

    def __eq__(self, other) -> bool:
        """Check equality."""
        if not isinstance(other, KeplerOptions):
            return False
        if self._raw_dict is not None and other._raw_dict is not None:
            return self._raw_dict == other._raw_dict
        if self._raw_dict is not None or other._raw_dict is not None:
            return False
        return self._kwargs == other._kwargs


class KeplerConfig:
    """Configure Kepler.gl map and rendering settings.

    Controls map appearance, data optimization, and layer blending behavior.

    :param raw_dict: Native Kepler.gl config dictionary (if provided, kwargs ignored)
    :type raw_dict: Optional[Dict[str, Any]]
    :param cull_unused_columns: Remove unused columns from datasets (default: True)
    :type cull_unused_columns: Optional[bool]
    :param overlay_blending: Blend mode - 'normal', 'additive', 'subtractive' (default: 'normal')
    :type overlay_blending: Optional[Literal['normal', 'additive', 'subtractive']]
    :param tile_style: Base map tile style configuration
    :type tile_style: Optional[Dict[str, Any]]
    :param auto_graph_renderer_switching: Enable automatic graph renderer switching, which allows Graphistry to hide Kepler node and edge layers depending on the mode (default: True)
    :type auto_graph_renderer_switching: Optional[bool]

    **Example: Structured parameters**
        ::

            from graphistry import KeplerConfig

            # Optimize data transfer
            cfg = KeplerConfig(cull_unused_columns=True)

            # Additive blending for heatmaps
            cfg = KeplerConfig(overlay_blending='additive')

            # Custom dark base map
            cfg = KeplerConfig(
                tile_style={
                    "id": "dark",
                    "label": "Dark Mode",
                    "url": "mapbox://styles/mapbox/dark-v10"
                }
            )

    **Example: Using raw_dict**
        ::

            # Pass native format
            cfg = KeplerConfig({
                "cullUnusedColumns": True,
                "overlayBlending": "additive"
            })
    """

    _raw_dict: Optional[Dict[str, Any]]
    _kwargs: Dict[str, Any]

    # Overload for raw_dict mode
    @overload
    def __init__(
        self,
        raw_dict: Dict[str, Any]
    ) -> None:
        ...

    # Overload for structured mode
    @overload
    def __init__(
        self,
        raw_dict: None = None,
        *,
        cull_unused_columns: Optional[bool] = None,
        overlay_blending: Optional[Literal['normal', 'additive', 'subtractive']] = None,
        tile_style: Optional[Dict[str, Any]] = None,
        auto_graph_renderer_switching: Optional[bool] = None
    ) -> None:
        ...

    # Actual implementation
    def __init__(
        self,
        raw_dict: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        if raw_dict is not None:
            if not isinstance(raw_dict, dict):
                raise TypeError(f"raw_dict must be a dict, got {type(raw_dict)}")
            self._raw_dict = raw_dict
            self._kwargs = {}
        else:
            self._raw_dict = None
            self._kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary format for Kepler.gl."""
        if self._raw_dict is not None:
            return self._raw_dict
        # Convert snake_case keys to camelCase
        return {snake_to_camel(k): v for k, v in self._kwargs.items()}

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        if self._raw_dict is not None:
            return f"KeplerConfig(raw_dict={self._raw_dict!r})"
        return f"KeplerConfig({self._kwargs!r})"

    def __eq__(self, other) -> bool:
        """Check equality."""
        if not isinstance(other, KeplerConfig):
            return False
        if self._raw_dict is not None and other._raw_dict is not None:
            return self._raw_dict == other._raw_dict
        if self._raw_dict is not None or other._raw_dict is not None:
            return False
        return self._kwargs == other._kwargs


class KeplerEncoding:
    """Immutable container for complete Kepler.gl configuration.

    Combines datasets, layers, options, and config into a complete Kepler visualization
    configuration. Follows an immutable builder pattern where methods return new instances
    rather than modifying in place.

    :param datasets: List of dataset configurations
    :type datasets: Optional[List[KeplerDataset]]
    :param layers: List of layer configurations
    :type layers: Optional[List[KeplerLayer]]
    :param options: Visualization options
    :type options: Optional[KeplerOptions]
    :param config: Map configuration settings
    :type config: Optional[KeplerConfig]

    **Example: Building configuration with method chaining**
        ::

            from graphistry import KeplerEncoding, KeplerDataset, KeplerLayer

            encoding = (
                KeplerEncoding()
                .with_dataset(KeplerDataset(id="companies", type="nodes"))
                .with_dataset(KeplerDataset(id="relationships", type="edges"))
                .with_layer(KeplerLayer({
                    "id": "points",
                    "type": "point",
                    "config": {
                        "dataId": "companies",
                        "columns": {"lat": "latitude", "lng": "longitude"}
                    }
                }))
                .with_options(center_map=True, read_only=False)
                .with_config(cull_unused_columns=True)
            )

    **Example: Using with Plotter**
        ::

            import graphistry

            g = graphistry.nodes(df, 'id')
            g = g.encode_kepler(encoding)
            g.plot()
    """

    def __init__(
        self,
        datasets: Optional[List[KeplerDataset]] = None,
        layers: Optional[List[KeplerLayer]] = None,
        options: Optional[KeplerOptions] = None,
        config: Optional[KeplerConfig] = None
    ):
        self.datasets: List[KeplerDataset] = datasets or []
        self.layers: List[KeplerLayer] = layers or []
        self.options: KeplerOptions = options if options is not None else KeplerOptions()
        self.config: KeplerConfig = config if config is not None else KeplerConfig()

    def with_dataset(self, dataset: KeplerDataset) -> 'KeplerEncoding':
        """Return a new KeplerEncoding with the dataset appended.

        :param dataset: KeplerDataset to append
        :type dataset: KeplerDataset
        :return: New KeplerEncoding instance with the dataset added
        :rtype: KeplerEncoding
        """
        # Create new instance with appended dataset
        return KeplerEncoding(
            datasets=self.datasets + [dataset],
            layers=self.layers,
            options=self.options,
            config=self.config
        )

    def with_layer(self, layer: KeplerLayer) -> 'KeplerEncoding':
        """Return a new KeplerEncoding with the layer appended.

        :param layer: KeplerLayer to append
        :type layer: KeplerLayer
        :return: New KeplerEncoding instance with the layer added
        :rtype: KeplerEncoding
        """
        # Create new instance with appended layer
        return KeplerEncoding(
            datasets=self.datasets,
            layers=self.layers + [layer],
            options=self.options,
            config=self.config
        )

    @overload
    def with_options(self, options: KeplerOptions) -> 'KeplerEncoding':
        ...

    @overload
    def with_options(
        self,
        options: None = None,
        *,
        center_map: Optional[bool] = None,
        read_only: Optional[bool] = None
    ) -> 'KeplerEncoding':
        ...

    def with_options(self, options: Optional[KeplerOptions] = None, **kwargs) -> 'KeplerEncoding':
        """Return a new KeplerEncoding with updated options.

        :param options: KeplerOptions object to replace current options
        :type options: Optional[KeplerOptions]
        :param center_map: Auto-center map on data
        :type center_map: Optional[bool]
        :param read_only: Disable map interactions
        :type read_only: Optional[bool]
        :return: New KeplerEncoding instance with updated options
        :rtype: KeplerEncoding
        """
        if options is not None:
            new_options = options
        else:
            # Merge kwargs with existing options
            existing_dict = self.options.to_dict()
            new_dict = {**existing_dict, **kwargs}
            new_options = KeplerOptions(raw_dict=new_dict)

        return KeplerEncoding(
            datasets=self.datasets,
            layers=self.layers,
            options=new_options,
            config=self.config
        )

    @overload
    def with_config(self, config: KeplerConfig) -> 'KeplerEncoding':
        ...

    @overload
    def with_config(
        self,
        config: None = None,
        *,
        cull_unused_columns: Optional[bool] = None,
        overlay_blending: Optional[Literal['normal', 'additive', 'subtractive']] = None,
        tile_style: Optional[Dict[str, Any]] = None,
        auto_graph_renderer_switching: Optional[bool] = None
    ) -> 'KeplerEncoding':
        ...

    def with_config(self, config: Optional[KeplerConfig] = None, **kwargs) -> 'KeplerEncoding':
        """Return a new KeplerEncoding with updated config.

        :param config: KeplerConfig object to replace current config
        :type config: Optional[KeplerConfig]
        :param cull_unused_columns: Remove columns not used by layers
        :type cull_unused_columns: Optional[bool]
        :param overlay_blending: Blend mode - 'normal', 'additive', 'subtractive'
        :type overlay_blending: Optional[Literal['normal', 'additive', 'subtractive']]
        :param tile_style: Base map tile style configuration
        :type tile_style: Optional[Dict[str, Any]]
        :param auto_graph_renderer_switching: Enable automatic graph renderer switching
        :type auto_graph_renderer_switching: Optional[bool]
        :return: New KeplerEncoding instance with updated config
        :rtype: KeplerEncoding
        """
        if config is not None:
            new_config = config
        else:
            # Merge kwargs with existing config
            existing_dict = self.config.to_dict()
            new_dict = {**existing_dict, **kwargs}
            new_config = KeplerConfig(raw_dict=new_dict)

        return KeplerEncoding(
            datasets=self.datasets,
            layers=self.layers,
            options=self.options,
            config=new_config
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary format for Kepler.gl.

        :return: Dictionary with datasets, layers, options, and config
        :rtype: Dict[str, Any]
        """
        result = {
            'datasets': [d.to_dict() for d in self.datasets] if self.datasets else [],
            'layers': [layer.to_dict() for layer in self.layers] if self.layers else [],
            'options': self.options.to_dict(),
            'config': self.config.to_dict()
        }

        return result

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (
            f"KeplerEncoding("
            f"datasets={self.datasets!r}, "
            f"layers={self.layers!r}, "
            f"options={self.options!r}, "
            f"config={self.config!r})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = []
        if self.datasets:
            parts.append(f"{len(self.datasets)} datasets")
        if self.layers:
            parts.append(f"{len(self.layers)} layers")
        return f"KeplerEncoding({', '.join(parts) if parts else 'empty'})"

    def __eq__(self, other) -> bool:
        """Check equality based on datasets, layers, options, and config."""
        if not isinstance(other, KeplerEncoding):
            return False
        return (
            self.datasets == other.datasets
            and self.layers == other.layers
            and self.options == other.options
            and self.config == other.config
        )
