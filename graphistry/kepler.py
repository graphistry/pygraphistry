"""
Kepler.gl encoding classes for managing datasets and layers.

This module provides classes for building Kepler.gl visualizations using an immutable pattern.
Type specifications are documented in kepler_types.py (specification only, not used at runtime).
"""

from typing import Optional, List, Dict, Any
import uuid


class KeplerDataset:
    """
    Represents a Kepler.gl dataset with type-specific configuration.

    Supports multiple dataset types:
    - 'nodes': Node data from graph
    - 'edges': Edge data from graph
    - 'countries': Country/region polygons
    - 'states'/'provinces'/'firstOrderAdminRegions': State/province polygons
    - None: Native Kepler dataset

    Args:
        raw_dict: Optional raw dictionary containing a native Kepler.gl dataset configuration.
                  If provided, the dict is passed through unmodified to Kepler.gl, allowing
                  direct use of native Kepler dataset formats. All other parameters are ignored.
        id: Dataset identifier (required for serialization)
        type: Dataset type (nodes, edges, countries, etc.)
        label: Optional display label (defaults to id)
        include: Optional list of columns to include
        exclude: Optional list of columns to exclude
        **kwargs: Type-specific parameters

    Example:
        >>> dataset = KeplerDataset(id="my-dataset", type="nodes")
        >>> dataset = KeplerDataset(id="countries", type="countries", resolution=50)
        >>> # Pass through native Kepler dataset dict
        >>> native_kepler = {"info": {"id": "my-dataset"}, "data": {...}}
        >>> dataset = KeplerDataset(native_kepler)
    """

    _raw_dict: Optional[Dict[str, Any]]
    id: Optional[str]
    type: Optional[str]
    label: Optional[str]
    _kwargs: Dict[str, Any]

    def __init__(
        self,
        raw_dict: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        type: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs
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
    """
    Represents a Kepler.gl layer with native Kepler configuration.

    Currently only supports raw dictionary pass-through mode.

    Supports layer types: point, arc, line, grid, hexagon, geojson, cluster,
    icon, heatmap, hexagonId, trip, s2

    Args:
        raw_dict: Raw dictionary containing a native Kepler.gl layer configuration.
                  The dict is passed through unmodified to Kepler.gl.

    Example:
        >>> # Pass through native Kepler layer dict
        >>> native_layer = {
        ...     "id": "layer-1",
        ...     "type": "point",
        ...     "config": {
        ...         "dataId": "my-dataset",
        ...         "columns": {"lat": "latitude", "lng": "longitude"}
        ...     }
        ... }
        >>> layer = KeplerLayer(native_layer)
    """

    _raw_dict: Dict[str, Any]
    # id: Optional[str]
    # type: Optional[str]
    # label: Optional[str]
    # _kwargs: Dict[str, Any]

    def __init__(
        self,
        raw_dict: Dict[str, Any]
    ):
        if not isinstance(raw_dict, dict):
            raise TypeError(f"raw_dict must be a dict, got {type(raw_dict)}")
        self._raw_dict = raw_dict
        # Extract and store as public properties
        # self.id = None
        # self.type = None
        # self.label = None
        # self._kwargs = {}

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


class KeplerEncoding:
    """
    Immutable container for Kepler.gl encoding configuration.

    Follows the immutable pattern used in GFQL Chain. Operations return new instances
    rather than modifying in place.

    Args:
        datasets: List of KeplerDataset objects
        layers: List of KeplerLayer objects
        options: Kepler options (centerMap, readOnly, etc.)
        config: Kepler config (cullUnusedColumns, overlayBlending, etc.)

    Example:
        >>> encoding = KeplerEncoding()
        >>> encoding = encoding.with_dataset(KeplerDataset(id="data1", type="nodes"))
        >>> encoding = encoding.with_layer(KeplerLayer(id="layer1", type="point", dataId="data1"))
        >>> encoding.to_dict()
    """

    def __init__(
        self,
        datasets: Optional[List[KeplerDataset]] = None,
        layers: Optional[List[KeplerLayer]] = None,
        options: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.datasets: List[KeplerDataset] = datasets or []
        self.layers: List[KeplerLayer] = layers or []
        self.options: Dict[str, Any] = options or {}
        self.config: Dict[str, Any] = config or {}

    def with_dataset(self, dataset: KeplerDataset) -> 'KeplerEncoding':
        """
        Return a new KeplerEncoding with the dataset appended.

        Args:
            dataset: KeplerDataset to append

        Returns:
            New KeplerEncoding instance with the dataset added
        """
        # Create new instance with appended dataset
        return KeplerEncoding(
            datasets=self.datasets + [dataset],
            layers=self.layers,
            options=self.options,
            config=self.config
        )

    def with_layer(self, layer: KeplerLayer) -> 'KeplerEncoding':
        """
        Return a new KeplerEncoding with the layer appended.

        Args:
            layer: KeplerLayer to append

        Returns:
            New KeplerEncoding instance with the layer added
        """
        # Create new instance with appended layer
        return KeplerEncoding(
            datasets=self.datasets,
            layers=self.layers + [layer],
            options=self.options,
            config=self.config
        )

    def with_options(self, **options) -> 'KeplerEncoding':
        """
        Return a new KeplerEncoding with updated options.

        Args:
            **options: Options to set (centerMap, readOnly, keepExistingConfig)

        Returns:
            New KeplerEncoding instance with updated options
        """
        new_options = {**self.options, **options}
        return KeplerEncoding(
            datasets=self.datasets,
            layers=self.layers,
            options=new_options,
            config=self.config
        )

    def with_config(self, **config) -> 'KeplerEncoding':
        """
        Return a new KeplerEncoding with updated config.

        Args:
            **config: Config to set (cullUnusedColumns, overlayBlending, tileStyle)

        Returns:
            New KeplerEncoding instance with updated config
        """
        new_config = {**self.config, **config}
        return KeplerEncoding(
            datasets=self.datasets,
            layers=self.layers,
            options=self.options,
            config=new_config
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary format for Kepler.gl.

        Returns:
            Dictionary with datasets, layers, options, and config
        """
        result = {
            'datasets': [d.to_dict() for d in self.datasets] if self.datasets else [],
            'layers': [layer.to_dict() for layer in self.layers] if self.layers else [],
            'options': self.options if self.options else {},
            'config': self.config if self.config else {}
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
