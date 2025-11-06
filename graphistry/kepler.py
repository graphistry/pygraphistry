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
        id: Dataset identifier (required for serialization)
        type: Dataset type (nodes, edges, countries, etc.)
        label: Optional display label (defaults to id)
        include: Optional list of columns to include
        exclude: Optional list of columns to exclude
        **kwargs: Type-specific parameters

    Example:
        >>> dataset = KeplerDataset(id="my-dataset", type="nodes")
        >>> dataset = KeplerDataset(id="countries", type="countries", resolution=50)
    """

    def __init__(
        self,
        id: Optional[str] = None,
        type: Optional[str] = None,
        label: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        **kwargs
    ):
        self.id = id
        self.type = type
        self.label = label
        self.include = include
        self.exclude = exclude
        self.kwargs = kwargs  # Store type-specific parameters

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary format for Kepler.gl."""
        result: Dict[str, Any] = {
            'info': {
                'id': self.id,
                **(({'label': self.label} if self.label else {}))
            }
        }

        if self.type:
            result['type'] = self.type

        if self.include is not None:
            result['include'] = self.include

        if self.exclude is not None:
            result['exclude'] = self.exclude

        # Add type-specific parameters
        result.update(self.kwargs)

        return result

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        parts = [f"id={self.id!r}"]
        if self.type:
            parts.append(f"type={self.type!r}")
        if self.label:
            parts.append(f"label={self.label!r}")
        return f"KeplerDataset({', '.join(parts)})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        type_str = f":{self.type}" if self.type else ""
        return f"KeplerDataset(id={self.id}{type_str})"

    def __eq__(self, other) -> bool:
        """Check equality based on all properties."""
        if not isinstance(other, KeplerDataset):
            return False
        return (
            self.id == other.id
            and self.type == other.type
            and self.label == other.label
            and self.include == other.include
            and self.exclude == other.exclude
            and self.kwargs == other.kwargs
        )


class KeplerLayer:
    """
    Represents a Kepler.gl layer with type-specific configuration.

    Supports layer types: point, arc, line, grid, hexagon, geojson, cluster,
    icon, heatmap, hexagonId, trip, s2

    Args:
        id: Layer identifier (required for serialization)
        type: Layer type (point, arc, line, etc.)
        dataId: Dataset ID this layer references
        label: Optional display label
        columns: Column mappings (lat, lng, lat0, lng0, etc.)
        **kwargs: Additional layer configuration

    Example:
        >>> layer = KeplerLayer(
        ...     id="my-layer",
        ...     type="point",
        ...     dataId="my-dataset",
        ...     columns={'lat': 'latitude', 'lng': 'longitude'}
        ... )
    """

    def __init__(
        self,
        id: Optional[str] = None,
        type: Optional[str] = None,
        dataId: Optional[str] = None,
        label: Optional[str] = None,
        columns: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        self.id = id
        self.type = type
        self.dataId = dataId
        self.label = label
        self.columns = columns
        self.kwargs = kwargs  # Store additional config

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary format for Kepler.gl."""
        result: Dict[str, Any] = {
            'id': self.id,
            'type': self.type,
            'config': {
                'dataId': self.dataId,
                **(({'label': self.label} if self.label else {})),
                **(({'columns': self.columns} if self.columns else {}))
            }
        }

        # Merge additional config into config object
        if self.kwargs:
            result['config'].update(self.kwargs)

        return result

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        parts = [f"id={self.id!r}"]
        if self.type:
            parts.append(f"type={self.type!r}")
        if self.dataId:
            parts.append(f"dataId={self.dataId!r}")
        return f"KeplerLayer({', '.join(parts)})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        type_str = f":{self.type}" if self.type else ""
        return f"KeplerLayer(id={self.id}{type_str})"

    def __eq__(self, other) -> bool:
        """Check equality based on all properties."""
        if not isinstance(other, KeplerLayer):
            return False
        return (
            self.id == other.id
            and self.type == other.type
            and self.dataId == other.dataId
            and self.label == other.label
            and self.columns == other.columns
            and self.kwargs == other.kwargs
        )


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

        Auto-generates an ID if the dataset doesn't have one.

        Args:
            dataset: KeplerDataset to append

        Returns:
            New KeplerEncoding instance with the dataset added
        """
        # Auto-generate ID if not provided
        if dataset.id is None:
            dataset = KeplerDataset(
                id=self._generate_id('dataset'),
                type=dataset.type,
                label=dataset.label,
                include=dataset.include,
                exclude=dataset.exclude,
                **dataset.kwargs
            )

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

        Auto-generates an ID if the layer doesn't have one.

        Args:
            layer: KeplerLayer to append

        Returns:
            New KeplerEncoding instance with the layer added
        """
        # Auto-generate ID if not provided
        if layer.id is None:
            layer = KeplerLayer(
                id=self._generate_id('layer'),
                type=layer.type,
                dataId=layer.dataId,
                label=layer.label,
                columns=layer.columns,
                **layer.kwargs
            )

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

    def _generate_id(self, prefix: str) -> str:
        """
        Generate a unique ID for datasets or layers.

        Args:
            prefix: Prefix for the ID (e.g., 'dataset' or 'layer')

        Returns:
            Unique ID string
        """
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

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
