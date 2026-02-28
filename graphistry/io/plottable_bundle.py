"""
Plottable bundle adapter for serializing/deserializing Plottable objects.

Domain-aware layer that maps Plottable fields to bundle artifacts.
Provides to_file() and from_file() with field group constants and
tripwire-testable canonical field lists.
"""
import copy
import json
import os
import platform
import warnings
from collections import UserDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

from graphistry.io.bundle import (
    BundleReadReport,
    BundleWriteReport,
    _require_pydantic,
    detect_format,
    finalize_bundle,
    prepare_bundle_dir,
    read_df_parquet,
    read_manifest,
    sha256_file,
    write_df_parquet,
    write_manifest,
    zip_to_dir,
)

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable

# ---------------------------------------------------------------------------
# Field group constants — canonical lists for tripwire tests
# ---------------------------------------------------------------------------

TIER1_DF_FIELDS: List[str] = [
    "_edges",
    "_nodes",
]

TIER1_BINDING_FIELDS: List[str] = [
    "_source",
    "_destination",
    "_node",
    "_edge",
    "_edge_title",
    "_edge_label",
    "_edge_color",
    "_edge_source_color",
    "_edge_destination_color",
    "_edge_size",
    "_edge_weight",
    "_edge_icon",
    "_edge_opacity",
    "_point_title",
    "_point_label",
    "_point_color",
    "_point_size",
    "_point_weight",
    "_point_icon",
    "_point_opacity",
    "_point_x",
    "_point_y",
    "_point_longitude",
    "_point_latitude",
]

TIER1_DISPLAY_FIELDS: List[str] = [
    "_height",
    "_render",
    "_url_params",
    "_name",
    "_description",
    "_style",
    "_complex_encodings",
]

TIER1_REMOTE_FIELDS: List[str] = [
    "_dataset_id",
    "_url",
    "_nodes_file_id",
    "_edges_file_id",
    "_privacy",
]

TIER2_DF_FIELDS: List[str] = [
    "_node_embedding",
    "_node_features",
    "_node_features_raw",
    "_node_target",
    "_node_target_raw",
    "_edge_embedding",
    "_edge_features",
    "_edge_features_raw",
    "_edge_target",
    "_edge_target_raw",
    "_weighted_edges_df",
    "_weighted_edges_df_from_nodes",
    "_weighted_edges_df_from_edges",
    "_xy",
]

TIER2_JSON_ALGO_FIELDS: List[str] = [
    "_umap_engine",
    "_umap_params",
    "_umap_fit_kwargs",
    "_umap_transform_kwargs",
    "_n_components",
    "_metric",
    "_n_neighbors",
    "_min_dist",
    "_spread",
    "_local_connectivity",
    "_repulsion_strength",
    "_negative_sample_rate",
    "_suffix",
    "_dbscan_engine",
    "_dbscan_params",
    "_collapse_node_col",
    "_collapse_src_col",
    "_collapse_dst_col",
]

TIER2_JSON_KG_FIELDS: List[str] = [
    "_relation",
    "_use_feat",
    "_triplets",
    "_kg_embed_dim",
]

TIER2_JSON_LAYOUT_FIELDS: List[str] = [
    "_partition_offsets",
]

TIER2_JSON_INDEX_FIELDS: List[str] = [
    "_entity_to_index",
    "_index_to_entity",
]

# Objects that can't be serialized as JSON/parquet in v1
TIER3_FIELDS: List[str] = [
    "_umap",
    "_node_encoder",
    "_node_target_encoder",
    "_edge_encoder",
    "_weighted_adjacency",
    "_weighted_adjacency_nodes",
    "_weighted_adjacency_edges",
    "_adjacency",
    "_dbscan_nodes",
    "_dbscan_edges",
]

# Never serialized: session/auth/driver/DGL state
NEVER_FIELDS: List[str] = [
    "session",
    "_pygraphistry",
    "_bolt_driver",
    "_tigergraph",
    "DGL_graph",
    "_dgl_graph",
]

ALL_KNOWN_FIELDS: List[str] = sorted(set(
    TIER1_DF_FIELDS
    + TIER1_BINDING_FIELDS
    + TIER1_DISPLAY_FIELDS
    + TIER1_REMOTE_FIELDS
    + TIER2_DF_FIELDS
    + TIER2_JSON_ALGO_FIELDS
    + TIER2_JSON_KG_FIELDS
    + TIER2_JSON_LAYOUT_FIELDS
    + TIER2_JSON_INDEX_FIELDS
    + TIER3_FIELDS
    + NEVER_FIELDS
))


SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_val(val: Any) -> Any:
    """Convert values that aren't directly JSON-serializable."""
    if val is None:
        return None
    if isinstance(val, UserDict):
        return dict(val)
    if isinstance(val, dict):
        return {k: _safe_json_val(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_safe_json_val(v) for v in val]
    if isinstance(val, (str, int, float, bool)):
        return val
    # Fall back to str for unserializable types
    try:
        json.dumps(val)
        return val
    except (TypeError, ValueError):
        return str(val)


def _get_field(g: 'Plottable', field_name: str) -> Any:
    """Safely get a field from a Plottable, returning None if missing."""
    try:
        return getattr(g, field_name)
    except AttributeError:
        return None


def _collect_json_fields(
    g: 'Plottable', field_names: List[str]
) -> Dict[str, Any]:
    """Collect fields as a JSON-safe dict, omitting None values."""
    out: Dict[str, Any] = {}
    for name in field_names:
        val = _get_field(g, name)
        if val is not None:
            out[name] = _safe_json_val(val)
    return out


# ---------------------------------------------------------------------------
# to_file
# ---------------------------------------------------------------------------

def to_file(
    g: 'Plottable',
    path: str,
    format: Optional[str] = None,
) -> Tuple['Plottable', BundleWriteReport]:
    """Save a Plottable graph to disk as a bundle (directory or zip).

    :param g: Plottable object to serialize
    :param path: Destination path (directory or .zip file)
    :param format: None for directory (default), "zip" for zip archive
    :return: Tuple of (original Plottable, BundleWriteReport)
    :raises RuntimeError: If edges DataFrame is missing
    :raises ImportError: If pydantic >= 2.0 is not installed
    """
    _require_pydantic()
    report = BundleWriteReport()

    if _get_field(g, '_edges') is None:
        raise RuntimeError(
            "Cannot save bundle: edges DataFrame is required. "
            "Set edges with g.edges(df, 'src', 'dst') first."
        )

    bundle_dir = prepare_bundle_dir(path, format)

    try:
        artifacts: Dict[str, Dict[str, str]] = {}
        files: Dict[str, str] = {}

        # --- Tier 1 DataFrames ---
        edges_art = write_df_parquet(g._edges, '_edges', bundle_dir, report)
        if edges_art is None:
            raise RuntimeError("Failed to write edges parquet — aborting bundle save")
        artifacts['_edges'] = edges_art
        files[edges_art['path']] = edges_art['sha256']

        nodes_art = write_df_parquet(
            _get_field(g, '_nodes'), '_nodes', bundle_dir, report
        )
        if nodes_art is not None:
            artifacts['_nodes'] = nodes_art
            files[nodes_art['path']] = nodes_art['sha256']

        # --- Tier 2 DataFrames ---
        for field_name in TIER2_DF_FIELDS:
            val = _get_field(g, field_name)
            if val is not None and isinstance(val, pd.DataFrame):
                art = write_df_parquet(val, field_name, bundle_dir, report)
                if art is not None:
                    artifacts[field_name] = art
                    files[art['path']] = art['sha256']
            elif val is not None:
                report.warnings.append(
                    f"{field_name}: not a DataFrame ({type(val).__name__}), skipping"
                )
                report.artifacts_skipped.append(field_name)

        # --- Build manifest ---
        from graphistry.io.metadata import serialize_plottable_metadata
        try:
            from graphistry._version import get_versions
            graphistry_version = get_versions()["version"]
        except Exception:
            graphistry_version = "unknown"

        manifest: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "graphistry_version": graphistry_version,
            "plottable_metadata": serialize_plottable_metadata(g),
            "settings": {
                "height": _get_field(g, '_height'),
                "render": _get_field(g, '_render'),
                "url_params": _safe_json_val(_get_field(g, '_url_params')),
            },
            "remote": {
                "dataset_id": _get_field(g, '_dataset_id'),
                "url": _get_field(g, '_url'),
                "nodes_file_id": _get_field(g, '_nodes_file_id'),
                "edges_file_id": _get_field(g, '_edges_file_id'),
                "privacy": _safe_json_val(_get_field(g, '_privacy')),
            },
            "algorithm_config": _collect_json_fields(g, TIER2_JSON_ALGO_FIELDS),
            "kg_config": _collect_json_fields(g, TIER2_JSON_KG_FIELDS),
            "layout": _collect_json_fields(g, TIER2_JSON_LAYOUT_FIELDS),
            "graph_indices": _collect_json_fields(g, TIER2_JSON_INDEX_FIELDS),
            "artifacts": artifacts,
            "files": files,
        }

        write_manifest(manifest, bundle_dir)
        finalize_bundle(bundle_dir, path, format)

    except Exception:
        # Clean up temp dir on failure if zipping
        if format == "zip" and bundle_dir != path:
            import shutil
            shutil.rmtree(bundle_dir, ignore_errors=True)
        raise

    return (g, report)


# ---------------------------------------------------------------------------
# from_file
# ---------------------------------------------------------------------------

def from_file(
    path: str,
    restore_remote: bool = False,
) -> Tuple['Plottable', BundleReadReport]:
    """Load a Plottable graph from a bundle on disk.

    :param path: Path to bundle directory or .zip file
    :param restore_remote: If True, restore remote server state (dataset_id, url, etc.)
    :return: Tuple of (Plottable, BundleReadReport)
    :raises ImportError: If pydantic >= 2.0 is not installed
    :raises FileNotFoundError: If path doesn't exist
    """
    _require_pydantic()
    report = BundleReadReport()

    fmt = detect_format(path)
    bundle_dir = path
    tmp_dir = None

    try:
        if fmt == "zip":
            import tempfile
            tmp_dir = tempfile.mkdtemp(prefix="graphistry_bundle_read_")
            zip_to_dir(path, tmp_dir)
            bundle_dir = tmp_dir

        manifest = read_manifest(bundle_dir)
        artifacts = manifest.get("artifacts", {})
        file_shas = manifest.get("files", {})

        # --- Verify file integrity ---
        for rel_path, expected_sha in file_shas.items():
            abs_path = os.path.join(bundle_dir, rel_path)
            if not os.path.exists(abs_path):
                report.warnings.append(f"Missing file: {rel_path}")
                report.integrity_ok = False
                continue
            actual_sha = sha256_file(abs_path)
            if actual_sha != expected_sha:
                report.warnings.append(
                    f"SHA256 mismatch: {rel_path} "
                    f"(expected {expected_sha[:16]}..., got {actual_sha[:16]}...)"
                )
                report.integrity_ok = False

        # --- Hydration sequence ---
        import graphistry
        from graphistry.io.metadata import deserialize_plottable_metadata

        g: 'Plottable' = graphistry.bind()  # type: ignore[assignment]

        # Load edges
        if '_edges' in artifacts:
            art = artifacts['_edges']
            edges_df = read_df_parquet(
                art['path'], bundle_dir, art.get('sha256'), report
            )
            if edges_df is not None:
                report.artifacts_loaded.append('_edges')
                g = g.edges(edges_df)
        else:
            report.warnings.append("No edges artifact in manifest")

        # Load nodes
        if '_nodes' in artifacts:
            art = artifacts['_nodes']
            nodes_df = read_df_parquet(
                art['path'], bundle_dir, art.get('sha256'), report
            )
            if nodes_df is not None:
                report.artifacts_loaded.append('_nodes')
                g = g.nodes(nodes_df)

        # Apply plottable metadata (bindings, encodings, name, desc, style)
        plottable_metadata = manifest.get("plottable_metadata", {})
        if plottable_metadata:
            g = deserialize_plottable_metadata(plottable_metadata, g)  # type: ignore[assignment]

        # Create final copy for direct field assignment
        result = copy.copy(g)

        # Restore settings directly (avoid .settings() merge semantics)
        settings = manifest.get("settings", {})
        if "height" in settings and settings["height"] is not None:
            result._height = settings["height"]
        if "render" in settings and settings["render"] is not None:
            result._render = settings["render"]
        if "url_params" in settings and settings["url_params"] is not None:
            result._url_params = settings["url_params"]

        # --- Load Tier 2 DF artifacts ---
        for field_name in TIER2_DF_FIELDS:
            if field_name in artifacts:
                art = artifacts[field_name]
                df = read_df_parquet(
                    art['path'], bundle_dir, art.get('sha256'), report
                )
                if df is not None:
                    setattr(result, field_name, df)
                    report.artifacts_loaded.append(field_name)
                else:
                    report.artifacts_skipped.append(field_name)

        # --- Restore Tier 2 JSON fields ---
        algo_config = manifest.get("algorithm_config", {})
        for field_name in TIER2_JSON_ALGO_FIELDS:
            if field_name in algo_config:
                setattr(result, field_name, algo_config[field_name])

        kg_config = manifest.get("kg_config", {})
        for field_name in TIER2_JSON_KG_FIELDS:
            if field_name in kg_config:
                setattr(result, field_name, kg_config[field_name])

        layout = manifest.get("layout", {})
        for field_name in TIER2_JSON_LAYOUT_FIELDS:
            if field_name in layout:
                setattr(result, field_name, layout[field_name])

        graph_indices = manifest.get("graph_indices", {})
        for field_name in TIER2_JSON_INDEX_FIELDS:
            if field_name in graph_indices:
                setattr(result, field_name, graph_indices[field_name])

        # --- Remote state ---
        remote = manifest.get("remote", {})
        has_remote = any(
            remote.get(k) is not None
            for k in ["dataset_id", "url", "nodes_file_id", "edges_file_id"]
        )
        if has_remote:
            if restore_remote:
                if remote.get("dataset_id") is not None:
                    result._dataset_id = remote["dataset_id"]
                if remote.get("url") is not None:
                    result._url = remote["url"]
                if remote.get("nodes_file_id") is not None:
                    result._nodes_file_id = remote["nodes_file_id"]
                if remote.get("edges_file_id") is not None:
                    result._edges_file_id = remote["edges_file_id"]
                if remote.get("privacy") is not None:
                    result._privacy = remote["privacy"]
            else:
                report.remote_state_skipped = True
                warnings.warn(
                    "Bundle contains remote server state (dataset_id, url, etc.) "
                    "which was not restored. Pass restore_remote=True to restore it.",
                    UserWarning,
                    stacklevel=2,
                )

        return (result, report)

    finally:
        if tmp_dir is not None:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
