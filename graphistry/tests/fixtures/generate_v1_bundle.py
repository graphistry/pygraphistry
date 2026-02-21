#!/usr/bin/env python
"""Generate the v1_bundle golden fixture.

Run once to create/refresh the fixture files:
    python graphistry/tests/fixtures/generate_v1_bundle.py
"""
import hashlib
import json
import os

import pandas as pd


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    fixture_dir = os.path.join(here, 'v1_bundle')
    data_dir = os.path.join(fixture_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Create edges
    edges_df = pd.DataFrame({
        's': ['a', 'b', 'c'],
        'd': ['b', 'c', 'a'],
        'w': [1.0, 2.0, 3.0],
    })
    edges_path = os.path.join(data_dir, '_edges.parquet')
    edges_df.to_parquet(edges_path)

    # Create nodes
    nodes_df = pd.DataFrame({
        'id': ['a', 'b', 'c'],
        'label': ['Node A', 'Node B', 'Node C'],
    })
    nodes_path = os.path.join(data_dir, '_nodes.parquet')
    nodes_df.to_parquet(nodes_path)

    # Create xy (tier 2)
    xy_df = pd.DataFrame({
        'x': [0.1, 0.2, 0.3],
        'y': [0.4, 0.5, 0.6],
    })
    xy_path = os.path.join(data_dir, '_xy.parquet')
    xy_df.to_parquet(xy_path)

    edges_sha = sha256_file(edges_path)
    nodes_sha = sha256_file(nodes_path)
    xy_sha = sha256_file(xy_path)

    manifest = {
        'schema_version': '1.0',
        'created_at': '2025-01-01T00:00:00+00:00',
        'python_version': '3.10.0',
        'graphistry_version': '0.35.0',
        'plottable_metadata': {
            'bindings': {
                'source': 's',
                'destination': 'd',
                'node': 'id',
                'edge_weight': 'w',
            },
            'encodings': {
                'edge_weight': 'w',
            },
            'metadata': {
                'name': 'Golden Test Graph',
                'description': 'A test graph for v1 bundle compatibility',
            },
        },
        'settings': {
            'height': 600,
            'render': 'g',
            'url_params': {'info': 'true', 'play': '2000'},
        },
        'remote': {
            'dataset_id': 'golden_dataset_123',
            'url': 'https://hub.graphistry.com/graph/golden_dataset_123',
            'nodes_file_id': None,
            'edges_file_id': None,
            'privacy': None,
        },
        'algorithm_config': {
            '_n_components': 2,
            '_metric': 'euclidean',
        },
        'kg_config': {},
        'layout': {},
        'graph_indices': {},
        'artifacts': {
            '_edges': {
                'kind': 'parquet',
                'path': 'data/_edges.parquet',
                'sha256': edges_sha,
            },
            '_nodes': {
                'kind': 'parquet',
                'path': 'data/_nodes.parquet',
                'sha256': nodes_sha,
            },
            '_xy': {
                'kind': 'parquet',
                'path': 'data/_xy.parquet',
                'sha256': xy_sha,
            },
        },
        'files': {
            'data/_edges.parquet': edges_sha,
            'data/_nodes.parquet': nodes_sha,
            'data/_xy.parquet': xy_sha,
        },
    }

    manifest_path = os.path.join(fixture_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'Golden fixture generated at {fixture_dir}')
    print(f'  edges sha: {edges_sha}')
    print(f'  nodes sha: {nodes_sha}')
    print(f'  xy sha:    {xy_sha}')


if __name__ == '__main__':
    main()
