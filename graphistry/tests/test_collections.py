# -*- coding: utf-8 -*-

import json
import pytest
from urllib.parse import quote, unquote

import graphistry
from graphistry.collections_helpers import collection_intersection, collection_set
from graphistry.validate.validate_collections import normalize_collections_url_params


def decode_collections(encoded: str):
    return json.loads(unquote(encoded))


def test_collections_encodes_and_normalizes():
    g = graphistry.bind()
    node_filter = graphistry.n({"subscribed_to_newsletter": True})
    collections = [
        {
            "type": "set",
            "id": "newsletter_subscribers",
            "name": "Newsletter Subscribers",
            "node_color": "#32CD32",
            "expr": {
                "type": "gfql_chain",
                "gfql": [node_filter],
            },
        }
    ]

    g2 = g.collections(
        collections=collections,
        show_collections=True,
        collections_global_node_color="#00FF00",
        collections_global_edge_color="#00AA00",
    )

    decoded = decode_collections(g2._url_params["collections"])
    assert decoded == [
        {
            "type": "set",
            "id": "newsletter_subscribers",
            "name": "Newsletter Subscribers",
            "node_color": "#32CD32",
            "expr": {
                "type": "gfql_chain",
                "gfql": [node_filter.to_json()],
            },
        }
    ]
    assert g2._url_params["showCollections"] is True
    assert g2._url_params["collectionsGlobalNodeColor"] == "00FF00"
    assert g2._url_params["collectionsGlobalEdgeColor"] == "00AA00"


def test_collection_helpers_build_sets_and_intersections():
    collections = [
        collection_set(
            expr=[graphistry.n({"vip": True})],
            id="vip",
            name="VIP",
            node_color="#FFAA00",
        ),
        collection_intersection(
            sets=["vip"],
            name="VIP Intersection",
            node_color="#00BFFF",
        ),
    ]

    g2 = graphistry.bind().collections(collections=collections)
    decoded = decode_collections(g2._url_params["collections"])
    assert decoded[0]["type"] == "set"
    assert decoded[0]["expr"]["type"] == "gfql_chain"
    assert decoded[1]["type"] == "intersection"
    assert decoded[1]["expr"] == {"type": "intersection", "sets": ["vip"]}


def test_collection_set_wraps_ast_expr():
    collection = collection_set(expr=graphistry.n({"vip": True}), id="vip")
    assert collection["expr"]["type"] == "gfql_chain"
    assert collection["expr"]["gfql"][0]["type"] == "Node"


def test_collections_accepts_chain_and_preserves_dataset_id():
    chain = graphistry.Chain([graphistry.n({"type": "user"})])
    g1 = graphistry.bind(dataset_id="dataset_123")

    g2 = g1.collections(collections={"type": "set", "expr": chain})

    decoded = decode_collections(g2._url_params["collections"])
    assert decoded == [
        {
            "type": "set",
            "expr": {
                "type": "gfql_chain",
                "gfql": [graphistry.n({"type": "user"}).to_json()],
            },
        }
    ]
    assert g2._dataset_id == "dataset_123"


def test_collections_encode_false_keeps_string():
    raw = json.dumps([{"type": "intersection", "expr": {"type": "intersection", "sets": ["a"]}}], separators=(",", ":"))
    encoded = quote(raw, safe="")
    g2 = graphistry.bind().collections(collections=encoded, encode=False)
    assert g2._url_params["collections"] == encoded


def test_collections_accepts_wire_protocol_chain():
    chain_json = {
        "type": "Chain",
        "chain": [
            {
                "type": "Node",
                "filter_dict": {
                    "type": "user"
                }
            }
        ]
    }
    g2 = graphistry.bind().collections(collections={"type": "set", "expr": chain_json})
    decoded = decode_collections(g2._url_params["collections"])
    assert decoded == [
        {
            "type": "set",
            "expr": {
                "type": "gfql_chain",
                "gfql": chain_json["chain"],
            },
        }
    ]


def test_collections_validation_strict_raises():
    bad_collections = [{"type": "set", "expr": [{"filter_dict": {"a": 1}}]}]
    with pytest.raises(ValueError):
        graphistry.bind().collections(collections=bad_collections, validate="strict")


def test_plot_url_param_validation_autofix_warns():
    bad = '[{"type":"set","expr":[{"filter_dict":{"a":1}}]}]'
    with pytest.warns(RuntimeWarning):
        normalized = normalize_collections_url_params({"collections": bad}, validate="autofix", warn=True)
    assert "collections" not in normalized or normalized["collections"].startswith("%5B")
