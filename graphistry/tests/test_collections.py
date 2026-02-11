# -*- coding: utf-8 -*-

import json
import pytest
from urllib.parse import unquote

import graphistry
from graphistry.collections import collection_intersection, collection_set
from graphistry.validate.validate_collections import normalize_collections_url_params


def decode_collections(encoded: str):
    return json.loads(unquote(encoded))


def collections_url_params(collections, **kwargs):
    return graphistry.bind().collections(collections=collections, **kwargs)._url_params


def test_collections_encodes_and_normalizes():
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

    url_params = collections_url_params(
        collections,
        show_collections=True,
        collections_global_node_color="#00FF00",
        collections_global_edge_color="#00AA00",
    )

    decoded = decode_collections(url_params["collections"])
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
    assert url_params["showCollections"] is True
    assert url_params["collectionsGlobalNodeColor"] == "00FF00"
    assert url_params["collectionsGlobalEdgeColor"] == "00AA00"


@pytest.mark.parametrize("expr", [graphistry.n({"vip": True}), [graphistry.n({"vip": True})]])
def test_collection_set_wraps_ast_expr(expr):
    collection = collection_set(expr=expr, id="vip")
    assert collection["expr"]["type"] == "gfql_chain"
    assert collection["expr"]["gfql"][0]["type"] == "Node"


def test_collection_helpers_build_sets_and_intersections():
    collections = [
        collection_set(expr=[graphistry.n({"vip": True})], id="vip", name="VIP", node_color="#FFAA00"),
        collection_intersection(sets=["vip"], id="vip_intersection", name="VIP Intersection", node_color="#00BFFF"),
    ]
    decoded = decode_collections(collections_url_params(collections)["collections"])
    assert decoded[0]["type"] == "set"
    assert decoded[0]["expr"]["type"] == "gfql_chain"
    assert decoded[1]["expr"] == {"type": "intersection", "sets": ["vip"]}


def test_collections_accepts_chain_and_preserves_dataset_id():
    node = graphistry.n({"type": "user"})
    chain = graphistry.Chain([node])
    g2 = graphistry.bind(dataset_id="dataset_123").collections(collections={"type": "set", "id": "my_set", "expr": chain})
    decoded = decode_collections(g2._url_params["collections"])
    assert decoded == [
        {
            "type": "set",
            "id": "my_set",
            "expr": {
                "type": "gfql_chain",
                "gfql": [node.to_json()],
            },
        }
    ]
    assert g2._dataset_id == "dataset_123"


def test_collections_string_input_is_encoded():
    # Include a set so the intersection has valid references
    raw = '[{"type":"set","id":"a","expr":{"type":"gfql_chain","gfql":[{"type":"Node"}]}},{"type":"intersection","id":"b","expr":{"type":"intersection","sets":["a"]}}]'
    url_params = collections_url_params(raw)
    assert url_params["collections"].startswith("%5B")
    decoded = decode_collections(url_params["collections"])
    # Node normalizes to include filter_dict: {}
    assert decoded == [
        {
            "type": "set",
            "id": "a",
            "expr": {"type": "gfql_chain", "gfql": [{"type": "Node", "filter_dict": {}}]},
        },
        {
            "type": "intersection",
            "id": "b",
            "expr": {"type": "intersection", "sets": ["a"]},
        }
    ]


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
    decoded = decode_collections(
        collections_url_params({"type": "set", "id": "users", "expr": chain_json})["collections"]
    )
    assert decoded == [
        {
            "type": "set",
            "id": "users",
            "expr": {
                "type": "gfql_chain",
                "gfql": chain_json["chain"],
            },
        }
    ]


def test_collections_accepts_let_expr():
    dag = graphistry.let({"seed": graphistry.n({"type": "user"})})
    decoded = decode_collections(
        collections_url_params({"type": "set", "id": "users", "expr": dag})["collections"]
    )
    assert decoded[0]["expr"]["type"] == "gfql_chain"
    assert decoded[0]["expr"]["gfql"][0]["type"] == "Let"


def test_collections_drop_unexpected_fields_autofix():
    collections = [
        {
            "type": "set",
            "id": "vip_set",
            "expr": [graphistry.n({"vip": True})],
            "unexpected": "drop-me",
        }
    ]
    decoded = decode_collections(
        collections_url_params(collections, validate="autofix", warn=False)["collections"]
    )
    assert "unexpected" not in decoded[0]


def test_collections_show_collections_coerces_autofix():
    g2 = graphistry.bind().collections(show_collections="true", validate="autofix")
    assert g2._url_params["showCollections"] is True


def test_collections_show_collections_strict_raises():
    with pytest.raises(ValueError):
        graphistry.bind().collections(show_collections="maybe", validate="strict")


def test_collections_validation_strict_raises():
    # Missing 'type' field in GFQL op causes validation error
    bad_collections = [{"type": "set", "id": "bad_set", "expr": [{"filter_dict": {"a": 1}}]}]
    with pytest.raises(ValueError):
        graphistry.bind().collections(collections=bad_collections, validate="strict")


def test_collections_autofix_drops_invalid_colors():
    collections = [
        {
            "type": "set",
            "id": "vip_set",
            "expr": [graphistry.n({"vip": True})],
            "node_color": 123,
            "edge_color": {"bad": True},
        }
    ]
    with pytest.warns(RuntimeWarning):
        url_params = collections_url_params(collections, validate="autofix", warn=True)
    decoded = decode_collections(url_params["collections"])
    assert "node_color" not in decoded[0]
    assert "edge_color" not in decoded[0]


def test_collections_autofix_drops_invalid_gfql_ops():
    # Collection with invalid GFQL op (missing 'type' field) gets dropped in autofix
    collections = [
        {
            "type": "set",
            "id": "bad_set",
            "expr": [graphistry.n({"vip": True}), {"filter_dict": {"a": 1}}],
        }
    ]
    with pytest.warns(RuntimeWarning):
        url_params = collections_url_params(collections, validate="autofix", warn=True)
    # Collection dropped due to invalid GFQL, so no collections key or empty
    assert "collections" not in url_params or decode_collections(url_params["collections"]) == []


def test_plot_url_param_validation_autofix_warns():
    bad = '[{"type":"set","expr":[{"filter_dict":{"a":1}}]}]'
    with pytest.warns(RuntimeWarning):
        normalized = normalize_collections_url_params({"collections": bad}, validate="autofix", warn=True)
    assert "collections" not in normalized or normalized["collections"].startswith("%5B")


def test_collections_autofix_generates_missing_ids():
    # Collections without IDs get auto-generated IDs in autofix mode (kebab-case)
    collections = [
        {"type": "set", "expr": [graphistry.n({"a": 1})]},
        {"type": "intersection", "expr": {"type": "intersection", "sets": ["set-0"]}},
    ]
    with pytest.warns(RuntimeWarning):
        url_params = collections_url_params(collections, validate="autofix", warn=True)
    decoded = decode_collections(url_params["collections"])
    assert decoded[0]["id"] == "set-0"
    assert decoded[1]["id"] == "intersection-1"


def test_collections_intersection_of_intersections():
    # Backend supports intersections-of-intersections (DAG structure)
    collections = [
        {"type": "set", "id": "set_a", "expr": [graphistry.n({"a": 1})]},
        {"type": "set", "id": "set_b", "expr": [graphistry.n({"b": 1})]},
        {"type": "intersection", "id": "inter_ab", "expr": {"type": "intersection", "sets": ["set_a", "set_b"]}},
        {"type": "intersection", "id": "inter_of_inter", "expr": {"type": "intersection", "sets": ["set_a", "inter_ab"]}},
    ]
    url_params = collections_url_params(collections)
    decoded = decode_collections(url_params["collections"])
    assert len(decoded) == 4
    assert decoded[3]["id"] == "inter_of_inter"
    assert decoded[3]["expr"]["sets"] == ["set_a", "inter_ab"]


def test_collections_intersection_self_reference_rejected():
    # Intersection cannot reference itself
    collections = [
        {"type": "set", "id": "set_a", "expr": [graphistry.n({"a": 1})]},
        {"type": "intersection", "id": "bad_inter", "expr": {"type": "intersection", "sets": ["set_a", "bad_inter"]}},
    ]
    with pytest.raises(ValueError):
        collections_url_params(collections, validate="strict")


def test_collections_intersection_cycle_rejected():
    # Cycles in intersection DAG are rejected
    collections = [
        {"type": "set", "id": "set_a", "expr": [graphistry.n({"a": 1})]},
        {"type": "intersection", "id": "inter_a", "expr": {"type": "intersection", "sets": ["set_a", "inter_b"]}},
        {"type": "intersection", "id": "inter_b", "expr": {"type": "intersection", "sets": ["set_a", "inter_a"]}},
    ]
    with pytest.raises(ValueError):
        collections_url_params(collections, validate="strict")


def test_collections_intersection_cycle_autofix_drops():
    # In autofix mode, cyclic intersections are dropped
    collections = [
        {"type": "set", "id": "set_a", "expr": [graphistry.n({"a": 1})]},
        {"type": "intersection", "id": "inter_a", "expr": {"type": "intersection", "sets": ["set_a", "inter_b"]}},
        {"type": "intersection", "id": "inter_b", "expr": {"type": "intersection", "sets": ["set_a", "inter_a"]}},
    ]
    with pytest.warns(RuntimeWarning):
        url_params = collections_url_params(collections, validate="autofix", warn=True)
    decoded = decode_collections(url_params["collections"])
    # Both cyclic intersections dropped, only set remains
    assert len(decoded) == 1
    assert decoded[0]["id"] == "set_a"


def test_collections_malformed_ast_autofix_drops():
    # AST from_json uses bare asserts - these should be caught, not crash
    # {"type": "Let"} missing required 'bindings' field
    from graphistry.validate.validate_collections import normalize_collections
    collections = [
        {"type": "set", "id": "good", "expr": [{"type": "Node"}]},
        {"type": "set", "id": "bad-let", "expr": [{"type": "Let"}]},  # missing bindings
    ]
    result = normalize_collections(collections, validate="autofix", warn=False)
    ids = [c.get("id") for c in result]
    assert "good" in ids
    assert "bad-let" not in ids


def test_collections_malformed_ast_strict_raises():
    from graphistry.validate.validate_collections import normalize_collections
    collections = [{"type": "set", "id": "bad", "expr": [{"type": "Let"}]}]
    with pytest.raises(ValueError):
        normalize_collections(collections, validate="strict")
