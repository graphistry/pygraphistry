"""
Synthetic test fixtures for Microsoft Sentinel Graph API responses.

These fixtures mimic the structure of actual Sentinel Graph API responses
for testing purposes without relying on real threat intelligence data.
"""

import json
from typing import Dict, Any, List


def _create_node_json(node_id: str, label: str, properties: Dict[str, Any]) -> str:
    """Helper to create a JSON-encoded node string."""
    node = {
        "_id": node_id,
        "_label": label,
        **properties
    }
    return json.dumps(node)


def _create_edge_json(
    edge_id: str,
    source_id: str,
    target_id: str,
    label: str,
    properties: Dict[str, Any]
) -> str:
    """Helper to create a JSON-encoded edge string."""
    edge = {
        "_id": edge_id,
        "_sourceId": source_id,
        "_targetId": target_id,
        "_label": label,
        **properties
    }
    return json.dumps(edge)


def _create_graph_node(node_id: str) -> Dict[str, Any]:
    """Helper to create a Graph.Nodes entry."""
    return {
        "Id": node_id,
        "Properties": [],
        "Labels": []
    }


def _wrap_response(cols: List[str]) -> Dict[str, Any]:
    """
    Wrap Cols list in full response structure.

    Args:
        cols: List of JSON-encoded strings (nodes and edges)

    Returns:
        Full response dict matching Sentinel Graph API structure
    """
    # Extract node IDs for Graph.Nodes section
    node_ids = []
    for col_value in cols:
        try:
            obj = json.loads(col_value)
            if "_label" in obj and "_sourceId" not in obj:
                node_ids.append(obj["_id"])
        except json.JSONDecodeError:
            pass

    return {
        "Graph": {
            "Nodes": [_create_graph_node(nid) for nid in node_ids],
            "Edges": []
        },
        "RawData": {
            "Rows": [
                {
                    "Cols": [{"Value": val} for val in cols]
                }
            ]
        }
    }


def get_minimal_response() -> Dict[str, Any]:
    """
    Minimal valid response with 1 node and 0 edges.
    Tests basic parsing functionality.
    """
    cols = [
        _create_node_json("node1", "Entity", {"name": "TestEntity"})
    ]
    return _wrap_response(cols)


def get_simple_graph_response() -> Dict[str, Any]:
    """
    Simple graph with 3 nodes and 2 edges forming a chain: A -> B -> C
    Tests basic node and edge extraction.
    """
    cols = [
        _create_node_json("node1", "Person", {"name": "Alice", "age": 30}),
        _create_edge_json("edge1", "node1", "node2", "KNOWS", {"since": 2020}),
        _create_node_json("node2", "Person", {"name": "Bob", "age": 25}),
        _create_edge_json("edge2", "node2", "node3", "WORKS_WITH", {"department": "Engineering"}),
        _create_node_json("node3", "Person", {"name": "Charlie", "age": 35})
    ]
    return _wrap_response(cols)


def get_duplicate_nodes_response() -> Dict[str, Any]:
    """
    Response with duplicate nodes having varying levels of completeness.
    Tests deduplication logic that keeps the most complete record.
    """
    cols = [
        # First occurrence - minimal data
        _create_node_json("node1", "Person", {"name": "Alice"}),
        _create_edge_json("edge1", "node1", "node2", "KNOWS", {}),
        # Second occurrence - more complete data (should be kept)
        _create_node_json("node1", "Person", {
            "name": "Alice",
            "age": 30,
            "email": "alice@example.com",
            "department": "Sales"
        }),
        _create_node_json("node2", "Person", {"name": "Bob"}),
        # Third occurrence - less complete than second
        _create_node_json("node1", "Person", {"name": "Alice", "age": 30})
    ]
    return _wrap_response(cols)


def get_malformed_response() -> Dict[str, Any]:
    """
    Response with some malformed JSON entries mixed with valid ones.
    Tests error handling and defensive parsing.
    """
    cols = [
        _create_node_json("node1", "Person", {"name": "Valid"}),
        "This is not valid JSON {{{",
        _create_edge_json("edge1", "node1", "node2", "RELATES", {}),
        '{"incomplete": "missing required fields"}',
        _create_node_json("node2", "Person", {"name": "AlsoValid"})
    ]
    return _wrap_response(cols)


def get_empty_response() -> Dict[str, Any]:
    """
    Valid response with empty results.
    Tests handling of queries that return no data.
    """
    return {
        "Graph": {
            "Nodes": [],
            "Edges": []
        },
        "RawData": {
            "Rows": []
        }
    }


def get_complex_graph_response() -> Dict[str, Any]:
    """
    Complex graph with multiple node types, edge types, and rich properties.
    Tests handling of diverse real-world scenarios.
    """
    cols = [
        # Organization nodes
        _create_node_json("org1", "Organization", {
            "name": "TechCorp",
            "industry": "Technology",
            "founded": 2010,
            "employees": 5000
        }),
        _create_node_json("org2", "Organization", {
            "name": "DataCo",
            "industry": "Analytics",
            "founded": 2015
        }),

        # Person nodes
        _create_node_json("person1", "Person", {
            "name": "Alice",
            "age": 30,
            "role": "Engineer",
            "email": "alice@techcorp.com"
        }),
        _create_node_json("person2", "Person", {
            "name": "Bob",
            "age": 35,
            "role": "Manager"
        }),
        _create_node_json("person3", "Person", {
            "name": "Charlie",
            "age": 28,
            "role": "Analyst"
        }),

        # Location nodes
        _create_node_json("loc1", "Location", {
            "city": "San Francisco",
            "country": "USA",
            "coordinates": "37.7749,-122.4194"
        }),

        # Employment edges
        _create_edge_json("emp1", "person1", "org1", "EMPLOYED_BY", {
            "start_date": "2020-01-15",
            "position": "Senior Engineer"
        }),
        _create_edge_json("emp2", "person2", "org1", "EMPLOYED_BY", {
            "start_date": "2018-06-01",
            "position": "Engineering Manager"
        }),
        _create_edge_json("emp3", "person3", "org2", "EMPLOYED_BY", {
            "start_date": "2021-03-10"
        }),

        # Relationship edges
        _create_edge_json("rel1", "person1", "person2", "REPORTS_TO", {
            "since": "2020-01-15"
        }),
        _create_edge_json("rel2", "person1", "person3", "COLLABORATES_WITH", {
            "projects": ["DataPipeline", "Analytics"]
        }),

        # Location edges
        _create_edge_json("loc_edge1", "org1", "loc1", "LOCATED_IN", {
            "office_type": "Headquarters"
        }),
        _create_edge_json("loc_edge2", "org2", "loc1", "LOCATED_IN", {
            "office_type": "Branch"
        })
    ]
    return _wrap_response(cols)


def get_edge_only_response() -> Dict[str, Any]:
    """
    Response with edges but no corresponding nodes (orphan edges).
    Tests handling of incomplete graph data.
    """
    cols = [
        _create_edge_json("edge1", "missing_node1", "missing_node2", "RELATES", {
            "type": "orphan"
        }),
        _create_edge_json("edge2", "missing_node2", "missing_node3", "CONNECTS", {
            "strength": 0.8
        })
    ]
    return _wrap_response(cols)


def get_response_with_special_characters() -> Dict[str, Any]:
    """
    Response with special characters, unicode, and edge cases in properties.
    Tests robust string handling.
    """
    cols = [
        _create_node_json("node1", "Person", {
            "name": "José García",
            "bio": "Engineer with 10+ years experience\nSpecializes in: Data & Analytics",
            "tags": ["Python", "ML/AI", "Cloud"],
            "special_chars": "Test: @#$%^&*()_+-={}[]|\\:;\"'<>,.?/"
        }),
        _create_edge_json("edge1", "node1", "node2", "MENTIONS", {
            "context": "Discussed \"data quality\" & 'performance issues'",
            "emoji": "👍🚀💯"
        }),
        _create_node_json("node2", "Document", {
            "title": "Q1 Report",
            "content": "Revenue: $1,000,000.00\nGrowth: 25%"
        })
    ]
    return _wrap_response(cols)


def get_response_with_null_properties() -> Dict[str, Any]:
    """
    Response with null/None values in properties.
    Tests handling of missing or null data.
    """
    cols = [
        _create_node_json("node1", "Person", {
            "name": "Alice",
            "age": None,
            "email": "alice@example.com",
            "phone": None,
            "department": None
        }),
        _create_edge_json("edge1", "node1", "node2", "KNOWS", {
            "since": None,
            "strength": 0.5,
            "notes": None
        }),
        _create_node_json("node2", "Person", {
            "name": "Bob",
            "age": 30
        })
    ]
    return _wrap_response(cols)


def _create_sys_node_json(
    node_id: str,
    label: str,
    sys_label: str,
    properties: Dict[str, Any]
) -> str:
    """Helper to create a JSON-encoded node with sys_* fields (Sentinel Graph API format)."""
    node = {
        "id": node_id,
        "sys_id": node_id,
        "label": label,
        "sys_label": sys_label,
        **properties
    }
    return json.dumps(node)


def _create_sys_edge_json(
    source_id: str,
    target_id: str,
    edge_type: str,
    source_label: str,
    target_label: str,
    properties: Dict[str, Any]
) -> str:
    """Helper to create a JSON-encoded edge with sys_* fields (Sentinel Graph API format)."""
    edge = {
        "type": edge_type,
        "sys_label": edge_type,
        "sys_sourceId": source_id,
        "sys_sourceLabel": source_label,
        "sys_targetId": target_id,
        "sys_targetLabel": target_label,
        "sys_edge_id": edge_type,
        **properties
    }
    return json.dumps(edge)


def get_sentinel_graph_api_response() -> Dict[str, Any]:
    """
    Response using Sentinel Graph API field naming (sys_* prefix).
    Tests compatibility with actual Microsoft Sentinel Graph API responses.

    Mimics authentication events: User -> AUTH_ATTEMPT_FROM -> IPAddress
    """
    cols = [
        # User node
        _create_sys_node_json("user1@example.com", "trusted-service-user", "User", {
            "displayName": "Alice User",
            "z_processed_at": "2025-01-15T10:00:00.0000000Z",
            "TimeGenerated": "2025-01-15T09:59:00.0000000Z"
        }),
        # Auth edge
        _create_sys_edge_json(
            "user1@example.com", "192.168.1.100",
            "AUTH_ATTEMPT_FROM", "User", "IPAddress",
            {"failureCount": 5, "successCount": 100}
        ),
        # IP node
        _create_sys_node_json("192.168.1.100", "192.168.1.100", "IPAddress", {
            "title": "192.168.1.100",
            "z_processed_at": "2025-01-15T10:00:00.0000000Z"
        }),
        # Another user
        _create_sys_node_json("user2@example.com", "trusted-service-user", "User", {
            "displayName": "Bob User"
        }),
        # Auth edge from second user
        _create_sys_edge_json(
            "user2@example.com", "10.0.0.50",
            "AUTH_ATTEMPT_FROM", "User", "IPAddress",
            {"failureCount": 0, "successCount": 50}
        ),
        # Second IP
        _create_sys_node_json("10.0.0.50", "10.0.0.50", "IPAddress", {
            "title": "10.0.0.50"
        })
    ]

    # Wrap in response structure - note sys_* format doesn't use Graph.Nodes typically
    return {
        "Graph": {
            "Nodes": [],
            "Edges": []
        },
        "RawData": {
            "Rows": [
                {
                    "Cols": [{"Value": val, "Metadata": {}, "Path": None} for val in cols]
                }
            ],
            "ColumnNames": ["n", "e", "m"]
        }
    }
