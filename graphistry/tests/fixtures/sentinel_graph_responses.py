"""
Test fixtures for Microsoft Sentinel Graph API responses.
Matches the public preview API format:
  https://learn.microsoft.com/en-us/azure/sentinel/datalake/graph-rest-api
"""

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(node_id: str, labels: List[str], properties: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": node_id, "labels": labels, "properties": properties}


def _make_edge(
    edge_id: str,
    source_id: str,
    target_id: str,
    labels: List[str],
    properties: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "id": edge_id,
        "sourceId": source_id,
        "targetId": target_id,
        "labels": labels,
        "properties": properties,
    }


def _wrap_graph_response(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    raw_tables: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {
        "status": 200,
        "result": {
            "graph": {"nodes": nodes, "edges": edges},
            "rawData": {"tables": raw_tables or []},
        },
        "correlationId": "test-correlation-id-0000",
    }


# ---------------------------------------------------------------------------
# Graph-format fixtures
# ---------------------------------------------------------------------------

def get_minimal_response() -> Dict[str, Any]:
    """1 node, 0 edges"""
    nodes = [
        _make_node("node-001", ["Device"], {"hostname": "laptop-01", "os": "Windows 11"}),
    ]
    return _wrap_graph_response(nodes, [])


def get_simple_graph_response() -> Dict[str, Any]:
    """3 nodes (A, B, C) and 2 edges (A->B, B->C)"""
    nodes = [
        _make_node("node-a", ["User"], {"name": "Alice", "department": "Engineering"}),
        _make_node("node-b", ["Group"], {"name": "Admins", "memberCount": 5}),
        _make_node("node-c", ["Resource"], {"name": "FileShare", "path": "/data"}),
    ]
    edges = [
        _make_edge("edge-ab", "node-a", "node-b", ["MemberOf"], {"since": "2024-01-01"}),
        _make_edge("edge-bc", "node-b", "node-c", ["HasAccess"], {"permission": "read"}),
    ]
    return _wrap_graph_response(nodes, edges)


def get_duplicate_nodes_response() -> Dict[str, Any]:
    """Same node id appears twice; the more-complete record should be kept."""
    nodes = [
        # Sparse first occurrence
        _make_node("node-dup", ["User"], {"name": "Bob"}),
        # Richer second occurrence (more properties)
        _make_node("node-dup", ["User"], {"name": "Bob", "email": "bob@contoso.com", "department": "IT"}),
        _make_node("node-other", ["User"], {"name": "Carol"}),
    ]
    edges = [
        _make_edge("edge-001", "node-dup", "node-other", ["Knows"], {}),
    ]
    return _wrap_graph_response(nodes, edges)


def get_malformed_response() -> Dict[str, Any]:
    """One node entry is missing the required 'id' field and should be skipped."""
    nodes = [
        _make_node("node-valid", ["User"], {"name": "Dave"}),
        # Missing 'id' — parser should skip this
        {"labels": ["Broken"], "properties": {"name": "No ID here"}},
    ]
    edges = [
        _make_edge("edge-001", "node-valid", "node-valid", ["SelfLoop"], {}),
    ]
    return _wrap_graph_response(nodes, edges)


def get_empty_response() -> Dict[str, Any]:
    """Valid response envelope with no nodes or edges."""
    return _wrap_graph_response([], [])


def get_complex_graph_response() -> Dict[str, Any]:
    """Multiple node types and edge types with rich properties."""
    nodes = [
        _make_node("user-001", ["User"], {
            "name": "Aino Rebane",
            "email": "aino.rebane@contoso.com",
            "department": "Engineering",
            "jobTitle": "Senior Engineer",
        }),
        _make_node("user-002", ["User"], {
            "name": "Marco Silva",
            "email": "marco.silva@contoso.com",
            "department": "Security",
        }),
        _make_node("group-001", ["Group"], {
            "name": "Administrators",
            "description": "System administrators",
            "memberCount": 25,
        }),
        _make_node("resource-001", ["Resource", "FileShare"], {
            "name": "FinanceData",
            "path": "/shares/finance",
            "sensitivity": "Confidential",
        }),
        _make_node("device-001", ["Device"], {
            "hostname": "workstation-42",
            "os": "Windows 11",
            "lastSeen": "2024-03-01T12:00:00Z",
        }),
    ]
    edges = [
        _make_edge("e-001", "user-001", "group-001", ["MemberOf"], {"assignedDate": "2024-01-15"}),
        _make_edge("e-002", "user-002", "group-001", ["MemberOf"], {"assignedDate": "2024-02-01"}),
        _make_edge("e-003", "group-001", "resource-001", ["HasAccess"], {"permission": "full"}),
        _make_edge("e-004", "user-001", "device-001", ["Uses"], {"primary": True}),
        _make_edge("e-005", "user-001", "user-002", ["CollaboratesWith"], {"projectCount": 3}),
    ]
    return _wrap_graph_response(nodes, edges)


def get_edge_only_response() -> Dict[str, Any]:
    """Edges referencing node IDs that are not present in the nodes list."""
    edges = [
        _make_edge("orphan-edge-001", "ghost-node-a", "ghost-node-b", ["Relates"], {}),
    ]
    return _wrap_graph_response([], edges)


def get_response_with_special_characters() -> Dict[str, Any]:
    """Node and edge properties containing unicode, special chars, and emoji."""
    nodes = [
        _make_node("node-unicode", ["User"], {
            "name": "Jose Garcia",
            "city": "Sao Paulo",
            "notes": "resume & Japanese test",
            "path": "C:\\Users\\test\\file.txt",
            "json_field": '{"key": "value with quotes"}',
        }),
    ]
    edges = [
        _make_edge("edge-unicode", "node-unicode", "node-unicode", ["SelfRef"], {
            "description": "Edge with Chinese and Arabic text",
        }),
    ]
    return _wrap_graph_response(nodes, edges)


def get_response_with_null_properties() -> Dict[str, Any]:
    """Properties dict may contain None values."""
    nodes = [
        _make_node("node-nulls", ["User"], {
            "name": "Eve",
            "email": None,
            "department": None,
            "role": "analyst",
        }),
    ]
    edges = [
        _make_edge("edge-nulls", "node-nulls", "node-nulls", ["SelfLoop"], {
            "weight": None,
            "label": "test",
        }),
    ]
    return _wrap_graph_response(nodes, edges)


# ---------------------------------------------------------------------------
# Table-format fixture (rawData.tables secondary path)
# ---------------------------------------------------------------------------

def get_table_format_response() -> Dict[str, Any]:
    """Response using rawData table format only (graph section is empty).

    Note: table-format edges use 'sourceOid'/'targetOid', not 'sourceId'/'targetId'.
    """
    tables = [
        {
            "tableName": "PrimaryResult",
            "columns": [
                {"columnName": "n", "dataType": "dynamic"},
                {"columnName": "r", "dataType": "dynamic"},
                {"columnName": "m", "dataType": "dynamic"},
            ],
            "rows": [
                [
                    {
                        "oid": "table-node-001",
                        "labels": ["User"],
                        "properties": {"name": "Alice", "department": "Engineering"},
                    },
                    {
                        "oid": "table-edge-001",
                        "labels": ["HasRole"],
                        "sourceOid": "table-node-001",
                        "targetOid": "table-node-002",
                        "properties": {"assignedDate": "2024-01-15"},
                    },
                    {
                        "oid": "table-node-002",
                        "labels": ["Group"],
                        "properties": {"name": "Administrators", "memberCount": 25},
                    },
                ]
            ],
        }
    ]
    return _wrap_graph_response([], [], raw_tables=tables)


# ---------------------------------------------------------------------------
# Graph list endpoint fixture
# ---------------------------------------------------------------------------

def get_graph_list_response() -> Dict[str, Any]:
    """Response from GET /graphs/graph-instances?graphTypes=Custom"""
    return {
        "value": [
            {
                "name": "TestGraph",
                "mapFileName": None,
                "mapFileVersion": None,
                "graphDefinitionName": "TestDefinition",
                "graphDefinitionVersion": "1.0",
                "refreshFrequency": "PT1H",
                "createTime": "2024-01-01T00:00:00Z",
                "lastUpdateTime": "2024-03-01T12:00:00Z",
                "lastSnapshotTime": "2024-03-01T11:00:00Z",
                "lastSnapshotRequestTime": "2024-03-01T10:55:00Z",
                "instanceStatus": "Ready",
            },
            {
                "name": "StagingGraph",
                "mapFileName": None,
                "mapFileVersion": None,
                "graphDefinitionName": "StagingDefinition",
                "graphDefinitionVersion": "0.9",
                "refreshFrequency": "PT6H",
                "createTime": "2024-03-01T08:00:00Z",
                "lastUpdateTime": "2024-03-01T08:00:00Z",
                "lastSnapshotTime": None,
                "lastSnapshotRequestTime": None,
                "instanceStatus": "Creating",
            },
        ]
    }
