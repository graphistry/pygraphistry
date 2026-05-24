"""Test script for GFQL validation helpers."""

import pandas as pd

from graphistry import edges, nodes
from graphistry.compute.ast import n, e_forward, e_reverse
from graphistry.compute.gfql.validate import (
    validate_syntax, validate_schema, validate_query,
    extract_schema_from_dataframes, format_validation_errors,
    Schema, ValidationIssue
)
from graphistry.compute.predicates.numeric import gt, lt
from graphistry.compute.predicates.str import contains


def test_syntax_validation():
    """Test syntax validation without data."""
    print("=== Testing Syntax Validation ===\n")
    
    # Valid query
    valid_chain = [n({"type": "person"}), e_forward(), n()]
    issues = validate_syntax(valid_chain)
    print(f"Valid chain issues: {len(issues)}")
    if issues:
        print(format_validation_errors(issues))
    
    # Invalid query - not a list
    try:
        issues = validate_syntax("not a list")
        print(f"\nInvalid type issues: {len(issues)}")
        print(format_validation_errors(issues))
    except Exception as e:
        print(f"Error: {e}")
    
    # Empty chain
    issues = validate_syntax([])
    print(f"\nEmpty chain issues: {len(issues)}")
    print(format_validation_errors(issues))
    
    # Invalid operation type
    issues = validate_syntax([n(), "not an operation", e_forward()])
    print(f"\nInvalid operation issues: {len(issues)}")
    print(format_validation_errors(issues))
    
    # Orphaned edge warning
    issues = validate_syntax([e_forward(), e_reverse()])
    print(f"\nOrphaned edge issues: {len(issues)}")
    print(format_validation_errors(issues))
    
    # Unbounded hops warning
    issues = validate_syntax([n(), e_forward(to_fixed_point=True), n()])
    print(f"\nUnbounded hops issues: {len(issues)}")
    print(format_validation_errors(issues))


def test_schema_validation():
    """Test schema validation with data."""
    print("\n\n=== Testing Schema Validation ===\n")
    
    # Create sample data
    nodes_df = pd.DataFrame({
        'id': ['a', 'b', 'c'],
        'type': ['person', 'person', 'company'],
        'age': [25, 30, 5],
        'name': ['Alice', 'Bob', 'Corp']
    })
    
    edges_df = pd.DataFrame({
        'source': ['a', 'b', 'c'],
        'target': ['b', 'c', 'a'], 
        'relationship': ['knows', 'works_at', 'employs'],
        'weight': [1.0, 2.0, 3.0]
    })
    
    # Extract schema
    schema = extract_schema_from_dataframes(nodes_df, edges_df)
    print(f"Schema: {schema}")
    print(f"Node columns: {list(schema.node_columns.keys())}")
    print(f"Edge columns: {list(schema.edge_columns.keys())}")
    
    # Valid query
    valid_chain = [n({"type": "person"}), e_forward({"relationship": "knows"}), n()]
    issues = validate_schema(valid_chain, schema)
    print(f"\nValid query issues: {len(issues)}")
    if issues:
        print(format_validation_errors(issues))
    
    # Column not found
    invalid_chain = [n({"nonexistent": "value"})]
    issues = validate_schema(invalid_chain, schema)
    print(f"\nColumn not found issues: {len(issues)}")
    print(format_validation_errors(issues))
    
    # Type mismatch - string predicate on numeric column
    type_mismatch_chain = [n({"age": contains("25")})]
    issues = validate_schema(type_mismatch_chain, schema)
    print(f"\nType mismatch issues: {len(issues)}")
    print(format_validation_errors(issues))
    
    # Valid numeric predicate
    valid_numeric_chain = [n({"age": gt(20)})]
    issues = validate_schema(valid_numeric_chain, schema)
    print(f"\nValid numeric predicate issues: {len(issues)}")
    if issues:
        print(format_validation_errors(issues))


def test_combined_validation():
    """Test combined validation."""
    print("\n\n=== Testing Combined Validation ===\n")
    
    nodes_df = pd.DataFrame({
        'id': range(5),
        'value': [10, 20, 30, 40, 50]
    })
    
    edges_df = pd.DataFrame({
        'source': [0, 1, 2, 3],
        'target': [1, 2, 3, 4],
        'weight': [1.0, 2.0, 3.0, 4.0]
    })
    
    # Query with both syntax and schema issues
    problematic_chain = [
        n({"missing_col": "value"}),  # Schema error
        e_forward(hops=-1),  # Syntax error
        n({"value": gt(25)})
    ]
    
    issues = validate_query(problematic_chain, nodes_df, edges_df)
    print(f"Combined validation issues: {len(issues)}")
    print(format_validation_errors(issues))
    
    # Show issue details
    print("\nIssue details:")
    for issue in issues:
        print(f"  - {issue.level}: {issue.message}")
        if issue.operation_index is not None:
            print(f"    at operation {issue.operation_index}")
        if issue.field:
            print(f"    field: {issue.field}")


def test_edge_validation():
    """Test edge-specific validation."""
    print("\n\n=== Testing Edge Validation ===\n")
    
    nodes_df = pd.DataFrame({
        'id': ['a', 'b', 'c'],
        'type': ['A', 'B', 'C']
    })
    
    edges_df = pd.DataFrame({
        'source': ['a', 'b'],
        'target': ['b', 'c'],
        'edge_type': ['follows', 'likes']
    })
    
    schema = extract_schema_from_dataframes(nodes_df, edges_df)
    
    # Valid edge query
    valid_chain = [
        n(),
        e_forward(
            edge_match={"edge_type": "follows"},
            source_node_match={"type": "A"},
            destination_node_match={"type": "B"}
        ),
        n()
    ]
    issues = validate_schema(valid_chain, schema)
    print(f"Valid edge query issues: {len(issues)}")
    if issues:
        print(format_validation_errors(issues))
    
    # Invalid edge column
    invalid_chain = [
        n(),
        e_forward({"missing_edge_col": "value"}),
        n()
    ]
    issues = validate_schema(invalid_chain, schema)
    print(f"\nInvalid edge column issues: {len(issues)}")
    print(format_validation_errors(issues))
    
    # Invalid node filter in edge
    invalid_node_filter = [
        n(),
        e_forward(source_node_match={"missing_node_col": "value"}),
        n()
    ]
    issues = validate_schema(invalid_node_filter, schema)
    print(f"\nInvalid node filter in edge issues: {len(issues)}")
    print(format_validation_errors(issues))


def test_syntax_filter_key_diagnostics_are_stable():
    issues = validate_syntax([
        n({1: "person"}),
        e_forward(
            edge_match={2: "knows"},
            source_node_match={3: "a"},
            destination_node_match={4: "b"},
        ),
        n(),
    ])

    errors = [issue for issue in issues if issue.error_type == "INVALID_FILTER_KEY"]
    assert [(issue.operation_index, issue.field, issue.message, issue.suggestion) for issue in errors] == [
        (0, "1", "Invalid filter key format: 1", "Filter keys must be strings"),
        (1, "edge_match.2", "Invalid filter key format: 2", "Filter keys must be strings"),
        (1, "source_node_match.3", "Invalid filter key format: 3", "Filter keys must be strings"),
        (1, "destination_node_match.4", "Invalid filter key format: 4", "Filter keys must be strings"),
    ]


def test_schema_filter_diagnostics_are_stable_across_node_edge_shapes():
    schema = Schema(
        node_columns={"id": "object", "age": "int64", "name": "object"},
        edge_columns={"relationship": "object", "weight": "float64"},
    )
    chain = [
        n({"missing_node": "value", "age": contains("25")}),
        e_forward(
            edge_match={"missing_edge": "value", "weight": contains("1")},
            source_node_match={"missing_source": "a"},
            destination_node_match={"missing_dest": "b"},
        ),
        n(),
    ]

    issues = validate_schema(chain, schema)

    assert [
        (issue.error_type, issue.operation_index, issue.field, issue.message)
        for issue in issues
    ] == [
        ("COLUMN_NOT_FOUND", 0, "missing_node", 'Column "missing_node" not found in node data'),
        ("TYPE_MISMATCH", 0, "age", 'Column "age" is int64 but predicate expects string'),
        ("COLUMN_NOT_FOUND", 1, "edge_match.missing_edge", 'Column "missing_edge" not found in edge data'),
        ("TYPE_MISMATCH", 1, "edge_match.weight", 'Column "weight" is float64 but predicate expects string'),
        ("COLUMN_NOT_FOUND", 1, "source_node_match.missing_source", 'Column "missing_source" not found in node data'),
        ("COLUMN_NOT_FOUND", 1, "destination_node_match.missing_dest", 'Column "missing_dest" not found in node data'),
    ]


def test_validation_issue_and_report_formatting_are_stable():
    error = ValidationIssue(
        "error",
        "Column problem",
        operation_index=2,
        field="age",
        suggestion="Fix age",
        error_type="TYPE_MISMATCH",
    )
    warning = ValidationIssue(
        "warning",
        "Slow hop",
        operation_index=3,
        field="hops",
        suggestion="Use fixed hops",
        error_type="UNBOUNDED_HOPS_WARNING",
    )

    assert repr(error) == (
        "ERROR: Column problem | at operation 2 | field: age | "
        "Suggestion: Fix age"
    )
    assert ValidationIssue("error", "Column problem") != ValidationIssue(
        "error", "Column problem"
    )
    assert error.to_dict() == {
        "level": "error",
        "message": "Column problem",
        "operation_index": 2,
        "field": "age",
        "suggestion": "Fix age",
        "error_type": "TYPE_MISMATCH",
    }
    assert format_validation_errors([error, warning]) == (
        "GFQL Validation Report:\n"
        "--------------------------------------------------\n"
        "\nERRORS (1):\n"
        "\n1. Column problem\n"
        "   Location: Operation 2\n"
        "   Field: age\n"
        "   💡 Fix age\n"
        "\nWARNINGS (1):\n"
        "\n1. Slow hop\n"
        "   Location: Operation 3\n"
        "   💡 Use fixed hops"
    )


if __name__ == "__main__":
    test_syntax_validation()
    test_schema_validation()
    test_combined_validation()
    test_edge_validation()
    
    print("\n\n=== All tests completed! ===")
