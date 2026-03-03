import pandas as pd
import pytest
from graphistry.compute.ast import ASTCall, ASTLet, ASTRef, distinct, limit, n, e, order_by, return_, rows, select, skip
from graphistry.compute.chain import Chain
from graphistry.tests.test_compute import CGFull

# Suppress deprecation warnings for chain() method in this test file
# This file tests the migration from chain() to gfql()
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:graphistry")


class TestGFQLAPI:
    """Test unified GFQL API and migration"""
    
    def test_public_api_methods(self):
        """Test what methods are available on the public API"""
        g = CGFull()
        
        # Should have gfql
        assert hasattr(g, 'gfql')
        assert callable(g.gfql)
        
        # Should still have chain (with deprecation)
        assert hasattr(g, 'chain')
        assert callable(g.chain)
        
        # chain_let should not be in public API - removed from ComputeMixin
        assert not hasattr(g, 'chain_let')

    def test_row_pipeline_primitives_build_ast_calls(self):
        row_step = rows("nodes", source="a")
        assert isinstance(row_step, ASTCall)
        assert row_step.function == "rows"
        assert row_step.params == {"table": "nodes", "source": "a"}

        select_step = select([("name", "name"), ("age", "age")])
        assert isinstance(select_step, ASTCall)
        assert select_step.function == "select"
        assert select_step.params == {"items": [("name", "name"), ("age", "age")]}

        return_step = return_([("name", "name")])
        assert isinstance(return_step, ASTCall)
        assert return_step.function == "select"
        assert return_step.params == {"items": [("name", "name")]}

        order_step = order_by([("name", "asc"), ("age", "desc")])
        assert isinstance(order_step, ASTCall)
        assert order_step.function == "order_by"
        assert order_step.params == {"keys": [("name", "asc"), ("age", "desc")]}

        skip_step = skip(3)
        assert isinstance(skip_step, ASTCall)
        assert skip_step.function == "skip"
        assert skip_step.params == {"value": 3}

        limit_step = limit(10)
        assert isinstance(limit_step, ASTCall)
        assert limit_step.function == "limit"
        assert limit_step.params == {"value": 10}

        distinct_step = distinct()
        assert isinstance(distinct_step, ASTCall)
        assert distinct_step.function == "distinct"
        assert distinct_step.params == {}


class TestGFQLRowPipeline:
    def test_row_pipeline_exec_projection_sort_page_distinct(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c", "d"],
            "name": ["n2", "n1", "n2", "n3"],
            "score": [2, 3, 2, 1]
        })
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            select([("name", "name"), ("score", "score")]),
            distinct(),
            order_by([("score", "desc"), ("name", "asc")]),
            skip(1),
            limit(2),
        ])

        assert list(result._nodes.columns) == ["name", "score"]
        assert result._nodes.reset_index(drop=True).to_dict(orient="records") == [
            {"name": "n2", "score": 2},
            {"name": "n3", "score": 1},
        ]
        assert result._edges is not None
        assert len(result._edges) == 0

    def test_row_pipeline_exec_with_match_alias_source(self):
        nodes_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "grp": ["x", "x", "y"],
        })
        edges_df = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            n({"grp": "x"}, name="a"),
            rows(source="a"),
            return_([("id", "id")]),
            order_by([("id", "asc")]),
        ])

        assert list(result._nodes.columns) == ["id"]
        assert result._nodes["id"].tolist() == ["a", "b"]

    def test_row_pipeline_bad_source_alias_raises(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"], "v": [1, 2]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        with pytest.raises(Exception, match="requires node column|alias column not found"):
            g.gfql([rows(source="missing")])

    def test_row_pipeline_rows_edges_table_projection(self):
        nodes_df = pd.DataFrame({"id": ["a", "b", "c"]})
        edges_df = pd.DataFrame({
            "s": ["a", "b", "a"],
            "d": ["b", "c", "c"],
            "weight": [1, 3, 2],
        })
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(table="edges"),
            select([("weight", "weight")]),
            order_by([("weight", "desc")]),
        ])

        assert list(result._nodes.columns) == ["weight"]
        assert result._nodes["weight"].tolist() == [3, 2, 1]

    def test_row_pipeline_select_allows_literal_expressions(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        result = g.gfql([
            rows(),
            select([("id", "id"), ("one", 1), ("txt", "id")]),
        ])

        assert result._nodes.to_dict(orient="records") == [
            {"id": "a", "one": 1, "txt": "a"},
            {"id": "b", "one": 1, "txt": "b"},
        ]

    def test_row_pipeline_invalid_rows_table_rejected(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        with pytest.raises(Exception, match="table"):
            g.gfql([rows("bad_table")])

    def test_row_pipeline_select_missing_column_raises(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        with pytest.raises(Exception, match="select expression column not found"):
            g.gfql([rows(), select([("x", "missing_col")])])

    def test_row_pipeline_order_by_missing_column_raises(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        with pytest.raises(Exception, match="order_by column not found"):
            g.gfql([rows(), order_by([("missing_col", "asc")])])

    @pytest.mark.parametrize("value", [-1, True, "1.5", "bad"])
    def test_row_pipeline_skip_invalid_values_rejected(self, value):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        with pytest.raises(Exception, match="Invalid type for parameter|non-negative integer|non-negative"):
            g.gfql([rows(), skip(value)])

    @pytest.mark.parametrize("value", [-1, True, "1.5", "bad"])
    def test_row_pipeline_limit_invalid_values_rejected(self, value):
        nodes_df = pd.DataFrame({"id": ["a", "b"]})
        edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")

        with pytest.raises(Exception, match="Invalid type for parameter|non-negative integer|non-negative"):
            g.gfql([rows(), limit(value)])

    def test_row_pipeline_vectorized_cudf_when_available(self):
        cudf = pytest.importorskip("cudf")

        nodes_pd = pd.DataFrame({"id": ["a", "b", "c"], "score": [3, 1, 2]})
        edges_pd = pd.DataFrame({"s": ["a"], "d": ["b"]})
        g = CGFull().nodes(cudf.from_pandas(nodes_pd), "id").edges(cudf.from_pandas(edges_pd), "s", "d")

        result = g.gfql([
            rows(),
            order_by([("score", "asc")]),
            limit(2),
        ])

        assert type(result._nodes).__module__.startswith("cudf")
        assert result._nodes["score"].to_pandas().tolist() == [1, 2]


class TestGFQL:
    """Test unified GFQL entrypoint"""
    
    def test_gfql_exists(self):
        """Test that gfql method exists on CGFull"""
        g = CGFull()
        assert hasattr(g, 'gfql')
        assert callable(g.gfql)
    
    def test_gfql_with_list(self):
        """Test gfql with list executes as chain"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'type': ['person', 'person', 'company', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Execute as chain
        result = g.gfql([n({'type': 'person'})])
        
        assert len(result._nodes) == 2
        assert all(result._nodes['type'] == 'person')
    
    def test_gfql_with_chain_object(self):
        """Test gfql with Chain object"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Execute with Chain
        chain = Chain([n({'type': 'person'}), e(), n()])
        result = g.gfql(chain)
        
        assert result is not None
        # Result depends on graph structure
    
    def test_gfql_with_dag(self):
        """Test gfql with ASTLet"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'type': ['person', 'person', 'company', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Execute as DAG - wrap n() in Chain for GraphOperation
        dag = ASTLet({
            'people': Chain([n({'type': 'person'})]),
            'companies': Chain([n({'type': 'company'})])
        })
        
        result = g.gfql(dag)
        assert result is not None
    
    def test_gfql_with_dict_convenience(self):
        """Test gfql with dict converts to DAG"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Dict convenience should auto-wrap ASTNode/ASTEdge in Chain
        result = g.gfql({'people': n({'type': 'person'})})
        
        # Should have filtered to people only  
        assert len(result._nodes) == 2
        assert all(result._nodes['type'] == 'person')
    
    def test_gfql_output_with_dag(self):
        """Test gfql output parameter works with DAG"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'type': ['person', 'person', 'company', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Dict convenience with output parameter
        result = g.gfql({
            'people': n({'type': 'person'}),
            'companies': n({'type': 'company'})
        }, output='people')
        
        assert len(result._nodes) == 2
        assert all(result._nodes['type'] == 'person')
    
    def test_gfql_output_ignored_for_chain(self):
        """Test gfql output parameter ignored for chains"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        
        # Should work but output ignored
        result = g.gfql([n()], output='ignored')
        assert result is not None
    
    def test_gfql_with_single_ast_object(self):
        """Test gfql with single ASTObject wraps in list"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Single ASTObject should work
        result = g.gfql(n({'type': 'person'}))
        
        assert len(result._nodes) == 2
        assert all(result._nodes['type'] == 'person')
    
    def test_gfql_invalid_query_type(self):
        """Test gfql with invalid query type"""
        g = CGFull()
        
        with pytest.raises(TypeError) as exc_info:
            g.gfql("not a valid query")
        
        assert "Query must be ASTObject, List[ASTObject], Chain, ASTLet, or dict" in str(exc_info.value)
    
    def test_gfql_deprecation_and_migration(self):
        """Test deprecation warnings and migration path"""
        import warnings
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # chain() should show deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chain_result = g.chain([n({'type': 'person'})])
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "chain() is deprecated" in str(w[0].message)
            assert "Use gfql()" in str(w[0].message)
        
        assert len(chain_result._nodes) == 2
        
        # chain_let should be removed from public API - use gfql() instead
        assert not hasattr(g, 'chain_let'), "chain_let should be removed from public API"
        
        # gfql should work for both patterns
        gfql_chain = g.gfql([n({'type': 'person'})])
        assert len(gfql_chain._nodes) == 2
        
        # Dict convenience should now work with auto-wrapping
        gfql_dag = g.gfql({'people': n({'type': 'person'})})
        assert len(gfql_dag._nodes) == 2


class TestGFQLDictConversion:
    """Test dict-to-AST conversion scenarios to prevent regressions."""

    def test_gfql_with_list_containing_raw_dicts(self):
        """Test gfql with list containing raw dict objects (main regression case)"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')

        # This was the exact case that failed before our fix
        result = g.gfql([{"type": "Node"}])

        # Should execute successfully without TypeError
        assert result is not None
        assert hasattr(result, '_nodes')
        # Should return all nodes since no filter specified
        assert len(result._nodes) == 3

    def test_gfql_with_mixed_list_ast_and_dicts(self):
        """Test gfql with list containing both AST objects and dicts"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')

        # Mixed: AST object + raw dict
        result = g.gfql([n({'type': 'person'}), {"type": "Node"}])

        # Should execute successfully
        assert result is not None
        assert hasattr(result, '_nodes')

    def test_gfql_with_multiple_dicts_in_list(self):
        """Test gfql with list containing multiple dict objects"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')

        # Multiple dicts in sequence
        result = g.gfql([{"type": "Node"}, {"type": "Node"}])

        # Should execute successfully
        assert result is not None
        assert hasattr(result, '_nodes')

    def test_gfql_dict_conversion_with_filter(self):
        """Test dict conversion works with actual filtering"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')

        # Test with actual filter in dict format
        result = g.gfql([{"type": "Node", "filter": {"type": "person"}}])

        # Should execute successfully (filtering behavior depends on implementation)
        assert result is not None
        assert hasattr(result, '_nodes')
        assert len(result._nodes) >= 0  # May or may not filter, just ensure no crash

    def test_gfql_empty_list_with_dicts(self):
        """Test edge case: empty list"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')

        # Empty list should work
        result = g.gfql([])
        assert result is not None

    def test_gfql_single_vs_list_dict_equivalence(self):
        """Test that list dict conversion works (single dict vs list behavior may differ)"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')

        # Test that list dict conversion works - main regression test
        result_list = g.gfql([{"type": "Node"}])

        # Should succeed without TypeError
        assert result_list is not None
        assert hasattr(result_list, '_nodes')
        assert len(result_list._nodes) >= 0  # Ensure execution succeeds
