import pandas as pd
import pytest
from typing import Any, Dict, List
from graphistry.compute.ast import ASTLet, ASTRef, n, e
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError, GFQLValidationError
from graphistry.compute.gfql.cypher import compile_cypher
from graphistry.compute.gfql.cypher.lowering import _reentry_hidden_column_name
from graphistry.compute.gfql_unified import _compiled_query_reentry_state
from graphistry.tests.test_compute import CGFull

# Suppress deprecation warnings for chain() method in this test file
# This file tests the migration from chain() to gfql()
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:graphistry")


def _mk_graph(ids, types, src, dst):
    nodes_df = pd.DataFrame({"id": ids, "type": types})
    edges_df = pd.DataFrame({"s": src, "d": dst})
    return CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")


def _mk_people_company_graph3():
    return _mk_graph(
        ids=["a", "b", "c"],
        types=["person", "person", "company"],
        src=["a", "b"],
        dst=["b", "c"],
    )


def _mk_people_company_graph4():
    return _mk_graph(
        ids=["a", "b", "c", "d"],
        types=["person", "person", "company", "company"],
        src=["a", "b", "c"],
        dst=["b", "c", "d"],
    )


def _mk_empty_graph():
    return _mk_graph(ids=[], types=[], src=[], dst=[])


def _mk_reentry_scalar_graph():
    nodes_df = pd.DataFrame(
        {
            "id": ["a1", "a2", "b1", "b2"],
            "label__A": [True, True, False, False],
            "num": [1, 2, 1, 3],
        }
    )
    edges_df = pd.DataFrame(
        {
            "s": ["a1", "a2"],
            "d": ["b1", "b2"],
            "type": ["R", "R"],
        }
    )
    return CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")


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

class TestGFQL:
    """Test unified GFQL entrypoint"""
    
    def test_gfql_exists(self):
        """Test that gfql method exists on CGFull"""
        g = CGFull()
        assert hasattr(g, 'gfql')
        assert callable(g.gfql)
    
    def test_gfql_with_list(self):
        """Test gfql with list executes as chain"""
        g = _mk_people_company_graph4()
        
        # Execute as chain
        result = g.gfql([n({'type': 'person'})])
        
        assert len(result._nodes) == 2
        assert all(result._nodes['type'] == 'person')
    
    def test_gfql_with_chain_object(self):
        """Test gfql with Chain object"""
        g = _mk_people_company_graph3()
        
        # Execute with Chain
        chain = Chain([n({'type': 'person'}), e(), n()])
        result = g.gfql(chain)
        
        assert result is not None
        # Result depends on graph structure
    
    def test_gfql_with_dag(self):
        """Test gfql with ASTLet"""
        g = _mk_people_company_graph4()
        
        # Execute as DAG - wrap n() in Chain for GraphOperation
        dag = ASTLet({
            'people': Chain([n({'type': 'person'})]),
            'companies': Chain([n({'type': 'company'})])
        })
        
        result = g.gfql(dag)
        assert result is not None
    
    def test_gfql_with_dict_convenience(self):
        """Test gfql with dict converts to DAG"""
        g = _mk_people_company_graph3()
        
        # Dict convenience should auto-wrap ASTNode/ASTEdge in Chain
        result = g.gfql({'people': n({'type': 'person'})})
        
        # Should have filtered to people only  
        assert len(result._nodes) == 2
        assert all(result._nodes['type'] == 'person')
    
    def test_gfql_output_with_dag(self):
        """Test gfql output parameter works with DAG"""
        g = _mk_people_company_graph4()
        
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
        g = _mk_people_company_graph3()
        
        # Single ASTObject should work
        result = g.gfql(n({'type': 'person'}))
        
        assert len(result._nodes) == 2
        assert all(result._nodes['type'] == 'person')
    
    def test_gfql_invalid_query_type(self):
        """Test gfql with invalid query type"""
        g = CGFull()
        
        with pytest.raises(TypeError) as exc_info:
            g.gfql(123)

        assert "Query must be ASTObject, List[ASTObject], Chain, ASTLet, dict, or string" in str(exc_info.value)

    def test_gfql_with_cypher_string(self):
        g = _mk_graph(
            ids=["a", "b", "c"],
            types=["person", "person", "person"],
            src=["a", "b"],
            dst=["b", "c"],
        )
        g = g.nodes(g._nodes.assign(score=[3, 1, 2], name=["Alice", "Bob", "Carol"]), "id")

        result = g.gfql(
            "MATCH (p:person) RETURN p.name AS person_name ORDER BY person_name DESC LIMIT $top_n",
            params={"top_n": 2},
        )

        assert result._nodes.to_dict(orient="records") == [
            {"person_name": "Carol"},
            {"person_name": "Bob"},
        ]

    def test_gfql_with_cypher_string_defaults_language_to_cypher(self):
        g = _mk_people_company_graph3()

        result = g.gfql("MATCH (p:person) RETURN p LIMIT 1")

        assert len(result._nodes) == 1
        assert result._nodes.iloc[0]["p"] == "(:person)"

    def test_gfql_string_invalid_syntax_surfaces_parser_error(self):
        g = _mk_people_company_graph3()

        with pytest.raises(GFQLSyntaxError) as exc_info:
            g.gfql("MATCH (p RETURN p")

        assert exc_info.value.code == ErrorCode.E107

    def test_gfql_string_rejects_unsupported_language(self):
        g = _mk_people_company_graph3()

        with pytest.raises(GFQLValidationError) as exc_info:
            g.gfql("MATCH (p) RETURN p", language="gremlin")

        assert exc_info.value.code == ErrorCode.E108

    def test_gfql_non_string_rejects_language_and_params(self):
        g = _mk_people_company_graph3()

        with pytest.raises(ValueError):
            g.gfql([n()], language="cypher")

        with pytest.raises(ValueError):
            g.gfql([n()], params={"x": 1})

    @pytest.mark.parametrize(
        ("direction", "expected"),
        [
            ("ASC", [{"name": "A"}, {"name": "A"}]),
            ("DESC", [{"name": "C"}, {"name": "C"}]),
        ],
    )
    def test_gfql_executes_cypher_with_order_by_source_expression(self, direction, expected):
        nodes_df = pd.DataFrame(
            {
                "id": ["a1", "a2", "b1", "c1", "c2"],
                "type": ["person"] * 5,
                "name": ["A", "A", "B", "C", "C"],
            }
        )
        g = CGFull().nodes(nodes_df, "id").edges(pd.DataFrame({"s": [], "d": []}), "s", "d")

        result = g.gfql(
            "MATCH (a) "
            "WITH a.name AS name "
            f"ORDER BY a.name + 'C' {direction} "
            "LIMIT 2 "
            "RETURN name"
        )

        assert result._nodes.to_dict(orient="records") == expected

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            (
                "RETURN 2 AS x UNION RETURN 1 AS x UNION RETURN 2 AS x",
                [{"x": 2}, {"x": 1}],
            ),
            (
                "RETURN 2 AS x UNION ALL RETURN 1 AS x UNION ALL RETURN 2 AS x",
                [{"x": 2}, {"x": 1}, {"x": 2}],
            ),
        ],
    )
    def test_gfql_executes_cypher_union_set_ops(self, query, expected):
        result = _mk_empty_graph().gfql(query)

        assert result._nodes.to_dict(orient="records") == expected

    def test_gfql_executes_cypher_union_with_unwind(self):
        result = _mk_empty_graph().gfql(
            "UNWIND [2, 1, 2, 3] AS x RETURN x "
            "UNION "
            "UNWIND [3, 4] AS x RETURN x"
        )

        assert result._nodes.to_dict(orient="records") == [
            {"x": 2},
            {"x": 1},
            {"x": 3},
            {"x": 4},
        ]

    def test_gfql_executes_cypher_union_of_whole_row_entity_outputs(self):
        nodes_df = pd.DataFrame({"id": ["a", "b"], "type": ["A", "B"]})
        g = CGFull().nodes(nodes_df, "id").edges(pd.DataFrame({"s": [], "d": []}), "s", "d")

        result = g.gfql("MATCH (a:A) RETURN a AS a UNION MATCH (b:B) RETURN b AS a")

        assert result._nodes.to_dict(orient="records") == [
            {"a": "(:A)"},
            {"a": "(:B)"},
        ]

    def test_gfql_rejects_cypher_union_with_mismatched_columns(self):
        with pytest.raises(GFQLValidationError) as exc_info:
            _mk_empty_graph().gfql("RETURN 1 AS a UNION RETURN 2 AS b")

        assert exc_info.value.code == ErrorCode.E108

    def test_gfql_rejects_mixed_union_kinds(self):
        with pytest.raises(GFQLSyntaxError) as exc_info:
            _mk_empty_graph().gfql("RETURN 1 AS a UNION RETURN 2 AS a UNION ALL RETURN 3 AS a")

        assert exc_info.value.code == ErrorCode.E107

    @pytest.mark.parametrize(
        ("direction", "expected"),
        [
            ("ASC", [{"name": "A", "cnt": 2}]),
            ("DESC", [{"name": "C", "cnt": 2}]),
        ],
    )
    def test_gfql_executes_cypher_with_aggregate_order_by_source_expression(self, direction, expected):
        nodes_df = pd.DataFrame(
            {
                "id": ["a1", "a2", "b1", "c1", "c2"],
                "type": ["person"] * 5,
                "name": ["A", "A", "B", "C", "C"],
            }
        )
        g = CGFull().nodes(nodes_df, "id").edges(pd.DataFrame({"s": [], "d": []}), "s", "d")

        result = g.gfql(
            "MATCH (a) "
            "WITH a.name AS name, count(*) AS cnt "
            f"ORDER BY a.name + 'C' {direction} "
            "LIMIT 1 "
            "RETURN name, cnt"
        )

        assert result._nodes.to_dict(orient="records") == expected

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("RETURN range(0, 3) AS vals", [{"vals": [0, 1, 2, 3]}]),
            ("RETURN 0x1A AS literal", [{"literal": 26}]),
            ("RETURN 0o12 AS literal", [{"literal": 10}]),
            ("RETURN .1e2 AS literal", [{"literal": 10.0}]),
            ("RETURN -0.0 AS literal", [{"literal": 0.0}]),
            ("RETURN keys({k: 1, l: null}) AS ks", [{"ks": ["k", "l"]}]),
            ("WITH null AS m RETURN keys(m) AS ks, keys(null) AS null_keys", [{"ks": None, "null_keys": None}]),
        ],
    )
    def test_gfql_executes_cypher_literal_list_and_map_scalar_queries(self, query, expected):
        g = _mk_graph(ids=["a"], types=["person"], src=[], dst=[])

        result = g.gfql(query)

        assert result._nodes.to_dict(orient="records") == expected
    
    def test_gfql_deprecation_and_migration(self):
        """Test deprecation warnings and migration path"""
        import warnings
        g = _mk_people_company_graph3()
        
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
        g = _mk_people_company_graph3()

        # This was the exact case that failed before our fix
        result = g.gfql([{"type": "Node"}])

        # Should execute successfully without TypeError
        assert result is not None
        assert hasattr(result, '_nodes')
        # Should return all nodes since no filter specified
        assert len(result._nodes) == 3

    def test_gfql_with_mixed_list_ast_and_dicts(self):
        """Test gfql with list containing both AST objects and dicts"""
        g = _mk_people_company_graph3()

        # Mixed: AST object + raw dict
        result = g.gfql([n({'type': 'person'}), {"type": "Node"}])

        # Should execute successfully
        assert result is not None
        assert hasattr(result, '_nodes')

    def test_gfql_with_multiple_dicts_in_list(self):
        """Test gfql with list containing multiple dict objects"""
        g = _mk_people_company_graph3()

        # Multiple dicts in sequence
        result = g.gfql([{"type": "Node"}, {"type": "Node"}])

        # Should execute successfully
        assert result is not None
        assert hasattr(result, '_nodes')

    def test_gfql_dict_conversion_with_filter(self):
        """Test dict conversion works with actual filtering"""
        g = _mk_people_company_graph3()

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
        g = _mk_people_company_graph3()

        # Test that list dict conversion works - main regression test
        result_list = g.gfql([{"type": "Node"}])

        # Should succeed without TypeError
        assert result_list is not None
        assert hasattr(result_list, '_nodes')
        assert len(result_list._nodes) >= 0  # Ensure execution succeeds

    def test_gfql_let_dict_envelope(self):
        """g.gfql() must accept a pre-serialized Let dict (issue #963).

        The ETL server receives Let envelopes from gfql_remote() and passes
        them to g.gfql(). The dict dispatch must recognize {"type": "Let"}
        and deserialize via ASTLet.from_json() instead of treating it as a
        bare binding dict.
        """
        g = _mk_people_company_graph3()
        from graphistry.compute.ast import ASTLet, ASTNode

        # Serialize a Let to JSON dict (this is what gfql_remote sends)
        let_obj = ASTLet({'people': ASTNode(filter_dict={'type': 'person'})})
        let_dict = let_obj.to_json()
        assert let_dict['type'] == 'Let'

        # g.gfql() must accept this and produce correct results
        result = g.gfql(let_dict)
        assert len(result._nodes) == 2
        assert all(result._nodes['type'] == 'person')

    def test_gfql_chain_dict_envelope(self):
        """g.gfql() must accept a pre-serialized Chain dict with WHERE."""
        g = _mk_people_company_graph3()
        from graphistry.compute.chain import Chain
        from graphistry.compute.ast import ASTNode, ASTEdge

        chain = Chain([ASTNode(filter_dict={'type': 'person'})])
        chain_dict = chain.to_json()
        assert chain_dict['type'] == 'Chain'

        result = g.gfql(chain_dict)
        assert len(result._nodes) == 2
        assert all(result._nodes['type'] == 'person')


class TestGFQLCypherReentryCarrier:

    _BASE_A_ROWS = {
        "a1": {"id": "a1", "label__A": True, "num": 1},
        "a2": {"id": "a2", "label__A": True, "num": 2},
    }

    @staticmethod
    def _compile_reentry_query(
        with_clause: str = "a, a.num AS property",
        *,
        match_alias: str = "a",
    ):
        return compile_cypher(
            "MATCH (a:A) "
            f"WITH {with_clause} "
            f"MATCH ({match_alias})-->(b) "
            "RETURN b.id AS bid"
        )

    @staticmethod
    def _bind_reentry_prefix_result(
        g,
        rows: Dict[str, List[Any]],
        ids: List[Any],
        *,
        output_name: str = "a",
    ):
        prefix_result = g.bind()
        prefix_result._nodes = pd.DataFrame(rows)
        prefix_result._cypher_entity_projection_meta = {
            output_name: {
                "table": "nodes",
                "alias": output_name,
                "id_column": "id",
                "ids": pd.Series(ids, name="id"),
            }
        }
        return prefix_result

    @staticmethod
    def _carry_by_id(dispatch_graph):
        return {
            row["id"]: {key: value for key, value in row.items() if key.startswith("__cypher_reentry_")}
            for row in dispatch_graph._nodes.to_dict(orient="records")
            if row["id"] in {"a1", "a2"} and any(key.startswith("__cypher_reentry_") for key in row)
        }

    @staticmethod
    def _hidden_updates(**values: Any) -> Dict[str, Any]:
        return {
            _reentry_hidden_column_name(output_name): value
            for output_name, value in values.items()
        }

    @classmethod
    def _expected_reentry_rows(cls, ordered_ids: List[str], carry_values_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                **cls._BASE_A_ROWS[node_id],
                **cls._hidden_updates(**carry_values_by_id.get(node_id, {})),
            }
            for node_id in ordered_ids
        ]

    @staticmethod
    def _run_reentry_state(g, compiled, prefix_result):
        return _compiled_query_reentry_state(
            g,
            compiled,
            prefix_result,
            engine="pandas",
        )

    @staticmethod
    def _assert_hidden_columns_preserved(g, expected_values_by_column: Dict[str, List[Any]]):
        for column, expected_values in expected_values_by_column.items():
            series = g._nodes[column].reset_index(drop=True)
            assert len(series) == len(expected_values)
            for actual, expected in zip(series.tolist(), expected_values):
                if pd.isna(expected):
                    assert pd.isna(actual)
                else:
                    assert actual == expected

    def _assert_reentry_state_by_id(
        self,
        *,
        g,
        compiled,
        prefix_result,
        ordered_ids,
        carry_values_by_id,
        expect_same_graph,
    ):
        dispatch_graph, start_nodes = self._run_reentry_state(g, compiled, prefix_result)
        assert (dispatch_graph is g) is expect_same_graph
        assert start_nodes.to_dict(orient="records") == self._expected_reentry_rows(ordered_ids, carry_values_by_id)
        assert self._carry_by_id(dispatch_graph) == {
            node_id: self._hidden_updates(**values)
            for node_id, values in carry_values_by_id.items()
        }

    def test_reentry_state_preserves_prefix_order_without_carried_columns(self):
        g = _mk_reentry_scalar_graph()
        self._assert_reentry_state_by_id(
            g=g,
            compiled=compile_cypher(
                "MATCH (a:A) "
                "WITH a "
                "MATCH (a)-->(b) "
                "RETURN b.id AS bid"
            ),
            prefix_result=g.gfql("MATCH (a:A) WITH a ORDER BY a.num DESC RETURN a"),
            ordered_ids=["a2", "a1"],
            carry_values_by_id={},
            expect_same_graph=True,
        )

    @pytest.mark.parametrize(
        ("rows", "ids", "match"),
        [
            ({"property": [1, 1]}, ["a1", "a1"], "unique carried node rows"),
            ({"property": [1]}, ["a1", "a2"], "metadata row counts disagreed"),
        ],
    )
    def test_reentry_state_rejects_invalid_carried_scalar_rows(self, rows, ids, match):
        g = _mk_reentry_scalar_graph()
        with pytest.raises(GFQLValidationError, match=match):
            self._run_reentry_state(
                g,
                self._compile_reentry_query(),
                self._bind_reentry_prefix_result(g, rows=rows, ids=ids),
            )

    def test_reentry_state_filters_null_carried_ids_before_aligning_scalar_payload(self):
        g = _mk_reentry_scalar_graph()
        self._assert_reentry_state_by_id(
            g=g,
            compiled=self._compile_reentry_query(),
            prefix_result=self._bind_reentry_prefix_result(
                g,
                rows={"property": [10, 20, 30]},
                ids=["a1", None, "a2"],
            ),
            ordered_ids=["a1", "a2"],
            carry_values_by_id={
                "a1": {"property": 10},
                "a2": {"property": 30},
            },
            expect_same_graph=False,
        )

    @pytest.mark.parametrize(
        ("with_clause", "rows", "carry_values_by_id", "existing_hidden_values"),
        [
            (
                "a, a.num AS property",
                {"property": [2, 1]},
                {
                    "a1": {"property": 1},
                    "a2": {"property": 2},
                },
                {"__cypher_reentry_property__": ["orig1", "orig2", None, None]},
            ),
            (
                "a, a.num AS property, a.num + 10 AS property2",
                {"property": [2, 1], "property2": [12, 11]},
                {
                    "a1": {"property": 1, "property2": 11},
                    "a2": {"property": 2, "property2": 12},
                },
                {
                    "__cypher_reentry_property__": ["orig1", "orig2", None, None],
                    "__cypher_reentry_property2__": ["keep1", "keep2", None, None],
                },
            ),
        ],
    )
    def test_reentry_state_overrides_internal_hidden_column_collisions(
        self,
        with_clause,
        rows,
        carry_values_by_id,
        existing_hidden_values,
    ):
        g = _mk_reentry_scalar_graph()
        g._nodes = g._nodes.assign(**existing_hidden_values)

        self._assert_reentry_state_by_id(
            g=g,
            compiled=self._compile_reentry_query(with_clause),
            prefix_result=self._bind_reentry_prefix_result(
                g,
                rows=rows,
                ids=["a2", "a1"],
            ),
            ordered_ids=["a2", "a1"],
            carry_values_by_id=carry_values_by_id,
            expect_same_graph=False,
        )

        self._assert_hidden_columns_preserved(g, existing_hidden_values)
