"""The index DDL accepts the Cypher optional spellings next to its own.

``CREATE INDEX [name] [IF NOT EXISTS] ... ON (prop)`` and ``DROP INDEX name [IF EXISTS]`` are
how Cypher (and the GQL implementations that add index DDL) spell the options; the GFQL forms
kept ``IF EXISTS`` before the name and took ``ON col`` without parentheses. Both spellings
parse to the same op; the kind-targeted ``FOR <kind>`` form and the ``GFQL`` token stay.
"""
import pytest

from graphistry.compute.gfql.index.cypher_ddl import parse_index_ddl
from graphistry.compute.gfql.index.wire import CreateIndex, DropIndex
from graphistry.tests.compute.gfql.index.test_index_ddl_in_one_call import ENGINES, _dst, _graph, _require, _served

SEEDED_HOP = "MATCH (m {id: 0})-[e]->(p) RETURN p"


@pytest.mark.parametrize("cypher_spelling,gfql_spelling", [
    ("CREATE GFQL INDEX IF NOT EXISTS FOR node_id", "CREATE GFQL INDEX FOR node_id"),
    ("CREATE GFQL INDEX my_idx IF NOT EXISTS FOR node_prop ON (id)", "CREATE GFQL INDEX my_idx FOR node_prop ON id"),
    ("CREATE GFQL INDEX FOR node_prop ON ( id )", "CREATE GFQL INDEX FOR node_prop ON id"),
    ("CREATE GFQL INDEX FOR node_prop ON(id)", "CREATE GFQL INDEX FOR node_prop ON id"),
    ("DROP GFQL INDEX my_idx IF EXISTS", "DROP GFQL INDEX IF EXISTS my_idx"),
    ("DROP GFQL INDEX IF EXISTS FOR node_prop ON (id)", "DROP GFQL INDEX IF EXISTS FOR node_prop ON id"),
])
def test_cypher_spelling_parses_to_the_same_op(cypher_spelling, gfql_spelling):
    assert parse_index_ddl(cypher_spelling) == parse_index_ddl(gfql_spelling)
    assert parse_index_ddl(cypher_spelling) is not None


def test_the_options_land_in_the_op():
    op = parse_index_ddl("CREATE GFQL INDEX my_idx IF NOT EXISTS FOR node_prop ON (score)")
    assert op == CreateIndex(kind="node_prop", column="score", name="my_idx")
    op = parse_index_ddl("DROP GFQL INDEX my_idx IF EXISTS")
    assert isinstance(op, DropIndex) and op.name == "my_idx" and op.missing_ok
    assert not parse_index_ddl("DROP GFQL INDEX my_idx").missing_ok


@pytest.mark.parametrize("bad", [
    "DROP GFQL INDEX IF NOT EXISTS FOR node_id",         # NOT EXISTS is a CREATE option
    "CREATE GFQL INDEX IF EXISTS FOR node_id",           # IF EXISTS is a DROP option
    "DROP GFQL INDEX IF EXISTS my_idx IF EXISTS",        # once
    "CREATE GFQL INDEX FOR node_prop ON (id",            # unbalanced
    "CREATE GFQL INDEX FOR node_prop ON id)",
    "CREATE GFQL INDEX IF NOT EXISTS my_idx FOR node_id",  # name comes before the option, as in Cypher
])
def test_misplaced_options_are_still_malformed(bad):
    with pytest.raises(ValueError, match="Malformed GFQL INDEX DDL"):
        parse_index_ddl(bad)


@pytest.mark.route_engaged("native-fast", "index-hop", "indexed-kernel", "cypher-fast")
@pytest.mark.parametrize("engine", ENGINES)
def test_cypher_spelling_builds_and_serves_in_one_call(engine):
    _require(engine)
    g = _graph(engine)
    fused = ("CREATE GFQL INDEX IF NOT EXISTS FOR edge_out_adj; CREATE GFQL INDEX seeds IF NOT EXISTS FOR node_id; "
             + SEEDED_HOP)
    two_step = g.gfql("CREATE GFQL INDEX FOR edge_out_adj").gfql("CREATE GFQL INDEX FOR node_id")
    expected = two_step.gfql_explain(SEEDED_HOP, engine=engine)
    report = g.gfql_explain(fused, engine=engine)
    assert report["used_index"] is True and _served(report) == _served(expected)
    assert _dst(g.gfql(fused, engine=engine)) == _dst(two_step.gfql(SEEDED_HOP, engine=engine))
