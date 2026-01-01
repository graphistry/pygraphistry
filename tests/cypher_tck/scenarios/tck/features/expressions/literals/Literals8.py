from graphistry.compute import e_forward, e_undirected, n

from tests.cypher_tck.models import Expected, GraphFixture, Scenario
from tests.cypher_tck.parse_cypher import graph_fixture_from_create

from tests.cypher_tck.scenarios.fixtures import (
    MATCH5_GRAPH,
    MATCH7_GRAPH_SINGLE,
    MATCH7_GRAPH_AB,
    MATCH7_GRAPH_ABC,
    MATCH7_GRAPH_REL,
    MATCH7_GRAPH_X,
    MATCH7_GRAPH_AB_X,
    MATCH7_GRAPH_LABELS,
    MATCH7_GRAPH_PLAYER_TEAM_BOTH,
    MATCH7_GRAPH_PLAYER_TEAM_SINGLE,
    MATCH7_GRAPH_PLAYER_TEAM_DIFF,
    WITH_ORDERBY4_GRAPH,
    BINARY_TREE_1_GRAPH,
    BINARY_TREE_2_GRAPH,
)


SCENARIOS = [
    Scenario(
        key='expr-literals8-1',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[1] Return an empty map',
        cypher='RETURN {} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-2',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[2] Return a map containing one value with alphabetic lower case key',
        cypher='RETURN {abc: 1} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{abc: 1}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-3',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[3] Return a map containing one value with alphabetic upper case key',
        cypher='RETURN {ABC: 1} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{ABC: 1}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-4',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[4] Return a map containing one value with alphabetic mixed case key',
        cypher='RETURN {aBCdeF: 1} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{aBCdeF: 1}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-5',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[5] Return a map containing one value with alphanumeric mixed case key',
        cypher='RETURN {a1B2c3e67: 1} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{a1B2c3e67: 1}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-6',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[6] Return a map containing a boolean',
        cypher='RETURN {k: false} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{k: false}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-7',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[7] Return a map containing a null',
        cypher='RETURN {k: null} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{k: null}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-8',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[8] Return a map containing a integer',
        cypher='RETURN {k: 1} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{k: 1}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-9',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[9] Return a map containing a hexadecimal integer',
        cypher='RETURN {F: -0x162CD4F6} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{F: -372036854}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-10',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[10] Return a map containing a octal integer',
        cypher='RETURN {k: 0o2613152366} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{k: 372036854}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-11',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[11] Return a map containing a float',
        cypher='RETURN {k: -.1e-5} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{k: -0.000001}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-12',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[12] Return a map containing a string',
        cypher="RETURN {k: 'ab: c, as#?lßdj '} AS literal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': "{k: 'ab: c, as#?lßdj '}"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-13',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[13] Return a map containing an empty map',
        cypher='RETURN {a: {}} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{a: {}}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-14',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[14] Return seven-deep nested maps',
        cypher='RETURN {a1: {a2: {a3: {a4: {a5: {a6: {}}}}}}} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{a1: {a2: {a3: {a4: {a5: {a6: {}}}}}}}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-15',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[15] Return 20-deep nested maps',
        cypher='RETURN {a1: {a2: {a3: {a4: {a5: {a6: {a7: {a8: {a9: {a10: {a11: {a12: {a13: {a14: {a15: {a16: {a17: {a18: {a19: {}}}}}}}}}}}}}}}}}}}} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{a1: {a2: {a3: {a4: {a5: {a6: {a7: {a8: {a9: {a10: {a11: {a12: {a13: {a14: {a15: {a16: {a17: {a18: {a19: {}}}}}}}}}}}}}}}}}}}}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-16',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[16] Return 40-deep nested maps',
        cypher='RETURN {a1: {a2: {a3: {a4: {a5: {a6: {a7: {a8: {a9: {a10: {a11: {a12: {a13: {a14: {a15: {a16: {a17: {a18: {a19: {a20: {a21: {a22: {a23: {a24: {a25: {a26: {a27: {a28: {a29: {a30: {a31: {a32: {a33: {a34: {a35: {a36: {a37: {a38: {a39: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '{a1: {a2: {a3: {a4: {a5: {a6: {a7: {a8: {a9: {a10: {a11: {a12: {a13: {a14: {a15: {a16: {a17: {a18: {a19: {a20: {a21: {a22: {a23: {a24: {a25: {a26: {a27: {a28: {a29: {a30: {a31: {a32: {a33: {a34: {a35: {a36: {a37: {a38: {a39: {}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-17',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[17] Return a map containing real and fake nested maps',
        cypher="RETURN { a : ' { b : ' , c : { d : ' ' } , d : ' } ' } AS literal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': "{a: ' { b : ', c: {d: ' '}, d: ' } '}"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-18',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[18] Return a complex map containing multiple mixed and nested values',
        cypher="RETURN  { data: [ {\n                  id: '0001',\n                  type: 'donut',\n                  name: 'Cake',\n                  ppu: 0.55,\n                  batters:\n                      {\n                          batter:\n                              [\n                                  { id: '1001', type: 'Regular' },\n                                  { id: '1002', type: 'Chocolate' },\n                                  { id: '1003', type: 'Blueberry' },\n                                  { id: '1004', type: 'Devils Food' }\n                              ]\n                      },\n                  topping:\n                      [\n                          { id: '5001', type: 'None' },\n                          { id: '5002', type: 'Glazed' },\n                          { id: '5005', type: 'Sugar' },\n                          { id: '5007', type: 'Powdered Sugar' },\n                          { id: '5006', type: 'Chocolate Sprinkles' },\n                          { id: '5003', type: 'Chocolate' },\n                          { id: '5004', type: 'Maple' }\n                      ]\n              },\n              {\n                  id: '0002',\n                  type: 'donut',\n                  name: 'Raised',\n                  ppu: 0.55,\n                  batters:\n                      {\n                          batter:\n                              [\n                                  { id: '1001', type: 'Regular' }\n                              ]\n                      },\n                  topping:\n                      [\n                          { id: '5001', type: 'None' },\n                          { id: '5002', type: 'Glazed' },\n                          { id: '5005', type: 'Sugar' },\n                          { id: '5003', type: 'Chocolate' },\n                          { id: '5004', type: 'Maple' }\n                      ]\n              },\n              {\n                  id: '0003',\n                  type: 'donut',\n                  name: 'Old Fashioned',\n                  ppu: 0.55,\n                  batters:\n                      {\n                          batter:\n                              [\n                                  { id: '1001', type: 'Regular' },\n                                  { id: '1002', type: 'Chocolate' }\n                              ]\n                      },\n                  topping:\n                      [\n                          { id: '5001', type: 'None' },\n                          { id: '5002', type: 'Glazed' },\n                          { id: '5003', type: 'Chocolate' },\n                          { id: '5004', type: 'Maple' }\n                      ]\n              } ] } AS literal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': "{data: [{id: '0001', type: 'donut', name: 'Cake', ppu: 0.55, batters: {batter: [{id: '1001', type: 'Regular'}, {id: '1002', type: 'Chocolate'}, {id: '1003', type: 'Blueberry'}, {id: '1004', type: 'Devils Food'}]}, topping: [{id: '5001', type: 'None'}, {id: '5002', type: 'Glazed'}, {id: '5005', type: 'Sugar'}, {id: '5007', type: 'Powdered Sugar'}, {id: '5006', type: 'Chocolate Sprinkles'}, {id: '5003', type: 'Chocolate'}, {id: '5004', type: 'Maple'}]}, {id: '0002', type: 'donut', name: 'Raised', ppu: 0.55, batters: {batter: [{id: '1001', type: 'Regular'}]}, topping: [{id: '5001', type: 'None'}, {id: '5002', type: 'Glazed'}, {id: '5005', type: 'Sugar'}, {id: '5003', type: 'Chocolate'}, {id: '5004', type: 'Maple'}]}, {id: '0003', type: 'donut', name: 'Old Fashioned', ppu: 0.55, batters: {batter: [{id: '1001', type: 'Regular'}, {id: '1002', type: 'Chocolate'}]}, topping: [{id: '5001', type: 'None'}, {id: '5002', type: 'Glazed'}, {id: '5003', type: 'Chocolate'}, {id: '5004', type: 'Maple'}]}]}"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-19',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[19] Fail on a map containing key starting with a number',
        cypher='RETURN {1B2c3e67:1} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-20',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[20] Fail on a map containing key with symbol',
        cypher='RETURN {k1#k: 1} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-21',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[21] Fail on a map containing key with dot',
        cypher='RETURN {k1.k: 1} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-22',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[22] Fail on a map containing unquoted string',
        cypher='RETURN {k1: k2} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-23',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[23] Fail on a map containing only a comma',
        cypher='RETURN {, } AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-24',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[24] Fail on a map containing a value without key',
        cypher='RETURN {1} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-25',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[25] Fail on a map containing a list without key',
        cypher='RETURN {[]} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-26',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[26] Fail on a map containing a map without key',
        cypher='RETURN {{}} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals8-27',
        feature_path='tck/features/expressions/literals/Literals8.feature',
        scenario='[27] Fail on a nested map with non-matching braces',
        cypher='RETURN {k: {k: {}} AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),
]
