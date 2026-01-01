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
        key='expr-literals7-1',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[1] Return an empty list',
        cypher='RETURN [] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-2',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[2] Return a list containing a boolean',
        cypher='RETURN [false] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '[false]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-3',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[3] Return a list containing a null',
        cypher='RETURN [null] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '[null]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-4',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[4] Return a list containing a integer',
        cypher='RETURN [1] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '[1]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-5',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[5] Return a list containing a hexadecimal integer',
        cypher='RETURN [-0x162CD4F6] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '[-372036854]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-6',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[6] Return a list containing a octal integer',
        cypher='RETURN [0o2613152366] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '[372036854]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-7',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[7] Return a list containing a float',
        cypher='RETURN [-.1e-5] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '[-0.000001]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-8',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[8] Return a list containing a string',
        cypher="RETURN ['abc, as#?lßdj '] AS literal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': "['abc, as#?lßdj ']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-9',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[9] Return a list containing an empty lists',
        cypher='RETURN [[]] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '[[]]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-10',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[10] Return seven-deep nested empty lists',
        cypher='RETURN [[[[[[[]]]]]]] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '[[[[[[[]]]]]]]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-11',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[11] Return 20-deep nested empty lists',
        cypher='RETURN [[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-12',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[12] Return 40-deep nested empty lists',
        cypher='RETURN [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-13',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[13] Return a list containing an empty map',
        cypher='RETURN [{}] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '[{}]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-14',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[14] Return a list containing multiple integer',
        cypher='RETURN [1, -2, 0o77, 0xA4C, 71034856] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '[1, -2, 63, 2636, 71034856]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-16',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[16] Return a list containing multiple mixed values',
        cypher="RETURN [2E-01, ', as#?lßdj ', null, 71034856, false] AS literal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': "[0.2, ', as#?lßdj ', null, 71034856, false]"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-17',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[17] Return a list containing real and fake nested lists',
        cypher="RETURN [null, [ ' a ', ' ' ], ' [ a ', ' [ ], ] ', ' [ ', [ ' ' ], ' ] ' ] AS literal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': "[null, [' a ', ' '], ' [ a ', ' [ ], ] ', ' [ ', [' '], ' ] ']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-18',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[18] Return a complex list containing multiple mixed and nested values',
        cypher="RETURN [ {\n                  id: '0001',\n                  type: 'donut',\n                  name: 'Cake',\n                  ppu: 0.55,\n                  batters:\n                      {\n                          batter:\n                              [\n                                  { id: '1001', type: 'Regular' },\n                                  { id: '1002', type: 'Chocolate' },\n                                  { id: '1003', type: 'Blueberry' },\n                                  { id: '1004', type: 'Devils Food' }\n                              ]\n                      },\n                  topping:\n                      [\n                          { id: '5001', type: 'None' },\n                          { id: '5002', type: 'Glazed' },\n                          { id: '5005', type: 'Sugar' },\n                          { id: '5007', type: 'Powdered Sugar' },\n                          { id: '5006', type: 'Chocolate Sprinkles' },\n                          { id: '5003', type: 'Chocolate' },\n                          { id: '5004', type: 'Maple' }\n                      ]\n              },\n              {\n                  id: '0002',\n                  type: 'donut',\n                  name: 'Raised',\n                  ppu: 0.55,\n                  batters:\n                      {\n                          batter:\n                              [\n                                  { id: '1001', type: 'Regular' }\n                              ]\n                      },\n                  topping:\n                      [\n                          { id: '5001', type: 'None' },\n                          { id: '5002', type: 'Glazed' },\n                          { id: '5005', type: 'Sugar' },\n                          { id: '5003', type: 'Chocolate' },\n                          { id: '5004', type: 'Maple' }\n                      ]\n              },\n              {\n                  id: '0003',\n                  type: 'donut',\n                  name: 'Old Fashioned',\n                  ppu: 0.55,\n                  batters:\n                      {\n                          batter:\n                              [\n                                  { id: '1001', type: 'Regular' },\n                                  { id: '1002', type: 'Chocolate' }\n                              ]\n                      },\n                  topping:\n                      [\n                          { id: '5001', type: 'None' },\n                          { id: '5002', type: 'Glazed' },\n                          { id: '5003', type: 'Chocolate' },\n                          { id: '5004', type: 'Maple' }\n                      ]\n              } ] AS literal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': "[{id: '0001', type: 'donut', name: 'Cake', ppu: 0.55, batters: {batter: [{id: '1001', type: 'Regular'}, {id: '1002', type: 'Chocolate'}, {id: '1003', type: 'Blueberry'}, {id: '1004', type: 'Devils Food'}]}, topping: [{id: '5001', type: 'None'}, {id: '5002', type: 'Glazed'}, {id: '5005', type: 'Sugar'}, {id: '5007', type: 'Powdered Sugar'}, {id: '5006', type: 'Chocolate Sprinkles'}, {id: '5003', type: 'Chocolate'}, {id: '5004', type: 'Maple'}]}, {id: '0002', type: 'donut', name: 'Raised', ppu: 0.55, batters: {batter: [{id: '1001', type: 'Regular'}]}, topping: [{id: '5001', type: 'None'}, {id: '5002', type: 'Glazed'}, {id: '5005', type: 'Sugar'}, {id: '5003', type: 'Chocolate'}, {id: '5004', type: 'Maple'}]}, {id: '0003', type: 'donut', name: 'Old Fashioned', ppu: 0.55, batters: {batter: [{id: '1001', type: 'Regular'}, {id: '1002', type: 'Chocolate'}]}, topping: [{id: '5001', type: 'None'}, {id: '5002', type: 'Glazed'}, {id: '5003', type: 'Chocolate'}, {id: '5004', type: 'Maple'}]}]"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-19',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[19] Fail on a list containing only a comma',
        cypher='RETURN [, ] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-20',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[20] Fail on a nested list with non-matching brackets',
        cypher='RETURN [[[]] AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals7-21',
        feature_path='tck/features/expressions/literals/Literals7.feature',
        scenario='[21] Fail on a nested list with missing commas',
        cypher="RETURN [[','[]',']] AS literal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),
]
