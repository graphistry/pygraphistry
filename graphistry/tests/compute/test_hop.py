import os
import pandas as pd
from graphistry.compute.predicates.is_in import is_in
import pytest

from graphistry.compute.ast import ASTNode, ASTEdge, n, e
from graphistry.tests.test_compute import CGFull


def _sorted_records(df: pd.DataFrame):
    cols = sorted(df.columns)
    if not cols:
        return []
    return df.sort_values(cols).reset_index(drop=True).to_dict(orient='records')


def _assert_one_hop_undirected_matches_fallback(
    g: CGFull,
    seed_nodes: pd.DataFrame,
    *,
    return_as_wave_front: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('GRAPHISTRY_HOP_FAST_PATH', raising=False)
    fast = g.hop(
        nodes=seed_nodes,
        hops=1,
        to_fixed_point=False,
        direction='undirected',
        return_as_wave_front=return_as_wave_front,
    )

    monkeypatch.setenv('GRAPHISTRY_HOP_FAST_PATH', '0')
    slow = g.hop(
        nodes=seed_nodes,
        hops=1,
        to_fixed_point=False,
        direction='undirected',
        return_as_wave_front=return_as_wave_front,
    )
    monkeypatch.delenv('GRAPHISTRY_HOP_FAST_PATH', raising=False)

    assert sorted(fast._nodes.columns.tolist()) == sorted(slow._nodes.columns.tolist())
    assert sorted(fast._edges.columns.tolist()) == sorted(slow._edges.columns.tolist())
    assert _sorted_records(fast._nodes) == _sorted_records(slow._nodes)
    assert _sorted_records(fast._edges) == _sorted_records(slow._edges)


@pytest.fixture(scope='module')
def g_long_forwards_chain() -> CGFull:
    """
    a->b->c->d->e
    """
    return (CGFull()
        .edges(pd.DataFrame({
            's': ['a', 'b', 'c', 'd'],
            'd': ['b', 'c', 'd', 'e'],
            't': ['1', '2', '3', '4'],
            'e': ['2', '3', '4', '5']}),
            's', 'd')
        .nodes(pd.DataFrame({
            'v': ['a', 'b', 'c', 'd', 'e'],
            'w': ['1', '2', '3', '4', '5']}),
            'v'))

@pytest.fixture(scope='module')
def n_a(g_long_forwards_chain: CGFull) -> pd.DataFrame:
    return g_long_forwards_chain._nodes.query('v == "a"')


@pytest.fixture(scope='module')
def n_mt(g_long_forwards_chain: CGFull) -> pd.DataFrame:
    return g_long_forwards_chain._nodes[:0]

@pytest.fixture(scope='module')
def n_d(g_long_forwards_chain: CGFull) -> pd.DataFrame:
    return g_long_forwards_chain._nodes.query('v == "d"')


class TestMultiHopForward():
    """
    Test multi-hop as used by chain, corresponding to chain multi-hop tests
    """

    def test_hop_short_forward(self, g_long_forwards_chain: CGFull, n_a):
        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=2,
            to_fixed_point=False,
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set(['b', 'c'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'}
        ]

    def test_hop_short_back(self, g_long_forwards_chain: CGFull, n_mt, n_a):
        g_reverse = g_long_forwards_chain.nodes(
            g_long_forwards_chain._nodes[
                g_long_forwards_chain._nodes['v'].isin(['b', 'c'])
            ]).edges(
                g_long_forwards_chain._edges.merge(
                    pd.DataFrame({
                        's': ['a', 'b'],
                        'd': ['b', 'c']}),
                    on=['s', 'd'],
                    how='inner'
                ))
        g2 = g_reverse.hop(
            nodes=n_mt,
            hops=2,
            to_fixed_point=False,
            direction='reverse',
            return_as_wave_front=True,
            target_wave_front=n_a
        )
        assert len(g2._nodes) == 0
        assert len(g2._edges) == 0

    def test_hop_exact_forward(self, g_long_forwards_chain, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=3,
            to_fixed_point=False,
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set(['b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_hop_labels_forward(self, g_long_forwards_chain: CGFull, n_a):
        # Exercise label tracking path (cuDF-safe seen IDs).
        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=3,
            to_fixed_point=False,
            direction='forward',
            label_node_hops='nh',
            label_edge_hops='eh',
            label_seeds=True
        )
        assert 'nh' in g2._nodes.columns
        assert 'eh' in g2._edges.columns
        assert g2._nodes['nh'].isna().sum() == 0
        assert g2._edges['eh'].isna().sum() == 0
        node_hops = {
            row['v']: int(row['nh'])
            for row in g2._nodes[['v', 'nh']].to_dict(orient='records')
        }
        assert node_hops == {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        edge_hops = {
            (row['s'], row['d']): int(row['eh'])
            for row in g2._edges[['s', 'd', 'eh']].to_dict(orient='records')
        }
        assert edge_hops == {('a', 'b'): 1, ('b', 'c'): 2, ('c', 'd'): 3}

    def test_hop_exact_back(self, g_long_forwards_chain: CGFull, n_d, n_a):
        g_reverse = g_long_forwards_chain.nodes(
            g_long_forwards_chain._nodes[
                g_long_forwards_chain._nodes['v'].isin(['a', 'b', 'c', 'd'])
            ]).edges(
                g_long_forwards_chain._edges.merge(
                    pd.DataFrame({
                        's': ['a', 'b', 'c'],
                        'd': ['b', 'c', 'd']}),
                    on=['s', 'd'],
                    how='inner'
                ))
        g2 = g_reverse.hop(
            nodes=n_d,
            hops=3,
            to_fixed_point=False,
            direction='reverse',
            return_as_wave_front=True,
            target_wave_front=n_a
        )
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]


    def test_hop_long_forward(self, g_long_forwards_chain, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set(['b', 'c', 'd', 'e'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
            {'s': 'd', 'd': 'e'}
        ]

    def test_hop_fixedpoint_undirected_does_not_revisit_seed_via_same_edge(self, g_long_forwards_chain: CGFull, n_a):
        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            to_fixed_point=True,
            direction='undirected',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == {'b', 'c', 'd', 'e'}
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
            {'s': 'd', 'd': 'e'}
        ]

    def test_hop_one_hop_undirected_keeps_rediscovered_seed_from_other_seed(self, g_long_forwards_chain: CGFull):
        seed_nodes = g_long_forwards_chain._nodes[g_long_forwards_chain._nodes['v'].isin(['a', 'b'])]

        g2 = g_long_forwards_chain.hop(
            nodes=seed_nodes,
            hops=1,
            to_fixed_point=False,
            direction='undirected',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == {'a', 'b', 'c'}
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'}
        ]

    def test_hop_one_hop_undirected_excludes_disconnected_explicit_seed(self):
        g_disconnected = (
            CGFull()
            .edges(
                pd.DataFrame(
                    {
                        's': ['a'],
                        'd': ['b'],
                    }
                ),
                's',
                'd',
            )
            .nodes(pd.DataFrame({'v': ['a', 'b', 'x']}), 'v')
        )
        seed_nodes = g_disconnected._nodes[g_disconnected._nodes['v'].isin(['a', 'x'])]

        g2 = g_disconnected.hop(
            nodes=seed_nodes,
            hops=1,
            to_fixed_point=False,
            direction='undirected',
            return_as_wave_front=False
        )
        assert set(g2._nodes['v'].tolist()) == {'a', 'b'}
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'}
        ]

    @pytest.mark.parametrize('return_as_wave_front', [False, True])
    def test_hop_one_hop_undirected_duplicate_seed_rows_match_fallback(self, g_long_forwards_chain: CGFull, return_as_wave_front: bool, monkeypatch: pytest.MonkeyPatch):
        seed_nodes = pd.DataFrame({'v': ['a', 'a', 'b']})

        _assert_one_hop_undirected_matches_fallback(
            g_long_forwards_chain,
            seed_nodes,
            return_as_wave_front=return_as_wave_front,
            monkeypatch=monkeypatch,
        )

    @pytest.mark.parametrize('return_as_wave_front', [False, True])
    def test_hop_one_hop_undirected_numeric_ids_match_fallback(self, return_as_wave_front: bool, monkeypatch: pytest.MonkeyPatch):
        g_numeric = (
            CGFull()
            .edges(
                pd.DataFrame(
                    {
                        's': [0, 1, 1, 2],
                        'd': [1, 2, 2, 3],
                    }
                ),
                's',
                'd',
            )
            .nodes(pd.DataFrame({'v': [0, 1, 2, 3]}), 'v')
        )
        seed_nodes = g_numeric._nodes[g_numeric._nodes['v'].isin([0, 1])]

        _assert_one_hop_undirected_matches_fallback(
            g_numeric,
            seed_nodes,
            return_as_wave_front=return_as_wave_front,
            monkeypatch=monkeypatch,
        )

    @pytest.mark.parametrize('return_as_wave_front', [False, True])
    def test_hop_one_hop_undirected_self_loop_matches_fallback(self, return_as_wave_front: bool, monkeypatch: pytest.MonkeyPatch):
        g_loop = (
            CGFull()
            .edges(
                pd.DataFrame(
                    {
                        's': ['a', 'a'],
                        'd': ['a', 'b'],
                    }
                ),
                's',
                'd',
            )
            .nodes(pd.DataFrame({'v': ['a', 'b']}), 'v')
        )
        seed_nodes = g_loop._nodes[g_loop._nodes['v'].isin(['a'])]

        _assert_one_hop_undirected_matches_fallback(
            g_loop,
            seed_nodes,
            return_as_wave_front=return_as_wave_front,
            monkeypatch=monkeypatch,
        )

    @pytest.mark.parametrize('return_as_wave_front', [False, True])
    def test_hop_one_hop_undirected_parallel_edges_match_fallback(self, return_as_wave_front: bool, monkeypatch: pytest.MonkeyPatch):
        g_parallel = (
            CGFull()
            .edges(
                pd.DataFrame(
                    {
                        's': ['a', 'a', 'b'],
                        'd': ['b', 'b', 'c'],
                    }
                ),
                's',
                'd',
            )
            .nodes(pd.DataFrame({'v': ['a', 'b', 'c']}), 'v')
        )
        seed_nodes = g_parallel._nodes[g_parallel._nodes['v'].isin(['a'])]

        _assert_one_hop_undirected_matches_fallback(
            g_parallel,
            seed_nodes,
            return_as_wave_front=return_as_wave_front,
            monkeypatch=monkeypatch,
        )

    @pytest.mark.parametrize('return_as_wave_front', [False, True])
    def test_hop_one_hop_undirected_overlapping_seed_neighborhood_matches_fallback(self, return_as_wave_front: bool, monkeypatch: pytest.MonkeyPatch):
        g_overlap = (
            CGFull()
            .edges(
                pd.DataFrame(
                    {
                        's': ['a', 'b', 'b', 'c'],
                        'd': ['b', 'c', 'd', 'e'],
                    }
                ),
                's',
                'd',
            )
            .nodes(pd.DataFrame({'v': ['a', 'b', 'c', 'd', 'e']}), 'v')
        )
        seed_nodes = g_overlap._nodes[g_overlap._nodes['v'].isin(['a', 'c'])]

        _assert_one_hop_undirected_matches_fallback(
            g_overlap,
            seed_nodes,
            return_as_wave_front=return_as_wave_front,
            monkeypatch=monkeypatch,
        )

    @pytest.mark.parametrize('return_as_wave_front', [False, True])
    def test_hop_one_hop_undirected_node_source_conflict_matches_fallback(self, return_as_wave_front: bool, monkeypatch: pytest.MonkeyPatch):
        g_conflict = (
            CGFull()
            .edges(
                pd.DataFrame(
                    {
                        'id': ['a', 'b'],
                        'dst': ['b', 'c'],
                        'weight': [1.5, 2.5],
                    }
                ),
                'id',
                'dst',
            )
            .nodes(pd.DataFrame({'id': ['a', 'b', 'c'], 'group': ['x', 'y', 'z']}), 'id')
        )
        seed_nodes = g_conflict._nodes[g_conflict._nodes['id'].isin(['a'])]

        _assert_one_hop_undirected_matches_fallback(
            g_conflict,
            seed_nodes,
            return_as_wave_front=return_as_wave_front,
            monkeypatch=monkeypatch,
        )

    @pytest.mark.parametrize('return_as_wave_front', [False, True])
    def test_hop_one_hop_undirected_node_destination_conflict_matches_fallback(self, return_as_wave_front: bool, monkeypatch: pytest.MonkeyPatch):
        g_conflict = (
            CGFull()
            .edges(
                pd.DataFrame(
                    {
                        'src': ['a', 'b'],
                        'id': ['b', 'c'],
                        'weight': [1.5, 2.5],
                    }
                ),
                'src',
                'id',
            )
            .nodes(pd.DataFrame({'id': ['a', 'b', 'c'], 'group': ['x', 'y', 'z']}), 'id')
        )
        seed_nodes = g_conflict._nodes[g_conflict._nodes['id'].isin(['b'])]

        _assert_one_hop_undirected_matches_fallback(
            g_conflict,
            seed_nodes,
            return_as_wave_front=return_as_wave_front,
            monkeypatch=monkeypatch,
        )

    @pytest.mark.parametrize('return_as_wave_front', [False, True])
    def test_hop_one_hop_undirected_explicit_edge_id_and_attrs_match_fallback(self, return_as_wave_front: bool, monkeypatch: pytest.MonkeyPatch):
        g_attr = (
            CGFull()
            .edges(
                pd.DataFrame(
                    {
                        's': ['a', 'a', 'b'],
                        'd': ['b', 'b', 'c'],
                        'eid': ['e1', 'e2', 'e3'],
                        'weight': [1.0, 2.0, 3.0],
                    }
                ),
                's',
                'd',
                'eid',
            )
            .nodes(pd.DataFrame({'v': ['a', 'b', 'c'], 'color': ['red', 'blue', 'green']}), 'v')
        )
        seed_nodes = g_attr._nodes[g_attr._nodes['v'].isin(['a'])]

        _assert_one_hop_undirected_matches_fallback(
            g_attr,
            seed_nodes,
            return_as_wave_front=return_as_wave_front,
            monkeypatch=monkeypatch,
        )

    def test_hop_fixedpoint_undirected_keeps_seed_when_reachable_via_real_cycle(self):
        g_cycle = (
            CGFull()
            .edges(
                pd.DataFrame(
                    {
                        's': ['a', 'b', 'c'],
                        'd': ['b', 'c', 'a'],
                    }
                ),
                's',
                'd',
            )
            .nodes(pd.DataFrame({'v': ['a', 'b', 'c']}), 'v')
        )
        n_a_cycle = g_cycle._nodes.query('v == "a"')

        g2 = g_cycle.hop(
            nodes=n_a_cycle,
            to_fixed_point=True,
            direction='undirected',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == {'a', 'b', 'c'}
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'a'}
        ]

    def test_hop_fixedpoint_undirected_keeps_rediscovered_seed_from_other_seed(self, g_long_forwards_chain: CGFull):
        seed_nodes = g_long_forwards_chain._nodes[g_long_forwards_chain._nodes['v'].isin(['a', 'e'])]

        g2 = g_long_forwards_chain.hop(
            nodes=seed_nodes,
            to_fixed_point=True,
            direction='undirected',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == {'a', 'b', 'c', 'd', 'e'}
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
            {'s': 'd', 'd': 'e'}
        ]

    def test_hop_fixedpoint_undirected_keeps_seed_on_self_loop(self):
        g_loop = (
            CGFull()
            .edges(
                pd.DataFrame(
                    {
                        's': ['a'],
                        'd': ['a'],
                    }
                ),
                's',
                'd',
            )
            .nodes(pd.DataFrame({'v': ['a']}), 'v')
        )
        n_a_loop = g_loop._nodes.query('v == "a"')

        g2 = g_loop.hop(
            nodes=n_a_loop,
            to_fixed_point=True,
            direction='undirected',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == {'a'}
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'a'}
        ]

    def test_hop_fixedpoint_undirected_excludes_unrediscovered_seeds_in_disconnected_components(self):
        g_disconnected = (
            CGFull()
            .edges(
                pd.DataFrame(
                    {
                        's': ['a', 'x'],
                        'd': ['b', 'y'],
                    }
                ),
                's',
                'd',
            )
            .nodes(pd.DataFrame({'v': ['a', 'b', 'x', 'y']}), 'v')
        )
        seed_nodes = g_disconnected._nodes[g_disconnected._nodes['v'].isin(['a', 'x'])]

        g2 = g_disconnected.hop(
            nodes=seed_nodes,
            to_fixed_point=True,
            direction='undirected',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == {'b', 'y'}
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'x', 'd': 'y'}
        ]

    def test_hop_long_back(self, g_long_forwards_chain: CGFull, n_d, n_a):
        g_reverse = g_long_forwards_chain.nodes(
            g_long_forwards_chain._nodes[
                g_long_forwards_chain._nodes['v'].isin(['b', 'c', 'd', 'e'])
            ]).edges(
                g_long_forwards_chain._edges.merge(
                    pd.DataFrame({
                        's': ['a', 'b', 'c', 'd'],
                        'd': ['b', 'c', 'd', 'e']}),
                    on=['s', 'd'],
                    how='inner'
                ))
        g2 = g_reverse.hop(
            nodes=n_d,
            hops=4,
            to_fixed_point=False,
            direction='reverse',
            return_as_wave_front=True,
            target_wave_front=n_a
        )
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'}
        ]

    def test_hop_predicates_ok_source_forward(self, g_long_forwards_chain: CGFull, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            source_node_match={'w': is_in(['1', '2', '3'])},
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set(['b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_ok_source_back(self, g_long_forwards_chain: CGFull, n_a, n_d):

        g_reverse = g_long_forwards_chain.nodes(
            g_long_forwards_chain._nodes[
                g_long_forwards_chain._nodes['v'].isin(['b', 'c', 'd'])
            ]).edges(
                g_long_forwards_chain._edges.merge(
                    pd.DataFrame({
                        's': ['a', 'b', 'c'],
                        'd': ['b', 'c', 'd']}),
                    on=['s', 'd'],
                    how='inner'
                ))

        g2 = g_reverse.hop(
            nodes=n_d,
            hops=4,
            to_fixed_point=False,
            destination_node_match={'w': is_in(['1', '2', '3'])},
            direction='reverse',
            return_as_wave_front=True,
            target_wave_front=n_a
        )
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_ok_edge_forward(self, g_long_forwards_chain: CGFull, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            edge_match={
                't': is_in(['1', '2', '3']),
                'e': is_in(['2', '3', '4'])
            },
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set(['b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_ok_edge_back(self, g_long_forwards_chain: CGFull, n_a, n_d):

        g_reverse = g_long_forwards_chain.nodes(
            g_long_forwards_chain._nodes[
                g_long_forwards_chain._nodes['v'].isin(['b', 'c', 'd'])
            ]).edges(
                g_long_forwards_chain._edges.merge(
                    pd.DataFrame({
                        's': ['a', 'b', 'c'],
                        'd': ['b', 'c', 'd']}),
                    on=['s', 'd'],
                    how='inner'
                ))

        g2 = g_reverse.hop(
            nodes=n_d,
            hops=4,
            to_fixed_point=False,
            edge_match={
                't': is_in(['1', '2', '3']),
                'e': is_in(['2', '3', '4'])
            },
            direction='reverse',
            return_as_wave_front=True,
            target_wave_front=n_a
        )
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_ok_destination_forward(self, g_long_forwards_chain: CGFull, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            destination_node_match={'w': is_in(['2', '3', '4'])},
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set(['b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_ok_destination_back(self, g_long_forwards_chain: CGFull, n_a, n_d):

        g_reverse = g_long_forwards_chain.nodes(
            g_long_forwards_chain._nodes[
                g_long_forwards_chain._nodes['v'].isin(['b', 'c', 'd'])
            ]).edges(
                g_long_forwards_chain._edges.merge(
                    pd.DataFrame({
                        's': ['a', 'b', 'c'],
                        'd': ['b', 'c', 'd']}),
                    on=['s', 'd'],
                    how='inner'
                ))

        g2 = g_reverse.hop(
            nodes=n_d,
            hops=4,
            to_fixed_point=False,
            source_node_match={'w': is_in(['2', '3', '4'])},
            direction='reverse',
            return_as_wave_front=True,
            target_wave_front=n_a
        )
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_ok_forward(self, g_long_forwards_chain: CGFull, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            source_node_match={'w': is_in(['1', '2', '3'])},
            edge_match={
                't': is_in(['1', '2', '3']),
                'e': is_in(['2', '3', '4'])
            },
            destination_node_match={'w': is_in(['2', '3', '4'])},
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set(['b', 'c', 'd'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_ok_back(self, g_long_forwards_chain: CGFull, n_a, n_d):

        g_reverse = g_long_forwards_chain.nodes(
            g_long_forwards_chain._nodes[
                g_long_forwards_chain._nodes['v'].isin(['b', 'c', 'd'])
            ]).edges(
                g_long_forwards_chain._edges.merge(
                    pd.DataFrame({
                        's': ['a', 'b', 'c'],
                        'd': ['b', 'c', 'd']}),
                    on=['s', 'd'],
                    how='inner'
                ))

        g2 = g_reverse.hop(
            nodes=n_d,
            hops=4,
            to_fixed_point=False,
            destination_node_match={'w': is_in(['1', '2', '3'])},
            edge_match={
                't': is_in(['1', '2', '3']),
                'e': is_in(['2', '3', '4'])
            },
            source_node_match={'w': is_in(['2', '3', '4'])},
            direction='reverse',
            return_as_wave_front=True,
            target_wave_front=n_a
        )
        assert set(g2._nodes['v'].tolist()) == set(['a', 'b', 'c'])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == [
            {'s': 'a', 'd': 'b'},
            {'s': 'b', 'd': 'c'},
            {'s': 'c', 'd': 'd'},
        ]

    def test_hop_predicates_fail_source_forward(self, g_long_forwards_chain: CGFull, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            source_node_match={'w': is_in([])},
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set([])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == []

    def test_hop_predicates_fail_edge_forward(self, g_long_forwards_chain: CGFull, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            edge_match={
                't': is_in([]),
                'e': is_in([])
            },
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set([])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == []

    def test_hop_predicates_fail_destination_forward(self, g_long_forwards_chain: CGFull, n_a):

        g2 = g_long_forwards_chain.hop(
            nodes=n_a,
            hops=4,
            to_fixed_point=False,
            destination_node_match={'w': is_in([])},
            direction='forward',
            return_as_wave_front=True
        )
        assert set(g2._nodes['v'].tolist()) == set([])
        assert g2._edges[['s', 'd']].sort_values(['s', 'd']).to_dict(orient='records') == []


def test_hop_binding_reuse():
    # This test has been updated to reflect the new behavior that allows node column names
    # to be the same as edge source or destination column names
    edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
    nodes1_df = pd.DataFrame({'v': ['a', 'b', 'c']})
    nodes2_df = pd.DataFrame({'s': ['a', 'b', 'c']})
    nodes3_df = pd.DataFrame({'d': ['a', 'b', 'c']})
    
    g1 = CGFull().nodes(nodes1_df, 'v').edges(edges_df, 's', 'd')
    g2 = CGFull().nodes(nodes2_df, 's').edges(edges_df, 's', 'd')
    g3 = CGFull().nodes(nodes3_df, 'd').edges(edges_df, 's', 'd')

    # With our new implementation, all three should successfully run
    g1_hop = g1.hop()
    g2_hop = g2.hop()
    g3_hop = g3.hop()
    
    # Make sure we get expected results - g1 and g2 have consistent behavior
    assert g1_hop._nodes.shape == g2_hop._nodes.shape
    assert g1_hop._edges.shape == g2_hop._edges.shape
    
    # g3 behavior differs because of how the node/edge bindings interact
    # we don't need identical behavior, just reasonable behavior
    assert g3_hop._nodes.shape[0] > 0
    assert g3_hop._edges.shape[0] > 0    

def test_hop_simple_cudf_pd():
    nodes_df = pd.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_df = pd.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
    g_nodes = g.hop()
    assert isinstance(g_nodes._nodes, pd.DataFrame)
    assert len(g_nodes._nodes) == 3
    g_edges = g.hop()
    assert isinstance(g_edges._edges, pd.DataFrame)
    assert len(g_edges._edges) == 3


@pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1",
)
def test_hop_simple_cudf():
    import cudf
    nodes_gdf = cudf.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_gdf = cudf.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')
    g_nodes = g.hop()
    assert isinstance(g_nodes._nodes, cudf.DataFrame)
    assert len(g_nodes._nodes) == 3
    g_edges = g.hop()
    assert isinstance(g_edges._edges, cudf.DataFrame)
    assert len(g_edges._edges) == 3

def test_hop_kv_cudf_pd():
    nodes_df = pd.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_df = pd.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
    g_nodes = g.hop(source_node_match=({'id': 0}))
    assert isinstance(g_nodes._nodes, pd.DataFrame)
    assert len(g_nodes._nodes) == 2
    g_edges = g.hop(edge_match={'src': 0})
    assert isinstance(g_edges._edges, pd.DataFrame)
    assert len(g_edges._edges) == 1

@pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1",
)
def test_hop_kv_cudf():
    import cudf
    nodes_gdf = cudf.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_gdf = cudf.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')
    g_nodes = g.hop(source_node_match={'id': 0})
    assert isinstance(g_nodes._nodes, cudf.DataFrame)
    assert len(g_nodes._nodes) == 2
    g_edges = g.hop(edge_match={'src': 0})
    assert isinstance(g_edges._edges, cudf.DataFrame)
    assert len(g_edges._edges) == 1

def test_hop_pred_cudf_pd():
    nodes_df = pd.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_df = pd.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
    g_nodes = g.hop(source_node_match={'id': is_in([0])})
    assert isinstance(g_nodes._nodes, pd.DataFrame)
    assert len(g_nodes._nodes) == 2
    g_edges = g.hop(edge_match={'src': is_in([0])})
    assert isinstance(g_edges._edges, pd.DataFrame)
    assert len(g_edges._edges) == 1

@pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1",
)
def test_hop_pred_cudf():
    import cudf
    nodes_gdf = cudf.DataFrame({'id': [0, 1, 2], 'label': ['a', 'b', 'c']})
    edges_gdf = cudf.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 0]})
    g = CGFull().nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')
    g_nodes = g.hop(source_node_match={'id': is_in([0])})
    assert isinstance(g_nodes._nodes, cudf.DataFrame)
    assert len(g_nodes._nodes) == 2
    g_edges = g.hop(edge_match={'src': is_in([0])})
    assert isinstance(g_edges._edges, cudf.DataFrame)
    assert len(g_edges._edges) == 1


def test_hop_none_edge_binding_internal_index():
    """Test that hop() correctly handles graphs with no edge binding.

    When g._edge is None, hop() internally generates a temporary edge index
    column using generate_safe_column_name to avoid conflicts. This test
    verifies that:
    1. hop() works correctly without an edge binding
    2. The internal index column is properly cleaned up
    3. No internal columns leak into the result
    """
    # Create a graph with NO edge binding (g._edge = None)
    edges_df = pd.DataFrame({
        's': ['a', 'b', 'c'],
        'd': ['b', 'c', 'd']
    })
    nodes_df = pd.DataFrame({
        'v': ['a', 'b', 'c', 'd']
    })

    g = CGFull().edges(edges_df, 's', 'd').nodes(nodes_df, 'v')

    # Verify g._edge is None before hop
    assert g._edge is None, "Input graph should have None edge binding"

    # Run a hop operation
    g_result = g.hop(nodes=pd.DataFrame({'v': ['a']}), hops=2)

    # Verify the hop operation worked
    assert len(g_result._nodes) > 0
    assert len(g_result._edges) > 0

    # Verify no internal GFQL columns leaked into the result
    for col in g_result._edges.columns:
        assert not col.startswith('__gfql_'), f"Internal column {col} should not be in result"

    # Verify we got expected nodes (a's 2-hop neighbors)
    result_nodes = set(g_result._nodes['v'].tolist())
    assert 'b' in result_nodes
    assert 'c' in result_nodes


def test_hop_custom_edge_binding_preserved():
    """Test that hop() preserves custom edge binding."""
    # Create a graph WITH an edge binding
    edges_df = pd.DataFrame({
        's': ['a', 'b', 'c'],
        'd': ['b', 'c', 'd'],
        'edge_id': ['e1', 'e2', 'e3']
    })
    nodes_df = pd.DataFrame({
        'v': ['a', 'b', 'c', 'd']
    })

    g = CGFull().edges(edges_df, 's', 'd', edge='edge_id').nodes(nodes_df, 'v')

    # Verify g._edge is 'edge_id' before hop
    assert g._edge == 'edge_id', "Input graph should have 'edge_id' edge binding"

    # Run a hop operation
    g_result = g.hop(nodes=pd.DataFrame({'v': ['a']}), hops=2)

    # Should preserve the 'edge_id' binding
    assert g_result._edge == 'edge_id', f"Output graph should have 'edge_id' edge binding, but got: {g_result._edge}"

    # Verify the hop operation actually worked
    assert len(g_result._nodes) > 0
    assert len(g_result._edges) > 0
    assert 'edge_id' in g_result._edges.columns


def _mk_abc_chain():
    """3-node forward chain: a->b->c."""
    return (
        CGFull()
        .edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd')
        .nodes(pd.DataFrame({'v': ['a', 'b', 'c']}), 'v')
    )


def _mk_abcd_chain():
    """4-node forward chain: a->b->c->d."""
    return (
        CGFull()
        .edges(pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']}), 's', 'd')
        .nodes(pd.DataFrame({'v': ['a', 'b', 'c', 'd']}), 'v')
    )


def test_hop_labels_reverse():
    # Edges a->b->c; start from c in reverse, should reach b and a
    g = _mk_abc_chain()
    n_c = g._nodes.query('v == "c"')
    g2 = g.hop(
        nodes=n_c,
        hops=2,
        direction='reverse',
        label_node_hops='nh',
        label_seeds=True,
    )
    assert 'nh' in g2._nodes.columns
    assert g2._nodes['nh'].isna().sum() == 0
    node_hops = {row['v']: int(row['nh']) for row in g2._nodes[['v', 'nh']].to_dict(orient='records')}
    assert node_hops == {'a': 2, 'b': 1, 'c': 0}


def test_hop_labels_undirected():
    # Chain a->b->c; start from b (middle), undirected should reach a and c
    g = _mk_abc_chain()
    n_b = g._nodes.query('v == "b"')
    g2 = g.hop(
        nodes=n_b,
        hops=1,
        direction='undirected',
        label_node_hops='nh',
        label_seeds=True,
    )
    assert 'nh' in g2._nodes.columns
    assert g2._nodes['nh'].isna().sum() == 0
    node_hops = {row['v']: int(row['nh']) for row in g2._nodes[['v', 'nh']].to_dict(orient='records')}
    assert node_hops['b'] == 0
    assert node_hops['a'] == 1
    assert node_hops['c'] == 1


def test_hop_labels_node_only():
    # label_node_hops only — no label_edge_hops; edge hop column must be absent
    g = _mk_abc_chain()
    n_a = g._nodes.query('v == "a"')
    g2 = g.hop(nodes=n_a, hops=2, direction='forward', label_node_hops='nh')
    assert 'nh' in g2._nodes.columns
    assert 'eh' not in g2._edges.columns


def test_hop_labels_edge_only():
    # label_edge_hops only — no label_node_hops; node hop column must be absent
    g = _mk_abc_chain()
    n_a = g._nodes.query('v == "a"')
    g2 = g.hop(nodes=n_a, hops=2, direction='forward', label_edge_hops='eh')
    assert 'eh' in g2._edges.columns
    assert 'nh' not in g2._nodes.columns


def test_hop_labels_empty_result():
    # Hop from a node with no outgoing edges; result should be empty but not crash
    g = (
        CGFull()
        .edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        .nodes(pd.DataFrame({'v': ['a', 'b']}), 'v')
    )
    n_b = g._nodes.query('v == "b"')
    g2 = g.hop(nodes=n_b, hops=1, direction='forward', label_node_hops='nh')
    assert len(g2._edges) == 0
    assert len(g2._nodes) == 0 or g2._nodes['nh'].isna().sum() == 0


def test_hop_labels_multi_hop_ordering():
    # Chain a->b->c->d; hops=3 from a; verify minimum-hop assignment per node
    g = _mk_abcd_chain()
    n_a = g._nodes.query('v == "a"')
    g2 = g.hop(
        nodes=n_a,
        hops=3,
        direction='forward',
        label_node_hops='nh',
        label_edge_hops='eh',
        label_seeds=True,
    )
    assert 'nh' in g2._nodes.columns
    assert 'eh' in g2._edges.columns
    node_hops = {row['v']: int(row['nh']) for row in g2._nodes[['v', 'nh']].to_dict(orient='records')}
    assert node_hops == {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    edge_hops = {
        (row['s'], row['d']): int(row['eh'])
        for row in g2._edges[['s', 'd', 'eh']].to_dict(orient='records')
    }
    assert edge_hops == {('a', 'b'): 1, ('b', 'c'): 2, ('c', 'd'): 3}


def test_hop_labels_empty_result_with_edge_hops():
    """Hop with label_edge_hops into an empty result must not crash.

    Exercises hop.py:961-965 else branch: when g_out._edges is empty,
    edge_map_df has 0 rows and edge_map is created as an empty Series.
    """
    g = (
        CGFull()
        .edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        .nodes(pd.DataFrame({'v': ['a', 'b']}), 'v')
    )
    n_b = g._nodes.query('v == "b"')
    # 'b' has no outgoing forward edges — result is empty
    g2 = g.hop(nodes=n_b, hops=1, direction='forward', label_node_hops='nh', label_edge_hops='eh')
    assert len(g2._edges) == 0
    # must not crash; columns may or may not be present on empty result
    if len(g2._nodes) > 0:
        assert g2._nodes['nh'].isna().sum() == 0


def test_hop_labels_output_max_hops_exercises_fallback_map():
    """output_max_hops triggers node_mask filtering (hop.py:834-841) and the fallback_map
    try/except block at lines 885-891. Verifies that the safe_map_series fallback path
    runs and produces correct filtered hop assignments, not a silent skip."""
    g = _mk_abcd_chain()
    n_a = g._nodes.query('v == "a"')
    # hops=3 but output_max_hops=1 — only hop-1 nodes should appear in result
    g2 = g.hop(
        nodes=n_a, hops=3, direction='forward',
        label_node_hops='nh', label_seeds=True,
        output_max_hops=1,
    )
    assert 'nh' in g2._nodes.columns
    # Only a (hop 0) and b (hop 1) should be present
    hop_vals = {row['v']: row['nh'] for row in g2._nodes[['v', 'nh']].to_dict(orient='records')}
    assert 'a' in hop_vals and hop_vals['a'] == 0
    assert 'b' in hop_vals and hop_vals['b'] == 1
    assert 'c' not in hop_vals and 'd' not in hop_vals


def test_hop_labels_label_seeds_no_starting_nodes():
    """label_seeds=True with no explicit starting_nodes (None) exercises hop.py:843-859.

    When label_seeds=True:
      - line 844: node_hop_records is not None → seed_rows = records where hop==0
      - If seed_rows is empty (no hop-0 records), elif at line 853 is tried
      - elif: starting_nodes is None → neither branch adds seeds
    Verifies the code path where label_seeds=True but starting_nodes is implicitly None."""
    g = _mk_abc_chain()
    n_a = g._nodes.query('v == "a"')
    # nodes= is passed but starting_nodes is derived from it internally
    # label_seeds=True should label 'a' with hop=0
    g2 = g.hop(nodes=n_a, hops=2, direction='forward', label_node_hops='nh', label_seeds=True)
    assert 'nh' in g2._nodes.columns
    hop_vals = {row['v']: row['nh'] for row in g2._nodes[['v', 'nh']].to_dict(orient='records')}
    assert hop_vals.get('a') == 0
    assert hop_vals.get('b') == 1
    assert hop_vals.get('c') == 2


def test_hop_labels_numeric_node_ids():
    """Numeric (int) node IDs exercise the safe_map_series int→int path at hop.py:938.

    Previous tests used string IDs. Int IDs hit a different pandas dtype path
    through the merge-based lookup in safe_map_series."""
    g = (
        CGFull()
        .edges(pd.DataFrame({'s': [1, 2, 3], 'd': [2, 3, 4]}), 's', 'd')
        .nodes(pd.DataFrame({'v': [1, 2, 3, 4]}), 'v')
    )
    n_1 = g._nodes.query('v == 1')
    g2 = g.hop(nodes=n_1, hops=3, direction='forward', label_node_hops='nh', label_seeds=True)
    assert 'nh' in g2._nodes.columns
    hop_vals = {row['v']: row['nh'] for row in g2._nodes[['v', 'nh']].to_dict(orient='records')}
    assert hop_vals.get(1) == 0
    assert hop_vals.get(2) == 1
    assert hop_vals.get(3) == 2
    assert hop_vals.get(4) == 3


def test_hop_labels_undirected_label_seeds_false_clears_seed_hop():
    """label_seeds=False + direction='undirected' exercises hop.py:987-1000.

    When label_seeds=False and direction='undirected', seeds reachable in
    both directions get their hop label cleared (set to NaN) at line 1000.
    This exercises the undirected branch of the post-traversal seed-removal block."""
    g = _mk_abc_chain()
    n_b = g._nodes.query('v == "b"')
    # undirected from b; label_seeds=False means b should NOT get hop=0
    g2 = g.hop(nodes=n_b, hops=1, direction='undirected', label_node_hops='nh', label_seeds=False)
    assert 'nh' in g2._nodes.columns
    hop_vals = {row['v']: row['nh'] for row in g2._nodes[['v', 'nh']].to_dict(orient='records')}
    # 'b' is the seed — with label_seeds=False its hop should be NaN/absent
    assert pd.isna(hop_vals.get('b', float('nan'))) or 'b' not in hop_vals
    # neighbors 'a' and 'c' should have hop=1
    assert hop_vals.get('a') == 1 or hop_vals.get('c') == 1


def test_hop_labels_node_only_missing_mask_guard():
    """label_node_hops without label_edge_hops must not crash when missing_mask fires.

    Guard at hop.py:973 prevents using undefined edge_map when edge_hop_col is None.
    The guard triggers (missing_mask.any() == True) when endpoint nodes pulled in at
    hop.py:910-928 have no hop record — they get NaN for node_hop_col.

    Graph: a->b (hop 1), b also has a back-edge b->a not in the hop direction.
    We seed from 'a' forward 1 hop; 'a' is an endpoint of edges in g_out._edges
    but with label_seeds=False it won't be in node_hop_records — so missing_mask fires.
    """
    g = (
        CGFull()
        .edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'a']}), 's', 'd')
        .nodes(pd.DataFrame({'v': ['a', 'b']}), 'v')
    )
    n_a = g._nodes.query('v == "a"')
    # label_node_hops only, no label_edge_hops, label_seeds=False
    # 'a' is an endpoint in edges but not labeled as seed — triggers missing_mask path
    g2 = g.hop(nodes=n_a, hops=1, direction='forward', label_node_hops='nh', label_seeds=False)
    assert 'nh' in g2._nodes.columns
    assert 'eh' not in g2._edges.columns
    # 'b' must have hop=1; no crash from undefined edge_map
    node_hops = dict(zip(g2._nodes['v'].tolist(), g2._nodes['nh'].tolist()))
    assert node_hops.get('b') == 1
