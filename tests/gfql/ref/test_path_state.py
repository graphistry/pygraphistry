"""Tests for PathState immutability and helper methods."""

import pandas as pd
import pytest
from types import MappingProxyType

from graphistry.compute.gfql.same_path_types import PathState


def idx(values):
    return pd.Index(values)


def empty_state() -> PathState:
    return PathState.from_mutable({}, {})


def _clone_state(state: PathState):
    return dict(state.allowed_nodes), dict(state.allowed_edges), dict(state.pruned_edges)


def _with_state(state: PathState, *, nodes=None, edges=None, pruned=None) -> PathState:
    nodes_out, edges_out, pruned_out = _clone_state(state)
    if nodes:
        nodes_out.update(nodes)
    if edges:
        edges_out.update(edges)
    if pruned:
        pruned_out.update(pruned)
    return PathState.from_mutable(nodes_out, edges_out, pruned_out)


def restrict_nodes(state: PathState, idx_key: int, keep):
    cur = state.allowed_nodes.get(idx_key)
    new = cur.intersection(keep) if cur is not None else keep
    return _with_state(state, nodes={idx_key: new})


def set_nodes(state: PathState, idx_key: int, nodes):
    return _with_state(state, nodes={idx_key: nodes})


def restrict_edges(state: PathState, idx_key: int, keep):
    cur = state.allowed_edges.get(idx_key)
    new = cur.intersection(keep) if cur is not None else keep
    return _with_state(state, edges={idx_key: new})


def set_edges(state: PathState, idx_key: int, edges):
    return _with_state(state, edges={idx_key: edges})


def with_pruned_edges(state: PathState, edge_idx: int, df):
    return _with_state(state, pruned={edge_idx: df})


def sync_to_mutable(state: PathState, mutable_nodes: dict, mutable_edges: dict) -> None:
    mutable_nodes.clear()
    mutable_nodes.update(state.allowed_nodes)
    mutable_edges.clear()
    mutable_edges.update(state.allowed_edges)


def sync_pruned_to_forward_steps(state: PathState, forward_steps) -> None:
    for edge_idx, df in state.pruned_edges.items():
        forward_steps[edge_idx]._edges = df


class _MockStep:
    def __init__(self):
        self._edges = None


class TestPathStateImmutability:
    def test_empty_creates_empty_state(self):
        state = empty_state()
        assert len(state.allowed_nodes) == 0
        assert len(state.allowed_edges) == 0
        assert len(state.pruned_edges) == 0

    def test_from_mutable_preserves_domains(self):
        mutable_nodes = {0: idx([1, 2, 3]), 1: idx([4, 5])}
        mutable_edges = {1: idx([10, 20])}

        state = PathState.from_mutable(mutable_nodes, mutable_edges)

        # Check types are frozen
        assert isinstance(state.allowed_nodes, MappingProxyType)
        assert isinstance(state.allowed_edges, MappingProxyType)
        for v in state.allowed_nodes.values():
            assert isinstance(v, pd.Index)
        for v in state.allowed_edges.values():
            assert isinstance(v, pd.Index)

        # Check values are correct
        assert state.allowed_nodes[0].equals(idx([1, 2, 3]))
        assert state.allowed_nodes[1].equals(idx([4, 5]))
        assert state.allowed_edges[1].equals(idx([10, 20]))

    def test_to_mutable_converts_back(self):
        state = PathState.from_mutable(
            {0: idx([1, 2]), 1: idx([3, 4])},
            {1: idx([10])},
        )

        nodes, edges = state.to_mutable()

        # Check types are mutable
        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)
        for v in nodes.values():
            assert isinstance(v, pd.Index)
        for v in edges.values():
            assert isinstance(v, pd.Index)

        # Check values
        assert nodes[0].equals(idx([1, 2]))
        assert nodes[1].equals(idx([3, 4]))
        assert edges[1].equals(idx([10]))

    def test_mapping_proxy_prevents_mutation(self):
        state = PathState.from_mutable({0: idx([1, 2])}, {})

        with pytest.raises(TypeError):
            state.allowed_nodes[0] = idx([99])  # type: ignore

        with pytest.raises(TypeError):
            state.allowed_nodes[99] = idx([1])  # type: ignore

    def test_frozen_dataclass_prevents_attribute_mutation(self):
        state = PathState.from_mutable({0: idx([1])}, {})

        with pytest.raises(AttributeError):
            state.allowed_nodes = MappingProxyType({})  # type: ignore


class TestPathStateRestrictNodes:

    def test_restrict_nodes_returns_new_object(self):
        s1 = PathState.from_mutable({0: idx([1, 2, 3])}, {})
        s2 = restrict_nodes(s1, 0, idx([2, 3, 4]))

        assert s1 is not s2
        assert set(s1.allowed_nodes[0]) == {1, 2, 3}  # Original unchanged
        assert set(s2.allowed_nodes[0]) == {2, 3}  # Intersection

    def test_restrict_nodes_preserves_other_indices(self):
        s1 = PathState.from_mutable({0: idx([1, 2]), 1: idx([3, 4])}, {2: idx([10])})
        s2 = restrict_nodes(s1, 0, idx([2]))

        assert set(s2.allowed_nodes[1]) == {3, 4}  # Unchanged
        assert set(s2.allowed_edges[2]) == {10}  # Unchanged

    def test_restrict_nodes_with_empty_current_uses_keep(self):
        s1 = empty_state()
        s2 = restrict_nodes(s1, 0, idx([1, 2]))

        assert set(s2.allowed_nodes[0]) == {1, 2}

    def test_restrict_nodes_returns_same_if_unchanged(self):
        s1 = PathState.from_mutable({0: idx([1, 2])}, {})
        s2 = restrict_nodes(s1, 0, idx([1, 2, 3, 4]))  # Superset

        # Since intersection equals original, could return same object
        # (implementation detail - either is fine)
        assert set(s2.allowed_nodes[0]) == {1, 2}


class TestPathStateRestrictEdges:

    def test_restrict_edges_returns_new_object(self):
        s1 = PathState.from_mutable({}, {1: idx([10, 20, 30])})
        s2 = restrict_edges(s1, 1, idx([20, 30, 40]))

        assert s1 is not s2
        assert set(s1.allowed_edges[1]) == {10, 20, 30}
        assert set(s2.allowed_edges[1]) == {20, 30}


class TestPathStateSetNodes:

    def test_set_nodes_replaces_value(self):
        s1 = PathState.from_mutable({0: idx([1, 2])}, {})
        s2 = set_nodes(s1, 0, idx([99, 100]))

        assert set(s1.allowed_nodes[0]) == {1, 2}
        assert set(s2.allowed_nodes[0]) == {99, 100}

    def test_set_nodes_adds_new_index(self):
        s1 = empty_state()
        s2 = set_nodes(s1, 5, idx([1, 2, 3]))

        assert 5 not in s1.allowed_nodes
        assert set(s2.allowed_nodes[5]) == {1, 2, 3}


class TestPathStateWithPrunedEdges:

    def test_with_pruned_edges_stores_df(self):
        df = pd.DataFrame({'a': [1, 2, 3]})

        s1 = empty_state()
        s2 = with_pruned_edges(s1, 1, df)

        assert 1 not in s1.pruned_edges
        assert 1 in s2.pruned_edges
        assert s2.pruned_edges[1] is df

    def test_with_pruned_edges_preserves_existing(self):
        df1 = pd.DataFrame({'a': [1]})
        df2 = pd.DataFrame({'b': [2]})

        s1 = with_pruned_edges(empty_state(), 1, df1)
        s2 = with_pruned_edges(s1, 3, df2)

        assert s2.pruned_edges[1] is df1
        assert s2.pruned_edges[3] is df2


class TestPathStateSyncMethods:

    def test_sync_to_mutable_updates_dicts(self):
        state = PathState.from_mutable(
            {0: idx([1, 2]), 1: idx([3])},
            {1: idx([10, 20])},
        )

        target_nodes: dict = {0: idx([99])}  # Will be replaced
        target_edges: dict = {}

        sync_to_mutable(state, target_nodes, target_edges)

        assert set(target_nodes[0]) == {1, 2}
        assert set(target_nodes[1]) == {3}
        assert set(target_edges[1]) == {10, 20}

    def test_sync_pruned_to_forward_steps(self):
        forward_steps = [_MockStep(), _MockStep(), _MockStep()]

        df1 = pd.DataFrame({'x': [1]})
        df2 = pd.DataFrame({'y': [2]})

        state = with_pruned_edges(with_pruned_edges(empty_state(), 0, df1), 2, df2)
        sync_pruned_to_forward_steps(state, forward_steps)

        assert forward_steps[0]._edges is df1
        assert forward_steps[1]._edges is None  # Unchanged
        assert forward_steps[2]._edges is df2


class TestPathStateRoundTrip:

    def test_mutable_to_immutable_to_mutable(self):
        original_nodes = {0: idx([1, 2, 3]), 2: idx([4, 5])}
        original_edges = {1: idx([10, 20]), 3: idx([30])}

        state = PathState.from_mutable(original_nodes, original_edges)
        nodes_back, edges_back = state.to_mutable()

        assert set(nodes_back[0]) == {1, 2, 3}
        assert set(nodes_back[2]) == {4, 5}
        assert set(edges_back[1]) == {10, 20}
        assert set(edges_back[3]) == {30}


class TestPathStateImmutabilityContracts:

    @pytest.mark.parametrize(
        "update_fn",
        [
            lambda state: restrict_nodes(state, 0, idx([2, 3])),
            lambda state: restrict_edges(state, 1, idx([10])),
            lambda state: set_nodes(state, 0, idx([99])),
            lambda state: set_edges(state, 1, idx([99])),
            lambda state: with_pruned_edges(state, 0, pd.DataFrame({'a': [1]})),
        ],
        ids=["restrict_nodes", "restrict_edges", "set_nodes", "set_edges", "with_pruned_edges"],
    )
    def test_pathstate_methods_return_new_objects(self, update_fn):
        s1 = PathState.from_mutable({0: idx([1, 2, 3])}, {1: idx([10, 20])})
        s2 = update_fn(s1)

        assert s1 is not s2
        assert set(s1.allowed_nodes[0]) == {1, 2, 3}
        assert set(s1.allowed_edges[1]) == {10, 20}
        assert 0 not in s1.pruned_edges

    def test_pathstate_cannot_be_modified_after_creation(self):
        state = PathState.from_mutable({0: idx([1, 2])}, {1: idx([10])})

        for field in ("allowed_nodes", "allowed_edges", "pruned_edges"):
            with pytest.raises(AttributeError):
                setattr(state, field, MappingProxyType({}))  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            state.allowed_nodes[0] = idx([99])  # type: ignore
        with pytest.raises(TypeError):
            state.allowed_nodes[99] = idx([1])  # type: ignore

    def test_from_mutable_creates_deep_copy(self):
        nodes = {0: idx([1, 2, 3])}
        edges = {1: idx([10, 20])}

        state = PathState.from_mutable(nodes, edges)

        # Modify original mutable data
        nodes[0] = idx([99])
        edges[1] = idx([99])

        # PathState should be unaffected (deep copy)
        assert set(state.allowed_nodes[0]) == {1, 2, 3}
        assert set(state.allowed_edges[1]) == {10, 20}

    def test_to_mutable_creates_independent_copy(self):
        state = PathState.from_mutable({0: idx([1, 2, 3])}, {1: idx([10, 20])})

        nodes, edges = state.to_mutable()

        # Modify the mutable copies
        nodes[0] = idx([99])
        edges[1] = idx([99])

        # Original PathState should be unaffected
        assert set(state.allowed_nodes[0]) == {1, 2, 3}
        assert set(state.allowed_edges[1]) == {10, 20}
