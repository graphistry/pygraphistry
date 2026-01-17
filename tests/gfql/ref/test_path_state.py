"""Tests for PathState immutability and helper methods."""

import pandas as pd
import pytest
from types import MappingProxyType

from graphistry.compute.gfql.same_path_types import PathState, _mp


def idx(values):
    return pd.Index(values)


class TestPathStateImmutability:
    """Test that PathState is truly immutable."""

    def test_empty_creates_empty_state(self):
        state = PathState.empty()
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
            state.allowed_nodes = _mp({})  # type: ignore


class TestPathStateRestrictNodes:
    """Test restrict_nodes returns new state with intersection."""

    def test_restrict_nodes_returns_new_object(self):
        s1 = PathState.from_mutable({0: idx([1, 2, 3])}, {})
        s2 = s1.restrict_nodes(0, idx([2, 3, 4]))

        assert s1 is not s2
        assert set(s1.allowed_nodes[0]) == {1, 2, 3}  # Original unchanged
        assert set(s2.allowed_nodes[0]) == {2, 3}  # Intersection

    def test_restrict_nodes_preserves_other_indices(self):
        s1 = PathState.from_mutable({0: idx([1, 2]), 1: idx([3, 4])}, {2: idx([10])})
        s2 = s1.restrict_nodes(0, idx([2]))

        assert set(s2.allowed_nodes[1]) == {3, 4}  # Unchanged
        assert set(s2.allowed_edges[2]) == {10}  # Unchanged

    def test_restrict_nodes_with_empty_current_uses_keep(self):
        s1 = PathState.empty()
        s2 = s1.restrict_nodes(0, idx([1, 2]))

        assert set(s2.allowed_nodes[0]) == {1, 2}

    def test_restrict_nodes_returns_same_if_unchanged(self):
        s1 = PathState.from_mutable({0: idx([1, 2])}, {})
        s2 = s1.restrict_nodes(0, idx([1, 2, 3, 4]))  # Superset

        # Since intersection equals original, could return same object
        # (implementation detail - either is fine)
        assert set(s2.allowed_nodes[0]) == {1, 2}


class TestPathStateRestrictEdges:
    """Test restrict_edges returns new state with intersection."""

    def test_restrict_edges_returns_new_object(self):
        s1 = PathState.from_mutable({}, {1: idx([10, 20, 30])})
        s2 = s1.restrict_edges(1, idx([20, 30, 40]))

        assert s1 is not s2
        assert set(s1.allowed_edges[1]) == {10, 20, 30}
        assert set(s2.allowed_edges[1]) == {20, 30}


class TestPathStateSetNodes:
    """Test set_nodes replaces the node set entirely."""

    def test_set_nodes_replaces_value(self):
        s1 = PathState.from_mutable({0: idx([1, 2])}, {})
        s2 = s1.set_nodes(0, idx([99, 100]))

        assert set(s1.allowed_nodes[0]) == {1, 2}
        assert set(s2.allowed_nodes[0]) == {99, 100}

    def test_set_nodes_adds_new_index(self):
        s1 = PathState.empty()
        s2 = s1.set_nodes(5, idx([1, 2, 3]))

        assert 5 not in s1.allowed_nodes
        assert set(s2.allowed_nodes[5]) == {1, 2, 3}


class TestPathStateWithPrunedEdges:
    """Test with_pruned_edges stores DataFrame."""

    def test_with_pruned_edges_stores_df(self):
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3]})

        s1 = PathState.empty()
        s2 = s1.with_pruned_edges(1, df)

        assert 1 not in s1.pruned_edges
        assert 1 in s2.pruned_edges
        assert s2.pruned_edges[1] is df

    def test_with_pruned_edges_preserves_existing(self):
        import pandas as pd
        df1 = pd.DataFrame({'a': [1]})
        df2 = pd.DataFrame({'b': [2]})

        s1 = PathState.empty().with_pruned_edges(1, df1)
        s2 = s1.with_pruned_edges(3, df2)

        assert s2.pruned_edges[1] is df1
        assert s2.pruned_edges[3] is df2


class TestPathStateSyncMethods:
    """Test sync methods for backward compatibility."""

    def test_sync_to_mutable_updates_dicts(self):
        state = PathState.from_mutable(
            {0: idx([1, 2]), 1: idx([3])},
            {1: idx([10, 20])},
        )

        target_nodes: dict = {0: idx([99])}  # Will be replaced
        target_edges: dict = {}

        state.sync_to_mutable(target_nodes, target_edges)

        assert set(target_nodes[0]) == {1, 2}
        assert set(target_nodes[1]) == {3}
        assert set(target_edges[1]) == {10, 20}

    def test_sync_pruned_to_forward_steps(self):
        import pandas as pd

        # Create mock forward_steps with _edges attribute
        class MockStep:
            def __init__(self):
                self._edges = None

        forward_steps = [MockStep(), MockStep(), MockStep()]

        df1 = pd.DataFrame({'x': [1]})
        df2 = pd.DataFrame({'y': [2]})

        state = PathState.empty().with_pruned_edges(0, df1).with_pruned_edges(2, df2)
        state.sync_pruned_to_forward_steps(forward_steps)

        assert forward_steps[0]._edges is df1
        assert forward_steps[1]._edges is None  # Unchanged
        assert forward_steps[2]._edges is df2


class TestPathStateRoundTrip:
    """Test conversion round-trips preserve data."""

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
    """Contract tests to ensure immutability is enforced at API boundaries."""

    def test_pathstate_methods_return_new_objects(self):
        """All PathState methods must return new objects, not mutate in place."""
        import pandas as pd

        s1 = PathState.from_mutable({0: idx([1, 2, 3])}, {1: idx([10, 20])})

        # restrict_nodes returns new object
        s2 = s1.restrict_nodes(0, idx([2, 3]))
        assert s1 is not s2
        assert set(s1.allowed_nodes[0]) == {1, 2, 3}  # Original unchanged

        # restrict_edges returns new object
        s3 = s1.restrict_edges(1, idx([10]))
        assert s1 is not s3
        assert set(s1.allowed_edges[1]) == {10, 20}  # Original unchanged

        # set_nodes returns new object
        s4 = s1.set_nodes(0, idx([99]))
        assert s1 is not s4
        assert set(s1.allowed_nodes[0]) == {1, 2, 3}  # Original unchanged

        # set_edges returns new object
        s5 = s1.set_edges(1, idx([99]))
        assert s1 is not s5
        assert set(s1.allowed_edges[1]) == {10, 20}  # Original unchanged

        # with_pruned_edges returns new object
        df = pd.DataFrame({'a': [1]})
        s6 = s1.with_pruned_edges(0, df)
        assert s1 is not s6
        assert 0 not in s1.pruned_edges  # Original unchanged

    def test_pathstate_cannot_be_modified_after_creation(self):
        """PathState fields cannot be modified after creation."""
        state = PathState.from_mutable({0: idx([1, 2])}, {1: idx([10])})

        # Cannot reassign fields (frozen dataclass)
        with pytest.raises(AttributeError):
            state.allowed_nodes = _mp({})  # type: ignore

        with pytest.raises(AttributeError):
            state.allowed_edges = _mp({})  # type: ignore

        with pytest.raises(AttributeError):
            state.pruned_edges = _mp({})  # type: ignore

        # Cannot modify MappingProxyType contents
        with pytest.raises(TypeError):
            state.allowed_nodes[0] = idx([99])  # type: ignore

        with pytest.raises(TypeError):
            state.allowed_nodes[99] = idx([1])  # type: ignore

    def test_from_mutable_creates_deep_copy(self):
        """from_mutable must not hold references to input mutable data."""
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
        """to_mutable must return data that doesn't affect original PathState."""
        state = PathState.from_mutable({0: idx([1, 2, 3])}, {1: idx([10, 20])})

        nodes, edges = state.to_mutable()

        # Modify the mutable copies
        nodes[0] = idx([99])
        edges[1] = idx([99])

        # Original PathState should be unaffected
        assert set(state.allowed_nodes[0]) == {1, 2, 3}
        assert set(state.allowed_edges[1]) == {10, 20}
