"""Tests for graphistry.io.plottable_bundle — Plottable serialization/hydration."""
import copy
import inspect
import os
import tempfile
import typing
import unittest
import warnings

import pandas as pd

import graphistry
from graphistry.io.plottable_bundle import (
    ALL_KNOWN_FIELDS,
    NEVER_FIELDS,
    SCHEMA_VERSION,
    TIER1_BINDING_FIELDS,
    TIER1_DF_FIELDS,
    TIER1_DISPLAY_FIELDS,
    TIER1_REMOTE_FIELDS,
    TIER2_DF_FIELDS,
    TIER2_JSON_ALGO_FIELDS,
    TIER2_JSON_INDEX_FIELDS,
    TIER2_JSON_KG_FIELDS,
    TIER2_JSON_LAYOUT_FIELDS,
    TIER3_FIELDS,
    from_file,
    to_file,
)
from graphistry.Plottable import Plottable
from graphistry.PlotterBase import PlotterBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph():
    """Build a basic graph for testing."""
    edges = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'a'], 'w': [1, 2, 3]})
    nodes = pd.DataFrame({'id': ['a', 'b', 'c'], 'label': ['A', 'B', 'C']})
    g = graphistry.edges(edges, 's', 'd').nodes(nodes, 'id')
    g = g.bind(edge_weight='w').name('Test Graph').description('A test')
    return g


def _make_graph_with_tier2():
    """Build a graph with Tier 2 fields populated."""
    g = _make_graph()
    result = copy.copy(g)
    result._xy = pd.DataFrame({'x': [0.1, 0.2, 0.3], 'y': [0.4, 0.5, 0.6]})
    result._node_embedding = pd.DataFrame({'e0': [1.0, 2.0, 3.0], 'e1': [4.0, 5.0, 6.0]})
    result._n_components = 5
    result._metric = 'cosine'
    return result


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------

class TestRoundtripDir(unittest.TestCase):
    """Test save/load roundtrip with directory format."""

    def test_basic_roundtrip(self):
        g = _make_graph()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'bundle')
            g2, wr = to_file(g, path)
            self.assertIs(g2, g)
            self.assertIn('_edges', wr.artifacts_written)
            self.assertIn('_nodes', wr.artifacts_written)

            g3, rr = from_file(path)
            self.assertTrue(rr.integrity_ok)

            # Verify Tier 1 bindings
            self.assertEqual(g3._source, 's')
            self.assertEqual(g3._destination, 'd')
            self.assertEqual(g3._node, 'id')
            self.assertEqual(g3._edge_weight, 'w')
            self.assertEqual(g3._name, 'Test Graph')
            self.assertEqual(g3._description, 'A test')

            # Verify edges data
            pd.testing.assert_frame_equal(g3._edges, g._edges)
            pd.testing.assert_frame_equal(g3._nodes, g._nodes)

    def test_settings_roundtrip(self):
        g = _make_graph()
        g = g.settings(height=800, url_params={'play': '0', 'info': 'true'})
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'bundle')
            to_file(g, path)
            g3, _ = from_file(path)
            self.assertEqual(g3._height, 800)
            self.assertEqual(g3._url_params, {'play': '0', 'info': 'true'})


class TestRoundtripZip(unittest.TestCase):
    """Test save/load roundtrip with zip format."""

    def test_zip_roundtrip(self):
        g = _make_graph()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'bundle.zip')
            g2, wr = to_file(g, path, format='zip')
            self.assertIn('_edges', wr.artifacts_written)

            g3, rr = from_file(path)
            self.assertTrue(rr.integrity_ok)
            self.assertEqual(g3._source, 's')
            self.assertEqual(g3._name, 'Test Graph')
            pd.testing.assert_frame_equal(g3._edges, g._edges)


class TestTier2Roundtrip(unittest.TestCase):
    """Test that Tier 2 DataFrames and JSON fields round-trip."""

    def test_tier2_df_roundtrip(self):
        g = _make_graph_with_tier2()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'bundle')
            _, wr = to_file(g, path)
            self.assertIn('_xy', wr.artifacts_written)
            self.assertIn('_node_embedding', wr.artifacts_written)

            g3, rr = from_file(path)
            self.assertIn('_xy', rr.artifacts_loaded)
            self.assertIn('_node_embedding', rr.artifacts_loaded)
            pd.testing.assert_frame_equal(g3._xy, g._xy)
            pd.testing.assert_frame_equal(g3._node_embedding, g._node_embedding)

    def test_tier2_json_roundtrip(self):
        g = _make_graph_with_tier2()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'bundle')
            to_file(g, path)
            g3, _ = from_file(path)
            self.assertEqual(g3._n_components, 5)
            self.assertEqual(g3._metric, 'cosine')


# ---------------------------------------------------------------------------
# Remote state tests
# ---------------------------------------------------------------------------

class TestRemoteState(unittest.TestCase):

    def test_remote_dropped_by_default(self):
        g = _make_graph()
        result = copy.copy(g)
        result._dataset_id = 'test_ds'
        result._url = 'https://example.com/graph'
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'bundle')
            with warnings.catch_warnings(record=True):
                to_file(result, path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                g3, rr = from_file(path)
            self.assertTrue(rr.remote_state_skipped)
            self.assertIsNone(g3._dataset_id)
            self.assertIsNone(g3._url)
            # Check warning was issued
            remote_warnings = [w for w in caught if 'remote server state' in str(w.message).lower()]
            self.assertTrue(len(remote_warnings) > 0)

    def test_remote_restored_when_requested(self):
        g = _make_graph()
        result = copy.copy(g)
        result._dataset_id = 'test_ds'
        result._url = 'https://example.com/graph'
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'bundle')
            to_file(result, path)
            g3, rr = from_file(path, restore_remote=True)
            self.assertFalse(rr.remote_state_skipped)
            self.assertEqual(g3._dataset_id, 'test_ds')
            self.assertEqual(g3._url, 'https://example.com/graph')


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_edges_required(self):
        g = graphistry.bind()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'bundle')
            with self.assertRaises(RuntimeError) as ctx:
                to_file(g, path)
            self.assertIn("edges", str(ctx.exception).lower())

    def test_partial_failure_tier2(self):
        """Non-DF in Tier 2 slot → skip + warn, rest succeed."""
        g = _make_graph()
        result = copy.copy(g)
        result._node_embedding = "not a dataframe"
        result._xy = pd.DataFrame({'x': [1.0], 'y': [2.0]})
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'bundle')
            _, wr = to_file(result, path)
            self.assertIn('_node_embedding', wr.artifacts_skipped)
            self.assertIn('_xy', wr.artifacts_written)

    def test_method_api(self):
        """Test that g.to_file() works as a method."""
        g = _make_graph()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'bundle')
            g2, wr = g.to_file(path)
            self.assertIs(g2, g)
            g3, rr = graphistry.from_file(path)
            self.assertEqual(g3._source, 's')


# ---------------------------------------------------------------------------
# Tripwire tests
# ---------------------------------------------------------------------------

class TestTripwire(unittest.TestCase):
    """Detect new fields added to Plottable/PlotterBase that aren't
    accounted for in the serialization field groups."""

    def _get_protocol_fields(self):
        """Extract field names from Plottable Protocol annotations."""
        hints = typing.get_type_hints(Plottable)
        return set(hints.keys())

    def _get_init_fields(self):
        """Extract field names set in PlotterBase.__init__."""
        source = inspect.getsource(PlotterBase.__init__)
        fields = set()
        for line in source.split('\n'):
            stripped = line.strip()
            if stripped.startswith('self.') and '=' in stripped:
                field = stripped.split('=')[0].strip().replace('self.', '')
                # Strip type annotations
                if ':' in field:
                    field = field.split(':')[0].strip()
                fields.add(field)
        return fields

    def test_no_unknown_protocol_fields(self):
        """Fail if Plottable Protocol has fields not in ALL_KNOWN_FIELDS."""
        protocol_fields = self._get_protocol_fields()
        known = set(ALL_KNOWN_FIELDS)
        # Exclude methods (only looking at data fields)
        unknown = protocol_fields - known
        # Filter out methods by checking if they're callable on the protocol
        data_unknown = set()
        for f in unknown:
            hint = typing.get_type_hints(Plottable).get(f)
            # If it's a callable type, skip it
            origin = getattr(hint, '__origin__', None)
            if origin is not None and origin is type:
                continue
            data_unknown.add(f)
        self.assertEqual(
            data_unknown, set(),
            f"New fields in Plottable Protocol not in ALL_KNOWN_FIELDS: {data_unknown}. "
            "Add them to the appropriate tier in plottable_bundle.py."
        )

    def test_no_unknown_init_fields(self):
        """Fail if PlotterBase.__init__ sets fields not in ALL_KNOWN_FIELDS."""
        init_fields = self._get_init_fields()
        known = set(ALL_KNOWN_FIELDS)
        unknown = init_fields - known
        self.assertEqual(
            unknown, set(),
            f"New fields in PlotterBase.__init__ not in ALL_KNOWN_FIELDS: {unknown}. "
            "Add them to the appropriate tier in plottable_bundle.py."
        )

    def test_no_phantom_fields(self):
        """Fail if ALL_KNOWN_FIELDS has fields not in Protocol or PlotterBase.__init__."""
        protocol_fields = self._get_protocol_fields()
        init_fields = self._get_init_fields()
        all_real = protocol_fields | init_fields
        known = set(ALL_KNOWN_FIELDS)
        phantom = known - all_real
        self.assertEqual(
            phantom, set(),
            f"Phantom fields in ALL_KNOWN_FIELDS not in Protocol or __init__: {phantom}. "
            "Remove them from plottable_bundle.py."
        )


# ---------------------------------------------------------------------------
# Golden fixture test
# ---------------------------------------------------------------------------

class TestGoldenFixture(unittest.TestCase):
    """Load the committed v1_bundle fixture and verify known values."""

    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = os.path.join(
            os.path.dirname(__file__), 'fixtures', 'v1_bundle'
        )
        if not os.path.exists(os.path.join(cls.fixture_dir, 'manifest.json')):
            # Auto-generate if missing (first run)
            from graphistry.tests.fixtures.generate_v1_bundle import main
            main()

    def test_load_golden_fixture(self):
        g, rr = from_file(self.fixture_dir)
        self.assertTrue(rr.integrity_ok)

        # Check bindings
        self.assertEqual(g._source, 's')
        self.assertEqual(g._destination, 'd')
        self.assertEqual(g._node, 'id')
        self.assertEqual(g._edge_weight, 'w')

        # Check metadata
        self.assertEqual(g._name, 'Golden Test Graph')
        self.assertEqual(g._description, 'A test graph for v1 bundle compatibility')

        # Check settings
        self.assertEqual(g._height, 600)
        self.assertEqual(g._url_params, {'info': 'true', 'play': '2000'})

        # Check edges data
        self.assertEqual(len(g._edges), 3)
        self.assertListEqual(list(g._edges['s']), ['a', 'b', 'c'])

        # Check nodes data
        self.assertEqual(len(g._nodes), 3)
        self.assertListEqual(list(g._nodes['id']), ['a', 'b', 'c'])

        # Check xy (tier 2)
        self.assertIsNotNone(g._xy)
        self.assertEqual(len(g._xy), 3)

        # Check algorithm config
        self.assertEqual(g._n_components, 2)
        self.assertEqual(g._metric, 'euclidean')

    def test_golden_remote_skipped_by_default(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            g, rr = from_file(self.fixture_dir)
        self.assertTrue(rr.remote_state_skipped)
        self.assertIsNone(g._dataset_id)

    def test_golden_remote_restored(self):
        g, rr = from_file(self.fixture_dir, restore_remote=True)
        self.assertEqual(g._dataset_id, 'golden_dataset_123')

    def test_schema_version(self):
        from graphistry.io.bundle import read_manifest
        manifest = read_manifest(self.fixture_dir)
        self.assertEqual(manifest['schema_version'], '1.0')


if __name__ == '__main__':
    unittest.main()
