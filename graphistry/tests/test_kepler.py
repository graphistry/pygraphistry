"""
Comprehensive tests for Kepler.gl encoding classes.
"""

import unittest
import pandas as pd
import graphistry
from graphistry.kepler import KeplerDataset, KeplerLayer, KeplerEncoding


class TestKeplerDataset(unittest.TestCase):
    """Tests for KeplerDataset class"""

    def test_init_nodes_dataset(self):
        """Test nodes dataset initialization"""
        dataset = KeplerDataset(id="test-dataset", type="nodes")
        self.assertEqual(dataset.id, "test-dataset")
        self.assertEqual(dataset.type, "nodes")

    def test_init_edges_dataset(self):
        """Test edges dataset initialization"""
        dataset = KeplerDataset(id="edges-ds", type="edges", map_node_coords=True)
        self.assertEqual(dataset.type, "edges")
        self.assertEqual(dataset._kwargs['map_node_coords'], True)

    def test_init_countries_dataset(self):
        """Test countries dataset initialization"""
        dataset = KeplerDataset(
            id="countries-ds",
            type="countries",
            resolution=50,
            include_countries=["USA", "Canada"]
        )
        self.assertEqual(dataset.type, "countries")
        self.assertEqual(dataset._kwargs['resolution'], 50)

    def test_init_without_id(self):
        """Test dataset initialization without ID auto-generates one"""
        dataset = KeplerDataset(type="nodes")
        self.assertIsNotNone(dataset.id)
        self.assertTrue(dataset.id.startswith("dataset-"))
        self.assertEqual(len(dataset.id), 16)  # "dataset-" + 8 hex chars

    def test_to_dict_nodes(self):
        """Test nodes dataset serialization"""
        dataset = KeplerDataset(id="test-dataset", type="nodes", label="Test Label")
        result = dataset.to_dict()
        self.assertEqual(result['info']['id'], "test-dataset")
        self.assertEqual(result['info']['label'], "Test Label")
        self.assertEqual(result['type'], "nodes")

    def test_to_dict_with_include(self):
        """Test dataset with include columns"""
        dataset = KeplerDataset(
            id="test-dataset",
            type="nodes",
            include=["col1", "col2"]
        )
        result = dataset.to_dict()
        self.assertEqual(result['include'], ["col1", "col2"])

    def test_equality(self):
        """Test dataset equality"""
        dataset1 = KeplerDataset(id="test-dataset", type="nodes")
        dataset2 = KeplerDataset(id="test-dataset", type="nodes")
        dataset3 = KeplerDataset(id="other-dataset", type="nodes")
        self.assertEqual(dataset1, dataset2)
        self.assertNotEqual(dataset1, dataset3)

    def test_raw_dict_passthrough(self):
        """Test raw_dict passes through without modification"""
        raw = {
            "info": {"id": "custom-dataset"},
            "type": "nodes",
            "customField": "customValue",
            "nested": {"data": [1, 2, 3]}
        }
        dataset = KeplerDataset(raw)
        result = dataset.to_dict()
        self.assertEqual(result, raw)
        self.assertIs(result, raw)  # Should be the same object

    def test_raw_dict_ignores_other_params(self):
        """Test that raw_dict ignores other parameters"""
        raw = {"info": {"id": "raw-id"}, "type": "edges"}
        # These params should be ignored
        dataset = KeplerDataset(raw, id="ignored", type="nodes", label="ignored")
        result = dataset.to_dict()
        self.assertEqual(result, raw)
        self.assertIsNone(dataset.id)
        self.assertIsNone(dataset.type)
        self.assertIsNone(dataset.label)

    def test_raw_dict_type_validation(self):
        """Test that raw_dict must be a dict"""
        with self.assertRaises(TypeError):
            KeplerDataset("not a dict")
        with self.assertRaises(TypeError):
            KeplerDataset(123)
        with self.assertRaises(TypeError):
            KeplerDataset(["a", "list"])

    def test_raw_dict_equality(self):
        """Test equality with raw_dict"""
        raw1 = {"info": {"id": "test"}, "type": "nodes"}
        raw2 = {"info": {"id": "test"}, "type": "nodes"}
        raw3 = {"info": {"id": "other"}, "type": "nodes"}

        dataset1 = KeplerDataset(raw1)
        dataset2 = KeplerDataset(raw2)
        dataset3 = KeplerDataset(raw3)
        normal_dataset = KeplerDataset(id="test", type="nodes")

        self.assertEqual(dataset1, dataset2)  # Same content
        self.assertNotEqual(dataset1, dataset3)  # Different content
        self.assertNotEqual(dataset1, normal_dataset)  # raw_dict vs normal


class TestKeplerLayer(unittest.TestCase):
    """Tests for KeplerLayer class (raw_dict only mode)"""

    def test_raw_dict_passthrough(self):
        """Test raw_dict passes through without modification"""
        raw = {
            "id": "custom-layer",
            "type": "hexagon",
            "config": {
                "dataId": "my-dataset",
                "columns": {"lat": "latitude", "lng": "longitude"}
            },
            "customField": "customValue"
        }
        layer = KeplerLayer(raw)
        result = layer.to_dict()
        self.assertEqual(result, raw)
        self.assertIs(result, raw)  # Should be the same object

    def test_extracts_id_and_type(self):
        """Test that id and type are extracted from raw_dict for repr"""
        raw = {"id": "my-layer", "type": "point", "config": {}}
        layer = KeplerLayer(raw)
        # Ensure repr works and shows id/type
        repr_str = repr(layer)
        self.assertIn("my-layer", repr_str)
        self.assertIn("point", repr_str)

    def test_raw_dict_type_validation(self):
        """Test that raw_dict must be a dict"""
        with self.assertRaises(TypeError):
            KeplerLayer("not a dict")
        with self.assertRaises(TypeError):
            KeplerLayer(123)
        with self.assertRaises(TypeError):
            KeplerLayer(["a", "list"])

    def test_raw_dict_equality(self):
        """Test equality with raw_dict"""
        raw1 = {"id": "layer1", "type": "point", "config": {}}
        raw2 = {"id": "layer1", "type": "point", "config": {}}
        raw3 = {"id": "layer2", "type": "point", "config": {}}

        layer1 = KeplerLayer(raw1)
        layer2 = KeplerLayer(raw2)
        layer3 = KeplerLayer(raw3)

        self.assertEqual(layer1, layer2)  # Same content
        self.assertNotEqual(layer1, layer3)  # Different content


class TestKeplerEncoding(unittest.TestCase):
    """Tests for KeplerEncoding container class"""

    def test_init_empty(self):
        """Test empty initialization"""
        encoding = KeplerEncoding()
        self.assertEqual(len(encoding.datasets), 0)
        self.assertEqual(len(encoding.layers), 0)
        self.assertEqual(encoding.options, {})
        self.assertEqual(encoding.config, {})

    def test_with_dataset_immutability(self):
        """Test that with_dataset returns new instance"""
        encoding1 = KeplerEncoding()
        dataset = KeplerDataset(id="test-dataset", type="nodes")
        encoding2 = encoding1.with_dataset(dataset)

        # Original should be unchanged
        self.assertEqual(len(encoding1.datasets), 0)
        # New instance should have dataset
        self.assertEqual(len(encoding2.datasets), 1)
        self.assertEqual(encoding2.datasets[0].id, "test-dataset")

    def test_with_layer_immutability(self):
        """Test that with_layer returns new instance"""
        encoding1 = KeplerEncoding()
        layer = KeplerLayer({"id": "test-layer", "type": "point", "config": {"dataId": "ds1"}})
        encoding2 = encoding1.with_layer(layer)

        # Original should be unchanged
        self.assertEqual(len(encoding1.layers), 0)
        # New instance should have layer
        self.assertEqual(len(encoding2.layers), 1)

    def test_with_options(self):
        """Test with_options method"""
        encoding1 = KeplerEncoding()
        encoding2 = encoding1.with_options(centerMap=False, readOnly=True)

        self.assertEqual(encoding1.options, {})
        self.assertEqual(encoding2.options, {'centerMap': False, 'readOnly': True})

    def test_with_config(self):
        """Test with_config method"""
        encoding1 = KeplerEncoding()
        encoding2 = encoding1.with_config(cullUnusedColumns=True, overlayBlending='additive')

        self.assertEqual(encoding1.config, {})
        self.assertEqual(encoding2.config, {'cullUnusedColumns': True, 'overlayBlending': 'additive'})

    def test_auto_generated_ids(self):
        """Test that IDs are auto-generated when not provided"""
        encoding = KeplerEncoding()
        dataset = KeplerDataset(type="nodes")  # No ID
        encoding = encoding.with_dataset(dataset)

        # Should have auto-generated ID
        self.assertIsNotNone(encoding.datasets[0].id)
        self.assertTrue(encoding.datasets[0].id.startswith("dataset-"))

    def test_to_dict_full(self):
        """Test serialization with all fields"""
        encoding = (KeplerEncoding()
                   .with_dataset(KeplerDataset(id="d1", type="nodes"))
                   .with_layer(KeplerLayer({"id": "l1", "type": "point", "config": {"dataId": "d1"}}))
                   .with_options(centerMap=True)
                   .with_config(cullUnusedColumns=False))

        result = encoding.to_dict()
        self.assertIn('datasets', result)
        self.assertIn('layers', result)
        self.assertIn('options', result)
        self.assertIn('config', result)
        self.assertEqual(len(result['datasets']), 1)
        self.assertEqual(len(result['layers']), 1)

    def test_to_dict_empty_includes_structure(self):
        """Test that empty encoding includes full structure"""
        encoding = KeplerEncoding()
        result = encoding.to_dict()
        # Empty encoding should return structure with empty arrays/dicts
        self.assertEqual(result, {
            'datasets': [],
            'layers': [],
            'options': {},
            'config': {}
        })

    def test_str(self):
        """Test str representation"""
        encoding = (KeplerEncoding()
                   .with_dataset(KeplerDataset(id="d1", type="nodes"))
                   .with_layer(KeplerLayer({"id": "l1", "type": "point", "config": {"dataId": "d1"}})))

        result = str(encoding)
        self.assertIn("1 datasets", result)
        self.assertIn("1 layers", result)

    def test_equality(self):
        """Test equality comparison"""
        encoding1 = KeplerEncoding().with_dataset(KeplerDataset(id="d1", type="nodes"))
        encoding2 = KeplerEncoding().with_dataset(KeplerDataset(id="d1", type="nodes"))
        encoding3 = KeplerEncoding().with_dataset(KeplerDataset(id="d2", type="nodes"))

        self.assertEqual(encoding1, encoding2)
        self.assertNotEqual(encoding1, encoding3)


class TestPlotterKeplerIntegration(unittest.TestCase):
    """Tests for Plotter integration with Kepler encoding"""

    def setUp(self):
        """Set up test graph"""
        edges = pd.DataFrame({
            'src': ['a', 'b', 'c'],
            'dst': ['b', 'c', 'd']
        })
        self.g = graphistry.edges(edges, 'src', 'dst')

    def test_encode_kepler_dataset_simple(self):
        """Test encode_kepler_dataset with just ID"""
        g2 = self.g.encode_kepler_dataset(id="test-dataset")

        # Check immutability - original should have no kepler encoding
        self.assertNotIn('pointKeplerEncoding', self.g._complex_encodings['node_encodings']['default'])

        # New instance should have dataset
        kepler = g2._complex_encodings['node_encodings']['default']['pointKeplerEncoding']
        self.assertEqual(len(kepler['datasets']), 1)
        self.assertEqual(kepler['datasets'][0]['info']['id'], 'test-dataset')

    def test_encode_kepler_dataset_typed(self):
        """Test encode_kepler_dataset with type"""
        g2 = self.g.encode_kepler_dataset(id="nodes-ds", type="nodes", label="My Nodes")

        kepler = g2._complex_encodings['node_encodings']['default']['pointKeplerEncoding']
        dataset = kepler['datasets'][0]
        self.assertEqual(dataset['info']['id'], "nodes-ds")
        self.assertEqual(dataset['type'], "nodes")
        self.assertEqual(dataset['info']['label'], "My Nodes")

    def test_encode_kepler_dataset_countries(self):
        """Test encode_kepler_dataset with countries type"""
        g2 = self.g.encode_kepler_dataset(
            id="countries",
            type="countries",
            resolution=50,
            include_countries=["USA", "Canada"]
        )

        kepler = g2._complex_encodings['node_encodings']['default']['pointKeplerEncoding']
        dataset = kepler['datasets'][0]
        self.assertEqual(dataset['type'], "countries")
        self.assertEqual(dataset['resolution'], 50)
        self.assertEqual(dataset['include_countries'], ["USA", "Canada"])

    def test_encode_kepler_layer_simple(self):
        """Test encode_kepler_layer with just ID"""
        g2 = self.g.encode_kepler_layer({"id": "test-layer", "type": "point"})

        # Check immutability
        self.assertNotIn('pointKeplerEncoding', self.g._complex_encodings['node_encodings']['default'])

        # New instance should have layer
        kepler = g2._complex_encodings['node_encodings']['default']['pointKeplerEncoding']
        self.assertEqual(len(kepler['layers']), 1)

    def test_encode_kepler_layer_full(self):
        """Test encode_kepler_layer with all parameters"""
        g2 = self.g.encode_kepler_layer({
            "id": "point-layer",
            "type": "point",
            "config": {
                "dataId": "nodes-ds",
                "label": "Points",
                "columns": {'lat': 'latitude', 'lng': 'longitude'}
            }
        })

        kepler = g2._complex_encodings['node_encodings']['default']['pointKeplerEncoding']
        layer = kepler['layers'][0]
        self.assertEqual(layer['id'], "point-layer")
        self.assertEqual(layer['type'], "point")
        self.assertEqual(layer['config']['dataId'], "nodes-ds")
        self.assertEqual(layer['config']['label'], "Points")
        self.assertEqual(layer['config']['columns']['lat'], 'latitude')

    def test_chaining(self):
        """Test chaining multiple kepler encodings"""
        g2 = (self.g
              .encode_kepler_dataset(id="dataset1", type="nodes")
              .encode_kepler_dataset(id="dataset2", type="edges")
              .encode_kepler_layer({"id": "layer1", "type": "point", "config": {"dataId": "dataset1"}}))

        kepler = g2._complex_encodings['node_encodings']['default']['pointKeplerEncoding']
        self.assertEqual(len(kepler['datasets']), 2)
        self.assertEqual(len(kepler['layers']), 1)
        self.assertEqual(kepler['datasets'][0]['type'], "nodes")
        self.assertEqual(kepler['datasets'][1]['type'], "edges")

    def test_auto_id_generation(self):
        """Test auto ID generation in plotter"""
        g2 = self.g.encode_kepler_dataset(type="nodes")  # No ID provided

        kepler = g2._complex_encodings['node_encodings']['default']['pointKeplerEncoding']
        self.assertEqual(len(kepler['datasets']), 1)
        self.assertIsNotNone(kepler['datasets'][0]['info']['id'])
        self.assertTrue(kepler['datasets'][0]['info']['id'].startswith("dataset-"))

    def test_dataset_id_invalidation(self):
        """Test that dataset_id is invalidated on kepler encoding"""
        # Mock a dataset_id
        g = self.g
        g._dataset_id = "old-dataset-id"

        g2 = g.encode_kepler_dataset(id="test-dataset", type="nodes")

        # dataset_id should be None on new instance
        self.assertIsNone(g2._dataset_id)

    def test_serialization_to_dict(self):
        """Test full serialization pipeline"""
        g2 = (self.g
              .encode_kepler_dataset(id="nodes", type="nodes", label="My Nodes")
              .encode_kepler_layer({
                  "id": "points",
                  "type": "point",
                  "config": {
                      "dataId": "nodes",
                      "columns": {'lat': 'lat', 'lng': 'lng'}
                  }
              }))

        kepler_dict = g2._complex_encodings['node_encodings']['default']['pointKeplerEncoding']

        # Check structure
        self.assertIn('datasets', kepler_dict)
        self.assertIn('layers', kepler_dict)

        # Check dataset
        self.assertEqual(len(kepler_dict['datasets']), 1)
        self.assertEqual(kepler_dict['datasets'][0]['info']['id'], 'nodes')
        self.assertEqual(kepler_dict['datasets'][0]['type'], 'nodes')

        # Check layer
        self.assertEqual(len(kepler_dict['layers']), 1)
        self.assertEqual(kepler_dict['layers'][0]['id'], 'points')
        self.assertEqual(kepler_dict['layers'][0]['type'], 'point')
        self.assertEqual(kepler_dict['layers'][0]['config']['dataId'], 'nodes')

    def test_encode_kepler_with_container(self):
        """Test encode_kepler with KeplerEncoding container"""
        from graphistry.kepler import KeplerEncoding, KeplerDataset, KeplerLayer

        # Build encoding using container
        kepler = (KeplerEncoding()
                 .with_dataset(KeplerDataset(id="test-data", type="nodes"))
                 .with_layer(KeplerLayer({"id": "test-layer", "type": "point", "config": {"dataId": "test-data"}}))
                 .with_options(centerMap=True)
                 .with_config(mapStyle='dark'))

        # Apply to plotter
        g2 = self.g.encode_kepler(kepler)

        # Check immutability
        self.assertNotIn('pointKeplerEncoding', self.g._complex_encodings['node_encodings']['default'])

        # Check result
        result = g2._complex_encodings['node_encodings']['default']['pointKeplerEncoding']
        self.assertEqual(len(result['datasets']), 1)
        self.assertEqual(result['datasets'][0]['info']['id'], 'test-data')
        self.assertEqual(len(result['layers']), 1)
        self.assertEqual(result['layers'][0]['id'], 'test-layer')
        self.assertEqual(result['options']['centerMap'], True)
        self.assertEqual(result['config']['mapStyle'], 'dark')

    def test_encode_kepler_with_dict(self):
        """Test encode_kepler with plain dict"""
        kepler_dict = {
            'datasets': [{'info': {'id': 'dict-data'}, 'type': 'nodes'}],
            'layers': [{'id': 'dict-layer', 'type': 'point', 'config': {'dataId': 'dict-data'}}],
            'options': {},
            'config': {}
        }

        g2 = self.g.encode_kepler(kepler_dict)

        # Check result
        result = g2._complex_encodings['node_encodings']['default']['pointKeplerEncoding']
        self.assertEqual(result['datasets'], kepler_dict['datasets'])
        self.assertEqual(result['layers'], kepler_dict['layers'])
        self.assertEqual(result['options'], kepler_dict['options'])
        self.assertEqual(result['config'], kepler_dict['config'])
        self.assertEqual(result['encodingType'], 'kepler')
        self.assertEqual(result['graphType'], 'point')

        # Verify user's dict wasn't mutated
        self.assertNotIn('encodingType', kepler_dict)
        self.assertNotIn('graphType', kepler_dict)

    def test_encode_kepler_replaces_existing(self):
        """Test that encode_kepler replaces existing encoding completely"""
        # First add some encodings
        g2 = self.g.encode_kepler_dataset(id="old-dataset")
        g2 = g2.encode_kepler_layer({"id": "old-layer", "type": "arc", "config": {"dataId": "old-dataset"}})

        # Now replace with new encoding
        from graphistry.kepler import KeplerEncoding, KeplerDataset
        kepler = KeplerEncoding().with_dataset(KeplerDataset(id="new-dataset"))
        g3 = g2.encode_kepler(kepler)

        # Check that old encodings are replaced
        result = g3._complex_encodings['node_encodings']['default']['pointKeplerEncoding']
        self.assertEqual(len(result['datasets']), 1)
        self.assertEqual(result['datasets'][0]['info']['id'], 'new-dataset')
        self.assertEqual(len(result['layers']), 0)  # No layers in new encoding


if __name__ == '__main__':
    unittest.main()
