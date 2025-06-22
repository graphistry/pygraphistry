"""Test temporal predicates work with filter_by_dict"""

import pandas as pd
import pytest
from datetime import datetime, date, time

from graphistry.compute import gt, lt, between, eq
from graphistry.compute.filter_by_dict import filter_by_dict
from graphistry.tests.test_compute import CGFull


class TestFilterByDictTemporal:
    """Test temporal predicates in filter_by_dict operations"""
    
    @pytest.fixture
    def temporal_graph(self):
        """Create a graph with temporal data"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'created': pd.to_datetime([
                '2023-01-01 10:00:00',
                '2023-03-15 14:30:00',
                '2023-06-01 09:00:00',
                '2023-09-20 16:45:00',
                '2023-12-25 12:00:00'
            ]),
            'expires': pd.to_datetime([
                '2024-01-01',
                '2024-03-15',
                '2024-06-01',
                '2024-09-20',
                '2024-12-25'
            ]),
            'status': ['active', 'active', 'expired', 'active', 'pending']
        })
        
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c', 'd'],
            'd': ['b', 'c', 'd', 'e'],
            'timestamp': pd.to_datetime([
                '2023-01-15 11:00:00',
                '2023-04-01 15:00:00',
                '2023-07-10 10:30:00',
                '2023-10-05 17:00:00'
            ]),
            'weight': [1.0, 2.5, 3.0, 4.5]
        })
        
        return CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
    
    def test_temporal_gt_predicate(self, temporal_graph):
        """Test greater than with datetime"""
        g = temporal_graph
        threshold = pd.Timestamp('2023-06-01')
        
        # Filter nodes created after June 1st
        filtered = filter_by_dict(g._nodes, {'created': gt(threshold)})
        assert len(filtered) == 2  # nodes c, d
        assert set(filtered['id'].tolist()) == {'d', 'e'}
    
    def test_temporal_lt_predicate(self, temporal_graph):
        """Test less than with datetime"""
        g = temporal_graph
        threshold = pd.Timestamp('2023-06-01')
        
        # Filter nodes created before June 1st
        filtered = filter_by_dict(g._nodes, {'created': lt(threshold)})
        assert len(filtered) == 2  # nodes a, b
        assert set(filtered['id'].tolist()) == {'a', 'b'}
    
    def test_temporal_between_predicate(self, temporal_graph):
        """Test between with datetime range"""
        g = temporal_graph
        start = pd.Timestamp('2023-03-01')
        end = pd.Timestamp('2023-10-01')
        
        # Filter nodes created between March and October
        filtered = filter_by_dict(g._nodes, {'created': between(start, end)})
        assert len(filtered) == 3  # nodes b, c, d
        assert set(filtered['id'].tolist()) == {'b', 'c', 'd'}
    
    def test_temporal_eq_predicate(self, temporal_graph):
        """Test equality with date (comparing just the date part)"""
        g = temporal_graph
        target_date = date(2023, 1, 1)
        
        # Filter nodes created on Jan 1st (regardless of time)
        # Convert date to datetime for comparison
        filtered = filter_by_dict(g._nodes, {
            'created': between(
                pd.Timestamp(target_date),
                pd.Timestamp(target_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            )
        })
        assert len(filtered) == 1
        assert filtered['id'].tolist() == ['a']
    
    def test_mixed_temporal_and_regular_filters(self, temporal_graph):
        """Test combining temporal predicates with regular filters"""
        g = temporal_graph
        
        # Active nodes created after March 1st
        filtered = filter_by_dict(g._nodes, {
            'created': gt(pd.Timestamp('2023-03-01')),
            'status': 'active'
        })
        assert len(filtered) == 2  # nodes b, d
        assert set(filtered['id'].tolist()) == {'b', 'd'}
    
    def test_temporal_edge_filtering(self, temporal_graph):
        """Test temporal predicates on edges"""
        g = temporal_graph
        
        # Edges with timestamps in Q2/Q3 2023
        filtered = filter_by_dict(g._edges, {
            'timestamp': between(
                pd.Timestamp('2023-04-01'),
                pd.Timestamp('2023-09-30')
            )
        })
        assert len(filtered) == 2
        assert set(filtered['s'].tolist()) == {'b', 'c'}
    
    def test_node_filter_method(self, temporal_graph):
        """Test filter_nodes_by_dict with temporal predicates"""
        g = temporal_graph
        
        # Use the graph method
        g2 = g.filter_nodes_by_dict({
            'created': gt(pd.Timestamp('2023-06-01')),
            'expires': lt(pd.Timestamp('2024-12-01'))
        })
        
        assert len(g2._nodes) == 1  # Only node d matches both conditions
        assert g2._nodes['id'].tolist() == ['d']
    
    def test_edge_filter_method(self, temporal_graph):
        """Test filter_edges_by_dict with temporal predicates"""  
        g = temporal_graph
        
        # Use the graph method
        g2 = g.filter_edges_by_dict({
            'timestamp': between(
                pd.Timestamp('2023-01-01'),
                pd.Timestamp('2023-06-30')
            ),
            'weight': gt(1.5)
        })
        
        assert len(g2._edges) == 1
        assert g2._edges['s'].tolist() == ['b']
        assert g2._edges['weight'].tolist() == [2.5]
    
    def test_timezone_aware_filtering(self):
        """Test filtering with timezone-aware timestamps"""
        import pytz
        
        utc = pytz.UTC
        # eastern = pytz.timezone('US/Eastern')  # Not used
        
        # Create timezone-aware data
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'timestamp': [
                pd.Timestamp('2023-01-01 10:00:00', tz=utc),
                pd.Timestamp('2023-01-01 15:00:00', tz=utc),  # 10:00 EST
                pd.Timestamp('2023-01-01 18:00:00', tz=utc),  # 13:00 EST
            ]
        })
        
        g = CGFull().nodes(nodes_df, 'id')
        
        # Filter with UTC threshold
        threshold = pd.Timestamp('2023-01-01 14:00:00', tz=utc)
        g2 = g.filter_nodes_by_dict({'timestamp': gt(threshold)})
        
        assert len(g2._nodes) == 2
        assert set(g2._nodes['id'].tolist()) == {'b', 'c'}
