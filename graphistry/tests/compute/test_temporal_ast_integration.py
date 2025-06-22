"""Integration tests for temporal predicates in AST and wire protocol"""

import pandas as pd
import pytest
from datetime import datetime, date, time

from graphistry import n, e_forward
from graphistry.compute import gt, lt, between
from graphistry.compute.ast import ASTNode, ASTEdge
from graphistry.compute.chain import Chain
from graphistry.compute.ast_temporal import DateTimeValue, DateValue, TimeValue


class TestTemporalASTIntegration:
    """Test temporal predicates work correctly in AST nodes and edges"""
    
    def test_temporal_in_node_filter_dict(self):
        """Test temporal predicates can be used in node filter_dict"""
        dt = datetime(2023, 1, 1, 12, 0, 0)
        pred = gt(pd.Timestamp(dt))
        
        # Create AST node with temporal predicate
        node = ASTNode(filter_dict={'timestamp': pred})
        
        # Test JSON serialization
        json_data = node.to_json()
        
        # Verify structure
        assert 'filter_dict' in json_data
        assert 'timestamp' in json_data['filter_dict']
        assert json_data['filter_dict']['timestamp']['type'] == 'GT'
        assert json_data['filter_dict']['timestamp']['val']['type'] == 'datetime'
        assert json_data['filter_dict']['timestamp']['val']['value'] == '2023-01-01T12:00:00'
        
        # Test deserialization
        node2 = ASTNode.from_json(json_data)
        assert node2.filter_dict is not None
        assert 'timestamp' in node2.filter_dict
        
    def test_temporal_in_edge_matches(self):
        """Test temporal predicates in all edge match types"""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)
        date_range = between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        
        # Create AST edge with temporal predicates
        edge = ASTEdge(
            direction='forward',
            edge_match={'created_date': date_range},
            source_node_match={'last_login': gt(pd.Timestamp(datetime(2023, 6, 1)))},
            destination_node_match={'expires': lt(pd.Timestamp(datetime(2024, 1, 1)))}
        )
        
        # Test JSON serialization
        json_data = edge.to_json()
        
        # Verify all match types have temporal predicates
        assert json_data['edge_match']['created_date']['type'] == 'Between'
        assert json_data['edge_match']['created_date']['lower']['type'] == 'datetime'
        assert json_data['edge_match']['created_date']['upper']['type'] == 'datetime'
        
        assert json_data['source_node_match']['last_login']['type'] == 'GT'
        assert json_data['source_node_match']['last_login']['val']['type'] == 'datetime'
        
        assert json_data['destination_node_match']['expires']['type'] == 'LT'
        assert json_data['destination_node_match']['expires']['val']['type'] == 'datetime'
        
        # Test deserialization
        edge2 = ASTEdge.from_json(json_data)
        assert edge2.edge_match is not None
        assert edge2.source_node_match is not None
        assert edge2.destination_node_match is not None
        
    def test_temporal_in_chain(self):
        """Test temporal predicates in a complete chain"""
        chain = Chain([
            n({'created': gt(pd.Timestamp('2023-01-01'))}),
            e_forward(
                edge_match={'timestamp': between(
                    pd.Timestamp('2023-01-01'), 
                    pd.Timestamp('2023-12-31')
                )},
                destination_node_match={'active_until': gt(pd.Timestamp('2023-06-01'))}
            ),
            n({'status': 'active'})
        ])
        
        # Test JSON serialization
        json_data = chain.to_json()
        
        # Verify chain structure
        assert len(json_data['chain']) == 3
        
        # Check first node has temporal predicate
        node1 = json_data['chain'][0]
        assert node1['filter_dict']['created']['type'] == 'GT'
        assert node1['filter_dict']['created']['val']['type'] == 'datetime'
        
        # Check edge has temporal predicates
        edge = json_data['chain'][1]
        assert edge['edge_match']['timestamp']['type'] == 'Between'
        assert edge['destination_node_match']['active_until']['type'] == 'GT'
        
        # Test deserialization
        chain2 = Chain.from_json(json_data)
        assert len(chain2.chain) == 3
        
    def test_temporal_value_objects_in_ast(self):
        """Test using temporal value objects directly"""
        dt_val = DateTimeValue("2023-01-01T12:00:00", "US/Eastern")
        date_val = DateValue("2023-01-01")
        time_val = TimeValue("12:00:00")
        
        # Create predicates with temporal value objects
        node = ASTNode(filter_dict={
            'datetime_col': gt(dt_val),
            'date_col': lt(date_val),
            'time_col': between(time_val, TimeValue("18:00:00"))
        })
        
        # Test serialization
        json_data = node.to_json()
        
        # Verify timezone is preserved
        assert json_data['filter_dict']['datetime_col']['val']['timezone'] == 'US/Eastern'
        
        # Verify date and time types
        assert json_data['filter_dict']['date_col']['val']['type'] == 'date'
        assert json_data['filter_dict']['time_col']['lower']['type'] == 'time'
        assert json_data['filter_dict']['time_col']['upper']['type'] == 'time'
        
    def test_mixed_predicate_types(self):
        """Test mixing temporal, numeric, and other predicate types"""
        from graphistry.compute.predicates.is_in import is_in
        
        node = ASTNode(filter_dict={
            'score': gt(0.5),  # numeric predicate
            'created': gt(pd.Timestamp('2023-01-01')),  # temporal predicate
            'category': 'A',  # regular value
            'tags': is_in(['tag1', 'tag2'])  # categorical predicate
        })
        
        json_data = node.to_json()
        
        # Verify each type is handled correctly
        assert json_data['filter_dict']['score']['type'] == 'GT'
        assert isinstance(json_data['filter_dict']['score']['val'], (int, float))
        
        assert json_data['filter_dict']['created']['type'] == 'GT'
        assert json_data['filter_dict']['created']['val']['type'] == 'datetime'
        
        assert json_data['filter_dict']['category'] == 'A'
        
        assert json_data['filter_dict']['tags']['type'] == 'IsIn'
        assert json_data['filter_dict']['tags']['options'] == ['tag1', 'tag2']
        
    def test_timezone_aware_datetime_in_ast(self):
        """Test timezone-aware datetime handling"""
        import pytz
        
        # Create timezone-aware timestamp
        eastern = pytz.timezone('US/Eastern')
        dt = eastern.localize(datetime(2023, 1, 1, 12, 0, 0))
        
        node = ASTNode(filter_dict={
            'timestamp': gt(pd.Timestamp(dt))
        })
        
        json_data = node.to_json()
        
        # Timezone should be preserved
        assert json_data['filter_dict']['timestamp']['val']['timezone'] == 'US/Eastern'
        
        # Deserialize and verify
        node2 = ASTNode.from_json(json_data)
        pred = node2.filter_dict['timestamp']
        assert hasattr(pred, 'val')
        assert isinstance(pred.val, DateTimeValue)
        assert pred.val.timezone == 'US/Eastern'
