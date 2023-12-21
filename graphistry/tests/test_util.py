from graphistry.util import assert_json_serializable

class TestAssertJsonSerializable():

    def test_primitives(self):
        assert_json_serializable(1)
        assert_json_serializable(1.0)
        assert_json_serializable('a')
        assert_json_serializable(True)
        assert_json_serializable(None)
    
    def test_list(self):
        assert_json_serializable([])
        assert_json_serializable([1])
        assert_json_serializable([1, 2])
        assert_json_serializable([1, 'a', True, None])
    
    def test_dict(self):
        assert_json_serializable({})
        assert_json_serializable({'a': 1})
        assert_json_serializable({'a': 1, 'b': 2})
        assert_json_serializable({'a': 1, 'b': 'b', 'c': True, 'd': None})
    
    def test_nested(self):
        assert_json_serializable({'a': [1]})
        assert_json_serializable({'a': {'b': 1}})
        assert_json_serializable({'a': [{'b': 1}]})
        assert_json_serializable({'a': [{'b': 1}, {'c': 2}]})
    
    def test_unserializable(self):

        try:
            assert_json_serializable(set())
            assert False, 'Expected exception on set'
        except AssertionError:
            pass

        try:
            assert_json_serializable({'a': set()})
            assert False, 'Expected exception on nested set'
        except AssertionError:
            pass

        try:
            assert_json_serializable({'a': [set()]})
            assert False, 'Expected exception on nested set'
        except AssertionError:
            pass

        try:
            assert_json_serializable({'a': [{'b': set()}]})
            assert False, 'Expected exception on nested set'
        except AssertionError:
            pass

        class Unserializable:
            pass

        try:
            assert_json_serializable(Unserializable())
            assert False, 'Expected exception on class'
        except AssertionError:
            pass
