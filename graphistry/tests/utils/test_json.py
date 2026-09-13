import json

import pytest

from graphistry.utils.json import assert_json_serializable, is_json_serializable

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
        class Unserializable:
            pass

        values = [set(), {'a': set()}, {'a': [set()]}, {'a': [{'b': set()}]}, Unserializable()]
        for value in values:
            with pytest.raises(AssertionError):
                assert_json_serializable(value)


@pytest.mark.parametrize('value', [
    None, False, True, '', 'unicode\u2603', '\ud800', 0, -1,
    -(2**64), -(2**64) + 1, 2**63, 2**64 - 1, 2**64, 2**65,
    0.0, float('nan'), float('inf'), float('-inf'),
    [], [1, None], {'nested': [True, 'text']}, (1, 2), set(), {'bad': set()}, object(),
])
def test_serializability_matches_standard_encoder(value):
    try:
        json.dumps(value)
    except TypeError:
        assert is_json_serializable(value) is False
    else:
        assert is_json_serializable(value) is True


def test_large_integer_keeps_encoder_limit_behavior():
    value = 10**5000
    try:
        json.dumps(value)
    except ValueError:
        with pytest.raises(ValueError):
            is_json_serializable(value)
    else:
        assert is_json_serializable(value) is True


def test_circular_container_keeps_encoder_error():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match='Circular reference'):
        is_json_serializable(value)


def test_integer_subclass_does_not_call_its_bit_length():
    class CustomInteger(int):
        def bit_length(self):
            raise AssertionError('Encoder does not call this override')

    assert is_json_serializable(CustomInteger(1)) is True


def test_metaclass_equality_cannot_admit_an_unsupported_object():
    class EqualToAnyType(type):
        def __eq__(cls, other):
            return True

    class Unsupported(metaclass=EqualToAnyType):
        pass

    assert is_json_serializable(Unsupported()) is False
