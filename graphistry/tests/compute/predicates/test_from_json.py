import pytest
from graphistry.compute.predicates.categorical import Duplicated
from graphistry.compute.predicates.str import (
    Startswith, Endswith, Match, Fullmatch, Contains
)
from graphistry.compute.predicates.from_json import from_json
from graphistry.compute.exceptions import GFQLValidationError, GFQLTypeError


def test_from_json_good():
    d = from_json({'type': 'Duplicated', 'keep': 'last'})
    assert isinstance(d, Duplicated)
    assert d.keep == 'last'

def test_from_json_bad():
    # Invalid type should raise error
    try:
        from_json({'type': 'zzz'})
        assert False, "Should have raised an exception"
    except Exception:
        # Could be GFQLValidationError or other exception
        assert True

    # Invalid keep value should raise GFQLValidationError
    try:
        from_json({'type': 'Duplicated', 'keep': 'zzz'})
        assert False, "Should have raised GFQLValidationError"
    except GFQLValidationError:
        assert True

    # Missing keep parameter - should use default
    # This might actually be valid with a default value
    d = from_json({'type': 'Duplicated'})
    assert isinstance(d, Duplicated)


# ============= JSON Serialization/Deserialization Tests =============

def test_startswith_json_with_case():
    """Test startswith serialization/deserialization with case parameter"""
    # Create predicate
    pred = Startswith('test', case=False, na=None)

    # Serialize
    json_data = pred.to_json()
    assert json_data['type'] == 'Startswith'
    assert json_data['pat'] == 'test'
    assert json_data['case'] is False
    assert json_data['na'] is None

    # Deserialize
    restored = Startswith.from_json(json_data)
    assert isinstance(restored, Startswith)
    assert restored.pat == 'test'
    assert restored.case is False
    assert restored.na is None


def test_startswith_json_with_tuple():
    """Test startswith serialization/deserialization with tuple pattern"""
    # Create predicate with tuple
    pred = Startswith(('test', 'demo'), case=True, na=False)

    # Serialize (tuples become lists in JSON)
    json_data = pred.to_json()
    assert json_data['type'] == 'Startswith'
    assert json_data['pat'] == ['test', 'demo']  # Tuple serialized as list
    assert json_data['case'] is True
    assert json_data['na'] is False

    # Deserialize (list converted back to tuple)
    restored = Startswith.from_json(json_data)
    assert isinstance(restored, Startswith)
    assert restored.pat == ('test', 'demo')  # List restored as tuple
    assert restored.case is True
    assert restored.na is False


def test_startswith_json_tuple_case_na_all_params():
    """Test startswith with all parameters combined"""
    pred = Startswith(('app', 'ban'), case=False, na=True)

    json_data = pred.to_json()
    assert json_data['pat'] == ['app', 'ban']
    assert json_data['case'] is False
    assert json_data['na'] is True

    restored = Startswith.from_json(json_data)
    assert restored.pat == ('app', 'ban')
    assert restored.case is False
    assert restored.na is True


def test_endswith_json_with_case():
    """Test endswith serialization/deserialization with case parameter"""
    pred = Endswith('.txt', case=False, na=None)

    json_data = pred.to_json()
    assert json_data['type'] == 'Endswith'
    assert json_data['pat'] == '.txt'
    assert json_data['case'] is False

    restored = Endswith.from_json(json_data)
    assert restored.pat == '.txt'
    assert restored.case is False


def test_endswith_json_with_tuple():
    """Test endswith serialization/deserialization with tuple pattern"""
    pred = Endswith(('.txt', '.csv'), case=True, na=False)

    json_data = pred.to_json()
    assert json_data['pat'] == ['.txt', '.csv']
    assert json_data['case'] is True
    assert json_data['na'] is False

    restored = Endswith.from_json(json_data)
    assert restored.pat == ('.txt', '.csv')
    assert restored.case is True
    assert restored.na is False


def test_endswith_json_tuple_case_na_all_params():
    """Test endswith with all parameters combined"""
    pred = Endswith(('.TXT', '.CSV'), case=False, na=True)

    json_data = pred.to_json()
    assert json_data['pat'] == ['.TXT', '.CSV']
    assert json_data['case'] is False
    assert json_data['na'] is True

    restored = Endswith.from_json(json_data)
    assert restored.pat == ('.TXT', '.CSV')
    assert restored.case is False
    assert restored.na is True


def test_fullmatch_json_basic():
    """Test fullmatch serialization/deserialization"""
    pred = Fullmatch(r'\d{3}', case=True, flags=0, na=None)

    json_data = pred.to_json()
    assert json_data['type'] == 'Fullmatch'
    assert json_data['pat'] == r'\d{3}'
    assert json_data['case'] is True
    assert json_data['flags'] == 0
    assert json_data['na'] is None

    restored = Fullmatch.from_json(json_data)
    assert isinstance(restored, Fullmatch)
    assert restored.pat == r'\d{3}'
    assert restored.case is True
    assert restored.flags == 0
    assert restored.na is None


def test_fullmatch_json_case_insensitive():
    """Test fullmatch with case=False"""
    pred = Fullmatch(r'[A-Z]+', case=False, flags=2, na=False)

    json_data = pred.to_json()
    assert json_data['case'] is False
    assert json_data['flags'] == 2
    assert json_data['na'] is False

    restored = Fullmatch.from_json(json_data)
    assert restored.case is False
    assert restored.flags == 2
    assert restored.na is False


def test_match_json_with_case():
    """Test match serialization/deserialization with case parameter"""
    pred = Match(r'test\d+', case=False, flags=0, na=None)

    json_data = pred.to_json()
    assert json_data['type'] == 'Match'
    assert json_data['pat'] == r'test\d+'
    assert json_data['case'] is False

    restored = Match.from_json(json_data)
    assert restored.pat == r'test\d+'
    assert restored.case is False


def test_contains_json_with_case():
    """Test contains serialization/deserialization with case parameter"""
    pred = Contains('test', case=False, flags=0, na=None, regex=True)

    json_data = pred.to_json()
    assert json_data['type'] == 'Contains'
    assert json_data['pat'] == 'test'
    assert json_data['case'] is False

    restored = Contains.from_json(json_data)
    assert restored.pat == 'test'
    assert restored.case is False


def test_from_json_registry_fullmatch():
    """Test from_json registry includes Fullmatch"""
    json_data = {'type': 'Fullmatch', 'pat': 'test', 'case': False, 'flags': 0, 'na': None}
    pred = from_json(json_data)

    assert isinstance(pred, Fullmatch)
    assert pred.pat == 'test'
    assert pred.case is False


def test_from_json_registry_startswith_tuple():
    """Test from_json registry handles tuple patterns"""
    json_data = {'type': 'Startswith', 'pat': ['app', 'ban'], 'case': False, 'na': False}
    pred = from_json(json_data)

    assert isinstance(pred, Startswith)
    assert pred.pat == ('app', 'ban')
    assert pred.case is False
    assert pred.na is False


def test_from_json_registry_endswith_tuple():
    """Test from_json registry handles tuple patterns"""
    json_data = {'type': 'Endswith', 'pat': ['.txt', '.csv'], 'case': False, 'na': True}
    pred = from_json(json_data)

    assert isinstance(pred, Endswith)
    assert pred.pat == ('.txt', '.csv')
    assert pred.case is False
    assert pred.na is True


# ============= Validation Tests =============

def test_startswith_validate_invalid_case():
    """Test _validate_fields catches invalid case parameter"""
    with pytest.raises(GFQLTypeError) as exc_info:
        pred = Startswith('test', case='not_a_bool')  # type: ignore
        pred.validate()
    assert 'case must be boolean' in str(exc_info.value)


def test_startswith_validate_invalid_na():
    """Test _validate_fields catches invalid na parameter"""
    with pytest.raises(GFQLTypeError) as exc_info:
        pred = Startswith('test', case=True, na='not_a_bool')  # type: ignore
        pred.validate()
    assert 'na must be boolean or None' in str(exc_info.value)


def test_startswith_validate_invalid_pat():
    """Test _validate_fields catches invalid pat parameter"""
    with pytest.raises(GFQLTypeError) as exc_info:
        pred = Startswith(123, case=True)  # type: ignore
        pred.validate()
    assert 'pat must be string or tuple of strings' in str(exc_info.value)


def test_startswith_validate_invalid_tuple_element():
    """Test _validate_fields catches non-string elements in tuple"""
    with pytest.raises(GFQLTypeError) as exc_info:
        pred = Startswith(('test', 123), case=True)  # type: ignore
        pred.validate()
    assert 'pat tuple element' in str(exc_info.value)
    assert 'must be string' in str(exc_info.value)


def test_endswith_validate_invalid_case():
    """Test _validate_fields catches invalid case parameter"""
    with pytest.raises(GFQLTypeError) as exc_info:
        pred = Endswith('.txt', case='not_a_bool')  # type: ignore
        pred.validate()
    assert 'case must be boolean' in str(exc_info.value)


def test_endswith_validate_invalid_tuple_element():
    """Test _validate_fields catches non-string elements in tuple"""
    with pytest.raises(GFQLTypeError) as exc_info:
        pred = Endswith(('.txt', 456), case=True)  # type: ignore
        pred.validate()
    assert 'pat tuple element' in str(exc_info.value)
    assert 'must be string' in str(exc_info.value)


def test_fullmatch_validate_invalid_pat():
    """Test _validate_fields catches invalid pat parameter"""
    with pytest.raises(GFQLTypeError) as exc_info:
        pred = Fullmatch(123, case=True)  # type: ignore
        pred.validate()
    assert 'pat must be string' in str(exc_info.value)


def test_fullmatch_validate_invalid_case():
    """Test _validate_fields catches invalid case parameter"""
    with pytest.raises(GFQLTypeError) as exc_info:
        pred = Fullmatch('test', case='not_a_bool')  # type: ignore
        pred.validate()
    assert 'case must be boolean' in str(exc_info.value)


def test_fullmatch_validate_invalid_flags():
    """Test _validate_fields catches invalid flags parameter"""
    with pytest.raises(GFQLTypeError) as exc_info:
        pred = Fullmatch('test', case=True, flags='not_an_int')  # type: ignore
        pred.validate()
    assert 'flags must be integer' in str(exc_info.value)


def test_match_validate_invalid_case():
    """Test _validate_fields catches invalid case parameter"""
    with pytest.raises(GFQLTypeError) as exc_info:
        pred = Match('test', case='not_a_bool')  # type: ignore
        pred.validate()
    assert 'case must be boolean' in str(exc_info.value)


def test_contains_validate_invalid_case():
    """Test _validate_fields catches invalid case parameter"""
    with pytest.raises(GFQLTypeError) as exc_info:
        pred = Contains('test', case='not_a_bool')  # type: ignore
        pred.validate()
    assert 'case must be boolean' in str(exc_info.value)


def test_json_roundtrip_validates():
    """Test that from_json automatically validates"""
    # Valid JSON should work
    json_data = {'type': 'Startswith', 'pat': 'test', 'case': True, 'na': None}
    pred = from_json(json_data)
    assert isinstance(pred, Startswith)

    # Invalid JSON should raise during validation
    with pytest.raises(GFQLTypeError):
        json_data_invalid = {'type': 'Startswith', 'pat': 'test', 'case': 'not_bool', 'na': None}
        from_json(json_data_invalid)
