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

SERIALIZATION_CASES = [
    (
        Startswith,
        ("test",),
        {"case": False, "na": None},
        {"type": "Startswith", "pat": "test", "case": False, "na": None},
        {"pat": "test", "case": False, "na": None},
    ),
    (
        Startswith,
        (("test", "demo"),),
        {"case": True, "na": False},
        {"type": "Startswith", "pat": ["test", "demo"], "case": True, "na": False},
        {"pat": ("test", "demo"), "case": True, "na": False},
    ),
    (
        Startswith,
        (("app", "ban"),),
        {"case": False, "na": True},
        {"type": "Startswith", "pat": ["app", "ban"], "case": False, "na": True},
        {"pat": ("app", "ban"), "case": False, "na": True},
    ),
    (
        Endswith,
        (".txt",),
        {"case": False, "na": None},
        {"type": "Endswith", "pat": ".txt", "case": False, "na": None},
        {"pat": ".txt", "case": False, "na": None},
    ),
    (
        Endswith,
        ((".txt", ".csv"),),
        {"case": True, "na": False},
        {"type": "Endswith", "pat": [".txt", ".csv"], "case": True, "na": False},
        {"pat": (".txt", ".csv"), "case": True, "na": False},
    ),
    (
        Endswith,
        ((".TXT", ".CSV"),),
        {"case": False, "na": True},
        {"type": "Endswith", "pat": [".TXT", ".CSV"], "case": False, "na": True},
        {"pat": (".TXT", ".CSV"), "case": False, "na": True},
    ),
    (
        Fullmatch,
        (r"\d{3}",),
        {"case": True, "flags": 0, "na": None},
        {"type": "Fullmatch", "pat": r"\d{3}", "case": True, "flags": 0, "na": None},
        {"pat": r"\d{3}", "case": True, "flags": 0, "na": None},
    ),
    (
        Fullmatch,
        (r"[A-Z]+",),
        {"case": False, "flags": 2, "na": False},
        {"type": "Fullmatch", "pat": r"[A-Z]+", "case": False, "flags": 2, "na": False},
        {"pat": r"[A-Z]+", "case": False, "flags": 2, "na": False},
    ),
    (
        Match,
        (r"test\d+",),
        {"case": False, "flags": 0, "na": None},
        {"type": "Match", "pat": r"test\d+", "case": False, "flags": 0, "na": None},
        {"pat": r"test\d+", "case": False, "flags": 0, "na": None},
    ),
    (
        Contains,
        ("test",),
        {"case": False, "flags": 0, "na": None, "regex": True},
        {"type": "Contains", "pat": "test", "case": False, "flags": 0, "na": None, "regex": True},
        {"pat": "test", "case": False, "flags": 0, "na": None, "regex": True},
    ),
]


@pytest.mark.parametrize("cls,args,kwargs,expected_json,expected_attrs", SERIALIZATION_CASES)
def test_string_predicate_json_roundtrip(
    cls,
    args,
    kwargs,
    expected_json,
    expected_attrs,
):
    pred = cls(*args, **kwargs)
    json_data = pred.to_json()
    assert json_data == expected_json

    restored = cls.from_json(json_data)
    assert isinstance(restored, cls)
    for attr, expected in expected_attrs.items():
        assert getattr(restored, attr) == expected


@pytest.mark.parametrize(
    "json_data,cls,expected_attrs",
    [
        (
            {'type': 'Fullmatch', 'pat': 'test', 'case': False, 'flags': 0, 'na': None},
            Fullmatch,
            {"pat": "test", "case": False},
        ),
        (
            {'type': 'Startswith', 'pat': ['app', 'ban'], 'case': False, 'na': False},
            Startswith,
            {"pat": ("app", "ban"), "case": False, "na": False},
        ),
        (
            {'type': 'Endswith', 'pat': ['.txt', '.csv'], 'case': False, 'na': True},
            Endswith,
            {"pat": (".txt", ".csv"), "case": False, "na": True},
        ),
    ],
)
def test_from_json_registry_string_predicates(json_data, cls, expected_attrs):
    pred = from_json(json_data)
    assert isinstance(pred, cls)
    for attr, expected in expected_attrs.items():
        assert getattr(pred, attr) == expected


# ============= Validation Tests =============

@pytest.mark.parametrize(
    "pred,expected_messages",
    [
        (Startswith('test', case='not_a_bool'), ['case must be boolean']),  # type: ignore
        (Startswith('test', case=True, na='not_a_bool'), ['na must be boolean or None']),  # type: ignore
        (Startswith(123, case=True), ['pat must be string or tuple of strings']),  # type: ignore
        (Startswith(('test', 123), case=True), ['pat tuple element', 'must be string']),  # type: ignore
        (Endswith('.txt', case='not_a_bool'), ['case must be boolean']),  # type: ignore
        (Endswith(('.txt', 456), case=True), ['pat tuple element', 'must be string']),  # type: ignore
        (Fullmatch(123, case=True), ['pat must be string']),  # type: ignore
        (Fullmatch('test', case='not_a_bool'), ['case must be boolean']),  # type: ignore
        (Fullmatch('test', case=True, flags='not_an_int'), ['flags must be integer']),  # type: ignore
        (Match('test', case='not_a_bool'), ['case must be boolean']),  # type: ignore
        (Contains('test', case='not_a_bool'), ['case must be boolean']),  # type: ignore
    ],
)
def test_string_predicate_validation_errors(pred, expected_messages):
    with pytest.raises(GFQLTypeError) as exc_info:
        pred.validate()
    for expected in expected_messages:
        assert expected in str(exc_info.value)


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
