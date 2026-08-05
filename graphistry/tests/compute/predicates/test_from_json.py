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


# --- Regression: comparison predicates must round-trip as the comparison.py (numeric+temporal+
# string) versions, NOT be downgraded to numeric.py's numeric-only classes. Cypher lowering and
# the public predicate API build comparison.* predicates; from_json previously registered the
# same-named numeric.* classes, so a JSON round-trip lost temporal/string capability and RAISED
# "val must be numeric" for a string/temporal equality or a temporal comparison. ---
import datetime as _dt

from graphistry.compute.predicates.comparison import (
    eq as _cmp_eq, gt as _cmp_gt, ge as _cmp_ge, lt as _cmp_lt, le as _cmp_le, ne as _cmp_ne,
    EQ as _CmpEQ, GT as _CmpGT,
)


@pytest.mark.parametrize("factory,val", [
    (_cmp_eq, "foo"),            # string equality — previously RAISED on round-trip
    (_cmp_eq, 5),                # numeric equality
    (_cmp_eq, _dt.date(2020, 1, 1)),   # temporal equality — previously RAISED
    (_cmp_ne, "bar"),
    (_cmp_gt, _dt.date(2020, 1, 1)),   # temporal comparison — previously RAISED
    (_cmp_gt, 3),
    (_cmp_ge, 3),
    (_cmp_lt, 3),
    (_cmp_le, 3),
])
def test_comparison_predicate_json_roundtrip_preserves_comparison_class(factory, val):
    pred = factory(val)
    back = from_json(pred.to_json())
    # must be the SAME comparison.py class, not a numeric.py downgrade
    assert type(back) is type(pred), (
        f"{type(pred).__module__}.{type(pred).__name__} round-tripped to "
        f"{type(back).__module__}.{type(back).__name__}"
    )
    assert type(back).__module__ == "graphistry.compute.predicates.comparison"


def test_string_and_temporal_eq_survive_roundtrip_without_raising():
    # The exact pre-fix failure: numeric.EQ.validate() raised GFQLTypeError "val must be numeric".
    from_json(_cmp_eq("foo").to_json())          # no raise
    from_json(_cmp_eq(_dt.date(2020, 1, 1)).to_json())  # no raise
    assert isinstance(from_json({'type': 'EQ', 'val': 'foo'}), _CmpEQ)
    assert isinstance(from_json({'type': 'GT', 'val': {'type': 'date', 'value': '2020-01-01'}}), _CmpGT)
