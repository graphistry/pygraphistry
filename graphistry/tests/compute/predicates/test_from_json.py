from graphistry.compute.predicates.categorical import Duplicated
from graphistry.compute.predicates.from_json import from_json


def test_from_json_good():
    d = from_json({'type': 'Duplicated', 'keep': 'last'})
    assert isinstance(d, Duplicated)
    assert d.keep == 'last'

def test_from_json_bad():
    try:
        from_json({'type': 'zzz'})
        assert False
    except AssertionError:
        assert True

    try:
        from_json({'type': 'Duplicated', 'keep': 'zzz'})
        assert False
    except AssertionError:
        assert True

    try:
        from_json({'type': 'Duplicated'})
        assert False
    except AssertionError:
        assert True
