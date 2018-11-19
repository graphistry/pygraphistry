import pytest
from graphistry.util import dict_util


def test_assign():
    original = {'a': 0, 'b': 0, 'c': 0}
    updates = {'a': 1,         'c': 3, 'd': 4}
    expected = {'a': 1, 'b': 0, 'c': 3}

    assert expected == dict_util.assign(original, updates)
