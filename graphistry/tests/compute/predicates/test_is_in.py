from graphistry.compute.predicates.is_in import IsIn, is_in


def test_is_in():

    d = is_in([1, 2, 3])
    assert isinstance(d, IsIn)
    assert d.options == [1, 2, 3]

    o = d.to_json()
    assert isinstance(o, dict)
    assert o['type'] == 'IsIn'

    d2 = IsIn.from_json(o)
    assert isinstance(d2, IsIn)
    assert d2.options == [1, 2, 3]
