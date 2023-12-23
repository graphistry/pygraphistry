from graphistry.compute.predicates.categorical import Duplicated, duplicated

def test_duplicated():

    d = duplicated('last')
    assert isinstance(d, Duplicated)
    assert d.keep == 'last'

    o = d.to_json()
    assert isinstance(o, dict)
    assert o['type'] == 'Duplicated'

    d2 = Duplicated.from_json(o)
    assert isinstance(d2, Duplicated)
    assert d2.keep == 'last'
