from graphistry.compute.predicates.str import IsUpper, isupper


def test_is_upper():
    
    d = isupper()
    assert isinstance(d, IsUpper)

    o = d.to_json()
    assert isinstance(o, dict)
    assert o['type'] == 'IsUpper'

    d2 = IsUpper.from_json(o)
    assert isinstance(d2, IsUpper)
