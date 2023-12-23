from graphistry.compute.predicates.numeric import GT, gt

def test_gt():

    d = gt(1)
    assert isinstance(d, GT)
    assert d.val == 1

    o = d.to_json()
    assert isinstance(o, dict)
    assert o['type'] == 'GT'
    assert o['val'] == 1

    d2 = GT.from_json(o)
    assert isinstance(d2, GT)
    assert d2.val == 1
