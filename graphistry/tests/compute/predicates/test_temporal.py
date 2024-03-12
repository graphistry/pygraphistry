from graphistry.compute.predicates.temporal import IsLeapYear, is_leap_year

def test_is_leap_year():
    
    d = is_leap_year()
    assert isinstance(d, IsLeapYear)

    o = d.to_json()
    assert isinstance(o, dict)
    assert o['type'] == 'IsLeapYear'

    d2 = IsLeapYear.from_json(o)
    assert isinstance(d2, IsLeapYear)
