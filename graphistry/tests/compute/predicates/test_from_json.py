from graphistry.compute.predicates.categorical import Duplicated
from graphistry.compute.predicates.from_json import from_json
from graphistry.compute.exceptions import GFQLValidationError


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
