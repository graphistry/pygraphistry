import os
import pytest

skip_if_travis = pytest.mark.skipif(
    os.environ.get('TRAVIS') is not None,
    reason='Cannot run on Travis CI'
)
