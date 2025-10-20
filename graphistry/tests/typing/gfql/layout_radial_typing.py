"""Static typing examples for ring_* GFQL calls.

Run `mypy` to ensure the examples stay well-typed.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphistry.compute.ast import call
    from graphistry.compute.ast import ASTCall

    # Continuous mode expects numeric-friendly parameters
    good_continuous: ASTCall = call('ring_continuous_layout', {
        'ring_col': 'score',
        'min_r': 100.0,
        'max_r': 1000.0,
    })

    # Categorical mode disallows continuous-only parameters (type ignored to flag misuse)
    bad_categorical = call('ring_categorical_layout', {  # type: ignore[arg-type]
        'num_rings': 4,
    })

    # Time mode with ISO ranges
    good_time: ASTCall = call('time_ring_layout', {
        'time_col': 'ts',
        'time_start': '2024-01-01T00:00:00',
        'time_end': '2024-01-01T01:00:00',
        'num_rings': 6,
    })
