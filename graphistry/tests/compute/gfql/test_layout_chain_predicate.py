from graphistry.compute import Chain, call, n
from graphistry.compute.gfql import (
    LAYOUT_FUNCTION_NAMES,
    LAYOUT_KINDS,
    RADIAL_LAYOUT_FUNCTION_NAMES,
    RADIAL_LAYOUT_KINDS,
    is_layout_chain,
    is_layout_kind,
)
from graphistry.compute.gfql.call.validation import SAFELIST_V1, _LAYOUT_CALL_KINDS


def test_layout_chain_public_exports() -> None:
    from graphistry.compute.gfql import (
        LAYOUT_FUNCTION_NAMES as gfql_layout_function_names,
        LAYOUT_KINDS as gfql_layout_kinds,
        RADIAL_LAYOUT_FUNCTION_NAMES as gfql_radial_layout_function_names,
        RADIAL_LAYOUT_KINDS as gfql_radial_layout_kinds,
        is_layout_chain as gfql_is_layout_chain,
    )

    assert gfql_is_layout_chain is is_layout_chain
    assert gfql_layout_function_names is LAYOUT_FUNCTION_NAMES
    assert gfql_layout_kinds is LAYOUT_KINDS
    assert gfql_radial_layout_function_names is RADIAL_LAYOUT_FUNCTION_NAMES
    assert gfql_radial_layout_kinds is RADIAL_LAYOUT_KINDS


def test_layout_chain_detects_chain_object() -> None:
    chain = Chain([
        n(),
        call('ring_categorical_layout', {'ring_col': 'kind'}),
    ])

    assert is_layout_chain(chain)
    assert is_layout_kind(chain) == 'ring_categorical'


def test_layout_chain_detects_wire_dict() -> None:
    chain = {
        'type': 'Chain',
        'chain': [
            {'type': 'Node', 'filter_dict': {}},
            {
                'type': 'Call',
                'function': 'time_ring_layout',
                'params': {'time_col': 'ts'},
            },
        ],
    }

    assert is_layout_chain(chain)
    assert is_layout_kind(chain) == 'time_ring'


def test_layout_chain_detects_direct_call_object() -> None:
    op = call('fa2_layout')

    assert is_layout_chain(op)
    assert is_layout_kind(op) == 'force_directed'
    assert is_layout_kind(op) not in RADIAL_LAYOUT_KINDS


def test_layout_chain_detects_direct_call_wire_dict() -> None:
    op = {'type': 'Call', 'function': 'circle_layout', 'params': {}}

    assert is_layout_chain(op)
    assert is_layout_kind(op) == 'circle'


def test_layout_chain_detects_nested_chain_wire_dict() -> None:
    chain = [
        {
            'type': 'Chain',
            'chain': [
                {'type': 'Call', 'function': 'mercator_layout', 'params': {}},
            ],
        },
    ]

    assert is_layout_chain(chain)
    assert is_layout_kind(chain) == 'mercator'


def test_layout_registry_covers_all_safelisted_gfql_layouts() -> None:
    assert LAYOUT_FUNCTION_NAMES == frozenset({
        'layout_cugraph',
        'layout_igraph',
        'layout_graphviz',
        'ring_continuous_layout',
        'ring_categorical_layout',
        'time_ring_layout',
        'fa2_layout',
        'group_in_a_box_layout',
        'circle_layout',
        'tree_layout',
        'mercator_layout',
        'modularity_weighted_layout',
    })
    assert LAYOUT_KINDS == frozenset({
        'cugraph',
        'igraph',
        'graphviz',
        'ring_continuous',
        'ring_categorical',
        'time_ring',
        'force_directed',
        'group_in_a_box',
        'circle',
        'tree',
        'mercator',
        'modularity_weighted',
    })
    assert is_layout_kind(call('group_in_a_box_layout')) == 'group_in_a_box'
    assert is_layout_kind(call('tree_layout')) == 'tree'


def test_radial_layout_registry_covers_canonical_radial_layouts() -> None:
    assert RADIAL_LAYOUT_KINDS == frozenset({
        'ring_continuous',
        'ring_categorical',
        'time_ring',
    })
    assert RADIAL_LAYOUT_FUNCTION_NAMES == frozenset({
        'ring_continuous_layout',
        'ring_categorical_layout',
        'time_ring_layout',
    })
    assert RADIAL_LAYOUT_FUNCTION_NAMES.issubset(LAYOUT_FUNCTION_NAMES)
    assert is_layout_kind(call('ring_continuous_layout')) in RADIAL_LAYOUT_KINDS


def test_layout_chain_false_for_non_layout_chain() -> None:
    chain = Chain([
        n(),
        call('get_degrees', {'col': 'degree'}),
    ])

    assert not is_layout_chain(chain)
    assert is_layout_kind(chain) is None


def test_layout_chain_false_for_empty_chain() -> None:
    assert not is_layout_chain([])
    assert is_layout_kind({'type': 'Chain', 'chain': []}) is None


def test_layout_chain_tolerates_mixed_type_chain() -> None:
    chain = [
        object(),
        {'type': 'Call', 'function': 'layout_graphviz', 'params': {}},
    ]

    assert is_layout_chain(chain)
    assert is_layout_kind(chain) == 'graphviz'


def test_layout_chain_ignores_unsupported_mixed_items() -> None:
    chain = [
        object(),
        {'type': 'Call', 'function': 'not_a_safelisted_call', 'params': {}},
    ]

    assert not is_layout_chain(chain)
    assert is_layout_kind(chain) is None


def test_layout_chain_does_not_substring_match_opaque_strings() -> None:
    assert not is_layout_chain("call('time_ring_layout', {'time_col': 'ts'})")
    assert is_layout_kind("time_ring_layout") is None


def test_layout_chain_metadata_tracks_safelist() -> None:
    assert LAYOUT_FUNCTION_NAMES == frozenset(_LAYOUT_CALL_KINDS)
    assert set(_LAYOUT_CALL_KINDS).issubset(SAFELIST_V1)


def test_layout_registry_does_not_export_unsupported_radial_aliases() -> None:
    assert frozenset({
        'radial_layout',
        'radial_categorical_layout',
        'radial_continuous_layout',
        'radial_time_layout',
    }).isdisjoint(LAYOUT_FUNCTION_NAMES)


def test_layout_registry_exports_python_layout_methods_as_gfql_calls() -> None:
    assert frozenset({
        'tree_layout',
        'circle_layout',
        'mercator_layout',
        'modularity_weighted_layout',
    }).issubset(LAYOUT_FUNCTION_NAMES)
