"""Validation functions for GFQL policy modifications."""

from typing import Any, Dict, Set
from .types import PolicyModification, Phase


def validate_modification(mod: Any, phase: Phase) -> PolicyModification:
    """Validate and type-check policy modifications.

    Args:
        mod: Modification dictionary from policy function
        phase: Current execution phase

    Returns:
        Validated PolicyModification

    Raises:
        ValueError: If modifications are invalid
    """
    # Check basic type
    if mod is None:
        return {}  # Empty modification is valid

    if not isinstance(mod, dict):
        raise ValueError(f"Modifications must be a dict, got {type(mod).__name__}")

    # Validate engine field
    if 'engine' in mod:
        engine = mod['engine']
        valid_engines = {'pandas', 'cudf', 'dask', 'dask_cudf', 'auto'}
        if engine not in valid_engines:
            raise ValueError(
                f"Invalid engine '{engine}'. Must be one of {valid_engines}"
            )

    # Validate params field
    if 'params' in mod:
        params = mod['params']
        if not isinstance(params, dict):
            raise ValueError(
                f"'params' must be a dict, got {type(params).__name__}"
            )

    # Query modifications only allowed in preload phase
    if 'query' in mod and phase != 'preload':
        raise ValueError(
            f"Query modifications only allowed in preload phase, not {phase}"
        )

    # Check for unknown fields
    allowed_fields: Set[str] = {'engine', 'params', 'query'}
    unknown_fields = set(mod.keys()) - allowed_fields

    if unknown_fields:
        raise ValueError(
            f"Unknown modification fields: {unknown_fields}. "
            f"Allowed fields: {allowed_fields}"
        )

    # Return typed result
    result: PolicyModification = {}

    if 'engine' in mod:
        result['engine'] = mod['engine']

    if 'params' in mod:
        result['params'] = mod['params']

    if 'query' in mod:
        result['query'] = mod['query']

    return result
