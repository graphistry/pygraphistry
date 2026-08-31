
import json
from math import isfinite
from typing import Any, Dict, List, Optional, Union

# For mypy 0.942, we need to handle recursive types more explicitly
# Using a simple base type that mypy can resolve
JSONVal = Union[
    None,
    bool,
    str,
    float,
    int,
    List[Any],  # Simplified for mypy compatibility
    Dict[str, Any]  # Simplified for mypy compatibility
]


def is_json_serializable(data):
    try:
        json.dumps(data)
        return True
    except TypeError:
        return False

def assert_json_serializable(data):
    assert is_json_serializable(data), f"Data is not JSON-serializable: {data}"


def find_non_finite(data: Any, path: str = '') -> Optional[str]:  # hygiene-ok: explicit-any -- scans an arbitrary JSON-shaped document
    """Locate the first NaN/infinity in a JSON-shaped value.

    ``json.dumps`` emits these as the non-standard ``NaN``/``Infinity`` literals,
    so ``is_json_serializable`` accepts them while a strict encoder rejects them.

    :param data: Value to scan.
    :param path: Dotted path of ``data`` within the enclosing document.
    :return: Path of the first non-finite float, or ``None`` when there is none.
    """
    if isinstance(data, float) and not isfinite(data):
        return path or '<root>'
    if isinstance(data, dict):
        for k, v in data.items():
            hit = find_non_finite(v, f'{path}.{k}' if path else str(k))
            if hit is not None:
                return hit
    elif isinstance(data, (list, tuple)):
        for i, v in enumerate(data):
            hit = find_non_finite(v, f'{path}[{i}]')
            if hit is not None:
                return hit
    return None

def serialize_to_json_val(obj: Any) -> JSONVal:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, tuple):
        # Convert tuples to lists for JSON serialization
        return [serialize_to_json_val(item) for item in obj]
    elif isinstance(obj, list):
        return [serialize_to_json_val(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_to_json_val(value) for key, value in obj.items()}
    else:
        raise TypeError(f"Unsupported type for to_json: {type(obj)}")
