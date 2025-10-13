
import json
from typing import Any, Dict, List, Union

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
