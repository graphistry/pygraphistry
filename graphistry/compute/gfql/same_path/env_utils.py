import os
from typing import Optional

_BOOL_TRUE = {"1", "true", "yes", "on"}


def env_lower(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip().lower()


def env_flag(name: str, default: bool = False) -> bool:
    raw = env_lower(name)
    return default if not raw else raw in _BOOL_TRUE


def env_optional_int(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def normalize_limit(value: Optional[float], default: Optional[float]) -> Optional[float]:
    value = default if value is None else value
    return None if value is not None and value <= 0 else value
