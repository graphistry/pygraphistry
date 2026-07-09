"""GFQL index policy validation."""
from __future__ import annotations

from typing import Literal, Optional, Tuple, cast

IndexPolicy = Literal["off", "use", "auto", "force"]
VALID_INDEX_POLICIES: Tuple[IndexPolicy, ...] = ("off", "use", "auto", "force")


def validate_index_policy(policy: Optional[str]) -> Optional[IndexPolicy]:
    """Validate a public ``index_policy`` value.

    ``None`` means the caller did not override the default planner behavior.
    """
    if policy is None:
        return None
    if policy not in VALID_INDEX_POLICIES:
        allowed = ", ".join(repr(p) for p in VALID_INDEX_POLICIES)
        raise ValueError(f"index_policy must be one of {allowed}; got {policy!r}")
    return cast(IndexPolicy, policy)
