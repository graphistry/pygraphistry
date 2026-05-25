from __future__ import annotations

from typing import Optional

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


NETWORKX_VERSION_SPEC = ">=2.5,<4"
SCIPY_VERSION_SPEC = ">=1.5,<2"

NETWORKX_EXTRA_REQUIREMENTS = (f"networkx{NETWORKX_VERSION_SPEC}",)
NETWORKX_SCIPY_EXTRA_REQUIREMENTS = (
    f"networkx{NETWORKX_VERSION_SPEC}",
    f"scipy{SCIPY_VERSION_SPEC}",
)


def _version_satisfies(version: str, spec: str) -> bool:
    try:
        parsed = Version(version)
    except InvalidVersion:
        return False
    return parsed in SpecifierSet(spec)


def networkx_version_error(version: Optional[str]) -> Optional[str]:
    installed = version or "unknown"
    if _version_satisfies(installed, NETWORKX_VERSION_SPEC):
        return None
    return f"NetworkX version {installed} is unsupported; install networkx{NETWORKX_VERSION_SPEC}."


def scipy_version_error(version: Optional[str]) -> Optional[str]:
    installed = version or "unknown"
    if _version_satisfies(installed, SCIPY_VERSION_SPEC):
        return None
    return f"SciPy version {installed} is unsupported for NetworkX-backed calls; install scipy{SCIPY_VERSION_SPEC}."
