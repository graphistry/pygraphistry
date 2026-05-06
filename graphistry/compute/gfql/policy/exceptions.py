"""GFQL Policy exceptions with enriched error data."""

from typing import Optional, Dict, Any, Union, TYPE_CHECKING
from .types import Phase

if TYPE_CHECKING:
    from .stats import GraphStats


class PolicyException(Exception):
    """Exception raised when policy denies operation.

    Attributes:
        phase: Phase where denial occurred (preload, postload, call)
        reason: Human-readable explanation of denial
        code: HTTP status code (default 403)
        query_type: Type of query (chain, dag, single)
        data_size: Size information (nodes, edges counts)
    """

    def __init__(
        self,
        phase: Phase,
        reason: str,
        code: int = 403,
        query_type: Optional[str] = None,
        data_size: Optional[Union[Dict[str, int], 'GraphStats']] = None
    ):
        """Initialize PolicyException with enriched error data.

        Args:
            phase: Phase where denial occurred
            reason: Human-readable explanation
            code: HTTP status code (default 403)
            query_type: Optional type of query being executed
            data_size: Optional data size information
        """
        self.phase = phase
        self.reason = reason
        self.code = code
        self.query_type = query_type
        self.data_size = data_size

        # Build detailed error message
        message = f"Policy denial in {phase}: {reason}"
        super().__init__(message)

    def to_dict(self) -> Dict[str, Union[int, str, Dict[str, int], 'GraphStats']]:
        """Convert exception to dictionary for JSON serialization.

        Returns:
            Dictionary with error details suitable for JSON response
        """
        result: Dict[str, Union[int, str, Dict[str, int], 'GraphStats']] = {
            "code": self.code,
            "phase": self.phase,
            "reason": self.reason
        }

        if self.query_type is not None:
            result["query_type"] = self.query_type

        if self.data_size is not None:
            result["data_size"] = self.data_size

        return result
