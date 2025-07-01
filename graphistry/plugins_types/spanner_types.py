from typing import List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
import os

if TYPE_CHECKING:
    from google.cloud.spanner_dbapi.connection import Connection
else:
    Connection = Any


class SpannerConnectionError(Exception):
    """Custom exception for errors related to Spanner connection."""
    pass

class SpannerQueryResult:
    """
    Encapsulates the results of a query, including metadata.

    :ivar list data: The raw query results.
    """

    def __init__(self, data: List[Any], column_names: List[str]):
        """
        Initializes a SpannerQueryResult instance.

        :param data: The raw query results.
        :type List[Any]
        :param column_names: a list of the column names from the cursor, defaults to None 
        :type: List[str]
        """
        self.data = data
        self.column_names = column_names


@dataclass
class SpannerConfig:
    project_id: Optional[str] = None
    instance_id: Optional[str] = None
    database_id: Optional[str] = None
    credentials_file: Optional[str] = None

    _client: Optional[Connection] = None

    def validate(self) -> None:
        if self._client is not None:
            return
        missing = []
        if not self.instance_id:
            missing.append("instance_id")
        if not self.database_id:
            missing.append("database_id")
        if missing:
            raise ValueError(f"SpannerConfig missing required field(s): {', '.join(missing)}")

        if self.project_id is None and self.credentials_file is None:
            raise ValueError("SpannerConfig requires `project_id` or `credentials_file`.")

        if self.credentials_file is not None and not os.path.isfile(self.credentials_file):
            raise FileNotFoundError(f"Credentials file not found: {self.credentials_file}")
