from typing import List, Any
from typing_extensions import TypedDict, NotRequired

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


class SpannerConfig(TypedDict):
    project_id: NotRequired[str]
    instance_id: str
    database_id: str
    credentials_file: NotRequired[str]
