from typing import List, TypedDict
from typing_extensions import NotRequired, Literal

Mode = Literal['private', 'organization', 'public']

class Privacy(TypedDict):
    mode: NotRequired[Mode]
    notify: NotRequired[bool]
    invited_users: NotRequired[List[str]]
    message: NotRequired[str]
