from typing import List
from typing_extensions import NotRequired, Literal, TypedDict

Mode = Literal['private', 'organization', 'public']

class Privacy(TypedDict):
    mode: NotRequired[Mode]
    notify: NotRequired[bool]
    invited_users: NotRequired[List[str]]
    message: NotRequired[str]
