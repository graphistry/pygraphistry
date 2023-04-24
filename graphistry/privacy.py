from typing import List, Literal, TypedDict
from typing_extensions import NotRequired

Mode = Literal['private', 'organization', 'public']

class Privacy(TypedDict):
    mode: NotRequired[Mode]
    notify: NotRequired[bool]
    invited_users: NotRequired[List[str]]
    message: NotRequired[str]
