from typing import List, Optional
from typing_extensions import Literal, TypedDict

Mode = Literal['private', 'organization', 'public']

ModeAction = Literal['10', '20']
MODE_ACTION_VIEW: ModeAction = '10'
MODE_ACTION_EDIT: ModeAction = '20'

class Privacy(TypedDict, total=False):
    mode: Optional[Mode]
    notify: Optional[bool]
    invited_users: Optional[List[str]]
    message: Optional[str]
    mode_action: Optional[ModeAction]
    
