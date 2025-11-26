from typing import List, Optional

try:
    # Prefer stdlib typing (Py3.8+) to avoid depending on typing_extensions for TypedDict
    from typing import Literal, TypedDict
except ImportError:  # pragma: no cover - fallback for older runtimes
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
    
