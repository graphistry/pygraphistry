from typing import Optional, Union

from graphistry.Plottable import RENDER_MODE_VALUES, Plottable, RenderModes, RenderModesConcrete
from graphistry.util import in_databricks, in_ipython


def resolve_render_mode(
    self: Plottable,
    render: Optional[Union[bool, RenderModes]],
) -> RenderModesConcrete:

    # cascade
    if render is None:
        render = self._render 

    # => RenderMode
    if isinstance(render, bool):
        render = "auto" if render else "url"    

    if render not in RENDER_MODE_VALUES:
        raise ValueError(f'Invalid render mode: {render}, expected one of {RENDER_MODE_VALUES}')

    # => RenderModeConcrete
    if render != "auto":
        return render

    if in_ipython():
        return "ipython"
    elif in_databricks():
        return "databricks"
    else:        
        try:
            import webbrowser
            webbrowser.get()  # Tries to find the default browser
            return "browser"
        except Exception:
            return "url"