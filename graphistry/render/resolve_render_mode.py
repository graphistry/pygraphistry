from typing import Optional, Union

from graphistry.Plottable import RENDER_MODE_CONCRETE_VALUES, Plottable, RenderModes, RenderModesConcrete
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

    # => RenderModeConcrete
    if render != "auto":
        assert render in RENDER_MODE_CONCRETE_VALUES
        return render  # type: ignore

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
