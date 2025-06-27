from typing import Any, Union
from graphistry.Plottable import Plottable
from graphistry.util import hash_memoize, cache_coercion, WeakValueWrapper, setup_logger

def check_set_memoize(
    g: Plottable, metadata, attribute: str, name: str = "", memoize: bool = True
) -> Union[bool, Any]:
    """
    Helper Memoize function that checks if metadata args have changed for object g -- which is unconstrained save
    for the fact that it must have `attribute`. If they have not changed, will return memoized version,
    if False, will continue with whatever pipeline it is in front.
    """

    logger = setup_logger(f"{__name__}.memoization")

    if not memoize:
        logger.debug("Memoization disabled")
        return False

    hashed = None
    weakref = getattr(g, attribute)
    try:
        hashed = hash_memoize(dict(data=metadata))
    except TypeError:
        logger.warning(
            f"! Failed {name} speedup attempt. Continuing without memoization speedups."
        )
    try:
        if hashed in weakref:
            logger.debug(f"{name} memoization hit: %s", hashed)
            return weakref[hashed].v
        else:
            logger.debug(
                f"{name} memoization miss for id (of %s): %s", len(weakref), hashed
            )
    except:
        logger.debug(f"Failed to hash {name} kwargs", exc_info=True)
        pass

    if memoize and (hashed is not None):
        w = WeakValueWrapper(g)
        cache_coercion(hashed, w)
        weakref[hashed] = w
    return False
