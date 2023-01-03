import hashlib
import logging
import os
import pandas as pd
import pandas.util as putil
import platform as p
import random
import string
import uuid
import warnings
from functools import lru_cache
from typing import Any

from .constants import VERBOSE, CACHE_COERCION_SIZE, TRACE


# #####################################

def global_logger():
    logger = logging.getLogger()
    return logger

def setup_logger(name, verbose=VERBOSE, fullpath=TRACE):
    #if fullpath:
    #    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ]\n   %(message)s\n"
    #else:
    #    FORMAT = " %(message)s\n"
    #logging.basicConfig(format=FORMAT)
    #logger = logging.getLogger()#f'graphistry.{name}')
    #if verbose is None:
    #    logger.setLevel(logging.ERROR)
    #else:
    #    logger.setLevel(logging.INFO if verbose else logging.DEBUG)
    #return logger
    return global_logger()


# #####################################
# Caching utils

_cache_coercion_val = None
@lru_cache(maxsize=CACHE_COERCION_SIZE)
def cache_coercion_helper(k):
    return _cache_coercion_val


def cache_coercion(k, v):
    """
        Holds references to last 100 used coercions
        Use with weak key/value dictionaries for actual lookups
    """
    global _cache_coercion_val
    _cache_coercion_val = v

    out = cache_coercion_helper(k)
    _cache_coercion_val = None
    return out


class WeakValueWrapper:
    def __init__(self, v):
        self.v = v


def hash_pdf(df: pd.DataFrame) -> str:
    # can be 20% faster via to_parquet (see lmeyerov issue in pandas gh), but unclear if always available
    return (
        hashlib.sha256(putil.hash_pandas_object(df, index=True).to_numpy().tobytes()).hexdigest()
        + hashlib.sha256(str(df.columns).encode('utf-8')).hexdigest()  # noqa: W503
    )


def hash_memoize_helper(v: Any) -> str:

    if isinstance(v, dict):
        rolling = '{'
        for k2, v2 in v.items():
            rolling += f'{k2}:{hash_memoize_helper(v2)},'
        rolling += '}'
    elif isinstance(v, list):
        rolling = '['
        for i in v:
            rolling += f'{hash_memoize_helper(i)},'
        rolling += ']'
    elif isinstance(v, tuple):
        rolling = '('
        for i in v:
            rolling += f'{hash_memoize_helper(i)},'
        rolling += ')'
    elif isinstance(v, bool):
        rolling = 'T' if v else 'F'
    elif isinstance(v, int):
        rolling = str(v)
    elif isinstance(v, float):
        rolling = str(v)
    elif isinstance(v, str):
        rolling = v
    elif v is None:
        rolling = 'N'
    elif isinstance(v, pd.DataFrame):
        rolling = hash_pdf(v)
    else:
        raise TypeError(f'Unsupported memoization type: {type(v)}')

    return rolling

def hash_memoize(v: Any) -> str:
    return hashlib.sha256(hash_memoize_helper(v).encode('utf-8')).hexdigest()

def check_set_memoize(g, metadata, attribute, name: str = '', memoize: bool = True):  # noqa: C901
    """
        Helper Memoize function that checks if metadata args have changed for object g -- which is unconstrained save
        for the fact that it must have `attribute`. If they have not changed, will return memoized version,
        if False, will continue with whatever pipeline it is in front.
    """
    
    logger = setup_logger(f'{__name__}.memoization')

    if not memoize:
        logger.debug('Memoization disabled')
        return False
    
    hashed = None
    weakref = getattr(g, attribute)
    try:
        hashed = hash_memoize(metadata)
    except TypeError:
        logger.warning(
            f'! Failed {name} speedup attempt. Continuing without memoization speedups.'
        )
    try:
        if hashed in weakref:
            logger.debug(f'{name} memoization hit: %s', hashed)
            return weakref[hashed].v
        else:
            logger.debug(f'{name} memoization miss for id (of %s): %s',
                         len(weakref), hashed)
    except:
        logger.debug(f'Failed to hash {name} kwargs', exc_info=True)
        pass
    
    if memoize and (hashed is not None):
        w = WeakValueWrapper(g)
        cache_coercion(hashed, w)
        weakref[hashed] = w
    return False


def make_iframe(url, height, extra_html="", override_html_style=None):
    id = uuid.uuid4()

    height_str = (
        f"{height}px"
        if isinstance(height, int) or isinstance(height, float)
        else str(height)
    )

    scrollbug_workaround = (
        """
            <script>
                try {
                  $("#%s").bind('mousewheel', function(e) { e.preventDefault(); });
                } catch (e) { console.error('exn catching scroll', e); }
            </script>
        """
        % id
    )

    style = None
    if override_html_style is not None:
        style = override_html_style
    else:
        style = (
            "width:100%%; height:%s; border: 1px solid #DDD; overflow: hidden"
            % height_str
        )

    iframe = """
            <iframe id="%s" src="%s"
                    allowfullscreen="true" webkitallowfullscreen="true" mozallowfullscreen="true"
                    oallowfullscreen="true" msallowfullscreen="true"
                    style="%s"
                    %s
            >
            </iframe>
        """ % (
        id,
        url,
        style,
        extra_html,
    )

    return iframe + scrollbug_workaround


def fingerprint():
    md5 = hashlib.md5()
    # Hostname, OS, CPU, MAC,
    data = [p.node(), p.system(), p.machine(), str(uuid.getnode())]
    md5.update("".join(data).encode("utf8"))

    from ._version import get_versions

    __version__ = get_versions()["version"]

    return "%s-pygraphistry-%s" % (md5.hexdigest()[:8], __version__)


def random_string(length):
    gibberish = [
        random.choice(string.ascii_uppercase + string.digits) for _ in range(length)
    ]
    return "".join(gibberish)


def in_ipython():
    try:
        if hasattr(__builtins__, "__IPYTHON__"):
            return True
    except NameError:
        pass
    try:
        from IPython import get_ipython

        cfg = get_ipython()
        if not (cfg is None) and ("IPKernelApp" in get_ipython().config):
            return True
    except ImportError:
        pass
    return False


def in_databricks():
    # FIXME: this is a hack
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        return True
    return False


def warn(msg):
    try:
        if in_ipython():
            import IPython

            IPython.utils.warn.warn(msg)
            return
    except:
        "ok"
    warnings.warn(RuntimeWarning(msg))


def error(msg):
    raise ValueError(msg)


def merge_two_dicts(a, b):
    c = a.copy()
    c.update(b)
    return c


def deprecated(message):
    """
    Marks a method as deprecated.

    :param message: Info regarding the deprecation.
    """

    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            warnings.warn(
                "{} is a deprecated function. {}".format(func.__name__, message),
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator


#
# def inspect_decorator(func, args, kwargs):
#     import inspect
#     frame = inspect.currentframe()
#     args, _, _, values = inspect.getargvalues(frame)
#     func_name = inspect.getframeinfo(frame)[2]
#     print(f'function name "{func_name}"')
#     for i in args:
#         print("    %s = %s" % (i, values[i]))
#     return [(i, values[i]) for i in args]
#
#
# # custom decorator
# def showargs_decorator(func):
#     import functools
#     # updates special attributes e.g. __name__, __doc__
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#
#       # call custom inspection logic
#       inspect_decorator(func, args, kwargs)
#
#       # calls original function
#       func(*args, **kwargs)
#
#     # matches name of inner function
#     return wrapper
