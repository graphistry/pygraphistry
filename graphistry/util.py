import hashlib, logging, os, pandas as pd, platform as p, random, string, sys, uuid, warnings
from typing import Any, Dict
from distutils.version import LooseVersion, StrictVersion
from functools import lru_cache

from .constants import VERBOSE, CACHE_COERCION_SIZE


# #####################################
def setup_logger(name, verbose=True):
    if verbose:
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ]\n   %(message)s\n"
    else:
        FORMAT = "   %(message)s\n"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
    return logger


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


def check_set_memoize(g, metadata, attribute, name: str = '', memoize: bool = True):  # noqa: C901
    """
        Helper Memoize function that checks if metadata args have changed for object g -- which is unconstrained save
        for the fact that it must have `attribute`. If they have not changed, will return memoized version,
        if False, will continue with whatever pipeline it is in front.
    """
    
    logger = setup_logger('memoization', verbose=VERBOSE)
    
    hashed = None
    weakref = getattr(g, attribute)
    try:
        hashed = (
            hashlib.sha256(str(metadata).encode('utf-8')).hexdigest()
        )
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


def cmp(x, y):
    return (x > y) - (x < y)


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


def compare_versions(v1, v2):
    try:
        return cmp(StrictVersion(v1), StrictVersion(v2))
    except ValueError:
        return cmp(LooseVersion(v1), LooseVersion(v2))


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
