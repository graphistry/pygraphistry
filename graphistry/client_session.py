import os
from typing import Any, Optional, Type, TypeVar, Union, overload, Literal, cast
from typing_extensions import deprecated
from functools import lru_cache
import json

from graphistry.privacy import Privacy
from . import util



config_paths = [
    os.path.join("/etc/graphistry", ".pygraphistry"),
    os.path.join(os.path.expanduser("~"), ".pygraphistry"),
    os.environ.get("PYGRAPHISTRY_CONFIG", ""),  # user-override path
]

ApiVersion = Literal[1, 3]

ENV_GRAPHISTRY_API_KEY = "GRAPHISTRY_API_KEY"

class ClientSession:

    def __init__(self) -> None:
        self.api_key: Optional[str] = get_from_env(ENV_GRAPHISTRY_API_KEY, str)
        self.api_token: Optional[str] = get_from_env("GRAPHISTRY_API_TOKEN", str)
        # self.api_token_refresh_ms: Optional[int] = None

        env_api_version = get_from_env("GRAPHISTRY_API_VERSION", int)
        if env_api_version is None:
            env_api_version = 1
        elif env_api_version not in [1, 3]:
            raise ValueError("Expected API version to be 1, 3, instead got (likely from GRAPHISTRY_API_VERSION): %s" % env_api_version)
        self.api_version: ApiVersion = cast(ApiVersion, env_api_version)  

        self.dataset_prefix: str = get_from_env("GRAPHISTRY_DATASET_PREFIX", str, "PyGraphistry/")
        self.hostname: str = get_from_env("GRAPHISTRY_HOSTNAME", str, "hub.graphistry.com")
        self.protocol: str = get_from_env("GRAPHISTRY_PROTOCOL", str, "https")
        self.client_protocol_hostname: Optional[str] = get_from_env("GRAPHISTRY_CLIENT_PROTOCOL_HOSTNAME", str)
        self.certificate_validation: bool = get_from_env("GRAPHISTRY_CERTIFICATE_VALIDATION", bool, True)
        self.store_token_creds_in_memory: bool = get_from_env("GRAPHISTRY_STORE_CREDS_IN_MEMORY", bool, True)
        self.privacy: Optional[Privacy] = None
        self.login_type: Optional[str] = None
        self.org_name: Optional[str] = None

        self.idp_name: Optional[str] = None
        self.sso_state: Optional[str] = None

        self.personal_key: Optional[str] = None
        self.personal_key_id: Optional[str] = None
        self.personal_key_secret: Optional[str] = None

        # NOTE: Still used as a global, perhaps use a session pattern
        self.encode_textual_batch_size: Optional[int] = None # encode_textual.batch_size

        # TODO: Migrate to a pattern like Kusto or Spanner
        self.bolt_driver: Optional[Any] = None


@deprecated("Use the session pattern instead")
def use_global_session() -> ClientSession:
    from .pygraphistry import PyGraphistry
    return PyGraphistry._session


T = TypeVar("T")

@overload
def get_from_env(name: str, expected_type: Type[T]) -> Optional[T]: ...
@overload  # when a *non-None* default is supplied, return is not Optional
def get_from_env(name: str, expected_type: Type[T], default: T) -> T: ...

def get_from_env(
    name: str,
    expected_type: Type[T],
    default: Union[T, None] = None,
) -> Union[T, None]:
    """
    Retrieve a configuration value for *name* with this precedence:
        1. The environment variable *name* (highest priority)
        2. The first matching key in any file listed in *config_paths*
        3. *default* (or None)

    The result is coerced to *expected_type*; invalid values are ignored
    with a warning rather than raising.
    """
    # 1) Process environment
    raw = os.environ.get(name)
    if raw is not None:
        val = _coerce(raw, expected_type)
        if val is not None:
            return val

    # 2) Config files
    file_cfg = _load_file_config()
    if name in file_cfg:
        val = _coerce(file_cfg[name], expected_type)
        if val is not None:
            return val

    # 3) Fallback
    return default

@lru_cache(maxsize=1)
def _load_file_config() -> dict:
    """
    Merge all readable JSON files in *config_paths*.
    The first key wins (later files override earlier ones).
    """
    merged: dict = {}
    for path in filter(bool, config_paths):                # skip '' entries
        try:
            with open(path) as fh:
                merged.update(json.load(fh))
        except FileNotFoundError:
            pass
        except ValueError as e:
            util.warn(f"Syntax error in {path}, skipping. ({e})")
    return merged


def strtobool(val: Any) -> bool:
    val = str(val).lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def _coerce(value: Any, expected_type: Type[T]) -> Optional[T]:
    """
    Convert *value* to *expected_type*; return None on failure.
    Supported coercions: str, int, float, bool, list, dict, plus
    a generic constructor fallback.
    """
    try:
        if expected_type is str:
            return str(value)              # type: ignore[return-value]
        if expected_type is bool:
            if isinstance(value, bool):
                return value               # type: ignore[return-value]
            return strtobool(value)  # type: ignore[return-value]
        if expected_type is int:
            return int(value)              # type: ignore[return-value]
        if expected_type is float:
            return float(value)            # type: ignore[return-value]
        if expected_type is list or expected_type is dict:
            if isinstance(value, expected_type):
                return value               # type: ignore[return-value]
            return json.loads(value)       # type: ignore[return-value]
        raise ValueError(f"Unsupported type: {expected_type.__name__}")
    except Exception as e:
        util.warn(f"Could not cast {value!r} to {expected_type.__name__}: {e}")
        return None
