import os
from dataclasses import is_dataclass, replace
from typing import Any, Optional, Literal, cast, Protocol, TypedDict, Dict, MutableMapping, Type, TypeVar, Union, overload, Iterator
from functools import lru_cache
import json
import warnings

from graphistry.privacy import Privacy
from . import util
from .plugins_types.spanner_types import SpannerConfig
from .plugins_types.kusto_types import KustoConfig



ApiVersion = Literal[1, 3]

ENV_GRAPHISTRY_API_KEY = "GRAPHISTRY_API_KEY"

config_paths = [
    os.path.join("/etc/graphistry", ".pygraphistry"),
    os.path.join(os.path.expanduser("~"), ".pygraphistry"),
    os.environ.get("PYGRAPHISTRY_CONFIG", ""),  # user-override path
]


class ClientSession:
    """
    Holds all configuration and authentication state for a Graphistry client.
    
    Session Isolation:
    Each GraphistryClient instance maintains its own ClientSession, providing
    isolation for concurrent and multi-tenant use cases. The session tracks:
    
    - Authentication state (tokens, API keys, SSO state)
    - Server configuration (hostname, protocol, API version)
    - Organization and privacy settings
    - Plugin configurations (Kusto, Spanner, etc.)
    
    Thread Safety:
    A ClientSession instance should only be used within a single concurrency
    context (thread, async task, etc.). For multi-threaded applications, create
    separate client instances for each thread.
    
    Token Management:
    Authentication tokens may be refreshed during plot() operations. The session
    maintains the current token state and handles refresh automatically.
    """
    def __init__(self) -> None:
        self._is_authenticated: bool = False
        self._tag = util.fingerprint()  # NOTE: Should this be unique per PyGraphistry.client()?

        self.api_key: Optional[str] = get_from_env(ENV_GRAPHISTRY_API_KEY, str)
        self.api_token: Optional[str] = get_from_env("GRAPHISTRY_API_TOKEN", str)
        # self.api_token_refresh_ms: Optional[int] = None

        env_api_version = get_from_env("GRAPHISTRY_API_VERSION", int)
        if env_api_version is None:
            env_api_version = 1
        elif env_api_version not in [1, 3]:
            raise ValueError("Expected API version to be 1, 3, instead got (likely from API_VERSION): %s" % env_api_version)
        self.api_version: ApiVersion = cast(ApiVersion, env_api_version)  

        self.dataset_prefix: str = get_from_env("GRAPHISTRY_DATASET_PREFIX", str, "PyGraphistry/")
        self.hostname: str = get_from_env("GRAPHISTRY_HOSTNAME", str, "hub.graphistry.com")
        self.protocol: str = get_from_env("GRAPHISTRY_PROTOCOL", str, "https")
        self.client_protocol_hostname: Optional[str] = get_from_env("GRAPHISTRY_CLIENT_PROTOCOL_HOSTNAME", str)
        self.certificate_validation: bool = get_from_env("GRAPHISTRY_CERTIFICATE_VALIDATION", bool, True)
        self.store_token_creds_in_memory: bool = get_from_env("GRAPHISTRY_STORE_CREDS_IN_MEMORY", bool, True)
        self.login_type: Optional[str] = None
        self.org_name: Optional[str] = None

        self.idp_name: Optional[str] = None
        self.sso_state: Optional[str] = None

        self.personal_key: Optional[str] = None
        self.personal_key_id: Optional[str] = None
        self.personal_key_secret: Optional[str] = None

        # NOTE: Still used as a global, perhaps use a session pattern
        self.encode_textual_batch_size: Optional[int] = None  # encode_textual.batch_size

        self.privacy: Optional[Privacy] = None

        # Plugin sessions
        # NOTE: These are dataclasses, so we shallow copy them
        self.kusto: Optional[KustoConfig] = None
        self.spanner: Optional[SpannerConfig] = None

        # TODO: Migrate to a pattern like Kusto or Spanner
        self._bolt_driver: Optional[Any] = None

    def copy(self) -> "ClientSession":
        """
        Create a copy of this ClientSession.

        Copies dicts and plugin sessions, but not their values.
        That way if a connection is created before the copy, it will be shared.
        """
        
        # NOTE: This is potentially fragile,
        # TODO: Adopt a more robust configuration pattern, perhaps with a library
        #       @lmeyerov likes: https://docs.dask.org/en/latest/configuration.html
        clone = ClientSession.__new__(ClientSession)

        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                clone.__dict__[k] = v.copy()
            elif is_dataclass(v) and not isinstance(v, type):
                clone.__dict__[k] = replace(v)
            else:
                clone.__dict__[k] = v

        return cast(ClientSession, clone)

    # NOTE: For backwards compatibility    
    def as_proxy(self) -> MutableMapping[str, Any]:
        return SessionProxy(self)





class DatasetInfo(TypedDict):
    name: str
    viztoken: str
    type: Literal["arrow", "vgraph"]



class AuthManagerProtocol(Protocol):
    session: ClientSession

    def _etl1(self, dataset: Any) -> DatasetInfo:
        ...

    def refresh(self, token: Optional[str] = None, fail_silent: bool = False) -> Optional[str]:
        ...
    
    def _viz_url(self, info: DatasetInfo, url_params: Dict[str, Any]) -> str:
        ...

    def certificate_validation(self, value: Optional[bool] = None) -> bool:
        ...
    
    def api_token(self, value: Optional[str] = None) -> Optional[str]:
        ...


def use_global_session() -> ClientSession:
    """Use the session pattern instead.

    .. deprecated::
       This function is deprecated. Use the session pattern instead.
    """
    warnings.warn("use_global_session() is deprecated. Use the session pattern instead.",
                  DeprecationWarning, stacklevel=2)
    from .pygraphistry import PyGraphistry
    return PyGraphistry.session


T = TypeVar("T")

@overload
def get_from_env(name: str, expected_type: Type[T]) -> Optional[T]:
    ...
@overload  # when a *non-None* default is supplied, return is not Optional
def get_from_env(name: str, expected_type: Type[T], default: T) -> T:
    ...

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



class SessionProxy(MutableMapping[str, Any]):
    """Dict view that stays in sync with its ClientSession."""
    __slots__ = ("_sess",)

    def __init__(self, sess: "ClientSession") -> None:
        self._sess = sess

    # --- Mapping interface -------------------------------------------------
    def __getitem__(self, k: str) -> Any:
        if k.startswith("_") or not hasattr(self._sess, k):
            raise KeyError(k)
        return getattr(self._sess, k)

    def __setitem__(self, k: str, v: Any) -> None:
        if k.startswith("_") or not hasattr(self._sess, k):
            raise KeyError(k)
        setattr(self._sess, k, v)

    def __delitem__(self, k: str) -> None:
        self.__setitem__(k, None)

    def __iter__(self) -> Iterator[str]:
        return (k for k in self._sess.__dict__ if not k.startswith("_"))

    def __len__(self) -> int:
        return len(self._sess.__dict__) - sum(k.startswith("_") for k in self._sess.__dict__)
