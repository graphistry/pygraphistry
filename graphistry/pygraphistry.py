from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from typing_extensions import Literal
from graphistry.Plottable import Plottable

"""Top-level import of class PyGraphistry as "Graphistry". Used to connect to the Graphistry server and then create a base plotter."""
import calendar, gzip, io, json, os, numpy as np, pandas as pd, requests, sys, time, warnings

from datetime import datetime

from .arrow_uploader import ArrowUploader
from .ArrowFileUploader import ArrowFileUploader

from . import util
from . import bolt_util
from .plotter import Plotter
from .util import in_databricks, setup_logger, in_ipython, make_iframe
from .exceptions import SsoRetrieveTokenTimeoutException

from .messages import (
    MSG_REGISTER_MISSING_PASSWORD,
    MSG_REGISTER_MISSING_USERNAME,
    MSG_REGISTER_MISSING_PKEY_SECRET,
    MSG_REGISTER_MISSING_PKEY_ID,
    MSG_REGISTER_ENTER_SSO_LOGIN
)


logger = setup_logger(__name__)


###############################################################################

SSO_GET_TOKEN_ELAPSE_SECONDS = 50

EnvVarNames = {
    "api_key": "GRAPHISTRY_API_KEY",
    #'api_token': 'GRAPHISTRY_API_TOKEN',
    #'username': 'GRAPHISTRY_USERNAME',
    #'password': 'GRAPHISTRY_PASSWORD',
    "api_version": "GRAPHISTRY_API_VERSION",
    "dataset_prefix": "GRAPHISTRY_DATASET_PREFIX",
    "hostname": "GRAPHISTRY_HOSTNAME",
    "protocol": "GRAPHISTRY_PROTOCOL",
    "client_protocol_hostname": "GRAPHISTRY_CLIENT_PROTOCOL_HOSTNAME",
    "certificate_validation": "GRAPHISTRY_CERTIFICATE_VALIDATION",
    "store_token_creds_in_memory": "GRAPHISTRY_STORE_CREDS_IN_MEMORY",
}

config_paths = [
    os.path.join("/etc/graphistry", ".pygraphistry"),
    os.path.join(os.path.expanduser("~"), ".pygraphistry"),
    os.environ.get("PYGRAPHISTRY_CONFIG", ""),
]

default_config = {
    "api_key": None,  # Dummy key
    "api_token": None,
    "api_token_refresh_ms": None,
    "api_version": 1,
    "dataset_prefix": "PyGraphistry/",
    "hostname": "hub.graphistry.com",
    "protocol": "https",
    "client_protocol_hostname": None,
    "certificate_validation": True,
    "store_token_creds_in_memory": True,
    # Do not call API when all None
    "privacy": None,
    "login_type": None
}


def _get_initial_config():
    config = default_config.copy()
    for path in config_paths:
        try:
            with open(path) as config_file:
                config.update(json.load(config_file))
        except ValueError as e:
            util.warn("Syntax error in %s, skipping. (%s)" % (path, e))
            pass
        except IOError:
            pass

    env_config = {k: os.environ.get(v) for k, v in EnvVarNames.items()}
    env_override = {k: v for k, v in env_config.items() if v is not None}
    config.update(env_override)
    if not config["certificate_validation"]:
        requests.packages.urllib3.disable_warnings()
    return config


def strtobool(val: Any) -> bool:
    val = str(val).lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))

class PyGraphistry(object):
    _config = _get_initial_config()
    _tag = util.fingerprint()
    _is_authenticated = False

    @staticmethod
    def authenticate():
        """Authenticate via already provided configuration (api=1,2).
        This is called once automatically per session when uploading and rendering a visualization.
        In api=3, if token_refresh_ms > 0 (defaults to 10min), this starts an automatic refresh loop.
        In that case, note that a manual .login() is still required every 24hr by default.
        """

        if PyGraphistry.api_version() == 3:
            if not (PyGraphistry.api_token() is None):
                PyGraphistry.refresh()
        else:
            key = PyGraphistry.api_key()
            # Mocks may set to True, so bypass in that case
            if (key is None) and (PyGraphistry._is_authenticated is False):
                util.error(
                    "In api=1 mode, API key not set explicitly in `register()` or available at "
                    + EnvVarNames["api_key"]  # noqa: W503
                )
            if not PyGraphistry._is_authenticated:
                PyGraphistry._check_key_and_version()
                PyGraphistry._is_authenticated = True

    @staticmethod
    def __reset_token_creds_in_memory():
        """Reset the token and creds in memory, used when switching hosts, switching register method"""

        PyGraphistry._config["api_key"] = None
        PyGraphistry._is_authenticated = False



    @staticmethod
    def not_implemented_thunk():
        raise Exception("Must call login() first")

    relogin = lambda: PyGraphistry.not_implemented_thunk()  # noqa: E731

    @staticmethod
    def login(username, password, org_name=None, fail_silent=False):
        """Authenticate and set token for reuse (api=3). If token_refresh_ms (default: 10min), auto-refreshes token.
        By default, must be reinvoked within 24hr."""
        logger.debug("@PyGraphistry login : org_name :{} vs PyGraphistry.org_name() : {}".format(org_name, PyGraphistry.org_name()))
        
        if not org_name:
            org_name = PyGraphistry.org_name()

        if PyGraphistry._config['store_token_creds_in_memory']:
            PyGraphistry.relogin = lambda: PyGraphistry.login(
                username, password, None, fail_silent
            )

        PyGraphistry._is_authenticated = False
        token = (
            ArrowUploader(
                server_base_path=PyGraphistry.protocol()
                + "://"                     # noqa: W503
                + PyGraphistry.server(),    # noqa: W503
                certificate_validation=PyGraphistry.certificate_validation(),
            )
            .login(username, password, org_name)
            .token
        )
        
        logger.debug("@PyGraphistry login After ArrowUploader.login: org_name :{} vs PyGraphistry.org_name() : {}".format(org_name, PyGraphistry.org_name()))

        PyGraphistry.api_token(token)
        PyGraphistry._is_authenticated = True

        return PyGraphistry.api_token()

    @staticmethod
    def pkey_login(personal_key_id, personal_key_secret, org_name=None, fail_silent=False):
        """Authenticate with personal key/secret and set token for reuse (api=3). If token_refresh_ms (default: 10min), auto-refreshes token.
        By default, must be reinvoked within 24hr."""

        if PyGraphistry._config['store_token_creds_in_memory']:
            PyGraphistry.relogin = lambda: PyGraphistry.pkey_login(
                personal_key_id, personal_key_secret, org_name if org_name else PyGraphistry.org_name(), fail_silent
            )

        PyGraphistry._is_authenticated = False
        token = (
            ArrowUploader(
                server_base_path=PyGraphistry.protocol()
                + "://"                     # noqa: W503
                + PyGraphistry.server(),    # noqa: W503
                certificate_validation=PyGraphistry.certificate_validation(),
            )
            .pkey_login(personal_key_id, personal_key_secret, org_name)
            .token
        )
        PyGraphistry.api_token(token)
        PyGraphistry._is_authenticated = True

        return PyGraphistry.api_token()

    @staticmethod
    def sso_login(org_name=None, idp_name=None, sso_timeout=SSO_GET_TOKEN_ELAPSE_SECONDS):
        """Authenticate with SSO and set token for reuse (api=3).

        :param org_name: Set login organization's name(slug). Defaults to user's personal organization.
        :type org_name: Optional[str]
        :param idp_name: Set sso login idp name. Default as None (for site-wide SSO / for the only idp record).
        :type idp_name: Optional[str]
        :param sso_timeout: Set sso login getting token timeout in seconds (blocking mode), set to None if non-blocking mode. Default as SSO_GET_TOKEN_ELAPSE_SECONDS.
        :type sso_timeout: Optional[int]
        :returns: None.
        :rtype: None

        SSO Login logic.

        """

        if PyGraphistry._config['store_token_creds_in_memory']:
            PyGraphistry.relogin = lambda: PyGraphistry.sso_login(
                org_name, idp_name, sso_timeout
            )

        PyGraphistry._is_authenticated = False
        arrow_uploader = ArrowUploader(
            server_base_path=PyGraphistry.protocol()
            + "://"                     # noqa: W503
            + PyGraphistry.server(),    # noqa: W503
            certificate_validation=PyGraphistry.certificate_validation(),
        ).sso_login(org_name, idp_name)

        try:
            if arrow_uploader.token:
                PyGraphistry.api_token(arrow_uploader.token)
                PyGraphistry._is_authenticated = True
                arrow_uploader.token = None
                return PyGraphistry.api_token()
        except Exception:  # required to log on
            # print("required to log on")
            PyGraphistry.sso_state(arrow_uploader.sso_state)

            auth_url = arrow_uploader.sso_auth_url
            # print("auth_url : {}".format(auth_url))
            if auth_url and not PyGraphistry.api_token():
                PyGraphistry._handle_auth_url(auth_url, sso_timeout)


    @staticmethod
    def _handle_auth_url(auth_url, sso_timeout):
        """Internal function to handle what to do with the auth_url 
           based on the client mode python/ipython console or notebook.

        :param auth_url: SSO auth url retrieved via API
        :type auth_url: str
        :param sso_timeout: Set sso login getting token timeout in seconds (blocking mode), set to None if non-blocking mode. Default as SSO_GET_TOKEN_ELAPSE_SECONDS.
        :type sso_timeout: Optional[int]
        :returns: None.
        :rtype: None

        SSO Login logic.

        """

        if in_ipython() or in_databricks():  # If run in notebook, just display the HTML
            # from IPython.core.display import HTML
            from IPython.display import display, HTML
            display(HTML(f'<a href="{auth_url}" target="_blank">Login SSO</a>'))
            print("Please click the above link to open browser to login")
            print("Please close browser tab after SSO login to back to notebook")
            # return HTML(make_iframe(auth_url, 20, extra_html=extra_html, override_html_style=override_html_style))
        else:
            print("Please minimize browser after SSO login to back to pygraphistry")

            import webbrowser
            input("Press Enter to open browser ...")
            # open browser to auth_url
            webbrowser.open(auth_url)

        if sso_timeout is not None:
            time.sleep(1)
            elapsed_time = 1
            token = None
            
            while True:
                token, org_name = PyGraphistry._sso_get_token()
                try:
                    if not token:
                        if elapsed_time % 10 == 1:
                            print("Waiting for token : {} seconds ...".format(sso_timeout - elapsed_time + 1))

                        time.sleep(1)
                        elapsed_time = elapsed_time + 1
                        if elapsed_time > sso_timeout:
                            raise SsoRetrieveTokenTimeoutException("[SSO] Get token timeout")
                    else:
                        break
                except SsoRetrieveTokenTimeoutException as toe:
                    logger.debug(toe, exc_info=1)
                    break
                except Exception:
                    token = None
            if token:
                # set org_name to sso org
                PyGraphistry._config['org_name'] = org_name

                print("Successfully get a token")
                return PyGraphistry.api_token()
            else:
                return None
        else:
            print("Please run graphistry.sso_get_token() to complete the authentication")


    @staticmethod
    def sso_get_token():
        """ Get authentication token in SSO non-blocking mode"""
        token, org_name = PyGraphistry._sso_get_token()
        # set org_name to sso org
        PyGraphistry._config['org_name'] = org_name
        return token
    
    @staticmethod
    def _sso_get_token():
        token = None
        # get token from API using state
        state = PyGraphistry.sso_state()
        # print("_sso_get_token : {}".format(state))
        arrow_uploader = ArrowUploader(
            server_base_path=PyGraphistry.protocol()
            + "://"                     # noqa: W503
            + PyGraphistry.server(),    # noqa: W503
            certificate_validation=PyGraphistry.certificate_validation(),
        ).sso_get_token(state)

        try:
            try:
                token = arrow_uploader.token
                org_name = arrow_uploader.org_name
            except Exception:
                pass
            logger.debug("jwt token :{}".format(token))
            # print("jwt token :{}".format(token))
            PyGraphistry.api_token(token or PyGraphistry._config['api_token'])
            # print("api_token() : {}".format(PyGraphistry.api_token()))
            PyGraphistry._is_authenticated = True
            token = PyGraphistry.api_token()
            # print("api_token() : {}".format(token))
            return token, org_name
        except:
            # raise
            pass
        return None, None

    @staticmethod
    def refresh(token=None, fail_silent=False):
        """Use self or provided JWT token to get a fresher one. If self token, internalize upon refresh."""
        using_self_token = token is None
        logger.debug("1. @PyGraphistry refresh, org_name: {}".format(PyGraphistry._config['org_name']))
        try:
            if PyGraphistry.store_token_creds_in_memory():
                logger.debug("JWT refresh via creds")
                logger.debug("2. @PyGraphistry refresh :relogin")
                return PyGraphistry.relogin()

            logger.debug("JWT refresh via token")
            if using_self_token:
                PyGraphistry._is_authenticated = False
            token = (
                ArrowUploader(
                    server_base_path=PyGraphistry.protocol()
                    + "://"                   # noqa: W503
                    + PyGraphistry.server(),  # noqa: W503
                    certificate_validation=PyGraphistry.certificate_validation(),
                )
                .refresh(PyGraphistry.api_token() if using_self_token else token)
                .token
            )
            if using_self_token:
                PyGraphistry.api_token(token)
                PyGraphistry._is_authenticated = True
            return PyGraphistry.api_token()
        except Exception as e:
            if not fail_silent:
                util.error("Failed to refresh token: %s" % str(e))

    @staticmethod
    def verify_token(token=None, fail_silent=False) -> bool:
        """Return True iff current or provided token is still valid"""
        using_self_token = token is None
        try:
            logger.debug("JWT refresh")
            if using_self_token:
                PyGraphistry._is_authenticated = False
            ok = ArrowUploader(
                server_base_path=PyGraphistry.protocol()
                + "://"                   # noqa: W503
                + PyGraphistry.server(),  # noqa: W503
                certificate_validation=PyGraphistry.certificate_validation(),
            ).verify(PyGraphistry.api_token() if using_self_token else token)
            if using_self_token:
                PyGraphistry._is_authenticated = ok
            return ok
        except Exception as e:
            if not fail_silent:
                util.error("Failed to verify token: %s" % str(e))
            return False

    @staticmethod
    def server(value=None):
        """Get the hostname of the server or set the server using hostname or aliases.
        Also set via environment variable GRAPHISTRY_HOSTNAME."""
        if value is None:
            return PyGraphistry._config["hostname"]

        # setter
        shortcuts = {}
        if value in shortcuts:
            resolved = shortcuts[value]
            PyGraphistry._config["hostname"] = resolved
            util.warn("Resolving alias %s to %s" % (value, resolved))
        else:
            PyGraphistry._config["hostname"] = value

    @staticmethod
    def store_token_creds_in_memory(value=None):
        """Cache credentials for JWT token access. Default off due to not being safe."""
        if value is None:
            return PyGraphistry._config["store_token_creds_in_memory"]
        else:
            v = bool(strtobool(value)) if isinstance(value, str) else value
            PyGraphistry._config["store_token_creds_in_memory"] = v

    @staticmethod
    def client_protocol_hostname(value=None):
        """Get/set the client protocol+hostname for when display urls (distinct from uploading).
        Also set via environment variable GRAPHISTRY_CLIENT_PROTOCOL_HOSTNAME.
        Defaults to hostname and no protocol (reusing environment protocol)"""

        if value is None:
            cfg_client_protocol_hostname = PyGraphistry._config[
                "client_protocol_hostname"
            ]
            # skip doing protocol by default to match notebook's protocol
            cph = (
                ("//" + PyGraphistry.server())
                if cfg_client_protocol_hostname is None
                else cfg_client_protocol_hostname
            )
            return cph
        else:
            PyGraphistry._config["client_protocol_hostname"] = value

    @staticmethod
    def api_key(value=None):
        """Set or get the API key.
        Also set via environment variable GRAPHISTRY_API_KEY."""

        if value is None:
            return PyGraphistry._config["api_key"]

        # setter
        if value is not PyGraphistry._config["api_key"]:
            PyGraphistry._config["api_key"] = value.strip()
            PyGraphistry._is_authenticated = False

    @staticmethod
    def api_token(value=None):
        """Set or get the API token.
        Also set via environment variable GRAPHISTRY_API_TOKEN."""

        if value is None:
            return PyGraphistry._config["api_token"]

        # setter
        if value is not PyGraphistry._config["api_token"]:
            PyGraphistry._config["api_token"] = value.strip()
            PyGraphistry._is_authenticated = False

    @staticmethod
    def api_token_refresh_ms(value=None):
        """Set or get the API token refresh interval in milliseconds.
        None and 0 interpreted as no refreshing."""

        if value is None:
            return PyGraphistry._config["api_token_refresh_ms"]

        # setter
        if value is not PyGraphistry._config["api_token_refresh_ms"]:
            PyGraphistry._config["api_token_refresh_ms"] = int(value)

    @staticmethod
    def protocol(value=None):
        """Set or get the protocol ('http' or 'https').
        Set automatically when using a server alias.
        Also set via environment variable GRAPHISTRY_PROTOCOL."""
        if value is None:
            return PyGraphistry._config["protocol"]
        # setter
        PyGraphistry._config["protocol"] = value

    @staticmethod
    def api_version(value=None):
        """Set or get the API version: 1 for 1.0 (deprecated), 3 for 2.0.
        Setting api=2 (protobuf) fully deprecated from the PyGraphistry client.
        Also set via environment variable GRAPHISTRY_API_VERSION."""
        
        import re
        if value is None:
            #if set by env var, interpret
            env_api_version = PyGraphistry._config["api_version"]
            if isinstance(env_api_version, str):
                if re.sub(r'\d+', '', env_api_version) == '':
                    value = int(env_api_version)
                else:
                    raise ValueError("Expected API version to be 1, 3, instead got (likely from GRAPHISTRY_API_VERSION): %s" % env_api_version)
            else:
                value = env_api_version

        if value not in [1, 3]:
            raise ValueError("Expected API version to be 1, 3, instead got: %s" % value)

        # setter
        PyGraphistry._config["api_version"] = value

        return value

    @staticmethod
    def certificate_validation(value=None):
        """Enable/Disable SSL certificate validation (True, False).
        Also set via environment variable GRAPHISTRY_CERTIFICATE_VALIDATION."""
        if value is None:
            return PyGraphistry._config["certificate_validation"]

        # setter
        v = bool(strtobool(value)) if isinstance(value, str) else value
        if not v:
            requests.packages.urllib3.disable_warnings()
        PyGraphistry._config["certificate_validation"] = v

    @staticmethod
    def set_bolt_driver(driver=None):
        PyGraphistry._config["bolt_driver"] = bolt_util.to_bolt_driver(driver)

    @staticmethod
    def register(
        key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        personal_key_id: Optional[str] = None,
        personal_key_secret: Optional[str] = None,
        server: Optional[str] = None,
        protocol: Optional[str] = None,
        api: Optional[Literal[1, 3]] = None,
        certificate_validation: Optional[bool] = None,
        bolt: Optional[Union[Dict, Any]] = None,
        token_refresh_ms: int = 10 * 60 * 1000,
        store_token_creds_in_memory: Optional[bool] = None,
        client_protocol_hostname: Optional[str] = None,
        org_name: Optional[str] = None,
        idp_name: Optional[str] = None,
        is_sso_login: Optional[bool] = False,
        sso_timeout: Optional[int] = SSO_GET_TOKEN_ELAPSE_SECONDS
    ):
        """API key registration and server selection

        Changing the key effects all derived Plotter instances.

        Provide one of key (deprecated api=1), username/password (api=3) or temporary token (api=3).

        :param key: API key (deprecated 1.0 API)
        :type key: Optional[str]
        :param username: Account username (2.0 API).
        :type username: Optional[str]
        :param password: Account password (2.0 API).
        :type password: Optional[str]
        :param token: Valid Account JWT token (2.0). Provide token, or username/password, but not both.
        :type token: Optional[str]
        :param personal_key_id: Personal Key id for service account.
        :type personal_key_id: Optional[str]
        :param personal_key_secret: Personal Key secret for service account.
        :type personal_key_secret: Optional[str]
        :param server: URL of the visualization server.
        :type server: Optional[str]
        :param protocol: Protocol to use for server uploaders, defaults to "https".
        :type protocol: Optional[str]
        :param api: API version to use, defaults to 1 (deprecated slow json 1.0 API), prefer 3 (2.0 API with Arrow+JWT)
        :type api: Optional[Literal[1, 3]]
        :param certificate_validation: Override default-on check for valid TLS certificate by setting to True.
        :type certificate_validation: Optional[bool]
        :param bolt: Neo4j bolt information. Optional driver or named constructor arguments for instantiating a new one.
        :type bolt: Union[dict, Any]
        :param protocol: Protocol used to contact visualization server, defaults to "https".
        :type protocol: Optional[str]
        :param token_refresh_ms: Ignored for now; JWT token auto-refreshed on plot() calls.
        :type token_refresh_ms: int
        :param store_token_creds_in_memory: Store username/password in-memory for JWT token refreshes (Token-originated have a hard limit, so always-on requires creds somewhere)
        :type store_token_creds_in_memory: Optional[bool]
        :param client_protocol_hostname: Override protocol and host shown in browser. Defaults to protocol/server or envvar GRAPHISTRY_CLIENT_PROTOCOL_HOSTNAME.
        :type client_protocol_hostname: Optional[str]
        :param org_name: Set login organization's name(slug). Defaults to user's personal organization.
        :type org_name: Optional[str]
        :param idp_name: Set sso login idp name. Default as None (for site-wide SSO / for the only idp record).
        :type idp_name: Optional[str]
        :param sso_timeout: Set sso login getting token timeout in seconds (blocking mode), set to None if non-blocking mode. Default as SSO_GET_TOKEN_ELAPSE_SECONDS.
        :type sso_timeout: Optional[int]
        :returns: None.
        :rtype: None

        **Example: Standard (2.0 api by org_name via SSO configured for site or for organization with only 1 IdP)**
                ::

                    import graphistry
                    graphistry.register(api=3, protocol='http', server='200.1.1.1', org_name="org-name", idp_name="idp-name")

        **Example: Standard (2.0 api by org_name via SSO IdP configured for an organization)**
                ::

                    import graphistry
                    graphistry.register(api=3, protocol='http', server='200.1.1.1', org_name="org-name")

        **Example: Standard (2.0 api by username/password with org_name)**
                ::

                    import graphistry
                    graphistry.register(api=3, protocol='http', server='200.1.1.1', username='person', password='pwd', org_name="org-name")

        **Example: Standard (2.0 api by username/password) without org_name**
                ::

                    import graphistry
                    graphistry.register(api=3, protocol='http', server='200.1.1.1', username='person', password='pwd')

        **Example: Standard (2.0 api by token)**
                ::

                    import graphistry
                    graphistry.register(api=3, protocol='http', server='200.1.1.1', token='abc')

        **Example: Standard (by personal_key_id/personal_key_secret)**
                ::

                    import graphistry
                    graphistry.register(api=3, protocol='http', server='200.1.1.1', personal_key_id='ZD5872XKNF', personal_key_secret='SA0JJ2DTVT6LLO2S')

        **Example: Remote browser to Graphistry-provided notebook server (2.0)**
                ::

                    import graphistry
                    graphistry.register(api=3, protocol='http', server='nginx', client_protocol_hostname='https://my.site.com', token='abc')

        **Example: Standard (1.0)**
                ::

                    import graphistry
                    graphistry.register(api=1, key="my api key")

        """
        PyGraphistry.api_version(api)
        PyGraphistry.api_token_refresh_ms(token_refresh_ms)
        PyGraphistry.api_key(key)
        PyGraphistry.server(server)
        PyGraphistry.protocol(protocol)
        PyGraphistry.client_protocol_hostname(client_protocol_hostname)
        PyGraphistry.certificate_validation(certificate_validation)
        PyGraphistry.store_token_creds_in_memory(store_token_creds_in_memory)
        PyGraphistry.set_bolt_driver(bolt)
        # Reset token creds
        PyGraphistry.__reset_token_creds_in_memory()
 
        if not (username is None) and not (password is None):
            PyGraphistry.login(username, password, org_name)
            PyGraphistry.api_token(token or PyGraphistry._config['api_token'])
            PyGraphistry.authenticate()
        elif (username is None and not (password is None)):
            raise Exception(MSG_REGISTER_MISSING_USERNAME)
        elif not (username is None) and password is None:
            raise Exception(MSG_REGISTER_MISSING_PASSWORD)
        elif not (personal_key_id is None) and not (personal_key_secret is None):
            PyGraphistry.pkey_login(personal_key_id, personal_key_secret, org_name=org_name)
            PyGraphistry.api_token(token or PyGraphistry._config['api_token'])
            PyGraphistry.authenticate()
        elif personal_key_id is None and not (personal_key_secret is None):
            raise Exception(MSG_REGISTER_MISSING_PKEY_ID)
        elif not (personal_key_id is None) and personal_key_secret is None:
            raise Exception(MSG_REGISTER_MISSING_PKEY_SECRET)
        elif not (token is None):
            PyGraphistry.api_token(token or PyGraphistry._config['api_token'])
        elif not (org_name is None) or is_sso_login:
            print(MSG_REGISTER_ENTER_SSO_LOGIN)
            PyGraphistry.sso_login(org_name, idp_name, sso_timeout=sso_timeout)

    @staticmethod
    def __check_login_type_to_reset_token_creds(
            origin_login_type: str,
            new_login_type: str,
        ):
        if origin_login_type != new_login_type:
            PyGraphistry.__reset_token_creds_in_memory()
        
    @staticmethod
    def privacy(
            mode: Optional[str] = None,
            notify: Optional[bool] = None,
            invited_users: Optional[List] = None,
            mode_action: Optional[str] = None,
            message: Optional[str] = None
        ):
        """Set global default sharing mode

        :param mode: Either "private" or "public" or "organization"
        :type mode: str
        :param notify: Whether to email the recipient(s) upon upload
        :type notify: bool
        :param invited_users: List of recipients, where each is {"email": str, "action": str} and action is "10" (view) or "20" (edit)
        :type invited_users: List
        :param mode_action: Only used when mode="organization", action for sharing within organization, "10" (view) or "20" (edit), default is "20"
        :type mode_action: str

        Requires an account with sharing capabilities.

        Shared datasets will appear in recipients' galleries.

        If mode is set to "private", only accounts in invited_users list can access. Mode "public" permits viewing by any user with the URL.

        Action "10" (view) gives read access, while action "20" (edit) gives edit access, like changing the sharing mode.

        When notify is true, uploads will trigger notification emails to invitees. Email will use visualization's ".name()"

        **Example: Limit visualizations to current user**

            ::

                import graphistry
                graphistry.register(api=3, username='myuser', password='mypassword')
                graphistry.privacy()  # default uploads to mode="private"

                #Subsequent uploads default to using .privacy() settings
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, direct=True)
                g = h['graph'].plot()


        **Example: Default to publicly viewable visualizations**

            ::

                import graphistry
                graphistry.register(api=3, username='myuser', password='mypassword')
                #graphistry.privacy(mode="public")  # can skip calling .privacy() for this default

                #Subsequent uploads default to using .privacy() settings
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, direct=True)
                g = h['graph'].plot()


        **Example: Default to sharing with select teammates, and keep notifications opt-in**

            ::

                import graphistry
                graphistry.register(api=3, username='myuser', password='mypassword')
                graphistry.privacy(
                    mode="private",
                    invited_users=[
                        {"email": "friend1@acme.org", "action": "10"}, # view
                        {"email": "friend2@acme.org", "action": "20"}, # edit
                    ],
                    notify=False)

                #Subsequent uploads default to using .privacy() settings
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, direct=True)
                g = h['graph'].plot()


        **Example: Keep visualizations public and email notifications upon upload**

            ::

                import graphistry
                graphistry.register(api=3, username='myuser', password='mypassword')
                graphistry.privacy(
                    mode="public",
                    invited_users=[
                        {"email": "friend1@acme.org", "action": "10"}, # view
                        {"email": "friend2@acme.org", "action": "20"}, # edit
                    ],
                    notify=True)

                #Subsequent uploads default to using .privacy() settings
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, direct=True)
                g = h['graph']
                g = g.name('my cool viz')  # For friendlier invitations
                g.plot()
        """

        PyGraphistry._config['privacy'] = {
            'mode': mode,
            'notify': notify,
            'invited_users': invited_users,
            'mode_action': mode_action,
            'message': message
        }

    @staticmethod
    def hypergraph(
        raw_events,
        entity_types: Optional[List[str]] = None,
        opts: dict = {},
        drop_na: bool = True,
        drop_edge_attrs: bool = False,
        verbose: bool = True,
        direct: bool = False,
        engine: str = "pandas",
        npartitions: Optional[int] = None,
        chunksize: Optional[int] = None,
    ):
        """Transform a dataframe into a hypergraph.

        :param raw_events: Dataframe to transform (pandas or cudf).
        :type raw_events: pandas.DataFrame
        :param Optional[list] entity_types: Columns (strings) to turn into nodes, None signifies all
        :param dict opts: See below
        :param bool drop_edge_attrs: Whether to include each row's attributes on its edges, defaults to False (include)
        :param bool verbose: Whether to print size information
        :param bool direct: Omit hypernode and instead strongly connect nodes in an event
        :param bool engine: String (pandas, cudf, ...) for engine to use
        :param Optional[int] npartitions: For distributed engines, how many coarse-grained pieces to split events into
        :param Optional[int] chunksize: For distributed engines, split events after chunksize rows

        Create a graph out of the dataframe, and return the graph components as dataframes,
        and the renderable result Plotter. Hypergraphs reveal relationships between rows and between column values.
        This transform is useful for lists of events, samples, relationships, and other structured high-dimensional data.

        Specify local compute engine by passing `engine='pandas'`, 'cudf', 'dask', 'dask_cudf' (default: 'pandas').
        If events are not in that engine's format, they will be converted into it.

        The transform creates a node for every unique value in the entity_types columns (default: all columns).
        If direct=False (default), every row is also turned into a node.
        Edges are added to connect every table cell to its originating row's node, or if direct=True, to the other nodes from the same row.
        Nodes are given the attribute 'type' corresponding to the originating column name, or in the case of a row, 'EventID'.
        Options further control the transform, such column category definitions for controlling whether values
        reocurring in different columns should be treated as one node,
        or whether to only draw edges between certain column type pairs.

        Consider a list of events. Each row represents a distinct event, and each column some metadata about an event.
        If multiple events have common metadata, they will be transitively connected through those metadata values.
        The layout algorithm will try to cluster the events together.
        Conversely, if an event has unique metadata, the unique metadata will turn into nodes that only have connections to the event node, and the clustering algorithm will cause them to form a ring around the event node.

        Best practice is to set EVENTID to a row's unique ID,
        SKIP to all non-categorical columns (or entity_types to all categorical columns),
        and CATEGORY to group columns with the same kinds of values.

        To prevent creating nodes for null values, set drop_na=True.
        Some dataframe engines may have undesirable null handling,
        and recommend replacing None values with np.nan .

        The optional ``opts={...}`` configuration options are:

        * 'EVENTID': Column name to inspect for a row ID. By default, uses the row index.
        * 'CATEGORIES': Dictionary mapping a category name to inhabiting columns. E.g., {'IP': ['srcAddress', 'dstAddress']}.  If the same IP appears in both columns, this makes the transform generate one node for it, instead of one for each column.
        * 'DELIM': When creating node IDs, defines the separator used between the column name and node value
        * 'SKIP': List of column names to not turn into nodes. For example, dates and numbers are often skipped.
        * 'EDGES': For direct=True, instead of making all edges, pick column pairs. E.g., {'a': ['b', 'd'], 'd': ['d']} creates edges between columns a->b and a->d, and self-edges d->d.


        :returns: {'entities': DF, 'events': DF, 'edges': DF, 'nodes': DF, 'graph': Plotter}
        :rtype: dict

        **Example: Connect user<-row->boss**

            ::

                import graphistry
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df)
                g = h['graph'].plot()

        **Example: Connect user->boss**

            ::

                import graphistry
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, direct=True)
                g = h['graph'].plot()

        **Example: Connect user<->boss**

            ::

                import graphistry
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, direct=True, opts={'EDGES': {'user': ['boss'], 'boss': ['user']}})
                g = h['graph'].plot()

        **Example: Only consider some columns for nodes**

            ::

                import graphistry
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, entity_types=['boss'])
                g = h['graph'].plot()

        **Example: Collapse matching user::<id> and boss::<id> nodes into one person::<id> node**

            ::

                import graphistry
                users_df = pd.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_df, opts={'CATEGORIES': {'person': ['user', 'boss']}})
                g = h['graph'].plot()

        **Example: Use cudf engine instead of pandas**

            ::

                import cudf, graphistry
                users_gdf = cudf.DataFrame({'user': ['a','b','x'], 'boss': ['x', 'x', 'y']})
                h = graphistry.hypergraph(users_gdf, engine='cudf')
                g = h['graph'].plot()

        """
        from . import hyper

        return hyper.Hypergraph().hypergraph(
            PyGraphistry,
            raw_events,
            entity_types,
            opts,
            drop_na,
            drop_edge_attrs,
            verbose,
            direct,
            engine=engine,
            npartitions=npartitions,
            chunksize=chunksize,
        )

    @staticmethod
    def infer_labels(self):
        """

        :return: Plotter w/neo4j

        * Prefers point_title/point_label if available
        * Fallback to node id
        * Raises exception if no nodes available, no likely candidates, and no matching node id fallback

        **Example**

                ::

                    import graphistry
                    g = graphistry.nodes(pd.read_csv('nodes.csv'), 'id_col').infer_labels()
                    g.plot()

        """
        return Plotter().infer_labels()

    @staticmethod
    def bolt(driver=None):
        """

        :param driver: Neo4j Driver or arguments for GraphDatabase.driver(**{...})**
        :return: Plotter w/neo4j

        Call this to create a Plotter with an overridden neo4j driver.

        **Example**

                ::

                    import graphistry
                    g = graphistry.bolt({ server: 'bolt://...', auth: ('<username>', '<password>') })

                ::

                    import neo4j
                    import graphistry

                    driver = neo4j.GraphDatabase.driver(...)

                    g = graphistry.bolt(driver)
        """
        return Plotter().bolt(driver)

    @staticmethod
    def cypher(query, params={}):
        """

        :param query: a cypher query
        :param params: cypher query arguments
        :return: Plotter with data from a cypher query. This call binds `source`, `destination`, and `node`.

        Call this to immediately execute a cypher query and store the graph in the resulting Plotter.

                ::

                    import graphistry
                    g = graphistry.bolt({ query='MATCH (a)-[r:PAYMENT]->(b) WHERE r.USD > 7000 AND r.USD < 10000 RETURN r ORDER BY r.USD DESC', params={ "AccountId": 10 })
        """
        return Plotter().cypher(query, params)

    @staticmethod
    def nodexl(xls_or_url, source="default", engine=None, verbose=False):
        """

        :param xls_or_url: file/http path string to a nodexl-generated xls, or a pandas ExcelFile() object
        :param source: optionally activate binding by string name for a known nodexl data source ('twitter', 'wikimedia')
        :param engine: optionally set a pandas Excel engine
        :param verbose: optionally enable printing progress by overriding to True

        """

        if not (engine is None):
            print("WARNING: Engine currently ignored, please contact if critical")

        return Plotter().nodexl(xls_or_url, source, engine, verbose)

    @staticmethod
    def gremlin(queries: Union[str, Iterable[str]]) -> Plottable:
        """Run one or more gremlin queries and get back the result as a graph object
        To support cosmosdb, sends as strings

        **Example: Login and plot**

            ::

                import graphistry
                (graphistry
                    .gremlin_client(my_gremlin_client)
                    .gremlin('g.E().sample(10)')
                    .fetch_nodes()  # Fetch properties for nodes
                    .plot())

        """
        return Plotter().gremlin(queries)

    @staticmethod
    def neptune(
        NEPTUNE_READER_HOST: Optional[str] = None,
        NEPTUNE_READER_PORT: Optional[str] = None,
        NEPTUNE_READER_PROTOCOL: Optional[str] = "wss",
        endpoint: Optional[str] = None,
        gremlin_client: Optional[Any] = None,
    ) -> Plotter:
        """
           Provide credentials as arguments, as environment variables, or by providing a gremlinpython client
           Environment variable names are the same as the constructor argument names
           If endpoint provided, do not need host/port/protocol
           If no client provided, create (connect)

        **Example: Login and plot via parrams**

            ::

                import graphistry
                (graphistry
                    .neptune(
                        NEPTUNE_READER_PROTOCOL='wss'
                        NEPTUNE_READER_HOST='neptunedbcluster-xyz.cluster-ro-abc.us-east-1.neptune.amazonaws.com'
                        NEPTUNE_READER_PORT='8182'
                    )
                    .gremlin('g.E().sample(10)')
                    .fetch_nodes()  # Fetch properties for nodes
                    .plot())

        **Example: Login and plot via env vars**

            ::

                import graphistry
                (graphistry
                    .neptune()
                    .gremlin('g.E().sample(10)')
                    .fetch_nodes()  # Fetch properties for nodes
                    .plot())

        **Example: Login and plot via endpoint**

            ::

                import graphistry
                (graphistry
                    .neptune(endpoint='wss://neptunedbcluster-xyz.cluster-ro-abc.us-east-1.neptune.amazonaws.com:8182/gremlin')
                    .gremlin('g.E().sample(10)')
                    .fetch_nodes()  # Fetch properties for nodes
                    .plot())

        **Example: Login and plot via client**

            ::

                import graphistry
                (graphistry
                    .neptune(gremlin_client=client)
                    .gremlin('g.E().sample(10)')
                    .fetch_nodes()  # Fetch properties for nodes
                    .plot())
        """
        return Plotter().neptune(
            NEPTUNE_READER_HOST=NEPTUNE_READER_HOST,
            NEPTUNE_READER_PORT=NEPTUNE_READER_PORT,
            NEPTUNE_READER_PROTOCOL=NEPTUNE_READER_PROTOCOL,
            endpoint=endpoint,
            gremlin_client=gremlin_client,
        )

    @staticmethod
    def cosmos(
        COSMOS_ACCOUNT: Optional[str] = None,
        COSMOS_DB: Optional[str] = None,
        COSMOS_CONTAINER: Optional[str] = None,
        COSMOS_PRIMARY_KEY: Optional[str] = None,
        gremlin_client: Any = None,
    ) -> Plotter:
        """Provide credentials as arguments, as environment variables, or by providing a gremlinpython client
        Environment variable names are the same as the constructor argument names
        If no client provided, create (connect)

        :param COSMOS_ACCOUNT: cosmos account
        :param COSMOS_DB: cosmos db name
        :param COSMOS_CONTAINER: cosmos container name
        :param COSMOS_PRIMARY_KEY: cosmos key
        :param gremlin_client: optional prebuilt client
        :return: Plotter with data from a cypher query. This call binds `source`, `destination`, and `node`.

        **Example: Login and plot**

            ::

                import graphistry
                (graphistry
                    .cosmos(
                        COSMOS_ACCOUNT='a',
                        COSMOS_DB='b',
                        COSMOS_CONTAINER='c',
                        COSMOS_PRIMARY_KEY='d')
                    .gremlin('g.E().sample(10)')
                    .fetch_nodes()  # Fetch properties for nodes
                    .plot())

        """
        return Plotter().cosmos(
            COSMOS_ACCOUNT=COSMOS_ACCOUNT,
            COSMOS_DB=COSMOS_DB,
            COSMOS_CONTAINER=COSMOS_CONTAINER,
            COSMOS_PRIMARY_KEY=COSMOS_PRIMARY_KEY,
            gremlin_client=gremlin_client,
        )

    @staticmethod
    def gremlin_client(gremlin_client: Any = None) -> Plotter:
        """Pass in a generic gremlin python client

        **Example: Login and plot**

            ::

                import graphistry
                from gremlin_python.driver.client import Client

                my_gremlin_client = Client(
                f'wss://MY_ACCOUNT.gremlin.cosmosdb.azure.com:443/',
                'g',
                username=f"/dbs/MY_DB/colls/{self.COSMOS_CONTAINER}",
                password=self.COSMOS_PRIMARY_KEY,
                message_serializer=GraphSONSerializersV2d0())

                (graphistry
                    .gremlin_client(my_gremlin_client)
                    .gremlin('g.E().sample(10)')
                    .fetch_nodes()  # Fetch properties for nodes
                    .plot())

        """
        return Plotter().gremlin_client(gremlin_client=gremlin_client)

    @staticmethod
    def drop_graph() -> Plotter:
        """
        Remove all graph nodes and edges from the database
        """
        return Plotter().drop_graph()

    @staticmethod
    def name(name):
        """Upload name

        :param name: Upload name
        :type name: str"""

        return Plotter().name(name)

    @staticmethod
    def description(description):
        """Upload description

        :param description: Upload description
        :type description: str"""

        return Plotter().description(description)

    @staticmethod
    def addStyle(bg=None, fg=None, logo=None, page=None):
        """Creates a base plotter with some style settings.

        For parameters, see ``plotter.addStyle``.

        :returns: Plotter
        :rtype: Plotter

        **Example**

            ::

                import graphistry
                graphistry.addStyle(bg={'color': 'black'})
        """

        return Plotter().addStyle(bg=bg, fg=fg, logo=logo, page=page)

    @staticmethod
    def style(bg=None, fg=None, logo=None, page=None):
        """Creates a base plotter with some style settings.

        For parameters, see ``plotter.style``.

        :returns: Plotter
        :rtype: Plotter

        **Example**

            ::

                import graphistry
                graphistry.style(bg={'color': 'black'})
        """

        return Plotter().style(bg=bg, fg=fg, logo=logo, page=page)

    @staticmethod
    def encode_point_color(
        column,
        palette=None,
        as_categorical=None,
        as_continuous=None,
        categorical_mapping=None,
        default_mapping=None,
        for_default=True,
        for_current=False,
    ):
        """Set point color with more control than bind()

        :param column: Data column name
        :type column: str

        :param palette: Optional list of color-like strings. Ex: ["black, "#FF0", "rgb(255,255,255)" ]. Used as a gradient for continuous and round-robin for categorical.
        :type palette: Optional[list]

        :param as_categorical: Interpret column values as categorical. Ex: Uses palette via round-robin when more values than palette entries.
        :type as_categorical: Optional[bool]

        :param as_continuous: Interpret column values as continuous. Ex: Uses palette for an interpolation gradient when more values than palette entries.
        :type as_continuous: Optional[bool]

        :param categorical_mapping: Mapping from column values to color-like strings. Ex: {"car": "red", "truck": #000"}
        :type categorical_mapping: Optional[dict]

        :param default_mapping: Augment categorical_mapping with mapping for values not in categorical_mapping. Ex: default_mapping="gray".
        :type default_mapping: Optional[str]

        :param for_default: Use encoding for when no user override is set. Default on.
        :type for_default: Optional[bool]

        :param for_current: Use encoding as currently active. Clearing the active encoding resets it to default, which may be different. Default on.
        :type for_current: Optional[bool]

        :returns: Plotter
        :rtype: Plotter

        **Example: Set a palette-valued column for the color, same as bind(point_color='my_column')**
            ::

                g2a = g.encode_point_color('my_int32_palette_column')
                g2b = g.encode_point_color('my_int64_rgb_column')

        **Example: Set a cold-to-hot gradient of along the spectrum blue, yellow, red**
            ::

                g2 = g.encode_point_color('my_numeric_col', palette=["blue", "yellow", "red"], as_continuous=True)

        **Example: Round-robin sample from 5 colors in hex format**
            ::

                g2 = g.encode_point_color('my_distinctly_valued_col', palette=["#000", "#00F", "#0F0", "#0FF", "#FFF"], as_categorical=True)

        **Example: Map specific values to specific colors, including with a default**
            ::

                g2a = g.encode_point_color('brands', categorical_mapping={'toyota': 'red', 'ford': 'blue'})
                g2a = g.encode_point_color('brands', categorical_mapping={'toyota': 'red', 'ford': 'blue'}, default_mapping='gray')

        """

        return Plotter().encode_point_color(
            column=column,
            palette=palette,
            as_categorical=as_categorical,
            as_continuous=as_continuous,
            categorical_mapping=categorical_mapping,
            default_mapping=default_mapping,
            for_default=for_default,
            for_current=for_current,
        )

    @staticmethod
    def encode_edge_color(
        column,
        palette=None,
        as_categorical=None,
        as_continuous=None,
        categorical_mapping=None,
        default_mapping=None,
        for_default=True,
        for_current=False,
    ):
        """Set edge color with more control than bind()

        :param column: Data column name
        :type column: str

        :param palette: Optional list of color-like strings. Ex: ["black, "#FF0", "rgb(255,255,255)" ]. Used as a gradient for continuous and round-robin for categorical.
        :type palette: Optional[list]

        :param as_categorical: Interpret column values as categorical. Ex: Uses palette via round-robin when more values than palette entries.
        :type as_categorical: Optional[bool]

        :param as_continuous: Interpret column values as continuous. Ex: Uses palette for an interpolation gradient when more values than palette entries.
        :type as_continuous: Optional[bool]

        :param categorical_mapping: Mapping from column values to color-like strings. Ex: {"car": "red", "truck": #000"}
        :type categorical_mapping: Optional[dict]

        :param default_mapping: Augment categorical_mapping with mapping for values not in categorical_mapping. Ex: default_mapping="gray".
        :type default_mapping: Optional[str]

        :param for_default: Use encoding for when no user override is set. Default on.
        :type for_default: Optional[bool]

        :param for_current: Use encoding as currently active. Clearing the active encoding resets it to default, which may be different. Default on.
        :type for_current: Optional[bool]

        :returns: Plotter
        :rtype: Plotter

        **Example: See encode_point_color**
        """

        return Plotter().encode_edge_color(
            column=column,
            palette=palette,
            as_categorical=as_categorical,
            as_continuous=as_continuous,
            categorical_mapping=categorical_mapping,
            default_mapping=default_mapping,
            for_default=for_default,
            for_current=for_current,
        )

    @staticmethod
    def encode_point_size(
        column,
        categorical_mapping=None,
        default_mapping=None,
        for_default=True,
        for_current=False,
    ):
        """Set point size with more control than bind()

        :param column: Data column name
        :type column: str

        :param categorical_mapping: Mapping from column values to numbers. Ex: {"car": 100, "truck": 200}
        :type categorical_mapping: Optional[dict]

        :param default_mapping: Augment categorical_mapping with mapping for values not in categorical_mapping. Ex: default_mapping=50.
        :type default_mapping: Optional[Union[int,float]]

        :param for_default: Use encoding for when no user override is set. Default on.
        :type for_default: Optional[bool]

        :param for_current: Use encoding as currently active. Clearing the active encoding resets it to default, which may be different. Default on.
        :type for_current: Optional[bool]

        :returns: Plotter
        :rtype: Plotter

        **Example: Set a numerically-valued column for the size, same as bind(point_size='my_column')**
            ::

                g2a = g.encode_point_size('my_numeric_column')

        **Example: Map specific values to specific colors, including with a default**
            ::

                g2a = g.encode_point_size('brands', categorical_mapping={'toyota': 100, 'ford': 200})
                g2b = g.encode_point_size('brands', categorical_mapping={'toyota': 100, 'ford': 200}, default_mapping=50)

        """

        return Plotter().encode_point_size(
            column=column,
            categorical_mapping=categorical_mapping,
            default_mapping=default_mapping,
            for_default=for_default,
            for_current=for_current,
        )

    @staticmethod
    def encode_point_icon(
        column,
        categorical_mapping=None,
        continuous_binning=None,
        default_mapping=None,
        comparator=None,
        for_default=True,
        for_current=False,
        as_text=False,
        blend_mode=None,
        style=None,
        border=None,
        shape=None,
    ):
        """Set node icon with more control than bind(). Values from Font Awesome 4 such as "laptop": https://fontawesome.com/v4.7.0/icons/

        :param column: Data column name
        :type column: str

        :param categorical_mapping: Mapping from column values to icon name strings. Ex: {"toyota": 'car', "ford": 'truck'}
        :type categorical_mapping: Optional[dict]

        :param default_mapping: Augment categorical_mapping with mapping for values not in categorical_mapping. Ex: default_mapping=50.
        :type default_mapping: Optional[Union[int,float]]

        :param for_default: Use encoding for when no user override is set. Default on.
        :type for_default: Optional[bool]

        :param for_current: Use encoding as currently active. Clearing the active encoding resets it to default, which may be different. Default on.
        :type for_current: Optional[bool]

        :param as_text: Values should instead be treated as raw strings, instead of icons and images. (Default False.)
        :type as_text: Optional[bool]

        :param blend_mode: CSS blend mode
        :type blend_mode: Optional[str]

        :param style: CSS filter properties - opacity, saturation, luminosity, grayscale, and more
        :type style: Optional[dict]

        :param border: Border properties - 'width', 'color', and 'storke'
        :type border: Optional[dict]

        :returns: Plotter
        :rtype: Plotter

        **Example: Set a string column of icons for the point icons, same as bind(point_icon='my_column')**
            ::

                g2a = g.encode_point_icon('my_icons_column')

        **Example: Map specific values to specific icons, including with a default**
            ::

                g2a = g.encode_point_icon('brands', categorical_mapping={'toyota': 'car', 'ford': 'truck'})
                g2b = g.encode_point_icon('brands', categorical_mapping={'toyota': 'car', 'ford': 'truck'}, default_mapping='question')

        **Example: Map countries to abbreviations**
            ::

                g2b = g.encode_point_icon('country_abbrev', as_text=True)
                g2b = g.encode_point_icon('country', as_text=True, categorical_mapping={'England': 'UK', 'America': 'US'}, default_mapping='')

        **Example: Border**
            ::

                g2b = g.encode_point_icon('country', border={'width': 3, color: 'black', 'stroke': 'dashed'}, 'categorical_mapping={'England': 'UK', 'America': 'US'})

        """

        return Plotter().encode_point_icon(
            column=column,
            categorical_mapping=categorical_mapping,
            continuous_binning=continuous_binning,
            default_mapping=default_mapping,
            comparator=comparator,
            for_default=for_default,
            for_current=for_current,
            as_text=as_text,
            blend_mode=blend_mode,
            style=style,
            border=border,
            shape=shape,
        )

    @staticmethod
    def encode_edge_icon(
        column,
        categorical_mapping=None,
        continuous_binning=None,
        default_mapping=None,
        comparator=None,
        for_default=True,
        for_current=False,
        as_text=False,
        blend_mode=None,
        style=None,
        border=None,
        shape=None,
    ):
        """Set edge icon with more control than bind(). Values from Font Awesome 4 such as "laptop": https://fontawesome.com/v4.7.0/icons/

        :param column: Data column name
        :type column: str

        :param categorical_mapping: Mapping from column values to icon name strings. Ex: {"toyota": 'car', "ford": 'truck'}
        :type categorical_mapping: Optional[dict]

        :param default_mapping: Augment categorical_mapping with mapping for values not in categorical_mapping. Ex: default_mapping=50.
        :type default_mapping: Optional[Union[int,float]]

        :param for_default: Use encoding for when no user override is set. Default on.
        :type for_default: Optional[bool]

        :param for_current: Use encoding as currently active. Clearing the active encoding resets it to default, which may be different. Default on.
        :type for_current: Optional[bool]

        :param as_text: Values should instead be treated as raw strings, instead of icons and images. (Default False.)
        :type as_text: Optional[bool]

        :param blend_mode: CSS blend mode
        :type blend_mode: Optional[str]

        :param style: CSS filter properties - opacity, saturation, luminosity, grayscale, and more
        :type style: Optional[dict]

        :param border: Border properties - 'width', 'color', and 'storke'
        :type border: Optional[dict]

        :returns: Plotter
        :rtype: Plotter

        **Example: Set a string column of icons for the edge icons, same as bind(edge_icon='my_column')**
            ::

                g2a = g.encode_edge_icon('my_icons_column')

        **Example: Map specific values to specific icons, including with a default**
            ::

                g2a = g.encode_edge_icon('brands', categorical_mapping={'toyota': 'car', 'ford': 'truck'})
                g2b = g.encode_edge_icon('brands', categorical_mapping={'toyota': 'car', 'ford': 'truck'}, default_mapping='question')

        **Example: Map countries to abbreviations**
            ::

                g2a = g.encode_edge_icon('country_abbrev', as_text=True)
                g2b = g.encode_edge_icon('country', categorical_mapping={'England': 'UK', 'America': 'US'}, default_mapping='')

        **Example: Border**
            ::

                g2b = g.encode_edge_icon('country', border={'width': 3, color: 'black', 'stroke': 'dashed'}, 'categorical_mapping={'England': 'UK', 'America': 'US'})

        """

        return Plotter().encode_edge_icon(
            column=column,
            categorical_mapping=categorical_mapping,
            continuous_binning=continuous_binning,
            default_mapping=default_mapping,
            comparator=comparator,
            for_default=for_default,
            for_current=for_current,
            as_text=as_text,
            blend_mode=blend_mode,
            style=style,
            border=border,
            shape=shape,
        )

    @staticmethod
    def encode_edge_badge(
        column,
        position="TopRight",
        categorical_mapping=None,
        continuous_binning=None,
        default_mapping=None,
        comparator=None,
        color=None,
        bg=None,
        fg=None,
        for_current=False,
        for_default=True,
        as_text=None,
        blend_mode=None,
        style=None,
        border=None,
        shape=None,
    ):

        return Plotter().encode_edge_badge(
            column=column,
            categorical_mapping=categorical_mapping,
            continuous_binning=continuous_binning,
            default_mapping=default_mapping,
            comparator=comparator,
            color=color,
            bg=bg,
            fg=fg,
            for_current=for_current,
            for_default=for_default,
            as_text=as_text,
            blend_mode=blend_mode,
            style=style,
            border=border,
            shape=shape,
        )

    @staticmethod
    def encode_point_badge(
        column,
        position="TopRight",
        categorical_mapping=None,
        continuous_binning=None,
        default_mapping=None,
        comparator=None,
        color=None,
        bg=None,
        fg=None,
        for_current=False,
        for_default=True,
        as_text=None,
        blend_mode=None,
        style=None,
        border=None,
        shape=None,
    ):

        return Plotter().encode_point_badge(
            column=column,
            categorical_mapping=categorical_mapping,
            continuous_binning=continuous_binning,
            default_mapping=default_mapping,
            comparator=comparator,
            color=color,
            bg=bg,
            fg=fg,
            for_current=for_current,
            for_default=for_default,
            as_text=as_text,
            blend_mode=blend_mode,
            style=style,
            border=border,
            shape=shape,
        )

    @staticmethod
    def bind(
        node=None,
        source=None,
        destination=None,
        edge_title=None,
        edge_label=None,
        edge_color=None,
        edge_weight=None,
        edge_icon=None,
        edge_size=None,
        edge_opacity=None,
        edge_source_color=None,
        edge_destination_color=None,
        point_title=None,
        point_label=None,
        point_color=None,
        point_weight=None,
        point_icon=None,
        point_size=None,
        point_opacity=None,
        point_x=None,
        point_y=None,
    ):
        """Create a base plotter.

        Typically called at start of a program. For parameters, see ``plotter.bind()`` .

        :returns: Plotter
        :rtype: Plotter

        **Example**

                ::

                    import graphistry
                    g = graphistry.bind()

        """

        return Plotter().bind(
            source=source,
            destination=destination,
            node=node,
            edge_title=edge_title,
            edge_label=edge_label,
            edge_color=edge_color,
            edge_size=edge_size,
            edge_weight=edge_weight,
            edge_icon=edge_icon,
            edge_opacity=edge_opacity,
            edge_source_color=edge_source_color,
            edge_destination_color=edge_destination_color,
            point_title=point_title,
            point_label=point_label,
            point_color=point_color,
            point_size=point_size,
            point_weight=point_weight,
            point_icon=point_icon,
            point_opacity=point_opacity,
            point_x=point_x,
            point_y=point_y,
        )

    @staticmethod
    def tigergraph(
        protocol="http",
        server="localhost",
        web_port=14240,
        api_port=9000,
        db=None,
        user="tigergraph",
        pwd="tigergraph",
        verbose=False,
    ):
        """Register Tigergraph connection setting defaults

        :param protocol: Protocol used to contact the database.
        :type protocol: Optional[str]
        :param server: Domain of the database
        :type server: Optional[str]
        :param web_port:
        :type web_port: Optional[int]
        :param api_port:
        :type api_port: Optional[int]
        :param db: Name of the database
        :type db: Optional[str]
        :param user:
        :type user: Optional[str]
        :param pwd:
        :type pwd: Optional[str]
        :param verbose: Whether to print operations
        :type verbose: Optional[bool]
        :returns: Plotter
        :rtype: Plotter


        **Example: Standard**
                ::

                    import graphistry
                    tg = graphistry.tigergraph(protocol='https', server='acme.com', db='my_db', user='alice', pwd='tigergraph2')

        """

        return Plotter().tigergraph(
            protocol, server, web_port, api_port, db, user, pwd, verbose
        )

    @staticmethod
    def gsql_endpoint(
        self, method_name, args={}, bindings=None, db=None, dry_run=False
    ):
        """Invoke Tigergraph stored procedure at a user-definend endpoint and return transformed Plottable

        :param method_name: Stored procedure name
        :type method_name: str
        :param args: Named endpoint arguments
        :type args: Optional[dict]
        :param bindings: Mapping defining names of returned 'edges' and/or 'nodes', defaults to @@nodeList and @@edgeList
        :type bindings: Optional[dict]
        :param db: Name of the database, defaults to value set in .tigergraph(...)
        :type db: Optional[str]
        :param dry_run: Return target URL without running
        :type dry_run: bool
        :returns: Plotter
        :rtype: Plotter

        **Example: Minimal**
                ::

                    import graphistry
                    tg = graphistry.tigergraph(db='my_db')
                    tg.gsql_endpoint('neighbors').plot()

        **Example: Full**
                ::

                    import graphistry
                    tg = graphistry.tigergraph()
                    tg.gsql_endpoint('neighbors', {'k': 2}, {'edges': 'my_edge_list'}, 'my_db').plot()

        **Example: Read data**
                ::

                    import graphistry
                    tg = graphistry.tigergraph()
                    out = tg.gsql_endpoint('neighbors')
                    (nodes_df, edges_df) = (out._nodes, out._edges)

        """

        return Plotter().gsql_endpoint(method_name, args, bindings, db, dry_run)

    @staticmethod
    def gsql(query, bindings=None, dry_run=False):
        """Run Tigergraph query in interpreted mode and return transformed Plottable

         :param query: Code to run
         :type query: str
         :param bindings: Mapping defining names of returned 'edges' and/or 'nodes', defaults to @@nodeList and @@edgeList
         :type bindings: Optional[dict]
         :param dry_run: Return target URL without running
         :type dry_run: bool
         :returns: Plotter
         :rtype: Plotter

         **Example: Minimal**
                 ::

                     import graphistry
                     tg = graphistry.tigergraph()
                     tg.gsql(\"\"\"
                     INTERPRET QUERY () FOR GRAPH Storage {

                         OrAccum<BOOL> @@stop;
                         ListAccum<EDGE> @@edgeList;
                         SetAccum<vertex> @@set;

                         @@set += to_vertex("61921", "Pool");

                         Start = @@set;

                         while Start.size() > 0 and @@stop == false do

                         Start = select t from Start:s-(:e)-:t
                         where e.goUpper == TRUE
                         accum @@edgeList += e
                         having t.type != "Service";
                         end;

                         print @@edgeList;
                     }
                     \"\"\").plot()

        **Example: Full**
                 ::

                     import graphistry
                     tg = graphistry.tigergraph()
                     tg.gsql(\"\"\"
                     INTERPRET QUERY () FOR GRAPH Storage {

                         OrAccum<BOOL> @@stop;
                         ListAccum<EDGE> @@edgeList;
                         SetAccum<vertex> @@set;

                         @@set += to_vertex("61921", "Pool");

                         Start = @@set;

                         while Start.size() > 0 and @@stop == false do

                         Start = select t from Start:s-(:e)-:t
                         where e.goUpper == TRUE
                         accum @@edgeList += e
                         having t.type != "Service";
                         end;

                         print @@my_edge_list;
                     }
                     \"\"\", {'edges': 'my_edge_list'}).plot()
        """

        return Plotter().gsql(query, bindings, dry_run)

    @staticmethod
    def nodes(nodes: Union[Callable, Any], node=None, *args, **kwargs) -> Plottable:
        """Specify the set of nodes and associated data.
        If a callable, will be called with current Plotter and whatever positional+named arguments

        Must include any nodes referenced in the edge list.

        :param nodes: Nodes and their attributes.
        :type nodes: Pandas dataframe or Callable

        :returns: Plotter
        :rtype: Plotter

        **Example**
            ::

                import graphistry

                es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                g = graphistry
                    .bind(source='src', destination='dst')
                    .edges(es)

                vs = pandas.DataFrame({'v': [0,1,2], 'lbl': ['a', 'b', 'c']})
                g = g.bind(node='v').nodes(vs)

                g.plot()

        **Example**
            ::

                import graphistry

                es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                g = graphistry.edges(es, 'src', 'dst')

                vs = pandas.DataFrame({'v': [0,1,2], 'lbl': ['a', 'b', 'c']})
                g = g.nodes(vs, 'v)

                g.plot()


        **Example**
            ::
                import graphistry

                def sample_nodes(g, n):
                    return g._nodes.sample(n)

                df = pandas.DataFrame({'id': [0,1,2], 'v': [1,2,0]})

                graphistry
                    .nodes(df, 'id')
                    ..nodes(sample_nodes, n=2)
                    ..nodes(sample_nodes, None, 2)  # equivalent
                    .plot()

        """
        return Plotter().nodes(nodes, node, *args, **kwargs)

    @staticmethod
    def edges(
        edges: Union[Callable, Any], source=None, destination=None, *args, **kwargs
    ) -> Plottable:
        """Specify edge list data and associated edge attribute values.
        If a callable, will be called with current Plotter and whatever positional+named arguments

        :param edges: Edges and their attributes, or transform from Plotter to edges
        :type edges: Pandas dataframe, NetworkX graph, or IGraph graph

        :returns: Plotter
        :rtype: Plotter

        **Example**
            ::

                import graphistry
                df = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                graphistry
                    .bind(source='src', destination='dst')
                    .edges(df)
                    .plot()

        **Example**
            ::

                import graphistry
                df = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
                graphistry
                    .edges(df, 'src', 'dst')
                    .plot()

        **Example**
            ::
                import graphistry

                def sample_edges(g, n):
                    return g._edges.sample(n)

                df = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})

                graphistry
                    .edges(df, 'src', 'dst')
                    .edges(sample_edges, n=2)
                    .edges(sample_edges, None, None, 2)  # equivalent
                    .plot()

        """
        return Plotter().edges(edges, source, destination, *args, **kwargs)

    @staticmethod
    def pipe(graph_transform: Callable, *args, **kwargs) -> Plottable:
        """Create new Plotter derived from current

        :param graph_transform:
        :type graph_transform: Callable

        **Example: Simple**
            ::

                import graphistry

                def fill_missing_bindings(g, source='src', destination='dst):
                    return g.bind(source=source, destination=destination)

                graphistry
                    .edges(pandas.DataFrame({'src': [0,1,2], 'd': [1,2,0]}))
                    .pipe(fill_missing_bindings, destination='d')  # binds 'src'
                    .plot()
        """

        return Plotter().pipe(graph_transform, *args, **kwargs)

    @staticmethod
    def graph(ig):

        return Plotter().graph(ig)

    @staticmethod
    def from_igraph(ig,
        node_attributes: Optional[List[str]] = None,
        edge_attributes: Optional[List[str]] = None,
        load_nodes = True, load_edges = True
    ):
        return Plotter().from_igraph(ig, node_attributes, edge_attributes, load_nodes, load_edges)
    from_igraph.__doc__ = Plotter.from_igraph.__doc__

    @staticmethod
    def from_cugraph(
        G,
        node_attributes: Optional[List[str]] = None,
        edge_attributes: Optional[List[str]] = None,
        load_nodes: bool = True, load_edges: bool = True,
        merge_if_existing: bool = True
    ):
        return Plotter().from_cugraph(G, node_attributes, edge_attributes, load_nodes, load_edges, merge_if_existing)
    from_cugraph.__doc__ = Plotter.from_cugraph.__doc__

    @staticmethod
    def settings(height=None, url_params={}, render=None):

        return Plotter().settings(height, url_params, render)

    @staticmethod
    def _etl_url():
        hostname = PyGraphistry._config["hostname"]
        protocol = PyGraphistry._config["protocol"]
        return "%s://%s/etl" % (protocol, hostname)

    @staticmethod
    def _check_url():
        hostname = PyGraphistry._config["hostname"]
        protocol = PyGraphistry._config["protocol"]
        return "%s://%s/api/check" % (protocol, hostname)

    @staticmethod
    def _viz_url(info, url_params):
        splash_time = int(calendar.timegm(time.gmtime())) + 15
        extra = "&".join([k + "=" + str(v) for k, v in list(url_params.items())])
        cph = PyGraphistry.client_protocol_hostname()
        pattern = "%s/graph/graph.html?dataset=%s&type=%s&viztoken=%s&usertag=%s&splashAfter=%s&%s"
        return pattern % (
            cph,
            info["name"],
            info["type"],
            info["viztoken"],
            PyGraphistry._tag,
            splash_time,
            extra,
        )

    @staticmethod
    def _switch_org_url(org_name):
        hostname = PyGraphistry._config["hostname"]
        protocol = PyGraphistry._config["protocol"]
        return "{}://{}/api/v2/o/{}/switch/".format(protocol, hostname, org_name)


    @staticmethod
    def _coerce_str(v):
        try:
            return str(v)
        except UnicodeDecodeError:
            print("UnicodeDecodeError")
            print("=", v, "=")
            x = v.decode("utf-8")
            print("x", x)
            return x

    @staticmethod
    def _get_data_file(dataset, mode):
        out_file = io.BytesIO()
        if mode == "json":
            json_dataset = None
            try:
                json_dataset = json.dumps(
                    dataset, ensure_ascii=False, cls=NumpyJSONEncoder
                )
            except TypeError:
                warnings.warn("JSON: Switching from NumpyJSONEncoder to str()")
                json_dataset = json.dumps(dataset, default=PyGraphistry._coerce_str)

            with gzip.GzipFile(fileobj=out_file, mode="w", compresslevel=9) as f:
                if sys.version_info < (3, 0) and isinstance(json_dataset, bytes):
                    f.write(json_dataset)
                else:
                    f.write(json_dataset.encode("utf8"))
        else:
            raise ValueError("Unknown mode:", mode)

        kb_size = len(out_file.getvalue()) // 1024
        if kb_size >= 5 * 1024:
            print("Uploading %d kB. This may take a while..." % kb_size)
            sys.stdout.flush()

        return out_file

    @staticmethod
    def _etl1(dataset):
        PyGraphistry.authenticate()

        headers = {"Content-Encoding": "gzip", "Content-Type": "application/json"}
        params = {
            "usertag": PyGraphistry._tag,
            "agent": "pygraphistry",
            "apiversion": "1",
            "agentversion": sys.modules["graphistry"].__version__,
            "key": PyGraphistry.api_key(),
        }

        out_file = PyGraphistry._get_data_file(dataset, "json")
        response = requests.post(
            PyGraphistry._etl_url(),
            out_file.getvalue(),
            headers=headers,
            params=params,
            verify=PyGraphistry._config["certificate_validation"],
        )
        response.raise_for_status()

        try:
            jres = response.json()
        except Exception:
            raise ValueError("Unexpected server response", response)

        if jres["success"] is not True:
            raise ValueError("Server reported error:", jres["msg"])
        else:
            return {
                "name": jres["dataset"],
                "viztoken": jres["viztoken"],
                "type": "vgraph",
            }


    @staticmethod
    def _check_key_and_version():
        params = {"text": PyGraphistry.api_key()}
        try:
            response = requests.get(
                PyGraphistry._check_url(),
                params=params,
                timeout=(3, 3),
                verify=PyGraphistry._config["certificate_validation"],
            )
            response.raise_for_status()
            jres = response.json()

            cver = sys.modules["graphistry"].__version__
            if (
                "pygraphistry" in jres
                and "minVersion" in jres["pygraphistry"]     # noqa: W503
                and "latestVersion" in jres["pygraphistry"]  # noqa: W503
            ):
                mver = jres["pygraphistry"]["minVersion"]
                lver = jres["pygraphistry"]["latestVersion"]

                from packaging.version import parse
                try:
                    if parse(mver) > parse(cver):
                        util.warn(
                            "Your version of PyGraphistry is no longer supported (installed=%s latest=%s). Please upgrade!"
                            % (cver, lver)
                        )
                    elif parse(lver) > parse(cver):
                        print(
                            "A new version of PyGraphistry is available (installed=%s latest=%s)."
                            % (cver, lver)
                        )
                except:
                    raise ValueError(f'Unexpected version value format when comparing {mver}, {cver}, and {lver}')
            if jres["success"] is not True:
                util.warn(jres["error"])
        except Exception:
            util.warn(
                "Could not contact %s. Are you connected to the Internet?"
                % PyGraphistry._config["hostname"]
            )

    @staticmethod
    def layout_settings(
        play: Optional[int] = None,
        locked_x: Optional[bool] = None,
        locked_y: Optional[bool] = None,
        locked_r: Optional[bool] = None,
        left: Optional[float] = None,
        top: Optional[float] = None,
        right: Optional[float] = None,
        bottom: Optional[float] = None,
        lin_log: Optional[bool] = None,
        strong_gravity: Optional[bool] = None,
        dissuade_hubs: Optional[bool] = None,
        edge_influence: Optional[float] = None,
        precision_vs_speed: Optional[float] = None,
        gravity: Optional[float] = None,
        scaling_ratio: Optional[float] = None,
    ):
        """Set layout options. Additive over previous settings.

        Corresponds to options at https://hub.graphistry.com/docs/api/1/rest/url/#urloptions

        **Example: Animated radial layout**

            ::

                import graphistry, pandas as pd
                edges = pd.DataFrame({'s': ['a','b','c','d'], 'boss': ['c','c','e','e']})
                nodes = pd.DataFrame({
                    'n': ['a', 'b', 'c', 'd', 'e'],
                    'y': [1,   1,   2,   3,   4],
                    'x': [1,   1,   0,   0,   0],
                })
                g = (graphistry
                    .edges(edges, 's', 'd')
                    .nodes(nodes, 'n')
                    .layout_settings(locked_r=True, play=2000)
                g.plot()
        """
        return Plotter().layout_settings(
            play,
            locked_x,
            locked_y,
            locked_r,
            left,
            top,
            right,
            bottom,
            lin_log,
            strong_gravity,
            dissuade_hubs,
            edge_influence,
            precision_vs_speed,
            gravity,
            scaling_ratio,
        )

    @staticmethod
    def org_name(value=None):
        """Set or get the org_name when register/login.
        """

        if value is None:
            if 'org_name' in PyGraphistry._config:
                return PyGraphistry._config['org_name']
            return None

        # setter, use switch_org instead
        if 'org_name' not in PyGraphistry._config or value is not PyGraphistry._config['org_name']:
            try: 
                PyGraphistry.switch_org(value.strip())
                # PyGraphistry._config['org_name'] = value.strip()
            except:
                raise Exception("Failed to switch organization")

    @staticmethod
    def idp_name(value=None):
        """Set or get the idp_name when register/login.
        """

        if value is None:
            if 'idp_name' in PyGraphistry._config:
                return PyGraphistry._config['idp_name']
            return None

        # setter
        if 'idp_name' not in PyGraphistry._config or value is not PyGraphistry._config['idp_name']:
            PyGraphistry._config['idp_name'] = value.strip()


    @staticmethod
    def sso_state(value=None):
        """Set or get the sso_state when register/sso login.
        """

        if value is None:
            if 'sso_state' in PyGraphistry._config:
                return PyGraphistry._config['sso_state']
            return None

        # setter
        if 'sso_state' not in PyGraphistry._config or value is not PyGraphistry._config['sso_state']:
            PyGraphistry._config['sso_state'] = value.strip()

    @staticmethod
    def scene_settings(
        menu: Optional[bool] = None,
        info: Optional[bool] = None,
        show_arrows: Optional[bool] = None,
        point_size: Optional[float] = None,
        edge_curvature: Optional[float] = None,
        edge_opacity: Optional[float] = None,
        point_opacity: Optional[float] = None,        
    ):
        return Plotter().scene_settings(
            menu,
            info,
            show_arrows,
            point_size,
            edge_curvature,
            edge_opacity,
            point_opacity
        )
    scene_settings.__doc__ = Plotter().scene_settings.__doc__


    @staticmethod
    def personal_key_id(value: Optional[str] = None):
        """Set or get the personal_key_id when register.
        """

        if value is None:
            if 'personal_key_id' in PyGraphistry._config:
                return PyGraphistry._config['personal_key_id']
            return None

        # setter
        if 'personal_key_id' not in PyGraphistry._config or value is not PyGraphistry._config['personal_key_id']:
            PyGraphistry._config['personal_key_id'] = value.strip()

    @staticmethod
    def personal_key_secret(value: Optional[str] = None):
        """Set or get the personal_key_secret when register.
        """

        if value is None:
            if 'personal_key_secret' in PyGraphistry._config:
                return PyGraphistry._config['personal_key_secret']
            return None

        # setter
        if 'personal_key_secret' not in PyGraphistry._config or value is not PyGraphistry._config['personal_key']:
            PyGraphistry._config['personal_key_secret'] = value.strip()

    @staticmethod
    def switch_org(value):
        # print(PyGraphistry._switch_org_url(value))
        response = requests.post(
            PyGraphistry._switch_org_url(value),
            data={'slug': value},
            headers={'Authorization': f'Bearer {PyGraphistry.api_token()}'},
            verify=PyGraphistry._config["certificate_validation"],
        )
        result = PyGraphistry._handle_api_response(response)

        if result is True:
            PyGraphistry._config['org_name'] = value.strip()
            logger.info("Switched to organization: {}".format(value.strip()))
        else:  # print the error message
            raise Exception(result)

    @staticmethod
    def _handle_api_response(response):
        try:
            json_response = response.json()
            if json_response.get('status', None) == 'OK':
                return True
            else:
                return json_response.get('message', '')
        except:
            logger.error('Error: %s', response, exc_info=True)
            raise Exception("Unknown Error")

        


client_protocol_hostname = PyGraphistry.client_protocol_hostname
store_token_creds_in_memory = PyGraphistry.store_token_creds_in_memory
server = PyGraphistry.server
protocol = PyGraphistry.protocol
register = PyGraphistry.register
sso_get_token = PyGraphistry.sso_get_token
privacy = PyGraphistry.privacy
login = PyGraphistry.login
refresh = PyGraphistry.refresh
api_token = PyGraphistry.api_token
verify_token = PyGraphistry.verify_token
bind = PyGraphistry.bind
addStyle = PyGraphistry.addStyle
style = PyGraphistry.style
encode_point_color = PyGraphistry.encode_point_color
encode_edge_color = PyGraphistry.encode_edge_color
encode_point_size = PyGraphistry.encode_point_size
encode_point_icon = PyGraphistry.encode_point_icon
encode_edge_icon = PyGraphistry.encode_edge_icon
encode_point_badge = PyGraphistry.encode_point_badge
encode_edge_badge = PyGraphistry.encode_edge_badge
infer_labels = PyGraphistry.infer_labels
name = PyGraphistry.name
description = PyGraphistry.description
edges = PyGraphistry.edges
nodes = PyGraphistry.nodes
pipe = PyGraphistry.pipe
graph = PyGraphistry.graph
settings = PyGraphistry.settings
hypergraph = PyGraphistry.hypergraph
bolt = PyGraphistry.bolt
cypher = PyGraphistry.cypher
nodexl = PyGraphistry.nodexl
tigergraph = PyGraphistry.tigergraph
cosmos = PyGraphistry.cosmos
neptune = PyGraphistry.neptune
gremlin = PyGraphistry.gremlin
gremlin_client = PyGraphistry.gremlin_client
drop_graph = PyGraphistry.drop_graph
gsql_endpoint = PyGraphistry.gsql_endpoint
gsql = PyGraphistry.gsql
layout_settings = PyGraphistry.layout_settings
org_name = PyGraphistry.org_name
idp_name = PyGraphistry.idp_name
sso_state = PyGraphistry.sso_state
scene_settings = PyGraphistry.scene_settings
from_igraph = PyGraphistry.from_igraph
from_cugraph = PyGraphistry.from_cugraph
personal_key_id = PyGraphistry.personal_key_id
personal_key_secret = PyGraphistry.personal_key_secret
switch_org = PyGraphistry.switch_org



class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, type(pd.NaT)):
            return None
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)
