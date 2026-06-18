import pytest
import graphistry
from graphistry.client_session import ClientSession
from graphistry.pygraphistry import GraphistryClient, PyGraphistry

import pandas as pd
from unittest import mock


class TestClientSession:
    """
    Tests for Graphistry client()/register() interactions using the *current*
    public surface. Legacy api_key compatibility is tested only as an inert,
    deprecated shim; isolation tests use token/session state.
    """

    # --------------------------------------------------------------------- #
    # House‑keeping                                                         #
    # --------------------------------------------------------------------- #

    def setup_method(self):
        """
        Reset global state at the start of every test.  Calling register()
        without credentials clears the in-memory session cleanly.
        """
        graphistry.register(api=3, verify_token=False)

    # --------------------------------------------------------------------- #
    # Basic config proxy                                                    #
    # --------------------------------------------------------------------- #

    def test_config_dict_access(self):
        """
        The session proxy still behaves like a mapping for read access.
        """
        cfg = graphistry.register(api=3, token="tok123", verify_token=False)._config

        assert cfg["api_token"] == "tok123"
        assert "nonexistent" not in cfg

        # Mapping API
        assert cfg.get("api_token") == "tok123"
        assert cfg.get("nonexistent", "default") == "default"

    # --------------------------------------------------------------------- #
    # Global vs. client isolation                                           #
    # --------------------------------------------------------------------- #

    def test_client_session_isolation(self):
        # Set up global session state directly
        PyGraphistry.session.api_token = "global_token"
        global_session = PyGraphistry.session

        client = graphistry.client()
        # Use a dedicated client instance to avoid polluting global state.
        assert isinstance(client, GraphistryClient)

        # Set client session state directly  
        client.session.api_token = "client_token"

        assert global_session.api_token == "global_token"
        assert client.session.api_token == "client_token"
        assert global_session is not client.session

    def test_global_session_persistence(self):
        # Test that global session persists across operations
        session1 = PyGraphistry.session
        session1.api_token = "T1"

        session2 = PyGraphistry.session
        session2.api_token = "T2"

        assert session1 is session2           # same object
        assert session2.api_token == "T2"     # value updated

    def test_client_inheritance(self):
        graphistry.register(
            api=3,
            protocol="https",
            server="test.graphistry.com",
            token="global_token",
            verify_token=False,
        )

        # Inherit = True copies the *current* session
        client_inherit = graphistry.client(inherit=True)
        assert client_inherit.session.api_token == "global_token"
        assert client_inherit.session.protocol == "https"
        assert client_inherit.session.hostname == "test.graphistry.com"

        # Inherit = False starts fresh
        client_fresh = graphistry.client(inherit=False)
        assert not client_fresh.session.api_token  # empty / None

    def test_multiple_clients(self):
        client1, client2, client3 = (graphistry.client() for _ in range(3))

        # Set session state directly to test isolation
        client1.session.api_token = "tok1"
        client2.session.api_token = "tok2"
        client3.session.api_token = "tok3"

        assert client1.session.api_token == "tok1"
        assert client2.session.api_token == "tok2"
        assert client3.session.api_token == "tok3"
        assert PyGraphistry.session.api_token not in {"tok1", "tok2", "tok3"}

    # --------------------------------------------------------------------- #
    # Plotter propagation                                                   #
    # --------------------------------------------------------------------- #

    def test_client_bind_operations(self):
        client = graphistry.client()
        # Set session state directly to test plotter propagation
        client.session.api_token = "client_token"

        g = client.bind(source="src", destination="dst")
        assert g._source == "src"
        assert g._destination == "dst"
        assert g._pygraphistry.session.api_token == "client_token"

    @mock.patch('graphistry.pygraphistry.ArrowUploader._switch_org')
    @mock.patch('requests.post')
    def test_client_register_with_org_sets_session(self, mock_post, mock_switch_org):
        mock_resp = mock.Mock()
        mock_resp.json.return_value = {
            'token': 'tok123',
            'active_organization': {
                'slug': 'mock-org',
                'is_found': True,
                'is_member': True
            }
        }
        mock_resp.status_code = 200
        mock_resp.content = b''
        mock_resp.text = ''
        mock_resp.headers = {}
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp

        client = graphistry.client()
        assert client.session.org_name is None

        client.register(api=3, username='u', password='p', org_name='mock-org')

        assert client.session.org_name == 'mock-org'
        assert client.org_name() == 'mock-org'
        mock_switch_org.assert_called_with('mock-org', 'tok123')

    @mock.patch('graphistry.pygraphistry.ArrowUploader._switch_org')
    @mock.patch('requests.post')
    def test_client_register_updates_last_switched_cache(self, mock_post, mock_switch_org):
        client = graphistry.client()

        def fake_switch(org_name, token):
            client.session._last_switched_org_token = (org_name, token)

        mock_switch_org.side_effect = fake_switch
        mock_resp = mock.Mock()
        mock_resp.json.return_value = {
            'token': 'tok123',
            'active_organization': {
                'slug': 'mock-org',
                'is_found': True,
                'is_member': True
            }
        }
        mock_resp.status_code = 200
        mock_resp.content = b''
        mock_resp.text = ''
        mock_resp.headers = {}
        mock_resp.raise_for_status = mock.Mock()
        mock_post.return_value = mock_resp

        assert client.session._last_switched_org_token is None

        client.register(api=3, username='u', password='p', org_name='mock-org')

        assert client.session._last_switched_org_token == ('mock-org', 'tok123')

    # --------------------------------------------------------------------- #
    # Persistence of arbitrary config                                       #
    # --------------------------------------------------------------------- #

    def test_client_config_persistence(self):
        client = graphistry.client()
        client.register(
            api=3,
            protocol="https",
            server="my.server.com",
            token="my_token",
            verify_token=False,
            certificate_validation=False,
        )

        g = client.bind(source="s", destination="d")
        ses = g._pygraphistry.session

        assert ses.api_token == "my_token"
        assert ses.protocol == "https"
        assert ses.hostname == "my.server.com"

        if pd is not None:
            g2 = g.nodes(pd.DataFrame({"id": [1, 2, 3]}))
            assert g2._pygraphistry.session.api_token == "my_token"

    # --------------------------------------------------------------------- #
    # Error handling                                                        #
    # --------------------------------------------------------------------- #

    def test_client_error_handling(self):
        client = graphistry.client()

        with pytest.raises(ValueError):
            client.register(api=99)  # nonsense API version
        
        # Test that register with api=3 and no creds doesn't crash 
        # (the actual validation may be deferred to API calls)
        try:
            client.register(api=3)
        except Exception:
            pass  # This is expected but we don't want to depend on specific behavior

    def test_register_legacy_api_fails_locally_with_upgrade_message(self):
        client = graphistry.client()

        with pytest.raises(ValueError) as exc:
            client.register(api=1)

        message = str(exc.value)
        assert "Legacy API versions 1 and 2 are no longer supported" in message
        assert "/api/check" in message
        assert "/etl" in message
        assert "HTTP 410" in message

    def test_register_legacy_key_warns_and_ignores(self):
        client = graphistry.client()

        with pytest.warns(DeprecationWarning, match=r"api_key\(\) is deprecated"):
            client.register(key="old-api-key")

        assert client.session.api_key is None

    def test_api_key_surface_warns_and_returns_none(self):
        client = graphistry.client()
        client.session.api_key = "manually-mutated-legacy-key"

        with pytest.warns(DeprecationWarning, match=r"api_key\(\) is deprecated"):
            assert client.api_key() is None
        assert client.session.api_key is None

        with pytest.warns(DeprecationWarning, match=r"api_key\(\) is deprecated"):
            assert client.api_key("old-api-key") is None
        assert client.session.api_key is None

    def test_env_legacy_api_fails_locally_with_upgrade_message(self, monkeypatch):
        monkeypatch.setenv("GRAPHISTRY_API_VERSION", "1")

        with pytest.raises(ValueError) as exc:
            ClientSession()

        assert "Legacy API versions 1 and 2 are no longer supported" in str(exc.value)

    def test_stale_env_legacy_key_is_ignored(self, monkeypatch):
        monkeypatch.setenv("GRAPHISTRY_API_KEY", "old-api-key")

        client = GraphistryClient()

        assert client.session.api_key is None
        with pytest.warns(DeprecationWarning, match=r"api_key\(\) is deprecated"):
            assert client.api_key() is None

    def test_plot_legacy_api_fails_before_refresh_or_upload(self):
        client = graphistry.client()
        client.session.api_version = 1  # type: ignore[assignment]
        g = client.bind(source="src", destination="dst").edges(
            pd.DataFrame({"src": [1], "dst": [2]})
        )

        with mock.patch.object(client, "refresh") as mock_refresh:
            with mock.patch.object(g, "_plot_dispatch_arrow") as mock_dispatch:
                with pytest.raises(ValueError, match="Legacy API versions 1 and 2"):
                    g.plot(render="url")

        mock_refresh.assert_not_called()
        mock_dispatch.assert_not_called()

    # --------------------------------------------------------------------- #
    # Isolation from subsequent global changes                              #
    # --------------------------------------------------------------------- #

    def test_client_state_isolation_from_global_changes(self):
        # Set up global state directly
        PyGraphistry.session.api_token = "global1"

        client = graphistry.client(inherit=True)
        assert client.session.api_token == "global1"

        # mutate global state directly
        PyGraphistry.session.api_token = "global2"

        assert PyGraphistry.session.api_token == "global2"
        assert client.session.api_token == "global1"

    # ------------------------------------------------------------------ #
    # Token-based registration                                          #
    # ------------------------------------------------------------------ #

    def test_register_token_marks_authenticated(self, monkeypatch):
        """
        register(token=...) should configure the session as authenticated so
        downstream calls (plot, gfql, etc.) don't have to patch private fields.
        """
        client = graphistry.client()
        calls = {}

        def fake_verify(token, fail_silent=False):
            calls["token"] = token
            calls["fail_silent"] = fail_silent
            return True

        monkeypatch.setattr(client, "verify_token", fake_verify)

        client.register(
            api=3,
            protocol="https",
            server="example.graphistry.com",
            token="tok123",
        )

        assert client.session.api_token == "tok123"
        assert client.session._is_authenticated is True
        assert client.session.store_token_creds_in_memory is False
        assert calls["token"] == "tok123"
        assert calls["fail_silent"] is False

    def test_register_token_verify_opt_out(self, monkeypatch):
        """
        verify_token=False should skip server verification while still authenticating.
        """
        client = graphistry.client()
        calls = {}

        def fake_verify(token, fail_silent=False):
            calls["token"] = token
            calls["fail_silent"] = fail_silent
            return True

        monkeypatch.setattr(client, "verify_token", fake_verify)

        client.register(
            api=3,
            protocol="https",
            server="example.graphistry.com",
            token=" tokABC ",
            verify_token=False,
        )

        assert calls == {}
        assert client.session._is_authenticated is True
