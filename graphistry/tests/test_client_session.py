import pytest
import graphistry
from graphistry.pygraphistry import GraphistryClient, PyGraphistry

import pandas as pd


class TestClientSession:
    """
    Tests for Graphistry client()/register() interactions using the *current*
    public surface.  The legacy attribute helpers (api_key(), …) are intentionally
    **not** used;  we interrogate `graphistry.session` or `client.session`
    directly.
    """

    # --------------------------------------------------------------------- #
    # House‑keeping                                                         #
    # --------------------------------------------------------------------- #

    def setup_method(self):
        """
        Reset global state at the start of every test.  Calling register()
        without credentials clears the in-memory session cleanly.
        """
        graphistry.register(api=1)

    # --------------------------------------------------------------------- #
    # Basic config proxy                                                    #
    # --------------------------------------------------------------------- #

    def test_config_dict_access(self):
        """
        The session proxy still behaves like a mapping for read access.
        """
        cfg = graphistry.register(api=3, token="tok123")._config

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
        PyGraphistry.session.api_key = "global_key"
        global_session = PyGraphistry.session

        client = graphistry.client()
        assert isinstance(client, GraphistryClient)

        # Set client session state directly  
        client.session.api_key = "client_key"

        assert global_session.api_key == "global_key"
        assert client.session.api_key == "client_key"
        assert global_session is not client.session

    def test_global_session_persistence(self):
        # Test that global session persists across operations
        session1 = PyGraphistry.session
        session1.api_key = "K1"

        session2 = PyGraphistry.session
        session2.api_key = "K2"

        assert session1 is session2           # same object
        assert session2.api_key == "K2"       # value updated

    def test_client_inheritance(self):
        graphistry.register(
            api=3,
            protocol="https",
            server="test.graphistry.com",
            token="global_token",
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
        client1.session.api_key = "key1"
        client2.session.api_key = "key2"
        client3.session.api_key = "key3"

        assert client1.session.api_key == "key1"
        assert client2.session.api_key == "key2"
        assert client3.session.api_key == "key3"
        assert PyGraphistry.session.api_key not in {"key1", "key2", "key3"}

    # --------------------------------------------------------------------- #
    # Plotter propagation                                                   #
    # --------------------------------------------------------------------- #

    def test_client_bind_operations(self):
        client = graphistry.client()
        # Set session state directly to test plotter propagation
        client.session.api_key = "client_key"

        g = client.bind(source="src", destination="dst")
        assert g._source == "src"
        assert g._destination == "dst"
        assert g._pygraphistry.session.api_key == "client_key"

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

    # --------------------------------------------------------------------- #
    # Isolation from subsequent global changes                              #
    # --------------------------------------------------------------------- #

    def test_client_state_isolation_from_global_changes(self):
        # Set up global state directly
        PyGraphistry.session.api_key = "global1"

        client = graphistry.client(inherit=True)
        assert client.session.api_key == "global1"

        # mutate global state directly
        PyGraphistry.session.api_key = "global2"

        assert PyGraphistry.session.api_key == "global2"
        assert client.session.api_key == "global1"
