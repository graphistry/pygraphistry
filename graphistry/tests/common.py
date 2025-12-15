import unittest
import graphistry
import graphistry.pygraphistry


class NoAuthTestCase(unittest.TestCase):
    """Test case that bypasses authentication for tests.

    WARNING: This class manipulates internal PyGraphistry state to bypass
    authentication. This is a deliberate hack that:
    - Is NOT thread-safe (but works with pytest-xdist process isolation)
    - Depends on internal implementation details
    - Could break if PyGraphistry refactors authentication

    The pattern works because:
    1. pytest-xdist runs workers in separate processes (not threads)
    2. Each process has its own PyGraphistry class instance
    3. Tests from the same file run in the same worker (--dist loadfile)

    Known issues:
    - If any test calls register(), it resets _is_authenticated globally
    - test_client_session.py affects all subsequent tests via _client_mode_enabled
    - Tests have order dependencies (e.g., test_ipython.py)

    TODO: Replace with proper mocking of authenticate() method
    """

    @classmethod
    def setUpClass(cls):
        # Register with a fake token to bypass auth for api=3
        # verify_token=False prevents actual token verification
        # store_token_creds_in_memory=True ensures relogin() is called on refresh failure
        graphistry.register(api=3, token="faketoken", verify_token=False, store_token_creds_in_memory=True)
        # HACK: Set _is_authenticated after register() to bypass auth
        # This is reset by register(), so we set it after
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        # HACK: Mock relogin to prevent "Must call login() first" error
        # when token refresh fails and tries to relogin (api=3 flow)
        graphistry.pygraphistry.PyGraphistry.relogin = lambda: "faketoken"

    def setUp(self):
        # Reset auth state before each test method
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
        graphistry.pygraphistry.PyGraphistry.relogin = lambda: "faketoken"
