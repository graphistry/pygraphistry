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
        # Register once per test class to set up the session
        graphistry.register(api=3, verify_token=False)
        # HACK: Set _is_authenticated after register() to bypass auth
        # This is reset by register(), so we set it after
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True

    def setUp(self):
        # Reset _is_authenticated before each test method
        # This handles cases where other tests might have called register()
        # or otherwise reset the authentication state
        graphistry.pygraphistry.PyGraphistry._is_authenticated = True
