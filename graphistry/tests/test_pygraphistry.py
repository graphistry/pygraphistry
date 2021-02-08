# -*- coding: utf-8 -*-

import unittest

from graphistry import PyGraphistry

#TODO mock requests for testing actual effectful code

class TestPyGraphistry_Auth(unittest.TestCase):
    def test_defaults(self):
        assert PyGraphistry.store_token_creds_in_memory()

    def test_overrides(self):
        PyGraphistry.register(store_token_creds_in_memory=None)
        assert PyGraphistry.store_token_creds_in_memory()
        PyGraphistry.register(store_token_creds_in_memory=False)
        assert PyGraphistry.store_token_creds_in_memory()
