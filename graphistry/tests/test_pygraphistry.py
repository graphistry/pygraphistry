# -*- coding: utf-8 -*-

import mock, pandas as pd, pytest, unittest

import graphistry
from common import NoAuthTestCase
from graphistry import PyGraphistry

#TODO mock requests for testing actual effectful code

class TestPyGraphistry_Auth(unittest.TestCase):
    def test_defaults(self):
        assert PyGraphistry.store_token_creds_in_memory() == True

    def test_overrides(self):
        PyGraphistry.register(store_token_creds_in_memory=None)
        assert PyGraphistry.store_token_creds_in_memory() == True
        PyGraphistry.register(store_token_creds_in_memory=False)
        assert PyGraphistry.store_token_creds_in_memory() == False
