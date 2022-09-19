# -*- coding: utf-8 -*-

import unittest, pytest

from graphistry import PyGraphistry

# TODO mock requests for testing actual effectful code


class TestPyGraphistry_Auth(unittest.TestCase):
    def test_defaults(self):
        assert PyGraphistry.store_token_creds_in_memory() is True

    def test_overrides(self):
        PyGraphistry.register(store_token_creds_in_memory=None)
        assert PyGraphistry.store_token_creds_in_memory() is True
        PyGraphistry.register(store_token_creds_in_memory=False)
        assert PyGraphistry.store_token_creds_in_memory() is False


def test_register_with_only_username(capfd):
    PyGraphistry.register(username='only_username')
    out, err = capfd.readouterr()
    assert out == "Hello World!"
