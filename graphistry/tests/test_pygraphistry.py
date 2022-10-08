# -*- coding: utf-8 -*-

import unittest, pytest

from graphistry.pygraphistry import PyGraphistry

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
    assert "username exists but missing password" in out

def test_register_with_only_password(capfd):
    PyGraphistry.register(password='only_password')
    out, err = capfd.readouterr()
    assert "password exist but missing username" in out

def test_register_with_only_personal_key_id(capfd):
    PyGraphistry.register(personal_key='only_personal_key_id')
    out, err = capfd.readouterr()
    assert "Error: personal key exists but missing personal key id" in out

def test_register_with_only_personal_key(capfd):
    PyGraphistry.register(personal_key='only_personal_key')
    out, err = capfd.readouterr()
    assert "Error: personal key id exists but missing personal key" in out
