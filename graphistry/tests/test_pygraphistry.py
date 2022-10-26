# -*- coding: utf-8 -*-

import unittest, pytest
from mock import patch

from graphistry.pygraphistry import PyGraphistry
from graphistry.messages import (
    MSG_REGISTER_MISSING_PASSWORD,
    MSG_REGISTER_MISSING_USERNAME,
    MSG_REGISTER_MISSING_PKEY_SECRET,
    MSG_REGISTER_MISSING_PKEY_ID
)


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
    with pytest.raises(Exception) as exc_info:
        PyGraphistry.register(username='only_username')

    assert str(exc_info.value) == MSG_REGISTER_MISSING_PASSWORD


def test_register_with_only_password(capfd):
    with pytest.raises(Exception) as exc_info:
        PyGraphistry.register(password='only_password')

    assert str(exc_info.value) == MSG_REGISTER_MISSING_USERNAME


def test_register_with_only_personal_key_id(capfd):
    with pytest.raises(Exception) as exc_info:
        PyGraphistry.register(personal_key_id='only_personal_key_id')

    assert str(exc_info.value) == MSG_REGISTER_MISSING_PKEY_SECRET


def test_register_with_only_personal_key_secret(capfd):
    with pytest.raises(Exception) as exc_info:
        PyGraphistry.register(personal_key_secret='only_personal_key_secret')

    assert str(exc_info.value) == MSG_REGISTER_MISSING_PKEY_ID


class FakeRequestResponse(object):
    def __init__(self, response):
        self.response = response
    def raise_for_status(self):
        pass

    def json(self):
        return self.response


switch_org_success_response = {
    "status": "OK",
    "message": "Switched to organization: success-org",
    "data": []
}


org_not_exist_response = {
    "status": "Failed",
    "message": "No such organization id 'not-exist-org'",
    "data": []
}

org_not_permitted_response = {
    "status": "Failed",
    "message": "Not authorized to organization 'not-permitted-org'",
    "data": []
}


@patch("requests.post", return_value=FakeRequestResponse(switch_org_success_response))
def test_switch_organization_success(mock_response, capfd):
    PyGraphistry.org_name("success-org")
    out, err = capfd.readouterr()
    assert "Switched to organization: success-org" in out


@patch("requests.post", return_value=FakeRequestResponse(org_not_exist_response))
def test_switch_organization_not_exist(mock_response, capfd):
    PyGraphistry.org_name("not-exist-org")
    out, err = capfd.readouterr()
    assert "No such organization id 'not-exist-org'" in out


@patch("requests.post", return_value=FakeRequestResponse(org_not_permitted_response))
def test_switch_organization_not_permitted(mock_response, capfd):
    PyGraphistry.org_name("not-permitted-org")
    out, err = capfd.readouterr()
    assert "Not authorized to organization 'not-permitted-org'" in out
