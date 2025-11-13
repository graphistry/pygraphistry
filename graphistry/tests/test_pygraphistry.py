# -*- coding: utf-8 -*-

import unittest, pytest
try:
    from mock import patch  # type: ignore
except ImportError:  # pragma: no cover - stdlib fallback
    from unittest.mock import patch
import graphistry

from graphistry.pygraphistry import PyGraphistry
from graphistry.messages import (
    MSG_REGISTER_MISSING_PASSWORD,
    MSG_REGISTER_MISSING_USERNAME,
    MSG_REGISTER_MISSING_PKEY_SECRET,
    MSG_REGISTER_MISSING_PKEY_ID,
    MSG_SWITCH_ORG_SUCCESS,
    MSG_SWITCH_ORG_NOT_FOUND,
    MSG_SWITCH_ORG_NOT_PERMITTED
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


@patch("graphistry.pygraphistry.ArrowUploader.login")
def test_login_switches_org(mock_login):
    mock_arrow = unittest.mock.MagicMock()
    mock_arrow.token = "tok123"
    mock_login.return_value = mock_arrow

    client = graphistry.client()
    with patch.object(client, "switch_org") as mock_switch:
        client.login("user", "pass", org_name="mock-org")

    mock_switch.assert_called_once_with("mock-org")


class FakeRequestResponse(object):
    def __init__(self, response, status_code: int):
        self.response = response
        self.status_code = status_code
    def raise_for_status(self):
        pass

    def json(self):
        return self.response


switch_org_success_response = {
    "status": "OK",
    "message": MSG_SWITCH_ORG_SUCCESS.format('success-org'),
    "data": []
}


org_not_exist_response = {
    "status": "Failed",
    "message": MSG_SWITCH_ORG_NOT_FOUND.format('not-exist-org'),
    "data": []
}

org_not_permitted_response = {
    "status": "Failed",
    "message": MSG_SWITCH_ORG_NOT_PERMITTED.format('not-permitted-org'),
    "data": []
}

# Print has been switch to logger.info
@patch("requests.post", return_value=FakeRequestResponse(switch_org_success_response, status_code=200))
def test_switch_organization_success(mock_response, capfd):
    PyGraphistry.org_name("success-org")
    out, err = capfd.readouterr()
    assert out == ''


@patch("requests.post", return_value=FakeRequestResponse(org_not_exist_response, status_code=404))
def test_switch_organization_not_exist(mock_response, capfd):
    org_name = "not-exist-org"
    with pytest.raises(Exception) as exc_info:
        PyGraphistry.org_name(org_name)

    assert str(exc_info.value) == "Failed to switch organization"

    # PyGraphistry.org_name("not-exist-org")
    # out, err = capfd.readouterr()
    # assert "Failed to switch organization" in out


@patch("requests.post", return_value=FakeRequestResponse(org_not_permitted_response, status_code=403))
def test_switch_organization_not_permitted(mock_response, capfd):
    org_name = "not-permitted-org"
    with pytest.raises(Exception) as exc_info:
        PyGraphistry.org_name(org_name)

    assert str(exc_info.value) == "Failed to switch organization"


    # PyGraphistry.org_name("not-permitted-org")
    # out, err = capfd.readouterr()
    # assert "Failed to switch organization" in out
