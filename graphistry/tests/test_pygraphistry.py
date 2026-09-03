# -*- coding: utf-8 -*-

import base64
import json
import time
from typing import Optional
import unittest, pytest
try:
    from mock import patch  # type: ignore
except ImportError:  # pragma: no cover - stdlib fallback
    from unittest.mock import patch
import graphistry

from graphistry.pygraphistry import PyGraphistry, GraphistryClient, SSO_POLL_MAX_CONSECUTIVE_ERRORS
from graphistry.exceptions import SsoPendingException
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
        fresh_client = GraphistryClient()
        assert fresh_client.store_token_creds_in_memory() is True

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


@patch("graphistry.pygraphistry.ArrowUploader.refresh")
def test_refresh_switches_org(mock_refresh):
    mock_arrow = unittest.mock.MagicMock()
    mock_arrow.token = "tok123"
    mock_refresh.return_value = mock_arrow

    client = graphistry.client()
    client.session.org_name = "mock-org"

    with patch.object(client, "switch_org") as mock_switch:
        client.refresh()

    mock_switch.assert_called_once_with("mock-org")


@patch("graphistry.pygraphistry.ArrowUploader.refresh")
def test_refresh_skips_switch_when_cached(mock_refresh):
    mock_arrow = unittest.mock.MagicMock()
    mock_arrow.token = "tok123"
    mock_refresh.return_value = mock_arrow

    client = graphistry.client()
    client.session.org_name = "mock-org"
    client.api_token("tok123")
    client.session.mark_org_verified("tok123", "mock-org")

    with patch.object(client, "switch_org") as mock_switch:
        client.refresh()

    mock_switch.assert_not_called()


@patch("graphistry.pygraphistry.ArrowUploader.refresh")
def test_refresh_switches_when_org_changes(mock_refresh):
    mock_arrow = unittest.mock.MagicMock()
    mock_arrow.token = "tok123"
    mock_refresh.return_value = mock_arrow

    client = graphistry.client()
    client.session.org_name = "new-org"
    client.api_token("tok123")
    client.session.mark_org_verified("tok123", "old-org")

    with patch.object(client, "switch_org") as mock_switch:
        client.refresh()

    mock_switch.assert_called_once_with("new-org")


def test_maybe_switch_org_cached_pair_skips():
    client = graphistry.client()
    client.api_token("tok123")
    client.session.mark_org_verified("tok123", "mock-org")

    with patch.object(client, "switch_org") as mock_switch:
        client._maybe_switch_org("mock-org")

    mock_switch.assert_not_called()


def test_maybe_switch_org_new_token_switches():
    client = graphistry.client()
    client.api_token("tok123")
    client.session.mark_org_verified("old-token", "mock-org")

    with patch.object(client, "switch_org") as mock_switch:
        client._maybe_switch_org("mock-org")

    mock_switch.assert_called_once_with("mock-org")


def test_maybe_switch_org_new_org_switches():
    client = graphistry.client()
    client.api_token("tok123")
    client.session.mark_org_verified("tok123", "other-org")

    with patch.object(client, "switch_org") as mock_switch:
        client._maybe_switch_org("mock-org")

    mock_switch.assert_called_once_with("mock-org")


def _fake_jwt(exp: Optional[float] = None) -> str:
    """Minimal unsigned-looking JWT with an optional exp claim, for exercising
    ClientSession's exp-aware verified-org cache without a real server."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = {} if exp is None else {"exp": exp}
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{header}.{body}.sig"


def test_switch_org_reuses_earlier_verified_token_for_different_org():
    """Org A was SSO-verified under token1; a later SSO login to org B mints
    token2 (now active). Switching back to org A should swap token1 back in
    locally instead of hitting the server, since token1 hasn't expired."""
    client = graphistry.client()
    token1 = _fake_jwt(exp=time.time() + 3600)
    token2 = _fake_jwt(exp=time.time() + 3600)

    client.api_token(token1)
    client.session.mark_org_verified(token1, "org-a")

    client.api_token(token2)
    client.session.mark_org_verified(token2, "org-b")
    client.session._is_authenticated = True

    with patch("graphistry.pygraphistry.switch_org_request") as mock_req:
        client.switch_org("org-a")

    mock_req.assert_not_called()
    assert client.api_token() == token1
    assert client.session.org_name == "org-a"
    # Swapping in an already-verified token must not de-authenticate the session,
    # else the next authenticate() refreshes and undoes the skipped round trip.
    assert client.session._is_authenticated is True


def test_switch_org_does_not_reuse_expired_cached_token():
    client = graphistry.client()
    expired_token = _fake_jwt(exp=time.time() - 10)
    active_token = _fake_jwt(exp=time.time() + 3600)

    client.api_token(expired_token)
    client.session.mark_org_verified(expired_token, "org-a")
    client.api_token(active_token)

    with patch("graphistry.pygraphistry.switch_org_request") as mock_req:
        client.switch_org("org-a")

    mock_req.assert_called_once()
    assert client.api_token() == active_token


class TestSsoPollErrorHandling:
    """_handle_auth_url's polling loop: keep waiting on SsoPendingException and
    network blips, propagate permanent errors."""

    def _client(self):
        client = graphistry.client()
        client.session.sso_state = "state-123"
        return client

    def test_pending_state_keeps_polling_until_login_completes(self):
        # The token endpoint answers "State is invalid" on every poll until the
        # user finishes the browser login -- that is the normal wait, not a failure.
        outcomes = [
            SsoPendingException("State is invalid"),
            SsoPendingException("State is invalid"),
            SsoPendingException("State is invalid"),
            SsoPendingException("State is invalid"),
            ("tok-final", "org-a"),
        ]

        def _poll():
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        client = self._client()
        with patch.object(client, "_sso_get_token", side_effect=_poll) as mock_get:
            with patch("graphistry.pygraphistry.time.sleep"):
                with patch.object(client, "api_token", return_value="tok-final"):
                    with patch.object(client, "_maybe_switch_org"):
                        out = client._handle_auth_url("http://auth", 30, None)

        # More pending polls than the network-error budget: pending must not be capped.
        assert mock_get.call_count == 5
        assert out == "tok-final"

    def test_pending_state_stops_at_timeout(self):
        client = self._client()
        with patch.object(
            client, "_sso_get_token",
            side_effect=SsoPendingException("State is invalid")
        ):
            with patch("graphistry.pygraphistry.time.sleep"):
                out = client._handle_auth_url("http://auth", 3, None)

        assert out is None

    def test_status_less_body_is_treated_as_pending(self):
        # A proxy/DRF error body with no 'status' key must not abort the login.
        outcomes = [
            SsoPendingException('{"detail": "Not found."}'),
            ("tok-final", "org-a"),
        ]

        def _poll():
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        client = self._client()
        with patch.object(client, "_sso_get_token", side_effect=_poll):
            with patch("graphistry.pygraphistry.time.sleep"):
                with patch.object(client, "api_token", return_value="tok-final"):
                    with patch.object(client, "_maybe_switch_org"):
                        out = client._handle_auth_url("http://auth", 30, None)

        assert out == "tok-final"

    def test_permanent_error_propagates_instead_of_timing_out(self):
        client = self._client()
        boom = Exception("SSO returned active_organization='org-b', but caller requested org_name='org-a'")

        with patch.object(client, "_sso_get_token", side_effect=boom) as mock_get:
            with patch("graphistry.pygraphistry.time.sleep"):
                with pytest.raises(Exception) as excinfo:
                    client._handle_auth_url("http://auth", 30, None)

        assert "active_organization" in str(excinfo.value)
        # Failed on the first poll rather than spinning out the full timeout.
        assert mock_get.call_count == 1

    def test_transient_network_error_is_retried_then_succeeds(self):
        client = self._client()
        import requests as _requests
        outcomes = [
            _requests.exceptions.ConnectionError("reset"),
            ("tok-final", "org-a"),
        ]

        def _poll():
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch.object(client, "_sso_get_token", side_effect=_poll):
            with patch("graphistry.pygraphistry.time.sleep"):
                with patch.object(client, "api_token", return_value="tok-final"):
                    with patch.object(client, "_maybe_switch_org"):
                        out = client._handle_auth_url("http://auth", 30, None)

        assert out == "tok-final"
        assert client.session.org_name == "org-a"

    def test_sustained_network_errors_give_up_before_timeout(self):
        client = self._client()
        import requests as _requests

        with patch.object(
            client, "_sso_get_token",
            side_effect=_requests.exceptions.ConnectionError("down")
        ) as mock_get:
            with patch("graphistry.pygraphistry.time.sleep"):
                with pytest.raises(_requests.exceptions.ConnectionError):
                    client._handle_auth_url("http://auth", 300, None)

        assert mock_get.call_count == SSO_POLL_MAX_CONSECUTIVE_ERRORS + 1


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


@patch("graphistry.pygraphistry.requests.get")
def test_from_dataset_id_hydrates_wrapped_plottable_metadata(mock_get):
    response = unittest.mock.MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": {
            "dataset_id": "wrapped_ds",
            "metadata": {
                "bindings": {"node": "id", "source": "src", "destination": "dst"},
                "encodings": {"point_color": "category"},
                "url_params": {"play": 0},
            },
        }
    }
    mock_get.return_value = response

    client = graphistry.client()
    out = client.from_dataset_id("wrapped_ds", api_token="tok_1")

    assert out._dataset_id == "wrapped_ds"
    assert out._node == "id"
    assert out._source == "src"
    assert out._destination == "dst"
    assert out._point_color == "category"
    assert out._url_params.get("play") == 0
    assert isinstance(out._url, str)
    assert "dataset=wrapped_ds" in out._url


@patch("graphistry.pygraphistry.requests.get")
def test_from_dataset_id_hydrates_legacy_shape(mock_get):
    response = unittest.mock.MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "dataset_id": "legacy_ds",
        "node_encodings": {
            "bindings": {
                "node": "id",
                "node_color": "group",
            },
            "complex": {
                "default": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "group",
                        "variation": "categorical",
                    }
                },
                "current": {},
            },
        },
        "edge_encodings": {
            "bindings": {"source": "s", "destination": "d", "edge_size": "weight"},
            "complex": {"default": {}, "current": {}},
        },
        "metadata": {"name": "Legacy graph", "description": "legacy payload"},
        "url_params": {"play": 0},
    }
    mock_get.return_value = response

    client = graphistry.client()
    out = client.from_dataset_id("legacy_ds", api_token="tok_2")

    assert out._dataset_id == "legacy_ds"
    assert out._node == "id"
    assert out._source == "s"
    assert out._destination == "d"
    assert out._point_color == "group"
    assert out._edge_size == "weight"
    assert out._name == "Legacy graph"
    assert out._description == "legacy payload"
    assert out._url_params.get("play") == 0
    assert out._complex_encodings["node_encodings"]["default"]["pointColorEncoding"]["attribute"] == "group"


@patch("graphistry.pygraphistry.requests.get")
def test_from_dataset_id_refreshes_when_token_missing(mock_get):
    response = unittest.mock.MagicMock()
    response.status_code = 200
    response.json.return_value = {"data": {"dataset_id": "token_ds"}}
    mock_get.return_value = response

    client = graphistry.client()
    client.session.api_token = "session_token"
    with patch.object(client, "refresh") as mock_refresh:
        out = client.from_dataset_id("token_ds")

    assert out._dataset_id == "token_ds"
    mock_refresh.assert_called_once()
    kwargs = mock_get.call_args[1]
    auth_header = kwargs["headers"]["Authorization"]
    assert auth_header == "Bearer session_token"


def test_from_dataset_id_rejects_empty_dataset_id():
    client = graphistry.client()
    with pytest.raises(ValueError) as exc_info:
        client.from_dataset_id("", api_token="tok")
    assert "dataset_id cannot be empty" in str(exc_info.value)
