# -*- coding: utf-8 -*-
import json
import unittest

try:
    import mock  # type: ignore
except ImportError:  # pragma: no cover - fallback for stdlib-only envs
    from unittest import mock

from graphistry.arrow_uploader import ArrowUploader
from graphistry.client_session import ClientSession
from graphistry.utils import requests as switch_org_requests_module
from graphistry.utils.requests import (
    OrgSwitchError,
    OrgSwitchIdpChallenge,
    switch_org_request,
)


class FakeResponse(object):
    """Minimal requests.Response stand-in for switch_org_request tests."""

    def __init__(self, status_code: int, body=None, raise_on_json: bool = False):
        self.status_code = status_code
        self._body = body
        self._raise_on_json = raise_on_json
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if self._raise_on_json:
            raise json.JSONDecodeError("no json", "", 0)
        return self._body


class TestSwitchOrgRequest(unittest.TestCase):
    @mock.patch.object(switch_org_requests_module.requests, "post")
    def test_success(self, mock_post):
        mock_post.return_value = FakeResponse(200, {"status": "OK", "data": []})
        # Should not raise
        switch_org_request("https://hub.graphistry.com", "my-org", "tok", True)
        args, kwargs = mock_post.call_args
        assert args[0] == "https://hub.graphistry.com/api/v2/o/my-org/switch/"
        assert kwargs["data"] == {"slug": "my-org"}
        assert kwargs["headers"]["Authorization"] == "Bearer tok"
        assert kwargs["verify"] is True

    @mock.patch.object(switch_org_requests_module.requests, "post")
    def test_http_error_status(self, mock_post):
        mock_post.return_value = FakeResponse(404, {"status": "Failed", "message": "not found"})
        with self.assertRaises(OrgSwitchError) as cm:
            switch_org_request("https://hub.graphistry.com", "no-such-org", "tok", True)
        assert cm.exception.status_code == 404
        assert cm.exception.org_name == "no-such-org"
        assert "404" in str(cm.exception)

    @mock.patch.object(switch_org_requests_module.requests, "post")
    def test_body_status_not_ok_with_detail(self, mock_post):
        mock_post.return_value = FakeResponse(200, {"status": "Failed", "message": "not permitted"})
        with self.assertRaises(OrgSwitchError) as cm:
            switch_org_request("https://hub.graphistry.com", "not-permitted-org", "tok", True)
        assert cm.exception.detail == "not permitted"
        assert "not permitted" in str(cm.exception)

    @mock.patch.object(switch_org_requests_module.requests, "post")
    def test_unparseable_json_body_treated_as_failure(self, mock_post):
        mock_post.return_value = FakeResponse(200, None, raise_on_json=True)
        with self.assertRaises(OrgSwitchError) as cm:
            switch_org_request("https://hub.graphistry.com", "my-org", "tok", True)
        # falls back to body={} -> status missing -> not 'OK'
        assert cm.exception.detail == ""

    @mock.patch.object(switch_org_requests_module.requests, "post")
    def test_idp_challenge(self, mock_post):
        idp = {"auth_url": "https://sso-idp-host/authorize?state=xxx"}
        mock_post.return_value = FakeResponse(
            200, {"status": "OK", "data": {"idp": idp}}
        )
        with self.assertRaises(OrgSwitchIdpChallenge) as cm:
            switch_org_request("https://hub.graphistry.com", "sso-org", "tok", True)
        assert cm.exception.org_name == "sso-org"
        assert cm.exception.idp == idp
        assert "sso-org" in str(cm.exception)
        assert "auth_url" in str(cm.exception)


class TestArrowUploaderSwitchOrg(unittest.TestCase):
    def _uploader(self):
        return ArrowUploader(client_session=ClientSession())

    def test_noop_without_org_or_token(self):
        au = self._uploader()
        with mock.patch("graphistry.arrow_uploader.switch_org_request") as mock_req:
            au._switch_org(None, "tok")
            au._switch_org("org", None)
            au._switch_org(None, None)
        mock_req.assert_not_called()

    def test_skips_when_already_switched(self):
        au = self._uploader()
        au._client_session._last_switched_org_token = ("my-org", "tok")
        with mock.patch("graphistry.arrow_uploader.switch_org_request") as mock_req:
            au._switch_org("my-org", "tok")
        mock_req.assert_not_called()

    def test_success_records_last_switched(self):
        au = self._uploader()
        with mock.patch("graphistry.arrow_uploader.switch_org_request") as mock_req:
            au._switch_org("my-org", "tok")
        mock_req.assert_called_once_with(au.server_base_path, "my-org", "tok", au.certificate_validation)
        assert au._client_session._last_switched_org_token == ("my-org", "tok")

    def test_org_switch_error_does_not_record(self):
        au = self._uploader()
        with mock.patch(
            "graphistry.arrow_uploader.switch_org_request",
            side_effect=OrgSwitchError("my-org", 404),
        ):
            au._switch_org("my-org", "tok")
        assert au._client_session._last_switched_org_token is None

    def test_idp_challenge_does_not_record(self):
        au = self._uploader()
        with mock.patch(
            "graphistry.arrow_uploader.switch_org_request",
            side_effect=OrgSwitchIdpChallenge("my-org", {"auth_url": "https://x"}),
        ):
            au._switch_org("my-org", "tok")
        assert au._client_session._last_switched_org_token is None

    def test_generic_exception_does_not_record(self):
        au = self._uploader()
        with mock.patch(
            "graphistry.arrow_uploader.switch_org_request",
            side_effect=RuntimeError("boom"),
        ):
            au._switch_org("my-org", "tok")
        assert au._client_session._last_switched_org_token is None


if __name__ == "__main__":
    unittest.main()
