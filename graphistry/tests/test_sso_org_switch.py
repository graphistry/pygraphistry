"""End-to-end sso_login() / switch_org() tests: /switch/ must be sent whenever
verification is recorded, and a challenged or skipped switch must not be marked."""
import base64
import json
import time
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from graphistry.client_session import ClientSession
from graphistry.pygraphistry import GraphistryClient


def _jwt(tag: str) -> str:
    h = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    b = base64.urlsafe_b64encode(
        json.dumps({"exp": time.time() + 3600, "tag": tag}).encode()
    ).rstrip(b"=").decode()
    return f"{h}.{b}.sig"


def _resp(body: Dict[str, Any], status: int = 200) -> mock.Mock:
    r = mock.Mock()
    r.status_code = status
    r.json.return_value = body
    r.text = json.dumps(body)
    return r


class _Server:
    """Records every POST; routes login / switch; serves the jwt poll."""

    def __init__(self, token: str, active_org: Optional[str] = None,
                 direct_token: bool = False, switch_body: Optional[Dict[str, Any]] = None):
        self.token = token
        self.active_org = active_org
        self.direct_token = direct_token
        self.switch_body = switch_body or {"status": "OK", "data": {}}
        self.posts: List[str] = []

    def post(self, url: str, *args: Any, **kwargs: Any) -> mock.Mock:
        self.posts.append(url)
        if "/switch/" in url:
            return _resp(self.switch_body)
        if "/sso/oidc/login" in url:
            if self.direct_token:
                return _resp({"status": "OK", "data": {"token": self.token}})
            return _resp({"status": "OK", "data": {"state": "s1", "auth_url": "http://auth"}})
        raise AssertionError("unexpected POST " + url)

    def get(self, url: str, *args: Any, **kwargs: Any) -> mock.Mock:
        if "/sso/oidc/jwt/" in url:
            data: Dict[str, Any] = {"token": self.token}
            if self.active_org is not None:
                data["active_organization"] = {"slug": self.active_org}
            return _resp({"status": "OK", "data": data})
        if "/my/organizations/" in url:
            return _resp({"results": []})
        raise AssertionError("unexpected GET " + url)

    @property
    def switches(self) -> List[str]:
        return [u for u in self.posts if "/switch/" in u]


def _login(server: _Server, org_name: Optional[str]) -> GraphistryClient:
    client = GraphistryClient()
    client.session.hostname = "hub.test"
    with mock.patch("requests.post", side_effect=server.post), \
            mock.patch("requests.get", side_effect=server.get), \
            mock.patch("time.sleep"):
        client.sso_login(org_name=org_name, sso_timeout=5)
    return client


class TestSsoLoginSendsSwitch:
    def test_layer1_server_bound_org_switches_once(self):
        tok = _jwt("l1")
        s = _Server(tok, active_org="x")
        c = _login(s, "x")
        assert len(s.switches) == 1 and s.switches[0].endswith("/o/x/switch/")
        assert c.session.is_org_verified(tok, "x")

    def test_layer2_caller_org_server_silent_switches(self):
        tok = _jwt("l2")
        s = _Server(tok, active_org=None)
        c = _login(s, "x")
        assert len(s.switches) == 1 and s.switches[0].endswith("/o/x/switch/")
        assert c.session.is_org_verified(tok, "x")

    def test_site_with_org_switches(self):
        tok = _jwt("site")
        s = _Server(tok, active_org="SITE")
        c = _login(s, "x")
        assert len(s.switches) == 1 and s.switches[0].endswith("/o/x/switch/")
        assert c.session.is_org_verified(tok, "x")

    def test_direct_token_with_org_switches(self):
        tok = _jwt("direct")
        s = _Server(tok, direct_token=True)
        c = _login(s, "x")
        assert len(s.switches) == 1 and s.switches[0].endswith("/o/x/switch/")
        assert c.session.is_org_verified(tok, "x")

    def test_idp_challenged_switch_is_not_marked_verified(self):
        tok = _jwt("challenged")
        challenge = {"status": "OK", "data": {"idp": {"auth_url": "http://idp"}}}
        s = _Server(tok, active_org=None, switch_body=challenge)
        c = _login(s, "x")
        assert len(s.switches) >= 1
        assert not c.session.is_org_verified(tok, "x")

    def test_challenged_direct_token_is_not_marked_verified(self):
        tok = _jwt("challenged-direct")
        challenge = {"status": "OK", "data": {"idp": {"auth_url": "http://idp"}}}
        s = _Server(tok, direct_token=True, switch_body=challenge)
        c = _login(s, "x")
        assert not c.session.is_org_verified(tok, "x")


class TestSwitchOrgCacheOnePerToken:
    def _client(self, tok: str) -> GraphistryClient:
        c = GraphistryClient()
        c.session.api_token = tok
        c.session.hostname = "hub.test"
        return c

    def test_a_b_a_sends_three_posts(self):
        tok = _jwt("aba")
        s = _Server(tok)
        c = self._client(tok)
        with mock.patch("requests.post", side_effect=s.post):
            c.switch_org("a")
            c.switch_org("b")
            c.switch_org("a")
        assert [u.split("/o/")[1] for u in s.switches] == ["a/switch/", "b/switch/", "a/switch/"]
        assert c.session.org_name == "a"

    def test_repeat_same_org_still_skips(self):
        tok = _jwt("rep")
        s = _Server(tok)
        c = self._client(tok)
        with mock.patch("requests.post", side_effect=s.post):
            c.switch_org("a")
            c.switch_org("a")
        assert len(s.switches) == 1

    def test_mark_evicts_other_orgs_for_same_token_only(self):
        sess = ClientSession()
        t1, t2 = _jwt("1"), _jwt("2")
        sess.mark_org_verified(t1, "a")
        sess.mark_org_verified(t2, "b")   # different token: a survives
        assert sess.is_org_verified(t1, "a") and sess.is_org_verified(t2, "b")
        sess.mark_org_verified(t1, "c")   # same token: a evicted
        assert not sess.is_org_verified(t1, "a")
        assert sess.is_org_verified(t1, "c") and sess.is_org_verified(t2, "b")

    def test_fast_path_2_swaps_in_older_token(self):
        t1, t2 = _jwt("1"), _jwt("2")
        s = _Server(t2)
        c = self._client(t2)
        c.session.mark_org_verified(t1, "a")
        c.session.mark_org_verified(t2, "b")
        with mock.patch("requests.post", side_effect=s.post):
            c.switch_org("a")
        assert s.switches == [] and c.session.api_token == t1


class TestStatusLessPollBodyIsNotPending:
    """A body with no 'status' (proxy page, DRF detail) must surface after the consecutive-error limit, not retry silently until timeout."""

    def test_poll_loop_gives_up_on_status_less_body(self):
        from graphistry.pygraphistry import SSO_POLL_MAX_CONSECUTIVE_ERRORS

        class S(_Server):
            def get(self, url: str, *a: Any, **k: Any) -> mock.Mock:
                if "/sso/oidc/jwt/" in url:
                    return _resp({"detail": "Not found."})
                return super().get(url, *a, **k)

        s = S(_jwt("x"))
        client = GraphistryClient()
        client.session.hostname = "hub.test"
        with mock.patch("requests.post", side_effect=s.post), \
                mock.patch("requests.get", side_effect=s.get), \
                mock.patch("time.sleep"):
            with pytest.raises(ValueError):
                client.sso_login(org_name="x", sso_timeout=SSO_POLL_MAX_CONSECUTIVE_ERRORS + 20)


class TestSwitchErrorKeepsServerMessage:
    def test_non_2xx_keeps_server_message(self):
        from graphistry.utils.requests import OrgSwitchError, switch_org_request

        r = _resp({"status": "Fail", "message": "Not authorized to organization 'x'"}, status=403)
        with mock.patch("requests.post", return_value=r):
            with pytest.raises(OrgSwitchError) as ei:
                switch_org_request("http://h", "x", "tok", True)
        assert ei.value.status_code == 403
        assert "Not authorized to organization 'x'" in str(ei.value)

    def test_non_json_non_2xx_still_raises_without_detail(self):
        from graphistry.utils.requests import OrgSwitchError, switch_org_request

        r = mock.Mock(status_code=502)
        r.json.side_effect = json.JSONDecodeError("no json", "", 0)
        with mock.patch("requests.post", return_value=r):
            with pytest.raises(OrgSwitchError) as ei:
                switch_org_request("http://h", "x", "tok", True)
        assert "HTTP 502" in str(ei.value)


class TestRequestedOrgNameCleared:
    def _client(self) -> GraphistryClient:
        c = GraphistryClient()
        c.session.hostname = "hub.test"
        return c

    def test_cleared_after_direct_token(self):
        s = _Server(_jwt("d"), direct_token=True)
        c = self._client()
        with mock.patch("requests.post", side_effect=s.post), mock.patch("requests.get", side_effect=s.get), mock.patch("time.sleep"):
            c.sso_login(org_name="x", sso_timeout=5)
        assert c.session.sso_requested_org_name is None

    def test_cleared_after_blocking_success(self):
        s = _Server(_jwt("b"), active_org="x")
        c = _login(s, "x")
        assert c.session.sso_requested_org_name is None

    def test_cleared_after_blocking_timeout(self):
        class Pending(_Server):
            def get(self, url: str, *a: Any, **k: Any) -> mock.Mock:
                if "/sso/oidc/jwt/" in url:
                    return _resp({"status": "ERR", "message": "State is invalid"})
                return super().get(url, *a, **k)

        s = Pending(_jwt("t"))
        c = self._client()
        with mock.patch("requests.post", side_effect=s.post), mock.patch("requests.get", side_effect=s.get), mock.patch("time.sleep"):
            c.sso_login(org_name="x", sso_timeout=3)
        assert c.session.sso_requested_org_name is None

    def test_cleared_after_blocking_error(self):
        class Broken(_Server):
            def get(self, url: str, *a: Any, **k: Any) -> mock.Mock:
                if "/sso/oidc/jwt/" in url:
                    return _resp({"detail": "Not found."})
                return super().get(url, *a, **k)

        s = Broken(_jwt("e"))
        c = self._client()
        with mock.patch("requests.post", side_effect=s.post), mock.patch("requests.get", side_effect=s.get), mock.patch("time.sleep"):
            with pytest.raises(ValueError):
                c.sso_login(org_name="x", sso_timeout=60)
        assert c.session.sso_requested_org_name is None

    def test_kept_for_non_blocking_then_cleared_by_sso_get_token(self):
        s = _Server(_jwt("nb"), active_org="x")
        c = self._client()
        with mock.patch("requests.post", side_effect=s.post), mock.patch("requests.get", side_effect=s.get), mock.patch("time.sleep"):
            c.sso_login(org_name="x", sso_timeout=None)
            assert c.session.sso_requested_org_name == "x"   # sso_get_token() still needs it
            assert c.sso_get_token()
        assert c.session.sso_requested_org_name is None

    def test_sso_get_token_pending_keeps_requested_org(self):
        from graphistry.exceptions import SsoPendingException

        class Pending(_Server):
            polls = 0

            def get(self, url: str, *a: Any, **k: Any) -> mock.Mock:
                if "/sso/oidc/jwt/" in url and Pending.polls == 0:
                    Pending.polls += 1
                    return _resp({"status": "ERR", "message": "State is invalid"})
                return super().get(url, *a, **k)

        s = Pending(_jwt("p"), active_org="x")
        c = self._client()
        with mock.patch("requests.post", side_effect=s.post), mock.patch("requests.get", side_effect=s.get), mock.patch("time.sleep"):
            c.sso_login(org_name="x", sso_timeout=None)
            with pytest.raises(SsoPendingException):
                c.sso_get_token()
            assert c.session.sso_requested_org_name == "x"   # second poll still targets x
            assert c.sso_get_token()
        assert c.session.sso_requested_org_name is None
