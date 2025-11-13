# -*- coding: utf-8 -*-

import graphistry, pandas as pd, pytest, unittest
try:
    import mock  # type: ignore
except ImportError:  # pragma: no cover - fallback for stdlib-only envs
    from unittest import mock

from graphistry import ArrowUploader
from graphistry.pygraphistry import PyGraphistry

# TODO mock requests for testing actual effectful code


class TestArrowUploader_Core(unittest.TestCase):
    def test_au_init_plain(self):
        au = ArrowUploader()
        with pytest.raises(Exception):
            au.token
        with pytest.raises(Exception):
            au.dataset_id
        assert au.edges is None
        assert au.nodes is None
        assert au.node_encodings == {"bindings": {}}
        assert au.edge_encodings == {"bindings": {}}
        assert len(au.name) > 0
        assert not (au.metadata is None)

    def test_au_init_args(self):
        n = pd.DataFrame({"n": []})
        e = pd.DataFrame({"e": []})
        sbp = "s"
        vbp = "v"
        name = "n"
        des = "d"
        t = "t"
        d = "d"
        ne = {"point_color": "c"}
        ee = {"edge_color": "c"}
        m = {"n": "n"}
        ce = False
        au = ArrowUploader(
            server_base_path=sbp,
            view_base_path=vbp,
            name=name,
            description=des,
            edges=e,
            nodes=n,
            node_encodings=ne,
            edge_encodings=ee,
            token=t,
            dataset_id=d,
            metadata=m,
            certificate_validation=ce,
        )
        assert au.server_base_path == sbp
        assert au.view_base_path == vbp
        assert au.name == name
        assert au.description == des
        assert au.edges is e
        assert au.nodes is n
        assert au.edge_encodings == ee
        assert au.node_encodings == ne
        assert au.token == t
        assert au.dataset_id == d
        assert au.certificate_validation == ce

    def test_au_n_enc_mt(self):
        g = graphistry.bind()
        au = ArrowUploader()
        assert au.g_to_node_encodings(g) == {"bindings": {}}

    def test_au_n_enc_full(self):
        g = graphistry.bind(
            node="n",
            point_color="c",
            point_size="s",
            point_title="t",
            point_label="l",
            point_weight="w",
            point_opacity="o",
            point_icon="i",
            point_x="x",
            point_y="y",
        )
        g = g.encode_point_color("c", ["green"], as_categorical=True)
        au = ArrowUploader()
        assert au.g_to_node_encodings(g) == {
            "bindings": {
                "node": "n",
                "node_color": "c",
                "node_size": "s",
                "node_title": "t",
                "node_label": "l",
                "node_weight": "w",
                "node_opacity": "o",
                "node_icon": "i",
                "node_x": "x",
                "node_y": "y",
            },
            "complex": {
                "default": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "c",
                        "variation": "categorical",
                        "colors": ["green"],
                    }
                }
            },
        }

    def test_au_e_enc_mt(self):
        g = graphistry.bind()
        au = ArrowUploader()
        assert au.g_to_edge_encodings(g) == {"bindings": {}}

    def test_au_e_enc_full(self):
        g = graphistry.bind(
            source="s",
            destination="d",
            edge_color="c",
            edge_title="t",
            edge_label="l",
            edge_weight="w",
            edge_opacity="o",
            edge_icon="i",
            edge_size="s",
            edge_source_color="sc",
            edge_destination_color="dc",
        )
        g = g.encode_edge_color("c", ["green"], as_categorical=True)
        au = ArrowUploader()
        assert au.g_to_edge_encodings(g) == {
            "bindings": {
                "source": "s",
                "destination": "d",
                "edge_color": "c",
                "edge_title": "t",
                "edge_label": "l",
                "edge_weight": "w",
                "edge_opacity": "o",
                "edge_icon": "i",
                "edge_size": "s",
                "edge_source_color": "sc",
                "edge_destination_color": "dc",
            },
            "complex": {
                "default": {
                    "edgeColorEncoding": {
                        "graphType": "edge",
                        "encodingType": "color",
                        "attribute": "c",
                        "variation": "categorical",
                        "colors": ["green"],
                    }
                }
            },
        }

    def test_cascade_privacy_settings_default_global(self):
        PyGraphistry.session.privacy = None
        PyGraphistry.privacy()
        au = ArrowUploader()
        assert au.cascade_privacy_settings() == ('private', False, [], '20', '')

    def test_cascade_privacy_settings_global_override(self):
        PyGraphistry.session.privacy = None
        PyGraphistry.privacy(mode="public", notify=True)
        au = ArrowUploader()
        assert au.cascade_privacy_settings() == ('public', True, [], '10', '')

    def test_cascade_privacy_settings_local_override(self):
        PyGraphistry.session.privacy = None
        g = graphistry.bind().privacy(mode="public", notify=True)
        au = ArrowUploader()
        assert au.cascade_privacy_settings(**g._privacy) == ('public', True, [], '10', '')

    def test_cascade_privacy_settings_local_override_cascade(self):
        PyGraphistry.session.privacy = None
        PyGraphistry.privacy()
        g = graphistry.bind().privacy(mode="public", notify=True)
        au = ArrowUploader()
        assert au.cascade_privacy_settings(**g._privacy) == ('public', True, [], '10', '')


class TestArrowUploader_Comms(unittest.TestCase):
    def _mock_response(
        self, status=200, content="CONTENT", json_data=None, raise_for_status=None
    ):

        mock_resp = mock.Mock()
        # mock raise_for_status call w/optional error
        mock_resp.raise_for_status = mock.Mock()
        if raise_for_status:
            mock_resp.raise_for_status.side_effect = raise_for_status
        # set status code and content
        mock_resp.status_code = status
        mock_resp.content = content
        # add json data if provided
        if json_data:
            mock_resp.json = mock.Mock(return_value=json_data)
        return mock_resp

    @mock.patch("requests.post")
    def test_login(self, mock_post):

        mock_resp = self._mock_response(json_data={"token": "123"})
        mock_post.return_value = mock_resp

        au = ArrowUploader()
        tok = au.login(username="u", password="p").token

        assert tok == "123"


    @mock.patch('requests.post')
    def test_login_with_org_success(self, mock_post):

        mock_resp = self._mock_response(
            json_data={
                'token': '123',
                'active_organization': {
                    "slug": "mock-org",
                    'is_found': True,
                    'is_member': True
                }
        })
        mock_post.return_value = mock_resp

        au = ArrowUploader()
        response = au.login(username="u", password="p", org_name="mock-org")
        tok = response.token
        assert tok == "123"
        assert PyGraphistry.org_name() == "mock-org"

    @mock.patch('graphistry.arrow_uploader.ArrowUploader._switch_org')
    @mock.patch('requests.post')
    def test_login_invokes_switch_org(self, mock_post, mock_switch):

        mock_resp = self._mock_response(
            json_data={
                'token': '123',
                'active_organization': {
                    "slug": "mock-org",
                    'is_found': True,
                    'is_member': True
                }
        })
        mock_post.return_value = mock_resp

        au = ArrowUploader()
        au.login(username="u", password="p", org_name="mock-org")

        mock_switch.assert_called_once_with("mock-org", "123")

    @mock.patch('requests.post')
    def test_login_with_org_updates_client_session(self, mock_post):

        mock_resp = self._mock_response(json_data={
                'token': '123',
                'active_organization': {
                    "slug": "mock-org",
                    'is_found': True,
                    'is_member': True
                }
        })
        mock_post.return_value = mock_resp

        client = graphistry.client()
        client.session.org_name = None
        PyGraphistry.session.org_name = None

        au = ArrowUploader(client_session=client.session)
        au.login(username="u", password="p", org_name="mock-org")

        assert client.session.org_name == "mock-org"
        assert PyGraphistry.org_name() == "mock-org"


    @mock.patch('requests.post')
    def test_login_with_org_old_server(self, mock_post):

        mock_resp = self._mock_response(json_data={'token': '123'})
        mock_post.return_value = mock_resp

        au = ArrowUploader()

        with pytest.raises(Exception):
            au.login(username="u", password="p", org_name="mock-org")

        with pytest.raises(Exception):
            au.token

    @mock.patch('requests.post')
    def test_login_with_org_invalid_org_name(self, mock_post):

        mock_resp = self._mock_response(
            json_data={
                'token': '123',
                'active_organization': {
                    "slug": "default-org",
                    'is_found': False,
                    'is_member': False
                }
        })
        mock_post.return_value = mock_resp

        au = ArrowUploader()

        with pytest.raises(Exception):
            au.login(username="u", password="p", org_name="mock-org")

        with pytest.raises(Exception):
            au.token

    @mock.patch('requests.post')
    def test_login_with_org_valid_org_name_not_member(self, mock_post):

        mock_resp = self._mock_response(
            json_data={
                'token': '123',
                'active_organization': {
                    "slug": "mock-org",
                    'is_found': True,
                    'is_member': False
                }
        })
        mock_post.return_value = mock_resp

        au = ArrowUploader()

        with pytest.raises(Exception):
            au.login(username="u", password="p", org_name="mock-org")

        with pytest.raises(Exception):
            au.token

    @mock.patch('requests.post')
    def test_sso_login_when_required_authentication(self, mock_post):

        mock_resp = self._mock_response(
            json_data={
                'status': 'OK',
                'message': 'Required login',
                'data': {
                    'auth_url': 'https://sso-idp-host/authorize?state=xxuixld',
                    'state': 'xxuixld'
                }
        })
        mock_post.return_value = mock_resp

        au = ArrowUploader()

        au.sso_login(org_name="mock-org", idp_name="mock-idp")

        au.sso_state == 'xxuixld'
        au.sso_auth_url == 'https://sso-idp-host/authorize?state=xxuixld'

        with pytest.raises(Exception):
            au.token

    @mock.patch('requests.post')
    def test_sso_login_when_already_authenticated(self, mock_post):

        mock_resp = self._mock_response(
            json_data={
                'status': 'OK',
                'message': 'User already authenticated',
                'data': {
                    #'state': 'xxuixld',
                    'token': '123',
                }
        })
        mock_post.return_value = mock_resp

        au = ArrowUploader()

        au.sso_login(org_name="mock-org", idp_name="mock-idp")
        #assert au.sso_state == 'xxuixld'
        assert au.token == '123'

    @mock.patch('requests.get')
    def test_sso_login_get_sso_token_ok(self, mock_get):

        mock_resp = self._mock_response(
            json_data={
                'status': 'OK',
                'message': 'State is valid',
                'data': {
                    'token': '123',
                    'active_organization': {
                        "slug": "mock-org",
                        'is_found': True,
                        'is_member': True
                    }
                }
        })
        mock_get.return_value = mock_resp

        au = ArrowUploader()

        au.sso_get_token(state='ignored-valid')
        assert au.token == '123'
