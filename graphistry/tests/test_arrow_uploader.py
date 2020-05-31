# -*- coding: utf-8 -*-

import mock, pandas as pd, pytest, unittest

import graphistry
from common import NoAuthTestCase
from graphistry import ArrowUploader

#TODO mock requests for testing actual effectful code

class TestArrowUploader_Core(unittest.TestCase):
    def test_au_init_plain(self):
        au = ArrowUploader()
        with pytest.raises(Exception):
            au.token
        with pytest.raises(Exception):
            au.dataset_id
        assert au.edges is None
        assert au.nodes is None
        assert au.node_encodings == {}
        assert au.edge_encodings == {}
        assert len(au.name) > 0
        assert not (au.metadata is None)
    
    def test_au_init_args(self):
        n = pd.DataFrame({'n': []})
        e = pd.DataFrame({'e': []})
        sbp = "s"
        vbp = "v"
        name = "n"
        t = "t"
        d = "d"
        ne = {"point_color": "c"}
        ee = {"edge_color": "c"}
        m = {"n": "n"}
        ce = False
        au = ArrowUploader(server_base_path=sbp, view_base_path=vbp,
            name = name,
            edges = e, nodes = n,
            node_encodings = ne, edge_encodings = ee,
            token = t, dataset_id = d,
            metadata = m,
            certificate_validation = ce)
        assert au.server_base_path == sbp
        assert au.view_base_path == vbp
        assert au.name == name
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
        assert au.g_to_node_encodings(g) == {}

    def test_au_n_enc_full(self):        
        g = graphistry.bind(node='n', point_color='c', point_size='s', point_title='t', point_label='l')
        au = ArrowUploader()
        assert au.g_to_node_encodings(g) == \
            {
                'node': 'n',
                'node_color': 'c',
                'node_size': 's',
                'node_title': 't',
                'node_label': 'l'
            }

    def test_au_e_enc_mt(self):
        g = graphistry.bind()
        au = ArrowUploader()
        assert au.g_to_edge_encodings(g) == {}

    def test_au_e_enc_full(self):        
        g = graphistry.bind(source='s', destination='d', edge_color='c', edge_title='t', edge_label='l', edge_weight='w')
        au = ArrowUploader()
        assert au.g_to_edge_encodings(g) == \
            {
                'source': 's',
                'destination': 'd',
                'edge_color': 'c',
                'edge_title': 't',
                'edge_label': 'l',
                'edge_weight': 'w'
            }


class TestArrowUploader_Comms(unittest.TestCase):

    def _mock_response(
            self,
            status=200,
            content="CONTENT",
            json_data=None,
            raise_for_status=None):

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
    
    @mock.patch('requests.post')
    def test_login(self, mock_post):

        mock_resp = self._mock_response(json_data={'token': '123'})
        mock_post.return_value = mock_resp

        au = ArrowUploader()
        tok = au.login(username="u", password="p").token

        assert tok == "123"

