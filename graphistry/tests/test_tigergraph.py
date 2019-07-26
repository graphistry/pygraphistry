# -*- coding: utf-8 -*-

import unittest
import graphistry
from mock import patch
from common import NoAuthTestCase

class TestTiger(NoAuthTestCase):
    def test_tg_init_plain(self):
        tg = graphistry.tigergraph()
        self.assertTrue(type(tg) == graphistry.plotter.Plotter)

    def test_tg_init_many(self):
        tg = graphistry.tigergraph(
            protocol = 'https',
            server = '127.0.0.1',
            web_port = 10000,
            api_port = 11000,
            db = 'z',
            user = 'tigergraph1',
            pwd = 'tigergraph2',
            verbose = False
        )
        self.assertTrue(type(tg) == graphistry.plotter.Plotter)

    def test_tg_endpoint_url_simple(self):
        tg = graphistry.tigergraph(
            protocol = 'https',
            server = '127.0.0.1',
            web_port = 10000,
            api_port = 11000,
            db = 'z',
            user = 'tigergraph1',
            pwd = 'tigergraph2',
            verbose = False
        )
        self.assertEqual(
            tg.gsql_endpoint('x', dry_run = True),
            'https://tigergraph1:tigergraph2@127.0.0.1:11000/query/z/x'
        )

    def test_tg_endpoint_url_1_arg(self):
        tg = graphistry.tigergraph(
            protocol = 'https',
            server = '127.0.0.1',
            web_port = 10000,
            api_port = 11000,
            db = 'z',
            user = 'tigergraph1',
            pwd = 'tigergraph2',
            verbose = False
        )
        self.assertEqual(
            tg.gsql_endpoint('x', {'f': 1}, dry_run = True),
            'https://tigergraph1:tigergraph2@127.0.0.1:11000/query/z/x?f=1'
        )     

    def test_tg_endpoint_url_3_arg(self):
        tg = graphistry.tigergraph(
            protocol = 'https',
            server = '127.0.0.1',
            web_port = 10000,
            api_port = 11000,
            db = 'z',
            user = 'tigergraph1',
            pwd = 'tigergraph2',
            verbose = False
        )
        #27 does not preserve order
        self.assertEqual(
            len(tg.gsql_endpoint('x', {'f': 1, 'ggg': 2, 'h': 33}, dry_run = True)),
            len('https://tigergraph1:tigergraph2@127.0.0.1:11000/query/z/x?f=1&ggg=2&h=33')
        )           
    
    def test_tg_gsql(self):
        tg = graphistry.tigergraph(
            protocol = 'https',
            server = '127.0.0.1',
            web_port = 10000,
            api_port = 11000,
            db = 'z',
            user = 'tigergraph1',
            pwd = 'tigergraph2',
            verbose = False
        )
        self.assertEqual(
            tg.gsql('x', dry_run = True),
            'https://tigergraph1:tigergraph2@127.0.0.1:10000/gsqlserver/interpreted_query'
        )  

