from __future__ import absolute_import
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import object
import sys
import gzip
import io
import json
import requests
import numpy

from . import util


class PyGraphistry(object):
    api_key = None
    _tag = util.fingerprint()
    _dataset_prefix = 'PyGraphistry/'
    _hostname = 'localhost:3000'

    @staticmethod
    def register(key, server='proxy-labs.graphistry.com'):
        PyGraphistry.api_key = key.strip()
        shortcuts = {'localhost': 'localhost:3000',
                     'staging': 'proxy-staging.graphistry.com',
                     'labs': 'proxy-labs.graphistry.com'}
        if server in shortcuts:
            PyGraphistry._hostname = shortcuts[server]
        else:
            PyGraphistry._hostname = server

    @staticmethod
    def bind(node=None, source=None, destination=None,
             edge_title=None, edge_label=None, edge_color=None, edge_weight=None,
             point_title=None, point_label=None, point_color=None, point_size=None):

        from . import plotter
        return plotter.Plotter().bind(source, destination, node, \
                              edge_title, edge_label, edge_color, edge_weight, \
                              point_title, point_label, point_color, point_size)

    @staticmethod
    def nodes(nodes):
        from . import plotter
        return plotter.Plotter().nodes(nodes)

    @staticmethod
    def edges(edges):
        from . import plotter
        return plotter.Plotter().edges(edges)

    @staticmethod
    def graph(ig):
        from . import plotter
        return plotter.Plotter().graph(ig)

    @staticmethod
    def settings(height=None, url_params={}):
        from . import plotter
        return plotter.Plotter().settings(height, url_params)

    @staticmethod
    def _etl_url():
        return 'http://%s/etl' % PyGraphistry._hostname

    @staticmethod
    def _viz_url(dataset_name, url_params):
        extra = '&'.join([ k + '=' + str(v) for k,v in list(url_params.items())])
        pattern = 'http://%s/graph/graph.html?dataset=%s&usertag=%s&key=%s&%s'
        return pattern % (PyGraphistry._hostname, dataset_name, PyGraphistry._tag, PyGraphistry.api_key, extra)

    @staticmethod
    def _etl(dataset):
        if PyGraphistry.api_key is None:
            raise ValueError('API key required')

        json_dataset = json.dumps(dataset, ensure_ascii=False, cls=NumpyJSONEncoder)
        headers = {'Content-Encoding': 'gzip', 'Content-Type': 'application/json'}
        params = {'usertag': PyGraphistry._tag, 'agent': 'pygraphistry', 'apiversion' : '1',
                  'agentversion': sys.modules['graphistry'].__version__,
                  'key': PyGraphistry.api_key}

        out_file = io.BytesIO()
        with gzip.GzipFile(fileobj=out_file, mode='w', compresslevel=9) as f:
            if sys.version_info < (3,0):
                try:
                    f.write(json_dataset)
                except UnicodeEncodeError:
                    f.write(json_dataset.encode('utf8'))
            else:
                f.write(json_dataset.encode('utf8'))

        size = len(out_file.getvalue()) / 1024
        if size >= 10 * 1024:
            print('Uploading %d kB. This may take a while...' % size)
        elif size > 100 * 1024:
            util.error('Dataset is too large (%d kB)!' % size)

        try:
            response = requests.post(PyGraphistry._etl_url(), out_file.getvalue(),
                                     headers=headers, params=params)
        except requests.exceptions.ConnectionError as e:
            raise ValueError('Connection Error:', e.message)
        except requests.exceptions.HTTPError as e:
            raise ValueError('HTTP Error:', e.message)

        jres = response.json()
        if (jres['success'] is not True):
            raise ValueError('Server reported error:', jres['msg'])
        else:
            return jres['dataset']


register = PyGraphistry.register
bind = PyGraphistry.bind
edges = PyGraphistry.edges
nodes = PyGraphistry.nodes
graph = PyGraphistry.graph
settings = PyGraphistry.settings


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
                return obj.tolist()
        elif isinstance(obj, numpy.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)
