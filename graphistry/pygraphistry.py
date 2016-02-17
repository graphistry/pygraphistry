"""Top-level import of class PyGraphistry as "Graphistry". Used to connect to the Graphistry server and then create a base plotter."""

from __future__ import absolute_import
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import object
import os
import sys
import calendar
import time
import gzip
import io
import json
import requests
import numpy

from . import util


EnvVarNames = {
    'api_key': 'GRAPHISTRY_API_KEY',
    'api_version': 'GRAPHISTRY_API_VERSION',
    'dataset_prefix': 'GRAPHISTRY_DATASET_PREFIX',
    'hostname': 'GRAPHISTRY_HOSTNAME',
    'protocol': 'GRAPHISTRY_PROTOCOL'
}


class PyGraphistry(object):
    _api_version = 1
    _api_key = os.environ.get(EnvVarNames['api_key'], None)
    _tag = util.fingerprint()
    _dataset_prefix = os.environ.get(EnvVarNames['dataset_prefix'], 'PyGraphistry/')
    _hostname = os.environ.get(EnvVarNames['hostname'], 'proxy-labs.graphistry.com')
    _protocol = os.environ.get(EnvVarNames['protocol'], None)

    _is_authenticated = False

    @staticmethod
    def authenticate():
        key = PyGraphistry.api_key()
        if key is None:
            raise RuntimeError('API key not set explicitly or available at ' + EnvVarNames['api_key'])
        if not PyGraphistry._is_authenticated:
            PyGraphistry._check_key_and_version()
            PyGraphistry._is_authenticated = True

    @staticmethod
    def server(value=None):
        if value is None:
            return PyGraphistry._hostname

        # setter
        shortcuts = {'localhost': 'localhost:3000',
                     'staging': 'proxy-staging.graphistry.com',
                     'labs': 'proxy-labs.graphistry.com'}
        if value in shortcuts:
            PyGraphistry._hostname = shortcuts[value]
        else:
            PyGraphistry._hostname = value

    @staticmethod
    def api_key(value=None):
        if value is None:
            return PyGraphistry._api_key

        # setter
        if value is not PyGraphistry._api_key:
            PyGraphistry._api_key = value.strip()
            PyGraphistry._is_authenticated = False

    @staticmethod
    def protocol(value=None):
        if value is None:
            return PyGraphistry._protocol
        # setter
        PyGraphistry._protocol = value

    @staticmethod
    def api_version(value=None):
        if value is None:
            return PyGraphistry._api_version
        # setter
        PyGraphistry._api_version = value

    @staticmethod
    def register(key=None, server='labs', protocol=None, api=1):
        """API key registration and server selection

        Changing the key effects all derived Plotter instances.

        :param key: API key.
        :type key: String.
        :param server: URL of the visualization server.
        :type server: Optional string.
        :param protocol: Protocol used to contact visualization server
        :type protocol: Optional string.
        :returns: None.
        :rtype: None.


        **Example: Standard**
                ::

                    import graphistry
                    graphistry.register(key="my api key")

        **Example: Developer**
                ::

                    import graphistry
                    graphistry.register('my api key', server='staging', protocol='https')


        **Example: Through environment variable**
                ::
                    export GRAPHISTRY_API_KEY = 'my api key'
                ::
                    import graphistry
                    graphistry.register()
        """
        PyGraphistry.api_key(key)
        PyGraphistry.server(server)
        PyGraphistry.api_version(api)
        PyGraphistry.protocol(protocol)
        PyGraphistry.authenticate()


    @staticmethod
    def bind(node=None, source=None, destination=None,
             edge_title=None, edge_label=None, edge_color=None, edge_weight=None,
             point_title=None, point_label=None, point_color=None, point_size=None):
        """Create a base plotter.

        Typically called at start of a program. For parameters, see ``plotter.bind()`` .

        :returns: Plotter.
        :rtype: Plotter.

        **Example**

                ::

                    import graphistry
                    g = graphistry.bind()

        """



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
    def _etl_url(datatype):
        if datatype == 'json':
            return 'http://%s/etl' % PyGraphistry._hostname
        elif datatype == 'vgraph':
            return 'http://%s/etlvgraph' % PyGraphistry._hostname


    @staticmethod
    def _check_url():
        return 'http://%s/api/check' % PyGraphistry._hostname


    @staticmethod
    def _viz_url(info, url_params):
        splash_time = int(calendar.timegm(time.gmtime())) + 15
        extra = '&'.join([ k + '=' + str(v) for k,v in list(url_params.items())])
        pattern = '//%s/graph/graph.html?dataset=%s&type=%s&viztoken=%s&usertag=%s&splashAfter=%s&%s'
        return pattern % (PyGraphistry._hostname, info['name'], info['type'],
                          info['viztoken'], PyGraphistry._tag, splash_time, extra)


    @staticmethod
    def _get_data_file(dataset, mode):
        out_file = io.BytesIO()
        if mode == 'json':
            json_dataset = json.dumps(dataset, ensure_ascii=False, cls=NumpyJSONEncoder)
            with gzip.GzipFile(fileobj=out_file, mode='w', compresslevel=9) as f:
                if sys.version_info < (3,0) and isinstance(json_dataset, str):
                    f.write(json_dataset)
                else:
                    f.write(json_dataset.encode('utf8'))
        elif mode == 'vgraph':
            bin_dataset = dataset.SerializeToString()
            with gzip.GzipFile(fileobj=out_file, mode='w', compresslevel=9) as f:
                f.write(bin_dataset)
        else:
            raise ValueError('Unknown mode:', mode)

        size = len(out_file.getvalue()) / 1024
        if size >= 5 * 1024:
            print('Uploading %d kB. This may take a while...' % size)
            sys.stdout.flush()
        elif size > 50 * 1024:
            util.error('Dataset is too large (%d kB)!' % size)

        return out_file


    @staticmethod
    def _etl1(dataset):
        PyGraphistry.authenticate()

        headers = {'Content-Encoding': 'gzip', 'Content-Type': 'application/json'}
        params = {'usertag': PyGraphistry._tag, 'agent': 'pygraphistry', 'apiversion' : '1',
                  'agentversion': sys.modules['graphistry'].__version__,
                  'key': PyGraphistry.api_key()}

        out_file = PyGraphistry._get_data_file(dataset, 'json')
        response = requests.post(PyGraphistry._etl_url('json'), out_file.getvalue(),
                                 headers=headers, params=params)
        response.raise_for_status()

        jres = response.json()
        if jres['success'] is not True:
            raise ValueError('Server reported error:', jres['msg'])
        else:
            return {'name': jres['dataset'], 'viztoken': jres['viztoken'], 'type': 'vgraph'}


    @staticmethod
    def _etl2(dataset):
        PyGraphistry.authenticate()

        vg = dataset['vgraph']
        encodings = dataset['encodings']
        attributes = dataset['attributes']
        metadata = {
            'name': dataset['name'],
            'datasources': [
                {
                    'type': 'vgraph',
                    'url': 'data0'
                }
            ],
            'nodes': [
                {
                    'count': vg.vertexCount,
                    'encodings': encodings['nodes'],
                    'attributes': attributes['nodes']
                }
            ],
            'edges': [
                {
                    'count': vg.edgeCount,
                    'encodings': encodings['edges'],
                    'attributes': attributes['edges']
                }
            ]
        }

        out_file = PyGraphistry._get_data_file(vg, 'vgraph')
        metadata_json = json.dumps(metadata, ensure_ascii=False, cls=NumpyJSONEncoder)
        parts = {
            'metadata': ('metadata', metadata_json, 'application/json'),
            'data0': ('data0', out_file.getvalue(), 'application/octet-stream')
        }

        params = {'usertag': PyGraphistry._tag, 'agent': 'pygraphistry', 'apiversion' : '2',
                  'agentversion': sys.modules['graphistry'].__version__,
                  'key': PyGraphistry.api_key()}
        response = requests.post(PyGraphistry._etl_url('json'), files=parts, params=params)
        response.raise_for_status()

        jres = response.json()
        if jres['success'] is not True:
            raise ValueError('Server reported error:', jres['msg'] if 'msg' in jres else 'No Message')
        else:
            return {'name': jres['dataset'], 'viztoken': jres['viztoken'], 'type': 'jsonMeta'}

    @staticmethod
    def _check_key_and_version():
        params = {'text': PyGraphistry.api_key()}
        try:
            response = requests.get(PyGraphistry._check_url(), params=params,
                                    timeout=(2,2))
            response.raise_for_status()
            jres = response.json()

            cver = sys.modules['graphistry'].__version__
            if  'pygraphistry' in jres and 'minVersion' in jres['pygraphistry'] and 'latestVersion' in jres['pygraphistry']:
                mver = jres['pygraphistry']['minVersion']
                lver = jres['pygraphistry']['latestVersion']
                if util.compare_versions(mver, cver) > 0:
                    util.warn('Your version of PyGraphistry is no longer supported (installed=%s latest=%s). Please upgrade!' % (cver, lver))
                elif util.compare_versions(lver, cver) > 0:
                    print('A new version of PyGraphistry is available (installed=%s latest=%s).' % (cver, lver))

            if jres['success'] is not True:
                util.warn(jres['error'])
        except Exception as e:
            util.warn('Could not contact %s. Are you connected to the Internet?' % PyGraphistry._hostname)


server = PyGraphistry.server
protocol = PyGraphistry.protocol
api_key = PyGraphistry.api_key
api_version = PyGraphistry.api_version
authenticate = PyGraphistry.authenticate
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
