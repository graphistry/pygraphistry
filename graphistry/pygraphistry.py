"""Top-level import of class PyGraphistry as "Graphistry". Used to connect to the Graphistry server and then create a base plotter."""

from __future__ import absolute_import
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import object
from builtins import str
from builtins import bytes
from past.utils import old_div
from past.builtins import basestring

import os
import sys
import calendar
import time
from datetime import datetime
from distutils.util import strtobool
import gzip
import io
import json
import requests
import pandas
import numpy

from . import util


EnvVarNames = {
    'api_key': 'GRAPHISTRY_API_KEY',
    'api_version': 'GRAPHISTRY_API_VERSION',
    'dataset_prefix': 'GRAPHISTRY_DATASET_PREFIX',
    'hostname': 'GRAPHISTRY_HOSTNAME',
    'protocol': 'GRAPHISTRY_PROTOCOL',
    'certificate_validation': 'GRAPHISTRY_CERTIFICATE_VALIDATION'
}

config_paths = [
    os.path.join('/etc/graphistry', '.pygraphistry'),
    os.path.join(os.path.expanduser('~'), '.pygraphistry'),
    os.environ.get('PYGRAPHISTRY_CONFIG', '')
]

default_config = {
    'api_key': None, # Dummy key
    'api_version': 1,
    'dataset_prefix': 'PyGraphistry/',
    'hostname': 'labs.graphistry.com',
    'protocol': 'https',
    'certificate_validation': True
}


def _get_initial_config():
    config = default_config.copy()
    for path in config_paths:
        try:
            with open(path) as config_file:
                config.update(json.load(config_file))
        except ValueError as e:
            util.warn('Syntax error in %s, skipping. (%s)' % (path, e.message))
            pass
        except IOError:
            pass

    env_config = {k: os.environ.get(v) for k, v in EnvVarNames.items()}
    env_override = {k: v for k, v in env_config.items() if v != None}
    config.update(env_override)
    return config


class PyGraphistry(object):
    _config = _get_initial_config()
    _tag = util.fingerprint()
    _is_authenticated = False

    @staticmethod
    def authenticate():
        """Authenticate via already provided configuration.
        This is called once automatically per session when uploading and rendering a visualization."""
        key = PyGraphistry.api_key()
        if key is None:
            util.error('API key not set explicitly in `register()` or available at ' + EnvVarNames['api_key'])
        if not PyGraphistry._is_authenticated:
            PyGraphistry._check_key_and_version()
            PyGraphistry._is_authenticated = True

    @staticmethod
    def server(value=None):
        """Get the hostname of the server or set the server using hostname or aliases.
        Supported aliases: 'localhost', 'staging', 'labs'.
        Also set via environment variable GRAPHISTRY_HOSTNAME."""
        if value is None:
            return PyGraphistry._config['hostname']

        # setter
        shortcuts = {'localhost': 'localhost:3000',
                     'staging': 'staging.graphistry.com',
                     'labs': 'labs.graphistry.com'}
        if value in shortcuts:
            PyGraphistry._config['hostname'] = shortcuts[value]
        else:
            PyGraphistry._config['hostname'] = value

    @staticmethod
    def api_key(value=None):
        """Set or get the API key.
        Also set via environment variable GRAPHISTRY_API_KEY."""

        if value is None:
            return PyGraphistry._config['api_key']

        # setter
        if value is not PyGraphistry._config['api_key']:
            PyGraphistry._config['api_key'] = value.strip()
            PyGraphistry._is_authenticated = False

    @staticmethod
    def protocol(value=None):
        """Set or get the protocol ('http' or 'https').
        Set automatically when using a server alias.
        Also set via environment variable GRAPHISTRY_PROTOCOL."""
        if value is None:
            return PyGraphistry._config['protocol']
        # setter
        PyGraphistry._config['protocol'] = value

    @staticmethod
    def api_version(value=None):
        """Set or get the API version (1 or 2).
        Also set via environment variable GRAPHISTRY_API_VERSION."""
        if value is None:
            return PyGraphistry._config['api_version']
        # setter
        PyGraphistry._config['api_version'] = value

    @staticmethod
    def certificate_validation(value=None):
        """Enable/Disable SSL certificate validation (True, False).
        Also set via environment variable GRAPHISTRY_CERTIFICATE_VALIDATION."""
        if value is None:
            return PyGraphistry._config['certificate_validation']

        # setter
        v = bool(strtobool(value)) if isinstance(value, basestring) else value
        if v == False:
            requests.packages.urllib3.disable_warnings()
        PyGraphistry._config['certificate_validation'] = v

    @staticmethod
    def register(key=None, server=None, protocol=None, api=None, certificate_validation=None):
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
        PyGraphistry.certificate_validation(certificate_validation)
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
    def _etl_url():
        hostname = PyGraphistry._config['hostname']
        protocol = PyGraphistry._config['protocol']
        return '%s://%s/etl' % (protocol, hostname)


    @staticmethod
    def _check_url():
        hostname = PyGraphistry._config['hostname']
        protocol = PyGraphistry._config['protocol']
        return '%s://%s/api/check' % (protocol, hostname)


    @staticmethod
    def _viz_url(info, url_params):
        splash_time = int(calendar.timegm(time.gmtime())) + 15
        extra = '&'.join([ k + '=' + str(v) for k,v in list(url_params.items())])
        pattern = '//%s/graph/graph.html?dataset=%s&type=%s&viztoken=%s&usertag=%s&splashAfter=%s&%s'
        return pattern % (PyGraphistry._config['hostname'], info['name'], info['type'],
                          info['viztoken'], PyGraphistry._tag, splash_time, extra)


    @staticmethod
    def _get_data_file(dataset, mode):
        out_file = io.BytesIO()
        if mode == 'json':
            json_dataset = json.dumps(dataset, ensure_ascii=False, cls=NumpyJSONEncoder)
            with gzip.GzipFile(fileobj=out_file, mode='w', compresslevel=9) as f:
                if sys.version_info < (3,0) and isinstance(json_dataset, bytes):
                    f.write(json_dataset)
                else:
                    f.write(json_dataset.encode('utf8'))
        elif mode == 'vgraph':
            bin_dataset = dataset.SerializeToString()
            with gzip.GzipFile(fileobj=out_file, mode='w', compresslevel=9) as f:
                f.write(bin_dataset)
        else:
            raise ValueError('Unknown mode:', mode)

        size = old_div(len(out_file.getvalue()), 1024)
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
        response = requests.post(PyGraphistry._etl_url(), out_file.getvalue(),
                                 headers=headers, params=params,
                                 verify=PyGraphistry._config['certificate_validation'])
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
        response = requests.post(PyGraphistry._etl_url(), files=parts, params=params,
                                 verify=PyGraphistry._config['certificate_validation'])
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
            response = requests.get(PyGraphistry._check_url(), params=params, timeout=(3,3),
                                    verify=PyGraphistry._config['certificate_validation'])
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
            util.warn('Could not contact %s. Are you connected to the Internet?' % PyGraphistry._config['hostname'])

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
        elif isinstance(obj, pandas.tslib.NaTType):
            return None
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)
