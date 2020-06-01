"""Top-level import of class PyGraphistry as "Graphistry". Used to connect to the Graphistry server and then create a base plotter."""

from __future__ import absolute_import
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import bytes, object, str
from past.utils import old_div
from past.builtins import basestring

import calendar, gzip, io, json, os, numpy, pandas, requests, sys, time, warnings

from datetime import datetime
from distutils.util import strtobool

from .arrow_uploader import ArrowUploader

from . import util
from . import bolt_util


EnvVarNames = {
    'api_key': 'GRAPHISTRY_API_KEY',
    #'api_token': 'GRAPHISTRY_API_TOKEN',
    #'username': 'GRAPHISTRY_USERNAME',
    #'password': 'GRAPHISTRY_PASSWORD',
    'api_version': 'GRAPHISTRY_API_VERSION',
    'dataset_prefix': 'GRAPHISTRY_DATASET_PREFIX',
    'hostname': 'GRAPHISTRY_HOSTNAME',
    'protocol': 'GRAPHISTRY_PROTOCOL',
    'client_protocol_hostname': 'GRAPHISTRY_CLIENT_PROTOCOL_HOSTNAME',
    'certificate_validation': 'GRAPHISTRY_CERTIFICATE_VALIDATION'
}

config_paths = [
    os.path.join('/etc/graphistry', '.pygraphistry'),
    os.path.join(os.path.expanduser('~'), '.pygraphistry'),
    os.environ.get('PYGRAPHISTRY_CONFIG', '')
]

default_config = {
    'api_key': None, # Dummy key
    'api_token': None,
    #'username': None,
    #'password': None,
    'api_version': 1,
    'dataset_prefix': 'PyGraphistry/',
    'hostname': 'labs.graphistry.com',
    'protocol': 'https',
    'client_protocol_hostname': None,
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
    if not config['certificate_validation']:
        requests.packages.urllib3.disable_warnings()
    return config


class PyGraphistry(object):
    _config = _get_initial_config()
    _tag = util.fingerprint()
    _is_authenticated = False


    @staticmethod
    def authenticate():
        """Authenticate via already provided configuration (api=1,2).
        This is called once automatically per session when uploading and rendering a visualization."""
        key = PyGraphistry.api_key()
        #Mocks may set to True, so bypass in that case
        if (key is None) and PyGraphistry._is_authenticated == False:
            util.error('API key not set explicitly in `register()` or available at ' + EnvVarNames['api_key'])
        if not PyGraphistry._is_authenticated:
            PyGraphistry._check_key_and_version()
            PyGraphistry._is_authenticated = True


    @staticmethod
    def login(username, password, fail_silent=False):
        """Authenticate and set token for reuse (api=3). Periodically call if risk of token going stale."""

        token = ArrowUploader(
            server_base_path=PyGraphistry.protocol() + '://' + PyGraphistry.server(),
            certificate_validation=PyGraphistry.certificate_validation())\
                .login(username, password).token
        PyGraphistry.api_token(token)
        PyGraphistry._is_authenticated = True



    @staticmethod
    def server(value=None):
        """Get the hostname of the server or set the server using hostname or aliases.
        Supported aliases: 'localhost', 'staging', 'labs'.
        Also set via environment variable GRAPHISTRY_HOSTNAME."""
        if value is None:
            return PyGraphistry._config['hostname']

        # setter
        shortcuts = {'dev': 'localhost:3000',
                     'staging': 'staging.graphistry.com',
                     'labs': 'labs.graphistry.com'}
        if value in shortcuts:
            resolved = shortcuts[value]
            PyGraphistry._config['hostname'] = resolved
            util.warn('Resolving alias %s to %s' % (value, resolved))
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
    def api_token(value=None):
        """Set or get the API token.
        Also set via environment variable GRAPHISTRY_API_TOKEN."""

        if value is None:
            return PyGraphistry._config['api_token']

        # setter
        if value is not PyGraphistry._config['api_token']:
            PyGraphistry._config['api_token'] = value.strip()
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
    def set_bolt_driver(driver=None):
        PyGraphistry._config['bolt_driver'] = bolt_util.to_bolt_driver(driver)


    @staticmethod
    def register(key=None, username=None, password=None, token=None, server=None, protocol=None, api=None, certificate_validation=None, bolt=None):
        """API key registration and server selection

        Changing the key effects all derived Plotter instances.

        Provide one of key (api=1,2) or username/password (api=3) or token (api=3). 

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
        PyGraphistry.set_bolt_driver(bolt)
        if not (username is None) and not (password is None):
            PyGraphistry.login(username, password)
        PyGraphistry.api_token(token)


    @staticmethod
    def hypergraph(raw_events, entity_types=None, opts={}, drop_na=True, drop_edge_attrs=False, verbose=True, direct=False):
        """Transform a dataframe into a hypergraph.

        :param Dataframe raw_events: Dataframe to transform
        :param List entity_types: Optional list of columns (strings) to turn into nodes, None signifies all
        :param Dict opts: See below
        :param bool drop_edge_attrs: Whether to include each row's attributes on its edges, defaults to False (include)
        :param bool verbose: Whether to print size information
        :param bool direct: Omit hypernode and instead strongly connect nodes in an event

        Create a graph out of the dataframe, and return the graph components as dataframes, 
        and the renderable result Plotter. It reveals relationships between the rows and between column values.
        This transform is useful for lists of events, samples, relationships, and other structured high-dimensional data.

        The transform creates a node for every row, and turns a row's column entries into node attributes. 
        If direct=False (default), every unique value within a column is also turned into a node. 
        Edges are added to connect a row's nodes to each of its column nodes, or if direct=True, to one another.
        Nodes are given the attribute 'type' corresponding to the originating column name, or in the case of a row, 'EventID'.


        Consider a list of events. Each row represents a distinct event, and each column some metadata about an event. 
        If multiple events have common metadata, they will be transitively connected through those metadata values. 
        The layout algorithm will try to cluster the events together. 
        Conversely, if an event has unique metadata, the unique metadata will turn into nodes that only have connections to the event node, and the clustering algorithm will cause them to form a ring around the event node.

        Best practice is to set EVENTID to a row's unique ID,
        SKIP to all non-categorical columns (or entity_types to all categorical columns),
        and CATEGORY to group columns with the same kinds of values.


        The optional ``opts={...}`` configuration options are:

        * 'EVENTID': Column name to inspect for a row ID. By default, uses the row index.
        * 'CATEGORIES': Dictionary mapping a category name to inhabiting columns. E.g., {'IP': ['srcAddress', 'dstAddress']}.  If the same IP appears in both columns, this makes the transform generate one node for it, instead of one for each column.
        * 'DELIM': When creating node IDs, defines the separator used between the column name and node value
        * 'SKIP': List of column names to not turn into nodes. For example, dates and numbers are often skipped.
        * 'EDGES': For direct=True, instead of making all edges, pick column pairs. E.g., {'a': ['b', 'd'], 'd': ['d']} creates edges between columns a->b and a->d, and self-edges d->d.


        :returns: {'entities': DF, 'events': DF, 'edges': DF, 'nodes': DF, 'graph': Plotter}
        :rtype: Dictionary

        **Example**

            ::

                import graphistry
                h = graphistry.hypergraph(my_df)
                g = h['graph'].plot()

        """
        from . import hyper
        return hyper.Hypergraph().hypergraph(PyGraphistry, raw_events, entity_types, opts, drop_na, drop_edge_attrs, verbose, direct)


    @staticmethod
    def bolt(driver = None):
        """

        :param driver: Neo4j Driver or arguments for GraphDatabase.driver(**{...})**
        :return: Plotter w/neo4j

        Call this to create a Plotter with an overridden neo4j driver.

        **Example**

                ::

                    import graphistry
                    g = graphistry.bolt({ server: 'bolt://...', auth: ('<username>', '<password>') })

                ::

                    import neo4j
                    import graphistry

                    driver = neo4j.GraphDatabase.driver(...)

                    g = graphistry.bolt(driver)
        """
        from . import plotter
        return plotter.Plotter().bolt(driver)


    @staticmethod
    def cypher(query, params = {}):
        """

        :param query: a cypher query
        :param params: cypher query arguments
        :return: Plotter with data from a cypher query. This call binds `source`, `destination`, and `node`.

        Call this to immediately execute a cypher query and store the graph in the resulting Plotter.

                ::

                    import graphistry
                    g = graphistry.bolt({ query='MATCH (a)-[r:PAYMENT]->(b) WHERE r.USD > 7000 AND r.USD < 10000 RETURN r ORDER BY r.USD DESC', params={ "AccountId": 10 })
        """
        from . import plotter
        return plotter.Plotter().cypher(query, params)


    @staticmethod
    def nodexl(xls_or_url, source='default', engine=None, verbose=False):
        """

        :param xls_or_url: file/http path string to a nodexl-generated xls, or a pandas ExcelFile() object
        :param source: optionally activate binding by string name for a known nodexl data source ('twitter', 'wikimedia')
        :param engine: optionally set a pandas Excel engine
        :param verbose: optionally enable printing progress by overriding to True

        """

        if not (engine is None):
            print('WARNING: Engine currently ignored, please contact if critical')

        from . import plotter
        return plotter.Plotter().nodexl(xls_or_url, source, engine, verbose)


    @staticmethod
    def bind(node=None, source=None, destination=None,
             edge_title=None, edge_label=None, edge_color=None, edge_weight=None, edge_icon=None, edge_size=None, edge_opacity=None,
             point_title=None, point_label=None, point_color=None, point_weight=None, point_icon=None, point_size=None, point_opacity=None):
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
                              edge_title=edge_title, edge_label=edge_label, edge_color=edge_color, 
                              edge_size=edge_size, edge_weight=edge_weight, edge_icon=edge_icon, edge_opacity=edge_opacity, \
                              point_title=point_title, point_label=point_label, point_color=point_color, 
                              point_size=point_size, point_weight=point_weight, point_icon=point_icon, point_opacity=point_opacity)


    @staticmethod
    def tigergraph(
        protocol = 'http',
        server = 'localhost',
        web_port = 14240,
        api_port = 9000,
        db = None,
        user = 'tigergraph',
        pwd = 'tigergraph',
        verbose = False
    ):
        """Register Tigergraph connection setting defaults
    
        :param protocol: Protocol used to contact the database.
        :type protocol: Optional string.
        :param server: Domain of the database
        :type server: Optional string.
        :param web_port: 
        :type web_port: Optional integer.
        :param api_port: 
        :type api_port: Optional integer.
        :param db: Name of the database
        :type db: Optional string.    
        :param user:
        :type user: Optional string.    
        :param pwd: 
        :type pwd: Optional string.
        :param verbose: Whether to print operations
        :type verbose: Optional bool.         
        :returns: Plotter.
        :rtype: Plotter.


        **Example: Standard**
                ::

                    import graphistry
                    tg = graphistry.tigergraph(protocol='https', server='acme.com', db='my_db', user='alice', pwd='tigergraph2')                    

        """
        from . import plotter
        return plotter.Plotter().tigergraph(protocol, server, web_port, api_port, db, user, pwd, verbose)


    @staticmethod
    def gsql_endpoint(self, method_name, args = {}, bindings = None, db = None, dry_run = False):
        """Invoke Tigergraph stored procedure at a user-definend endpoint and return transformed Plottable
    
        :param method_name: Stored procedure name
        :type method_name: String.
        :param args: Named endpoint arguments
        :type args: Optional dictionary.
        :param bindings: Mapping defining names of returned 'edges' and/or 'nodes', defaults to @@nodeList and @@edgeList
        :type bindings: Optional dictionary.
        :param db: Name of the database, defaults to value set in .tigergraph(...)
        :type db: Optional string.
        :param dry_run: Return target URL without running
        :type dry_run: Bool, defaults to False            
        :returns: Plotter.
        :rtype: Plotter.

        **Example: Minimal**
                ::

                    import graphistry
                    tg = graphistry.tigergraph(db='my_db')
                    tg.gsql_endpoint('neighbors').plot()

        **Example: Full**
                ::

                    import graphistry
                    tg = graphistry.tigergraph()
                    tg.gsql_endpoint('neighbors', {'k': 2}, {'edges': 'my_edge_list'}, 'my_db').plot()

        **Example: Read data**
                ::

                    import graphistry
                    tg = graphistry.tigergraph()
                    out = tg.gsql_endpoint('neighbors')
                    (nodes_df, edges_df) = (out._nodes, out._edges)

        """
        from . import plotter
        return plotter.Plotter().gsql_endpoint(method_name, args, bindings, db, dry_run)



    @staticmethod
    def gsql(query, bindings = None, dry_run = False):
        """Run Tigergraph query in interpreted mode and return transformed Plottable
    
        :param query: Code to run
        :type query: String.
        :param bindings: Mapping defining names of returned 'edges' and/or 'nodes', defaults to @@nodeList and @@edgeList
        :type bindings: Optional dictionary.
        :param dry_run: Return target URL without running
        :type dry_run: Bool, defaults to False        
        :returns: Plotter.
        :rtype: Plotter.

        **Example: Minimal**
                ::

                    import graphistry
                    tg = graphistry.tigergraph()
                    tg.gsql(\"\"\"
INTERPRET QUERY () FOR GRAPH Storage { 
    
    OrAccum<BOOL> @@stop;
    ListAccum<EDGE> @@edgeList;
    SetAccum<vertex> @@set;
    
    @@set += to_vertex("61921", "Pool");

    Start = @@set;

    while Start.size() > 0 and @@stop == false do

      Start = select t from Start:s-(:e)-:t
      where e.goUpper == TRUE
      accum @@edgeList += e
      having t.type != "Service";
    end;

    print @@edgeList;
  }
                    \"\"\").plot()

       **Example: Full**
                ::

                    import graphistry
                    tg = graphistry.tigergraph()
                    tg.gsql(\"\"\"
INTERPRET QUERY () FOR GRAPH Storage { 
    
    OrAccum<BOOL> @@stop;
    ListAccum<EDGE> @@edgeList;
    SetAccum<vertex> @@set;
    
    @@set += to_vertex("61921", "Pool");

    Start = @@set;

    while Start.size() > 0 and @@stop == false do

      Start = select t from Start:s-(:e)-:t
      where e.goUpper == TRUE
      accum @@edgeList += e
      having t.type != "Service";
    end;

    print @@my_edge_list;
  }
                    \"\"\", {'edges': 'my_edge_list'}).plot()
        """
        from . import plotter
        return plotter.Plotter().gsql(query, bindings, dry_run)



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
    def settings(height=None, url_params={}, render=None):
        from . import plotter
        return plotter.Plotter().settings(height, url_params, render)


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
        cfg_client_protocol_hostname = PyGraphistry._config['client_protocol_hostname']
        cph = ('//' + PyGraphistry._config['hostname']) if cfg_client_protocol_hostname is None else cfg_client_protocol_hostname
        pattern = '%s/graph/graph.html?dataset=%s&type=%s&viztoken=%s&usertag=%s&splashAfter=%s&%s'
        return pattern % (cph, info['name'], info['type'],
                          info['viztoken'], PyGraphistry._tag, splash_time, extra)


    @staticmethod
    def _coerce_str(v):
        try:
            return str(v)
        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            print('=', v, '=')
            x = v.decode('utf-8')
            print('x', x)
            return x

    @staticmethod
    def _get_data_file(dataset, mode):
        out_file = io.BytesIO()
        if mode == 'json':
            json_dataset = None
            try:
                json_dataset = json.dumps(dataset, ensure_ascii=False, cls=NumpyJSONEncoder)
            except TypeError:
                warnings.warn("JSON: Switching from NumpyJSONEncoder to str()")                
                json_dataset = json.dumps(dataset, default=PyGraphistry._coerce_str)

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
hypergraph = PyGraphistry.hypergraph
bolt = PyGraphistry.bolt
cypher = PyGraphistry.cypher
nodexl = PyGraphistry.nodexl
tigergraph = PyGraphistry.tigergraph
gsql_endpoint = PyGraphistry.gsql_endpoint
gsql = PyGraphistry.gsql


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
                return obj.tolist()
        elif isinstance(obj, numpy.generic):
            return obj.item()
        elif isinstance(obj, type(pandas.NaT)):
            return None
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)
