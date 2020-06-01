import io, json, logging, pandas as pd, pyarrow as pa, requests, sys

logger = logging.getLogger('ArrowUploader')

class ArrowUploader:
    
    @property
    def token(self) -> str:
        if self.__token is None:
            raise Exception("Not logged in")
        return self.__token

    @token.setter
    def token(self, token: str):
        self.__token = token

    @property
    def dataset_id(self) -> str:
        if self.__dataset_id is None:
            raise Exception("Must first create a dataset")
        return self.__dataset_id

    @dataset_id.setter
    def dataset_id(self, dataset_id: str):
        self.__dataset_id = dataset_id

    @property
    def server_base_path(self) -> str:
        return self.__server_base_path

    @server_base_path.setter
    def server_base_path(self, server_base_path: str):
        self.__server_base_path = server_base_path

    @property
    def view_base_path(self) -> str:
        return self.__view_base_path

    @view_base_path.setter
    def view_base_path(self, view_base_path: str):
        self.__view_base_path = view_base_path

    @property
    def edges(self) -> pa.Table:
        return self.__edges

    @edges.setter
    def edges(self, edges: pa.Table):
        self.__edges = edges

    @property
    def nodes(self) -> pa.Table:
        return self.__nodes

    @nodes.setter
    def nodes(self, nodes: pa.Table):
        self.__nodes = nodes

    @property
    def node_encodings(self):
        if self.__node_encodings is None:
            return {}
        return self.__node_encodings

    @node_encodings.setter
    def node_encodings(self, node_encodings):
        self.__node_encodings = node_encodings

    @property
    def edge_encodings(self):
        if self.__edge_encodings is None:
            return {}
        return self.__edge_encodings
    
    @edge_encodings.setter
    def edge_encodings(self, edge_encodings):
        self.__edge_encodings = edge_encodings

    @property
    def name(self) -> str:
        if self.__name is None:
            return "untitled"
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name


    @property
    def metadata(self):
        if self.__metadata is None:
            return {
                #'usertag': PyGraphistry._tag,
                #'key': PyGraphistry.api_key()
                'agent': 'pygraphistry',
                'apiversion' : '3',
                'agentversion': sys.modules['graphistry'].__version__,
            }
        return self.__metadata
    
    @metadata.setter
    def metadata(self, metadata):
        self.__metadata = metadata

    #########################################################################

    @property
    def certificate_validation(self):
        return self.__certificate_validation
    
    @certificate_validation.setter
    def certificate_validation(self, certificate_validation):
        self.__certificate_validation = certificate_validation


    ########################################################################3

    def __init__(self, 
            server_base_path='http://nginx', view_base_path='http://localhost',
            name = None,
            edges = None, nodes = None,
            node_encodings = None, edge_encodings = None,
            token = None, dataset_id = None,
            metadata = None,
            certificate_validation = True):
        self.__name = name
        self.__server_base_path = server_base_path
        self.__view_base_path = view_base_path
        self.__token = token
        self.__dataset_id = dataset_id
        self.__edges = edges
        self.__nodes = nodes
        self.__node_encodings = node_encodings
        self.__edge_encodings = edge_encodings
        self.__metadata = metadata
        self.__certificate_validation = certificate_validation
    
    def login(self, username, password):   
        base_path = self.server_base_path
        out = requests.post(
            f'{base_path}/api-token-auth/',
            verify=self.certificate_validation,
            json={'username': username, 'password': password})
        json_response = None
        try:
            json_response = out.json()
            if not ('token' in json_response):
                raise Exception(out.text)
        except Exception:
            logger.error('Error: %s', out, exc_info=True)
            raise Exception(out.text)
            
        self.token = out.json()['token']        
        return self
    
    def create_dataset(self, json):
        tok = self.token 
        
        res = requests.post(
            self.server_base_path + '/api/v2/upload/datasets/',
            verify=self.certificate_validation,
            headers={'Authorization': f'Bearer {tok}'},
            json=json)
             
        try:            
            out = res.json()
            if not out['success']:
                raise Exception(out)
        except Exception as e:
            logger.error('Failed creating dataset: %s', res.text, exc_info=True)
            raise e
        
        self.dataset_id = out['data']['dataset_id']

        return out
        
    #PyArrow's table.getvalues().to_pybytes() fails to hydrate some reason, 
    #  so work around by consolidate into a virtual file and sending that
    def arrow_to_buffer(self, table: pa.Table):
        b = io.BytesIO()
        writer = pa.RecordBatchFileWriter(b, table.schema)
        writer.write_table(table)
        writer.close()
        return b.getvalue()


    def maybe_bindings(self, g, bindings, base = {}):
        out = { **base }
        for old_field_name, new_field_name in bindings:
            try:
                val = getattr(g, old_field_name)
                if val is None:
                    continue
                else:
                    out[new_field_name] = val
            except AttributeError:
                continue
        logger.debug('bindings: %s', out)
        return out

    def g_to_node_encodings(self, g):
        node_encodings = self.maybe_bindings(
                g,
                [
                    ['_node', 'node'],
                    ['_point_color', 'node_color'],
                    ['_point_label', 'node_label'],
                    ['_point_opacity', 'node_opacity'],
                    ['_point_size', 'node_size'],
                    ['_point_title', 'node_title'],
                    ['_point_weight', 'node_weight'],
                    ['_point_icon', 'node_icon']
                ])
        if not (g._nodes is None):
            if 'x' in g._nodes:
                node_encodings['x'] = 'x'
            if 'y' in g._nodes:
                node_encodings['y'] = 'y'

        return node_encodings

    def g_to_edge_encodings(self, g):
        edge_encodings = self.maybe_bindings(
                g,
                [
                    ['_source', 'source'],
                    ['_destination', 'destination'],
                    ['_edge_color', 'edge_color'],
                    ['_edge_label', 'edge_label'],
                    ['_edge_opacity', 'edge_opacity'],
                    ['_edge_size', 'edge_size'],
                    ['_edge_title', 'edge_title'],
                    ['_edge_weight', 'edge_weight'],
                    ['_edge_icon', 'edge_icon']
                ])
        return edge_encodings

    
    def post(self):
        self.create_dataset({
            "node_encodings": {"bindings": self.node_encodings},
            "edge_encodings": {"bindings": self.edge_encodings},
            "metadata": self.metadata,
            "name": self.name
        })
        
        self.post_edges_arrow()
        
        if not (self.nodes is None):
            self.post_nodes_arrow()
        
        return self

    ###########################################


    def post_edges_arrow(self, arr=None, opts=''):
        if arr is None:
            arr = self.edges
        return self.post_arrow(arr, 'edges', opts) 

    def post_nodes_arrow(self, arr=None, opts=''):
        if arr is None:
            arr = self.nodes
        return self.post_arrow(arr, 'nodes', opts) 

    def post_arrow(self, arr, graph_type, opts=''):
        buf = self.arrow_to_buffer(arr)

        dataset_id = self.dataset_id
        tok = self.token
        base_path = self.server_base_path

        url = f'{base_path}/api/v2/upload/datasets/{dataset_id}/{graph_type}/arrow'
        if len(opts) > 0:
            url = f'{url}?{opts}'
        out = requests.post(
            url,
            verify=self.certificate_validation,
            headers={'Authorization': f'Bearer {tok}'},
            data=buf).json()
        
        if not out['success']:
            raise Exception(out)
            
        return out


    ###########################################


    def post_g(self, g, name=None):

        self.edge_encodings = self.g_to_edge_encodings(g)
        self.node_encodings = self.g_to_node_encodings(g)
        if not (name is None):
            self.name = name

        self.edges = pa.Table.from_pandas(g._edges, preserve_index=False).replace_schema_metadata({})
        if not (g._nodes is None):
            self.nodes = pa.Table.from_pandas(g._nodes, preserve_index=False).replace_schema_metadata({})

        return self.post()
    
    def post_edges_file(self, file_path, file_type='csv'):
        return self.post_file(file_path, 'edges', file_type)

    def post_nodes_file(self, file_path, file_type='csv'):
        return self.post_file(file_path, 'nodes', file_type)

    def post_file(self, file_path, graph_type='edges', file_type='csv'):

        dataset_id = self.dataset_id
        tok = self.token
        base_path = self.server_base_path

        with open(file_path, 'rb') as file:        
            out = requests.post(
                f'{base_path}/api/v2/upload/datasets/{dataset_id}/{graph_type}/{file_type}',
                verify=self.certificate_validation,
                headers={'Authorization': f'Bearer {tok}'},
                data=file.read()).json()
            if not out['success']:
                raise Exception(out)
            
            return out