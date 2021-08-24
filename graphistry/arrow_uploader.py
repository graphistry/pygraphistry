#Enable when we drop 3.6
#from __future__ import annotations
from typing import List, Optional

import io, logging, pyarrow as pa, requests, sys
from .ArrowFileUploader import ArrowFileUploader

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
            return {'bindings': {}}
        return self.__node_encodings

    @node_encodings.setter
    def node_encodings(self, node_encodings):
        self.__node_encodings = node_encodings

    @property
    def edge_encodings(self):
        if self.__edge_encodings is None:
            return {'bindings': {}}
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
    def description(self) -> str:
        if self.__description is None:
            return ""
        return self.__description

    @description.setter
    def description(self, description):
        self.__description = description


    @property
    def metadata(self):
        return {
            #'usertag': PyGraphistry._tag,
            #'key': PyGraphistry.api_key()
            'agent': 'pygraphistry',
            'apiversion' : '3',
            'agentversion': sys.modules['graphistry'].__version__,
            **(self.__metadata if not (self.__metadata is None) else {})
        }
    
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
            description = None,
            edges = None, nodes = None,
            node_encodings = None, edge_encodings = None,
            token = None, dataset_id = None,
            metadata = None,
            certificate_validation = True):
        self.__name = name
        self.__description = description
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

    def refresh(self, token=None):
        if token is None:
            token = self.token

        base_path = self.server_base_path
        out = requests.post(
            f'{base_path}/api-token-refresh/',
            verify=self.certificate_validation,
            json={'token': token})
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
    
    def verify(self, token=None) -> bool:
        if token is None:
            token = self.token

        base_path = self.server_base_path
        out = requests.post(
            f'{base_path}/api-token-verify/',
            verify=self.certificate_validation,
            json={'token': token})
        return out.status_code == requests.codes.ok

    def create_dataset(self, json):  # noqa: F811
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

    def g_to_node_bindings(self, g):
        bindings = self.maybe_bindings(  # noqa: E126
            g,  # noqa: E126
            [
                ['_node', 'node'],
                ['_point_color', 'node_color'],
                ['_point_label', 'node_label'],
                ['_point_opacity', 'node_opacity'],
                ['_point_size', 'node_size'],
                ['_point_title', 'node_title'],
                ['_point_weight', 'node_weight'],
                ['_point_icon', 'node_icon'],
                ['_point_x', 'node_x'],
                ['_point_y', 'node_y']
            ])

        return bindings

    def g_to_node_encodings(self, g):
        encodings = {
            'bindings': self.g_to_node_bindings(g)
        }
        for mode in ['current', 'default']:
            if len(g._complex_encodings['node_encodings'][mode].keys()) > 0:
                if not ('complex' in encodings):
                    encodings['complex'] = {}
                encodings['complex'][mode] = g._complex_encodings['node_encodings'][mode]
        return encodings


    def g_to_edge_bindings(self, g):
        bindings = self.maybe_bindings(  # noqa: E126
                g,  # noqa: E126
                [
                    ['_source', 'source'],
                    ['_destination', 'destination'],
                    ['_edge_color', 'edge_color'],
                    ['_edge_source_color', 'edge_source_color'],
                    ['_edge_destination_color', 'edge_destination_color'],
                    ['_edge_label', 'edge_label'],
                    ['_edge_opacity', 'edge_opacity'],
                    ['_edge_size', 'edge_size'],
                    ['_edge_title', 'edge_title'],
                    ['_edge_weight', 'edge_weight'],
                    ['_edge_icon', 'edge_icon']
                ])
        return bindings


    def g_to_edge_encodings(self, g):
        encodings = {
            'bindings': self.g_to_edge_bindings(g)
        }
        for mode in ['current', 'default']:
            if len(g._complex_encodings['edge_encodings'][mode].keys()) > 0:
                if not ('complex' in encodings):
                    encodings['complex'] = {}
                encodings['complex'][mode] = g._complex_encodings['edge_encodings'][mode]
        return encodings


    def post(self, as_files: bool = True, memoize: bool = True):
        """
        Note: likely want to pair with self.maybe_post_share_link(g)
        """
        if as_files:

            file_uploader = ArrowFileUploader(self)

            e_file_id, _ = file_uploader.create_and_post_file(self.edges, file_opts={'name': self.name + ' edges'})

            if not (self.nodes is None):
                n_file_id, _ = file_uploader.create_and_post_file(self.nodes, file_opts={'name': self.name + ' nodes'})

            self.create_dataset({
                "node_encodings": self.node_encodings,
                "edge_encodings": self.edge_encodings,
                "metadata": self.metadata,
                "name": self.name,
                "description": self.description,
                "edge_files": [ e_file_id ],
                **({"node_files": [ n_file_id ] if not (self.nodes is None) else []})
            })

        else:

            self.create_dataset({
                "node_encodings": self.node_encodings,
                "edge_encodings": self.edge_encodings,
                "metadata": self.metadata,
                "name": self.name,
                "description": self.description
            })
            
            self.post_edges_arrow()
            
            if not (self.nodes is None):
                self.post_nodes_arrow()
        
        return self


    ###########################################


    def cascade_privacy_settings(
        self,
        mode: Optional[str] = None,
        notify: Optional[bool] = None,
        invited_users: Optional[List] = None,
        message: Optional[str] = None
    ):
        """
        Cascade:
            - local (passed in)
            - global
            - hard-coded
        """

        from .pygraphistry import PyGraphistry
        global_privacy = PyGraphistry._config['privacy']
        if global_privacy is not None:
            if mode is None:
                mode = global_privacy['mode']
            if notify is None:
                notify = global_privacy['notify']
            if invited_users is None:
                invited_users = global_privacy['invited_users']
            if message is None:
                message = global_privacy['message']

        if mode is None:
            mode = 'private'
        if notify is None:
            notify = False
        if invited_users is None:
            invited_users = []
        if message is None:
            message = ''

        return mode, notify, invited_users, message


    def post_share_link(
        self,
        obj_pk: str,
        obj_type: str = 'dataset',
        privacy: Optional[dict] = None
    ):
        """
        Set sharing settings. Any settings not passed here will cascade from PyGraphistry or defaults
        """

        mode, notify, invited_users, message = self.cascade_privacy_settings(**(privacy or {}))

        path = self.server_base_path + '/api/v2/share/link/'
        tok = self.token
        res = requests.post(
            path,
            verify=self.certificate_validation,
            headers={'Authorization': f'Bearer {tok}'},
            json={
                'obj_pk': obj_pk,
                'obj_type': obj_type,
                'mode': mode,
                'notify': notify,
                'invited_users': invited_users,
                'message': message
            })

        if res.status_code != requests.codes.ok:
            logger.error('Failed setting sharing status (code %s): %s', res.status_code, res.text, exc_info=True)

        if res.status_code == 404:
            raise Exception(f'Code not find resource {path}; is your server location correct and does it support sharing?')

        if res.status_code == 403:
            raise Exception(f'Permission denied ({path}); do you have edit access and does your account having sharing enabled?')

        try:
            out = res.json()
            logger.debug('Server create file response: %s', out)
            if res.status_code != requests.codes.ok:
                res.raise_for_status()
        except Exception as e:
            logger.error('Unexpected error setting sharing settings: %s', res.text, exc_info=True)
            raise e

        logger.debug('Set privacy: mode %s, notify %s, users %s, message: %s', mode, notify, invited_users, message)
        
        return out


    ###########################################


    def post_edges_arrow(self, arr=None, opts=''):
        if arr is None:
            arr = self.edges
        return self.post_arrow(arr, 'edges', opts) 

    def post_nodes_arrow(self, arr=None, opts=''):
        if arr is None:
            arr = self.nodes
        return self.post_arrow(arr, 'nodes', opts) 

    def post_arrow(self, arr: pa.Table, graph_type: str, opts: str = ''):

        dataset_id = self.dataset_id
        tok = self.token
        sub_path = f'api/v2/upload/datasets/{dataset_id}/{graph_type}/arrow'

        try:
            resp = self.post_arrow_generic(sub_path, tok, arr, opts)
            out = resp.json()
            if not ('success' in out) or not out['success']:
                raise Exception('No success indicator in server response')
            return out
        except requests.exceptions.HTTPError as e:
            logger.error('Failed to post arrow to %s (%s)', sub_path, e.request.url, exc_info=True)
            logger.error('%s', e)
            logger.error('%s', e.response.text)
            raise e
        except Exception as e:
            logger.error('Failed to post arrow to %s', sub_path, exc_info=True)
            raise e

    def post_arrow_generic(self, sub_path: str, tok: str, arr: pa.Table, opts='') -> requests.Response:
        buf = self.arrow_to_buffer(arr)

        base_path = self.server_base_path

        url = f'{base_path}/{sub_path}'
        if len(opts) > 0:
            url = f'{url}?{opts}'
        resp = requests.post(
            url,
            verify=self.certificate_validation,
            headers={'Authorization': f'Bearer {tok}'},
            data=buf)
                    
        if resp.status_code != requests.codes.ok:
            resp.raise_for_status()

        return resp
    ###########################################


    #TODO refactor to be part of post()
    def maybe_post_share_link(self, g) -> bool:
        """
            Skip if never called .privacy()
            Return True/False based on whether called
        """
        from .pygraphistry import PyGraphistry
        logger.debug('Privacy: global (%s), local (%s)', PyGraphistry._config['privacy'] or 'None', g._privacy or 'None')
        if PyGraphistry._config['privacy'] is not None or g._privacy is not None:
            self.post_share_link(self.dataset_id, 'dataset', g._privacy)
            return True

        return False


    def post_g(self, g, name=None, description=None):
        """
        Warning: main post() does not call this
        """

        self.edge_encodings = self.g_to_edge_encodings(g)
        self.node_encodings = self.g_to_node_encodings(g)
        if not (name is None):
            self.name = name
        if not (description is None):
            self.description = description

        self.edges = pa.Table.from_pandas(g._edges, preserve_index=False).replace_schema_metadata({})
        if not (g._nodes is None):
            self.nodes = pa.Table.from_pandas(g._nodes, preserve_index=False).replace_schema_metadata({})

        out = self.post()
        self.maybe_post_share_link(g)

        return out
    
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
