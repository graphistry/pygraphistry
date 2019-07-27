from __future__ import absolute_import
import requests
import pandas as pd

def merge_dicts(x, y):
    return dict(list(x.items()) + list(y.items()))

class Tigeristry(object):
    """Tigergraph bindings class

        * Initialize with DB cfg
        * Register named stored procedures and graphistry bindings
        * Call stored procedures
        * Call interpreted queries

    """
  
    # ----------------- Helpers ----------------------- 

    def __log(self, v):
        if self.tiger_config['verbose']:
            print(v)
            
    # () -> 'http://site.com:9000'          
    def __base_url(self, mode = 'api'):
        port = self.tiger_config['web_port'] if mode == 'web' else self.tiger_config['api_port']
        who = \
            (self.tiger_config['user'] + ':' + self.tiger_config['pwd'] + '@') \
            if (not (self.tiger_config['user'] is None) and not (self.tiger_config['pwd'] is None)) \
            else ''
        return self.tiger_config['protocol'] + '://'+ who + self.tiger_config['server'] + ':' + str(port)
      
    def __check_initialized(self, graphistry):
        if (graphistry is None) or (graphistry._tigergraph is None):
            raise Exception("First register a tigergraph db via .tigergraph() or .register(tigergraph=)")

        
    # --------------------------------------------------

      
    def __init__(
        self,
        graphistry,
        protocol = 'http',
        server = 'localhost',
        web_port = 14240,
        api_port = 9000,
        db = None,
        user = 'tigergraph',
        pwd = 'tigergraph',
        verbose = False
    ):

        self.tiger_config = {
            'protocol': protocol,
            'server': server,
            'web_port': web_port,
            'api_port': api_port,
            'db': db,
            'user': user,
            'pwd': pwd,
            'verbose': verbose
        }

        self.__log('TG config: ' + str({k: v for k, v in self.tiger_config.items() if k not in ['pwd']}))
             

    # --------------------------------------------------    
    
    def __verify_and_unwrap_json_result(self, json):
      
        if json is None:
            raise Exception("No response!")
        elif not 'error' in json:
            raise Exception("Unexpected response format, no validity indicator", json)
        elif json['error']:
            raise Exception("Database returned error", json['message'] if 'message' in json else 'No message')
        elif not ('results' in json):
            raise Exception("No field results in database response")
                   
        return json['results']
    
    # str * ?dict * ?str => json graph
    def __gsql_endpoint(self, method_name, args = {}, db = None, dry_run = False):

        db = self.tiger_config['db'] if db is None else db
        if db is None:
            raise Exception("Must specify db in Tigeristry constructor or .__call()")

        base_url = self.__base_url('api')
        url = base_url + '/query/' + db + '/' + method_name
        if len(args.items()) > 0:
            url = url + '?' + '&'.join( [str(k) + '=' + str(v) for k, v in args.items()] )
        self.__log(url)

        if dry_run:            
            return url

        resp = requests.get(url)
        self.__log(resp)
        json = resp.json()

        return self.__verify_and_unwrap_json_result(json)

    def __json_to_graphistry(self, graphistry, json, bindings):        
        edges_df = pd.DataFrame({'from_id': [], 'to_id': []})
        edge_key = bindings['edges']
        edges = [x for x in json if edge_key in x]      
        if len(edges) > 0 and (edge_key in edges[0]):
            edges = edges[0][edge_key]
            edges_df = pd.DataFrame(edges)
            try:
                edges_df = edges_df.drop(columns=['attributes'])
                attrs = [x['attributes'] for x in edges]
                edges_df = pd.merge( edges_df, pd.DataFrame(attrs), left_index=True, right_index=True )
            except:
                self.__log('Failed to extract edge attrs')
            g = graphistry.bind(source='from_id', destination='to_id').edges(edges_df)
        
        nodes_df = pd.DataFrame({'type': [], 'node_id': []})
        node_key = bindings['nodes']
        nodes = [x for x in json if node_key in x]
        if len(nodes) > 0 and (node_key in nodes[0]):
            nodes = nodes[0][node_key]
            nodes_df = pd.DataFrame(nodes)
            try:
                nodes_df = nodes_df.drop(columns=['attributes'])
                attrs = [x['attributes'] for x in nodes]
                nodes_df = pd.merge( nodes_df, pd.DataFrame(attrs), left_index=True, right_index=True )
            except:
                self.__log('Failed to extract node attrs')
        else:        
            nodes_df = pd.DataFrame({'node_id': edges_df['from_id'].append(edges_df['to_id'])}) \
              .drop_duplicates().reset_index(drop=True)           
            from_types = nodes_df.merge(edges_df[['from_id', 'from_type']].rename(columns={'from_id': 'node_id', 'from_type': 'type'}), on='node_id', how='left')
            to_types = nodes_df.merge(edges_df[['to_id', 'to_type']].rename(columns={'to_id': 'node_id', 'to_type': 'type'}), on='node_id', how='left')
            nodes_df = nodes_df.merge(
                pd.DataFrame({'type': 
                              from_types.merge(to_types, left_index=True, right_index=True)\
                                .apply(
                                  lambda row: row['type_x'] if not pd.isna(row['type_x']) else row['type_y'],
                                  axis=1)}),
                left_index=True, right_index=True)              
        g = g.bind(node='node_id').nodes(nodes_df)
        return g
  
    def __gsql(self, query, dry_run = False):
        base_url = self.__base_url('web')
        url = base_url + '/gsqlserver/interpreted_query'
        self.__log(url)
        if dry_run == True:
            return url
        response = requests.post(url, data=query)
        json = response.json()
        return self.__verify_and_unwrap_json_result(json)


    # --------------------------------------------------

    # Tigeristry * Plotter * string * ?dict * ?dict * ?string => Plotter
    def gsql_endpoint(self, graphistry, method_name, args = {}, bindings = {}, db = None, dry_run = False):
        
        self.__check_initialized(graphistry)

        json = self.__gsql_endpoint(method_name, args, db, dry_run)

        if dry_run == True:
            url = json
            return url

        bindings = merge_dicts(
            {
              'edges': '@@edgeList',
              'nodes': '@@nodeList'
            },
            bindings
        )   

        return self.__json_to_graphistry(graphistry, json, bindings)

    # Tigeristry * Plotter * string * ?dict => Plotter
    def gsql(self, graphistry, query, bindings = {}, dry_run = False):

        self.__check_initialized(graphistry)

        json = self.__gsql(query, dry_run)

        if dry_run == True:
            url = json
            return url

        bindings = merge_dicts(
            {
              'edges': '@@edgeList',
              'nodes': '@@nodeList'
            },
            bindings
        )      

        return self.__json_to_graphistry(graphistry, json, bindings)