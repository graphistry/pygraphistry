import pandas as pd

class NodeXLGraphistryBase(object):

    source_to_mappings = None
    verbose = False
    graphistry = None

    def __init__(self, schema = {}, graphistry_binder = None, engine=None, verbose=None):

        if not (engine is None):
            print('WARNING: Engine currently ignored, please contact if critical')

        default_bindings = {
            'edges_df_transformer': NodeXLGraphistryBase.edges_df_transformer_default,
            'edge_bindings': NodeXLGraphistryBase.edge_bindings_default,
            'nodes_df_transformer': NodeXLGraphistryBase.nodes_df_transformer_default,
            'node_bindings': NodeXLGraphistryBase.node_bindings_default
        }    
        self.source_to_mappings = {
            'default': default_bindings,
            **{
                binding_name: {
                    **default_bindings,
                    **schema[binding_name]
                }
               for binding_name in schema.keys()
            }
        }
        if not (verbose is None):
            self.verbose = verbose

        if graphistry_binder is None:
            import graphistry
            self.graphistry = graphistry
        else:
            self.graphistry = graphistry_binder

        self.engine = engine

    @staticmethod
    def link_urls(series):
        return series.apply(lambda s: ' '.join(['<a href="%s" target="_blank">%s</a>' % (s, s) for s in str(s).split(' ')]))

    @staticmethod
    def embed_img(series):
        return series.apply(lambda s: '<a href="%s" target="_blank"><img src="%s"/></a>' % (s,s) if len(str(s)) > 0 and not str(s) == 'nan' else '')

    ####################################################


    @staticmethod
    def edges_df_transformer_default(edges_df):
        #FIXME weird pandas bug forcing here vs post, otherwise color col is offset one with last row as null
        edges_df = edges_df.assign(ColorInt=pd.Series(edges_df['Color'].factorize()[0]).apply(lambda r: r % 12))
        return edges_df

    @staticmethod
    def nodes_df_transformer_default(nodes_df):
        ##TODO factor out
        #FIXME weird pandas bug forcing here vs later, otherwise color col is offset one with last row as null
        nodes_df = nodes_df.assign(Color2=pd.Series(nodes_df['Vertex Group'].factorize()[0]).apply(lambda r: r % 12))
        nodes_df = nodes_df[1:]
        nodes_df = nodes_df.assign(**{
            'Custom Menu Item': nodes_df.apply(lambda row: '<a href="%s" target="_blank">%s</a>' % ( row['Custom Menu Item Action'], row['Custom Menu Item Text']), axis=1)
        })
        nodes_df = nodes_df.drop(columns=['Custom Menu Item Action', 'Custom Menu Item Text'])
        return nodes_df

    edge_bindings_default = {
        'source': 'Vertex 1',
        'destination': 'Vertex 2',
        'edge_color': 'ColorInt'
    }

    node_bindings_default = {
        'point_title': 'Label',
        'point_size': 'Size',
        #'point_opacity': 'Opacity',
        'node': 'Vertex',
        'point_color': 'Color2'
    }

    ###################################

    #xls from pd.ExcelFile(...)
    def xls_to_edges_df(self, xls, edges_df_transformer = None):
        if edges_df_transformer is None:
            edges_df_transformer = NodeXLGraphistryBase.edges_df_transformer_default
        raw_edges_df = pd.read_excel(xls, 'Edges')
        edge_header_dict = raw_edges_df[0:1].to_dict()
        edge_header_mapping = {x: edge_header_dict[x][0] for x in edge_header_dict.keys()}
        edges_df = raw_edges_df.rename(columns=edge_header_mapping)

        edges_df = edges_df_transformer(edges_df)

        edges_df = edges_df[1:]
        return edges_df

    def plot_edges_df(self, edges_df, edge_bindings = None):
        if edge_bindings is None:
            edge_bindings = NodeXLGraphistryBase.edge_bindings_default
        g = self.graphistry.edges(edges_df).bind(**edge_bindings)
        return g

    ## results of plot_edges_df and adds node bindings for richer plotter
    def plot_graph_df(self, edge_graph, nodes_df, node_bindings = None):
        if node_bindings is None:
            node_bindings = NodeXLGraphistryBase.node_bindings_default
        g = edge_graph
        g2 = g.nodes(nodes_df).bind(**node_bindings).settings(url_params={'play': 0})
        return g2
        
    def xls_to_nodes_df(self, xls, nodes_df_transformer = None):
        if nodes_df_transformer is None:
            nodes_df_transformer = NodeXLGraphistryBase.nodes_df_transformer_default
        raw_nodes_df = pd.read_excel(xls, 'Vertices')
        node_header_dict = raw_nodes_df[0:1].to_dict()
        node_header_mapping = {x: node_header_dict[x][0] for x in node_header_dict.keys()}

        ##x, y are not (yet) official passthrough bindings, but automation happens to pick these up 
        nodes_df = raw_nodes_df.rename(columns=node_header_mapping).rename(columns={'X': 'x', 'Y': 'y'})

        nodes_df = nodes_df_transformer(nodes_df)
        return nodes_df

    ##TODO can we infer source?
    # str * ?(str | dict) * ?bool => graphistry
    def xls(self, xls_or_url, source='default', verbose=None):

        verbose = self.verbose if verbose is None else verbose        
        p = print if verbose else (lambda x: 1)

        ## source is either undefined, a string, or a (partial) bindings object
        if type(source) == str and not source in self.source_to_mappings:
            p('Unknown source type', source)
            raise Exception('Unknown nodexl source type %s' % str(source))
        bindings = self.source_to_mappings[source] if type(source) == str else source
        
        p('Fetching...')
        xls = pd.ExcelFile(xls_or_url) if type(xls_or_url) == str else xls_or_url

        p('Formatting edges')
        edges_df = self.xls_to_edges_df(xls, bindings['edges_df_transformer'])

        p('Formatting nodes')
        nodes_df = self.xls_to_nodes_df(xls, bindings['nodes_df_transformer'])

        p('Setting up bindings')
        g1 = self.plot_edges_df(edges_df, bindings['edge_bindings'])
        g2 = self.plot_graph_df(g1, nodes_df, bindings['node_bindings'])

        p('Ready to upload and view! (%s nodes, %s edges, %s columns' % (len(nodes_df), len(edges_df), len(nodes_df.columns) + len(edges_df.columns)))

        return g2        


class NodeXLGraphistry(NodeXLGraphistryBase):

    def __init__(self, graphistry_binder=None, engine=None, verbose=None):

        if not (engine is None):
            print('WARNING: Engine currently ignored, please contact if critical')

        super().__init__({
                'simple': {
                    'edges_df_transformer': NodeXLGraphistry.simple_edges_df_transformer,
                    'edge_bindings': {
                        **(NodeXLGraphistryBase.edge_bindings_default)
                    },
                    'nodes_df_transformer': NodeXLGraphistry.simple_nodes_df_transformer,
                    'node_bindings': {
                        **(NodeXLGraphistryBase.node_bindings_default)
                    }
                },
                ##################################################
                #                                                #
                # Default bindings used where none are provided  #
                #                                                #
                ##################################################
                'simple2': { },

                'twitter': {
                    'nodes_df_transformer': NodeXLGraphistry.twitter_nodes_df_transformer
                },
                'mediawiki': {
                    'nodes_df_transformer': NodeXLGraphistry.mediawiki_nodes_df_transformer    
                }
            },
            graphistry_binder,
            engine,
            verbose)

    #######################################################
    #                                                     #
    #  Simple transformer demo                            #
    #                                                     #
    #  A binding that explicitly calls base defaults      #
    #                                                     #
    #######################################################

    @staticmethod
    def simple_edges_df_transformer(edges_df):
      edges_df = NodeXLGraphistryBase.edges_df_transformer_default(edges_df)
      return edges_df

    @staticmethod
    def simple_nodes_df_transformer(nodes_df):
      nodes_df = NodeXLGraphistryBase.nodes_df_transformer_default(nodes_df)
      return nodes_df

    #######################################################
    #                                                     #
    #  Twitter                                            #
    #                                                     #
    #######################################################

    @staticmethod
    def twitter_nodes_df_transformer(nodes_df):
        nodes_df = NodeXLGraphistryBase.nodes_df_transformer_default(nodes_df)
        return nodes_df.assign(**{
            **{col: NodeXLGraphistryBase.link_urls(nodes_df[col]) 
              for col in ['Domains in Tweet by Count', 'Domains in Tweet by Salience']},
            **{col: NodeXLGraphistryBase.embed_img(nodes_df[col])
              for col in ['Image File', 'Profile Background Image Url', 'Profile Banner Url']}
        })

    #######################################################
    #                                                     #
    #  Mediawiki                                          #
    #                                                     #
    #######################################################

    @staticmethod
    def mediawiki_nodes_df_transformer(nodes_df):
        nodes_df = NodeXLGraphistryBase.nodes_df_transformer_default(nodes_df)
        return nodes_df.assign(**{
            **{col: NodeXLGraphistryBase.embed_img(nodes_df[col])
              for col in ['Image File']}
        })
