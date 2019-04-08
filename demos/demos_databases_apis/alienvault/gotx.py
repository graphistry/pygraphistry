import pandas as pd

### Helpers
class GraphistryHelpers:
    @staticmethod
    def apply_dict(df, col, dictionary):
      return df[col].apply(lambda v: dictionary[v])

    @staticmethod
    def color_nodes_by_dict(g, col, dictionary):  
      node_colors_s = GraphistryHelpers.apply_dict(g._nodes, col, dictionary)
      return g.nodes(g._nodes.assign(p_colors=node_colors_s)).bind(point_color='p_colors')

    @staticmethod
    def color_edges_by_dict(g, col, dictionary):  
      edge_colors_s = GraphistryHelpers.apply_dict(g._edges, col, dictionary)
      return g.edges(g._edges.assign(e_colors=edge_colors_s)).bind(edge_color='e_colors')

    @staticmethod
    def ungroup_one(df, col_list_name, col_singleton_name):
        s = df.apply(lambda x: pd.Series(x[col_list_name]),axis=1).stack().reset_index(level=1, drop=True)
        s.name = col_singleton_name
        return df.drop(col_list_name, axis=1).join(s)  

    @staticmethod
    def ungroup(df, group_colname_to_singleton_colname):
        for (group_colname, singleton_colname) in group_colname_to_singleton_colname.items():
            if group_colname in df.columns:
                df = GraphistryHelpers.ungroup_one(df, group_colname, singleton_colname)
            else:
                print('Warning: no group column', group_colname)
        return df
      

class G_OTX:
  
    def __init__(self, graphistry, otx):
        self.graphistry = graphistry
        self.otx = otx
        self.tlp_to_color = {
              'red': 5,
              'amber': 10,
              'green': 2,
              'white': 10000
          }
        self.type_to_color = {
              'id': 60003, #silver
              'country': 11, #brown
              'industry': 0 #blue
          }
   
    def pulses_to_df(self, pulses):
        df = pd.io.json.json_normalize(pulses)
        df = df.rename(columns={'created': 'createdTime', 'modified': 'modifiedTime'})
        return df
      
    def pulses_to_indicators_df(self, pulses):
        indicators = []
        for pulse in pulses:
            for indicator in pulse.get('indicators'):
                indicators.append({'pulse_name': pulse.get('name'), 'indicator': indicator.get('indicator'), 'indicator_type': indicator.get('type')})    
        indicators_df = pd.DataFrame(indicators)    
        #print('# indicators', len(indicators_df))
        #indicators_df.sample(3)
        return indicators_df
     
      
    def industrymap(self, pulses_df, include_indicators=False):
        expanded_df = GraphistryHelpers.ungroup(pulses_df, {'industries': 'industry',  'targeted_countries': 'country'}).dropna(subset=['industry', 'country'])
        for c in ['industry', 'country']:
            expanded_df[c] = expanded_df[c].str.lower()
        #print('# rows', len(expanded_df))
        #print('# Reports with industry<>country', len(expanded_df['id'].unique()))
        #expanded_df[:3]
        hg = None
        if include_indicators:
            hg = self.graphistry.hypergraph(
                expanded_df.astype(str),
                ['industry', 'country', 'id'], 
                direct=True,
                opts={'EDGES': {
                   'id': ['industry', 'country']
            }})          
        else:
            hg = self.graphistry.hypergraph(
                expanded_df.astype(str),
                ['industry', 'country'], 
                direct=True,
                opts={'EDGES': {
                   'industry': ['country']
            }})
        g = hg['graph']
        g = GraphistryHelpers.color_edges_by_dict(g, 'tlp', self.tlp_to_color)
        g = GraphistryHelpers.color_nodes_by_dict(g, 'type', self.type_to_color)        
        return g

      
      
    def indicator_details_by_section_to_pulses_df(self, pulses):
        pulses_df = pd.DataFrame([self.otx.get_pulse_details(pulse.get('id')) for pulse in pulses.get('pulse_info').get('pulses')]) 
        return pulses_df
      
    def indicator_details_by_section_to_indicators_df(self, pulses):
        indicators_df = pd.DataFrame()
        for pulse in pulses.get('pulse_info').get('pulses'):
            p_details = self.otx.get_pulse_details(pulse.get('id'))
            chunk_df = pd.DataFrame([
                {'pulse_name': pulse.get('name'), 'indicator': indicator.get('indicator'), 'indicator_type': indicator.get('type')}
                for indicator in p_details.get('indicators')
            ])
            indicators_df = pd.concat([indicators_df, chunk_df], ignore_index=True)
        return indicators_df   
      
      
    def indicatormap(self, pulses_df, indicators_df):
      
      pulses_df = pulses_df.copy()
      for col in pulses_df.columns:#['industries', 'references', 'tags', 'targeted_countries', 'name']:
          if col in pulses_df.columns:
              pulses_df[col] = pulses_df[col].astype(str)

      indicators_df = indicators_df.copy()
      for col in indicators_df.columns:  
          indicators_df[col] = indicators_df[col].astype(str)
              

      hg = self.graphistry.hypergraph(
          indicators_df,
          ['indicator', 'pulse_name'],
          direct=True)

      g = hg['graph']

      #enrich indicator nodes
      nodes2 = g._nodes.set_index('indicator').join(indicators_df[['indicator', 'indicator_type']].set_index('indicator')).reset_index().rename(columns={'index': 'indicator'})
      g = g.nodes(nodes2)

      #enrich pulse nodes
      nodes2 = g._nodes.set_index('pulse_name').join(pulses_df.set_index('name')).reset_index().rename(columns={'index': 'pulse_name'})
      g = g.nodes(nodes2)

      #TODO hardcode to make stable
      lst = list(g._nodes['indicator_type'].unique())
      indicatortype_to_color = {i: lst.index(i) % 13 for i in lst}

      g = GraphistryHelpers.color_nodes_by_dict(g, 'indicator_type', indicatortype_to_color)

      return g
      

