import pandas as pd
import sys

### COMMON TO HYPERGRAPH AND SIMPLE GRAPH
def makeDefs(DEFS, opts={}):
    defs = {key: opts[key] if key in opts else DEFS[key] for key in DEFS}    
    base_skip = opts['SKIP'] if 'SKIP' in opts else defs['SKIP']
    skip = [x for x in base_skip] #copy
    defs['SKIP'] = skip
    for key in DEFS:
        if not defs[key] in skip:
            skip.append(defs[key])
    return defs

def screen_entities(events, entity_types, defs):
    base = entity_types if not entity_types == None else events.columns
    return [x for x in base if not x in defs['SKIP']]


def col2cat(cat_lookup, col):
    return cat_lookup[col] if col in cat_lookup else col

def make_reverse_lookup(categories):
    lookup = {}
    for category in categories:
        for col in categories[category]:
            lookup[col] = category
    return lookup



def valToSafeStr (v): 
    if sys.version_info < (3,0):        
        t = type(v)
        if t is unicode: # noqa: F821
            return v
        elif t is str:
            return v
        else:
            return repr(v)
    else:
        t = type(v)
        if t is str:
            return v
        else:
            return repr(v)        


#ex output: pd.DataFrame([{'val::state': 'CA', 'nodeType': 'state', 'nodeID': 'state::CA'}])
def format_entities(events, entity_types, defs, drop_na):
    cat_lookup = make_reverse_lookup(defs['CATEGORIES'])
    lst = sum([[{
                    col: v,
                    defs['TITLE']: v,
                    defs['NODETYPE']: col, 
                    defs['NODEID']: col2cat(cat_lookup, col) + defs['DELIM'] + valToSafeStr(v)
                } 
                for v in events[col].unique() if not drop_na or valToSafeStr(v) != 'nan'] for col in entity_types], [])
    df = pd.DataFrame(lst).drop_duplicates([defs['NODEID']])
    df[defs['CATEGORY']] = df[defs['NODETYPE']].apply(lambda col: col2cat(cat_lookup, col))
    return df

DEFS_HYPER = {
    'TITLE': 'nodeTitle',
    'DELIM': '::',
    'NODEID': 'nodeID',
    'ATTRIBID': 'attribID',
    'EVENTID': 'EventID',
    'SOURCE': 'src',
    'DESTINATION': 'dst',    
    'CATEGORY': 'category',
    'NODETYPE': 'type',
    'EDGETYPE': 'edgeType',
    'SKIP': [],
    'CATEGORIES': {} # { 'categoryName': ['colName', ...], ... }
}



#ex output: pd.DataFrame([{'edgeType': 'state', 'attribID': 'state::CA', 'eventID': 'eventID::0'}])
def format_hyperedges(events, entity_types, defs, drop_na, drop_edge_attrs):
    is_using_categories = len(defs['CATEGORIES'].keys()) > 0
    cat_lookup = make_reverse_lookup(defs['CATEGORIES'])

    subframes = []
    for col in sorted(entity_types):
        fields = list(set([defs['EVENTID']] + ([x for x in events.columns] if not drop_edge_attrs else [col])))
        raw = events[ fields ]
        if drop_na:
            raw = raw.dropna(axis=0, subset=[col])            
        raw = raw.copy()
        if len(raw):            
            if is_using_categories:
                raw[defs['EDGETYPE']] = raw.apply(lambda r: col2cat(cat_lookup, col), axis=1)
                raw[defs['CATEGORY']] = raw.apply(lambda r: col, axis=1)
            else:
                raw[defs['EDGETYPE']] = raw.apply(lambda r: col, axis=1)
            raw[defs['ATTRIBID']] = raw.apply(lambda r: col2cat(cat_lookup, col) + defs['DELIM'] + valToSafeStr(r[col]), axis=1)
            subframes.append(raw)

    if len(subframes):
        result_cols = list(set(
            ([x for x in events.columns.tolist() if not x == defs['NODETYPE']] 
                if not drop_edge_attrs 
                else [])
            + [defs['EDGETYPE'], defs['ATTRIBID'], defs['EVENTID']]
            + ([defs['CATEGORY']] if is_using_categories else []) ))
        out = pd.concat(subframes, ignore_index=True, sort=False).reset_index(drop=True)[ result_cols ]
        return out
    else:
        return pd.DataFrame([])


# [ str ] * {?'EDGES' : ?{str: [ str ] }} -> {str: [ str ]}
def direct_edgelist_shape(entity_types, defs):  
  if 'EDGES' in defs and not defs['EDGES'] is None:
      return defs['EDGES']
  else:
      out = {}
      for entity_i in range(len(entity_types)):
        out[ entity_types[entity_i] ] = entity_types[(entity_i + 1):]
      return out
  
      
#ex output: pd.DataFrame([{'edgeType': 'state', 'attribID': 'state::CA', 'eventID': 'eventID::0'}])
def format_direct_edges(events, entity_types, defs, edge_shape, drop_na, drop_edge_attrs):
    is_using_categories = len(defs['CATEGORIES'].keys()) > 0
    cat_lookup = make_reverse_lookup(defs['CATEGORIES'])

    subframes = []
    for col1 in sorted(edge_shape.keys()):
      for col2 in edge_shape[col1]:
          fields = list(set([defs['EVENTID']] + ([x for x in events.columns] if not drop_edge_attrs else [col1, col2])))
          raw = events[ fields ]
          if drop_na:
              raw = raw.dropna(axis=0, subset=[col1, col2])            
          raw = raw.copy()
          if len(raw):            
              if not drop_edge_attrs:
                  if is_using_categories:
                      raw[defs['EDGETYPE']] = raw.apply(lambda r: col2cat(cat_lookup, col1) + defs['DELIM'] + col2cat(cat_lookup, col2), axis=1)
                      raw[defs['CATEGORY']] = raw.apply(lambda r: col1 + defs['DELIM'] + col2, axis=1)
                  else:
                      raw[defs['EDGETYPE']] = raw.apply(lambda r: col1 + defs['DELIM'] + col2, axis=1)
              raw[defs['SOURCE']] = raw.apply(lambda r: col2cat(cat_lookup, col1) + defs['DELIM'] + valToSafeStr(r[col1]), axis=1)
              raw[defs['DESTINATION']] = raw.apply(lambda r: col2cat(cat_lookup, col2) + defs['DELIM'] + valToSafeStr(r[col2]), axis=1)
              subframes.append(raw)

    if len(subframes):
        result_cols = list(set(
            ([x for x in events.columns.tolist() if not x == defs['NODETYPE']] 
                if not drop_edge_attrs 
                else [])
            + [defs['EDGETYPE'], defs['SOURCE'], defs['DESTINATION'], defs['EVENTID']]
            + ([defs['CATEGORY']] if is_using_categories else []) ))
        out = pd.concat(subframes, ignore_index=True).reset_index(drop=True)[ result_cols ]
        return out
    else:
        return pd.DataFrame([])


def format_hypernodes(events, defs, drop_na):
    event_nodes = events.copy()
    event_nodes[defs['NODETYPE']] = defs['EVENTID']
    event_nodes[defs['CATEGORY']] = 'event'
    event_nodes[defs['NODEID']] = event_nodes[defs['EVENTID']]    
    event_nodes[defs['TITLE']] = event_nodes[defs['EVENTID']]    
    return event_nodes

def hyperbinding(g, defs, entities, event_entities, edges, source, destination):
    nodes = pd.concat([entities, event_entities], ignore_index=True, sort=False).reset_index(drop=True)
    return {
        'entities': entities,
        'events': event_entities,
        'edges': edges,
        'nodes': nodes,
        'graph': g\
            .bind(source=source, destination=destination).edges(edges)\
            .bind(node=defs['NODEID'], point_title=defs['TITLE']).nodes(nodes)
    }     

#turn lists etc to strs, and preserve nulls
def flatten_objs_inplace(df, cols):
   for c in cols:
        name = df[c].dtype.name
        if name == 'category':
            #Avoid warning
            df[c] = df[c].astype(str).where(~df[c].isnull(), df[c])
        elif name == 'object':
            df[c] = df[c].where(df[c].isnull(), df[c].astype(str))
 
###########        

class Hypergraph(object):        

    @staticmethod
    def hypergraph(g, raw_events, entity_types=None, opts={}, drop_na=True, drop_edge_attrs=False, verbose=True, direct=False):
        defs = makeDefs(DEFS_HYPER, opts)
        entity_types = screen_entities(raw_events, entity_types, defs)
        events = raw_events.copy().reset_index(drop=True)
        flatten_objs_inplace(events, entity_types)

        if defs['EVENTID'] in events.columns:
            events[defs['EVENTID']] = events.apply(
                lambda r: defs['EVENTID'] + defs['DELIM'] + valToSafeStr(r[defs['EVENTID']]), 
                axis=1)
        else:
            events[defs['EVENTID']] = events.reset_index().apply(
                lambda r: defs['EVENTID'] + defs['DELIM'] + valToSafeStr(r['index']),
                axis=1)
        events[defs['NODETYPE']] = 'event'
        
        entities = format_entities(events, entity_types, defs, drop_na)
        event_entities = None
        edges = None
        if direct:
            edge_shape = direct_edgelist_shape(entity_types, opts)
            event_entities = pd.DataFrame()
            edges = format_direct_edges(events, entity_types, defs, edge_shape, drop_na, drop_edge_attrs)
        else:        
            event_entities = format_hypernodes(events, defs, drop_na)
            edges = format_hyperedges(events, entity_types, defs, drop_na, drop_edge_attrs)
        if verbose:
            print('# links', len(edges))
            print('# events', len(events))
            print('# attrib entities', len(entities))
        return hyperbinding(
            g, defs, entities, event_entities, edges,
            defs['SOURCE'] if direct else defs['ATTRIBID'],
            defs['DESTINATION'] if direct else defs['EVENTID'])