import pandas as pd

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

#ex output: pd.DataFrame([{'val::state': 'CA', 'nodeType': 'state', 'nodeID': 'state::CA'}])
def format_entities(events, entity_types, defs, drop_na):
    cat_lookup = make_reverse_lookup(defs['CATEGORIES'])
    lst = sum([[{
                    col: v,
                    defs['TITLE']: v,
                    defs['NODETYPE']: col, 
                    defs['NODEID']: col2cat(cat_lookup, col) + defs['DELIM'] + str(v)
                } 
                for v in events[col].unique() if not drop_na or str(v) != 'nan'] for col in entity_types], [])
    df = pd.DataFrame(lst)
    df[defs['CATEGORY']] = df[defs['NODETYPE']].apply(lambda col: col2cat(cat_lookup, col))
    return df

DEFS_HYPER = {
    'TITLE': 'nodeTitle',
    'DELIM': '::',
    'NODEID': 'nodeID',
    'ATTRIBID': 'attribID',
    'EVENTID': 'EventID',
    'CATEGORY': 'category',
    'NODETYPE': 'type',
    'EDGETYPE': 'edgeType',
    'SKIP': [],
    'CATEGORIES': {} # { 'categoryName': ['colName', ...], ... }
}



#ex output: pd.DataFrame([{'edgeType': 'state', 'attribID': 'state::CA', 'eventID': 'eventID::0'}])
def format_hyperedges(events, entity_types, defs, drop_na, drop_edge_attrs):
    cat_lookup = make_reverse_lookup(defs['CATEGORIES'])
    subframes = []
    for col in entity_types:
        raw = events[[col, defs['EVENTID']]].copy()
        if drop_na:
            raw = raw.dropna()[[col, defs['EVENTID']]].copy()
        if len(raw):
            raw[defs['EDGETYPE']] = raw.apply(lambda r: col, axis=1)
            raw[defs['CATEGORY']] = raw.apply(lambda r: col2cat(cat_lookup, col), axis=1)
            raw[defs['ATTRIBID']] = raw.apply(lambda r: col2cat(cat_lookup, col) + defs['DELIM'] + str(r[col]), axis=1)
            subframes.append(raw)
    if len(subframes):
        return pd.concat(subframes).reset_index(drop=True)[[defs['EDGETYPE'], defs['ATTRIBID'], defs['EVENTID']]]
    return pd.DataFrame([])

def format_hypernodes(events, defs, drop_na):
    event_nodes = events.copy()
    event_nodes[defs['NODETYPE']] = defs['EVENTID']
    event_nodes[defs['NODEID']] = event_nodes[defs['EVENTID']]    
    event_nodes[defs['TITLE']] = event_nodes[defs['EVENTID']]    
    return event_nodes

def hyperbinding(g, defs, entities, event_entities, edges):
    nodes = pd.concat([entities, event_entities]).reset_index(drop=True)
    return {
        'entities': entities,
        'events': event_entities,
        'edges': edges,
        'nodes': nodes,
        'graph': g\
            .bind(source=defs['ATTRIBID'], destination=defs['EVENTID']).edges(edges)\
            .bind(node=defs['NODEID'], point_title=defs['TITLE']).nodes(nodes)
    }    

###########        

class Hypergraph(object):        

    @staticmethod
    def hypergraph(g, raw_events, entity_types=None, opts={}, drop_na=True, drop_edge_attrs=True, verbose=True):
        defs = makeDefs(DEFS_HYPER, opts)
        entity_types = screen_entities(raw_events, entity_types, defs)
        events = raw_events.copy().reset_index(drop=True)
        if defs['EVENTID'] in events.columns:
            events[defs['EVENTID']] = events.apply(
                lambda r: defs['EVENTID'] + defs['DELIM'] + str(r[defs['EVENTID']]), 
                axis=1)
        else:
            events[defs['EVENTID']] = events.reset_index().apply(
                lambda r: defs['EVENTID'] + defs['DELIM'] + str(r['index']), 
                axis=1)
        events[defs['NODETYPE']] = 'event'
        entities = format_entities(events, entity_types, defs, drop_na)
        event_entities = format_hypernodes(events, defs, drop_na)
        edges = format_hyperedges(events, entity_types, defs, drop_na, drop_edge_attrs)
        if verbose:
            print('# links', len(edges))
            print('# event entities', len(events))
            print('# attrib entities', len(entities))
        return hyperbinding(g, defs, entities, event_entities, edges)
