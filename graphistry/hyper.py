import pandas as pd
import sys

from graphistry.constants import BINDING, BINDING_DEFAULT


DEFS_HYPER = {
    BINDING.NODE_ID:    BINDING_DEFAULT.NODE_ID,
    BINDING.NODE_TITLE: BINDING_DEFAULT.NODE_TITLE,
    BINDING.EDGE_SRC:   BINDING_DEFAULT.EDGE_SRC,
    BINDING.EDGE_DST:   BINDING_DEFAULT.EDGE_DST,
    'DELIM':            '::',
    'ATTRIBID':         'attribID',
    'EVENTID':          'EventID',
    'CATEGORY':         'category',
    'NODETYPE':         'type',
    'EDGETYPE':         'edgeType',
    'SKIP': [
    ],
    'CATEGORIES': { # { 'categoryName': ['colName', ...], ... }
    } 
}


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


def valToSafeStr(value):
    if sys.version_info < (3,0):
        t = type(value)
        if t is unicode: # noqa: F821
            return value
        elif t is str:
            return value
        else:
            return repr(value)
    else:
        t = type(value)
        if t is str:
            return value
        else:
            return repr(value)


# ex output: pd.DataFrame([{'val::state': 'CA', 'nodeType': 'state', 'nodeID': 'state::CA'}])
def format_entities(events, entity_types, defs, drop_node_attributes):
    cat_lookup = make_reverse_lookup(defs['CATEGORIES'])
    df = pd.DataFrame([
        node for node in events_to_nodes(events, entity_types, cat_lookup, defs, drop_node_attributes)
    ])

    return df


def events_to_nodes(events, entity_types, category_lookup, defs, drop_node_attributes):
    for column in entity_types:
        category = col2cat(category_lookup, column)
        for value in events[column].unique():
            if valToSafeStr(value) == 'nan':
                continue
            node_id = category + defs['DELIM'] + valToSafeStr(value)
            if drop_node_attributes:
                yield {
                    defs[BINDING.NODE_ID]: node_id,
                }
            else:
                yield {
                    column: value,
                    defs[BINDING.NODE_ID]: node_id,
                    defs[BINDING.NODE_TITLE]: value,
                    defs['NODETYPE']: column,
                    defs['CATEGORY']: category
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
        out = pd.concat(subframes, ignore_index=True).reset_index(drop=True)[ result_cols ]
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
              raw[defs[BINDING.EDGE_SRC]] = raw.apply(lambda r: col2cat(cat_lookup, col1) + defs['DELIM'] + valToSafeStr(r[col1]), axis=1)
              raw[defs[BINDING.EDGE_DST]] = raw.apply(lambda r: col2cat(cat_lookup, col2) + defs['DELIM'] + valToSafeStr(r[col2]), axis=1)
              subframes.append(raw)

    if len(subframes):
        result_cols = list(set(
            ([x for x in events.columns.tolist() if not x == defs['NODETYPE']]
                if not drop_edge_attrs
                else [])
            + [defs['EDGETYPE'], defs[BINDING.EDGE_SRC], defs[BINDING.EDGE_DST], defs['EVENTID']]
            + ([defs['CATEGORY']] if is_using_categories else []) ))
        out = pd.concat(subframes, ignore_index=True).reset_index(drop=True)[ result_cols ]
        return out
    else:
        return pd.DataFrame([])


def format_hypernodes(events, defs):
    event_nodes = events.copy()
    event_nodes[defs['NODETYPE']] = defs['EVENTID']
    event_nodes[defs['CATEGORY']] = 'event'
    event_nodes[defs[BINDING.NODE_ID]] = event_nodes[defs['EVENTID']]
    event_nodes[defs[BINDING.NODE_TITLE]] = event_nodes[defs['EVENTID']]
    event_nodes.loc[:, event_nodes.dtypes == object] = event_nodes.loc[:, event_nodes.dtypes == object].where(pd.notnull(event_nodes), None)
    return event_nodes


class Hypergraph(object):

    @staticmethod
    def hypergraph(plotter, raw_events, entity_types=None, opts={}, drop_na=True, drop_edge_attrs=False, verbose=True, direct=False):
        defs = makeDefs(DEFS_HYPER, opts)
        entity_types = screen_entities(raw_events, entity_types, defs)
        events = raw_events.copy().reset_index(drop=True)
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
            event_entities = format_hypernodes(events, defs)
            edges = format_hyperedges(events, entity_types, defs, drop_na, drop_edge_attrs)

        if verbose:
            print('# links', len(edges))
            print('# events', len(events))
            print('# attrib entities', len(entities))


        nodes = pd.concat([entities, event_entities], ignore_index=True).reset_index(drop=True)

        # For any column containing object types, replace all NaN values with None (pyarrow doesn't convert it automatically)

        nodes.loc[:, nodes.dtypes == object] = nodes.loc[:, nodes.dtypes == object] \
            .where(pd.notnull(nodes), None) \
            .astype(str)

        edges.loc[:, edges.dtypes == object] = edges.loc[:, edges.dtypes == object] \
            .where(pd.notnull(edges), None) \
            .astype(str)

        plotter = plotter \
            .data(edges=edges, nodes=nodes) \
            .bind(
                source =      defs[BINDING.EDGE_SRC] if direct else defs['ATTRIBID'],
                destination = defs[BINDING.EDGE_DST] if direct else defs['EVENTID'],
                nodeId =      defs[BINDING.NODE_ID]
            )

        if not direct:
            plotter = plotter.bind(
                pointTitle = defs[BINDING.NODE_TITLE]
            )


        return {
            'entities': entities,
            'events': event_entities,
            'edges': edges,
            'nodes': nodes,
            'graph': plotter
        }
