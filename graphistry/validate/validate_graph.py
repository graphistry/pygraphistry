def check_node_dataframe_exists(g, verbose=True):
    if g._nodes is None:
        if verbose:
            print("Warning: graph was created with only edges. Skipping Node ID check if Node IDs match edge IDs. Use g2 = g.materialize_nodes() to force node df creation. Exiting.")
        return False
    return True


def check_node_id_defined(g, verbose=True):
    if g._node is None:
        if verbose:
            print("Invalid graph: Missing Node ID. Did you forget to specify the node ID in the .nodes() function? Exiting.")
        return False
    return True


def check_nan_node_ids(g, verbose=True):
    if g._nodes[g._node].isnull().any():
        if verbose:
            print("Invalid graph: Contains NaN Node IDs.")
        return False
    return True


def check_duplicate_node_ids(g, verbose=True):
    if g._nodes[g._node].duplicated().any():
        if verbose:
            print("Invalid graph: Contains duplicate Node IDs.")
        return False
    return True


def check_edge_sources_exist_in_nodes(g, verbose=True):
    if not g._edges[g._source].isin(g._nodes[g._node]).all():
        if verbose:
            print("Warning: Contains source edge IDs that do not exist in the node DataFrame. This can cause unexpected results.")
    return True


def check_edge_destinations_exist_in_nodes(g, verbose=True):
    if not g._edges[g._destination].isin(g._nodes[g._node]).all():
        if verbose:
            print("Warning: Contains destination edge IDs that do not exist in the node DataFrame. This can cause unexpected results.")
    return True


def validate_graph(g, verbose=True):
    if not check_node_dataframe_exists(g, verbose):
        return False
    if not check_node_id_defined(g, verbose):
        return False
    if not check_nan_node_ids(g, verbose):
        return False
    if not check_duplicate_node_ids(g, verbose):
        return False
    check_edge_sources_exist_in_nodes(g, verbose)  # Warnings only
    check_edge_destinations_exist_in_nodes(g, verbose)  # Warnings only

    if verbose:
        print("Graph is valid.")
    return True
