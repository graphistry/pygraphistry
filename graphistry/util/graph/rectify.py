import pyarrow as arrow


int32 = arrow.int32()
int64 = arrow.int64()


def rectify(
    edges,
    nodes,
    edge,
    node,
    edge_src,
    edge_dst,
    safe = True
):
    return _rectify_node_ids(
        edges=_rectify_edge_ids(
            edges=edges,
            edge=edge,
            safe=safe
        ),
        nodes=nodes,
        node=node,
        edge_src=edge_src,
        edge_dst=edge_dst,
        safe=safe
    )


def _rectify_edge_ids(
    edges,
    edge,
    safe = True
):

    edge_column = None
    edge_column_id = edges.schema.get_field_index(edge)

    if edge_column_id < 0:
        edge_column = arrow.column(edge, [[x for x in range(edges.num_rows)]]).cast(int32)
        return edges.append_column(edge_column)
    else:
        edge_column = edges.column(edge_column_id)

    if edge_column.type == int32:
        return edges

    if edge_column.type == int64:
        return edges.set_column(
            edge_column_id,
            edge_column.cast(int32, safe=safe)
        )

    return edges.set_column(
        edge_column_id,
        arrow.column(edge_column.name, [range(edges.num_rows)]).cast(
            int32, safe=safe)
    )


def _rectify_node_ids(
    edges,
    nodes,
    node,
    edge_src,
    edge_dst,
    safe=True
):
    # make sure id columns are int32, which may require one of the following:
    # - down-cast from int64
    # - create index via node column and map src/dst/node to an index.
    # - dictionary encode the column (not server support yet)
    node = nodes.schema.get_field_index(node)
    edge_src = edges.schema.get_field_index(edge_src)
    edge_dst = edges.schema.get_field_index(edge_dst)

    node_column = nodes.column(node)
    edge_src_column = edges.column(edge_src)
    edge_dst_column = edges.column(edge_dst)

    _assert_column_types_match(node_column, edge_src_column)
    _assert_column_types_match(node_column, edge_dst_column)

    # already good to go.
    if node_column.type == int32:
        return (edges, nodes)

    # convert int64 => int32 if no overflow.
    if node_column.type == int64:
        edges = edges \
            .set_column(edge_src, edge_src_column.cast(int32, safe=safe)) \
            .set_column(edge_dst, edge_dst_column.cast(int32, safe=safe))

        nodes = nodes \
            .set_column(node, node_column.cast(int32, safe=safe))

        return (edges, nodes)

    # replace existing src/dst/node columns with equivolent indices
    index_lookup = _index_by_value(node_column)
    edge_src_column = _map_column_to_index(index_lookup, edge_src_column)
    edge_dst_column = _map_column_to_index(index_lookup, edge_dst_column)
    node_column = _map_column_to_index(index_lookup, node_column)

    edges = edges \
        .set_column(edge_src, edge_src_column) \
        .set_column(edge_dst, edge_dst_column)

    nodes = nodes \
        .set_column(node, node_column)

    return (edges, nodes)


def _index_by_value(iterable):
    keys = {}
    for (index, value) in enumerate(iterable):
        keys[value] = index
    return keys


def _map_column_to_index(lookup, column):
    indicies = [lookup[value] for value in column]
    return arrow.column(column.name, [indicies]).cast(int32, safe=False)


def _assert_column_types_match(expected, actual):
    if actual.type == expected.type:
        return

    raise Exception(
        'column types mismatch (%s/%s). expected(%s) actual(%s)' % (
            expected.name,
            actual.name,
            expected.type,
            actual.type
        )
    )
