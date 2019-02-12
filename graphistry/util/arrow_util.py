import pandas
import pyarrow as arrow


# Consider where to move to_buffer
def table_to_buffer(table):
    sink = arrow.BufferOutputStream()
    writer = arrow.RecordBatchStreamWriter(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return sink.getvalue()


def to_arrow(source):
    if source is None:
        return None

    if isinstance(source, arrow.Table):
        return source

    if isinstance(source, pandas.DataFrame):
        return arrow.Table.from_pandas(source, preserve_index=False)

    raise Exception('unsupported data source type: %s' % type(source))
