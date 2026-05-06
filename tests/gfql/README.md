# GFQL tests

This folder hosts GFQL reference and parity tests (oracle enumerator,
executor inputs, and same-path plan coverage).

Cypher TCK conformance tests live in the external repo:
https://github.com/graphistry/tck-gfql

To run the TCK harness locally against a pygraphistry checkout:
```bash
git clone https://github.com/graphistry/tck-gfql.git
cd tck-gfql
PYGRAPHISTRY_PATH=/path/to/pygraphistry PYGRAPHISTRY_INSTALL=1 ./bin/ci.sh
```
