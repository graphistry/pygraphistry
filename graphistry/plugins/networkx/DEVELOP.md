# NetworkX / SciPy Optional Dependency Policy

The `networkx` extra supports `networkx>=2.5,<4`. Keep this range aligned with
`graphistry.plugins.networkx.policy.NETWORKX_VERSION_SPEC` and the local Cypher
`graphistry.nx.*` CALL tests.

SciPy is optional for NetworkX-backed calls. The `networkx-scipy` extra declares
the tested SciPy range, `scipy>=1.5,<2`, for environments that want NetworkX
algorithms to use SciPy-backed implementations when NetworkX chooses them. The
GFQL CALL surface must still keep no-SciPy fallbacks for algorithms that have
local fallbacks, such as `graphistry.nx.pagerank` and `graphistry.nx.hits`.

When adding NetworkX-backed procedures:

- Use `graphistry.plugins.networkx.policy` for supported-version checks.
- Update the `test-networkx-scipy-policy` CI matrix if the new procedure needs
  a narrower NetworkX/SciPy range.
- Add coverage for both the lower supported NetworkX bound and the current
  upper-bound resolver path.
