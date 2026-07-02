export const DEFAULT_PYODIDE_INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.0/full/";

const DEFAULT_PYODIDE_PACKAGES = [
  "micropip",
  "pandas",
  "requests",
  "packaging",
  "typing-extensions",
];
const DEFAULT_GRAPHISTRY_REQUIREMENTS = [
  "lark>=1.1,<2",
];

function setGlobals(pyodide, values) {
  for (const [key, value] of Object.entries(values)) {
    pyodide.globals.set(key, value);
  }
}

async function installGraphistryWheel(pyodide, graphistryWheel) {
  if (!graphistryWheel) {
    return;
  }

  let wheelTarget = graphistryWheel;
  if (typeof graphistryWheel !== "string") {
    wheelTarget = graphistryWheel.path || "/tmp/graphistry-pyodide.whl";
    pyodide.FS.writeFile(wheelTarget, graphistryWheel.data);
    setGlobals(pyodide, { _gfql_graphistry_wheel: wheelTarget });
    await pyodide.runPythonAsync(`
import sysconfig
import zipfile
from pathlib import PurePosixPath

with zipfile.ZipFile(_gfql_graphistry_wheel) as _gfql_wheel:
    for _gfql_member in _gfql_wheel.infolist():
        _gfql_path = PurePosixPath(_gfql_member.filename)
        if _gfql_path.is_absolute() or ".." in _gfql_path.parts:
            raise ValueError(f"Unsafe wheel member path: {_gfql_member.filename}")
    _gfql_wheel.extractall(sysconfig.get_paths()["purelib"])
`);
    return;
  }

  setGlobals(pyodide, { _gfql_graphistry_wheel: wheelTarget });
  await pyodide.runPythonAsync(`
import micropip
await micropip.install(_gfql_graphistry_wheel, deps=False)
`);
}

function mountRequirementWheels(pyodide, requirements) {
  return requirements.map((requirement, index) => {
    if (typeof requirement === "string") {
      return requirement;
    }
    const path = requirement.path || `/tmp/gfql-requirement-${index}.whl`;
    pyodide.FS.writeFile(path, requirement.data);
    return `emfs:${path}`;
  });
}

async function retryAsync(fn, { attempts = 3, delayMs = 1000 } = {}) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt < attempts) {
        await new Promise((resolve) => setTimeout(resolve, delayMs * attempt));
      }
    }
  }
  throw lastError;
}

export async function createGFQLRuntime({
  loadPyodide,
  indexURL,
  packageBaseUrl,
  pyodidePackages = DEFAULT_PYODIDE_PACKAGES,
  requirements = DEFAULT_GRAPHISTRY_REQUIREMENTS,
  graphistryWheel,
  stdout,
  stderr,
} = {}) {
  if (!loadPyodide) {
    throw new Error("createGFQLRuntime requires a loadPyodide function");
  }

  const loadPyodideOptions = { stdout, stderr };
  if (indexURL) {
    loadPyodideOptions.indexURL = indexURL;
  }
  if (packageBaseUrl) {
    loadPyodideOptions.packageBaseUrl = packageBaseUrl;
  }

  const pyodide = await loadPyodide(loadPyodideOptions);
  await retryAsync(() => pyodide.loadPackage(pyodidePackages));

  setGlobals(pyodide, { _gfql_requirements: mountRequirementWheels(pyodide, requirements) });
  await pyodide.runPythonAsync(`
import micropip
await micropip.install(_gfql_requirements)
`);
  await installGraphistryWheel(pyodide, graphistryWheel);

  await pyodide.runPythonAsync(`
import copy
import pandas as pd
from graphistry.compute.ComputeMixin import ComputeMixin
from graphistry.compute import e, ge

class GFQLMiniGraph(ComputeMixin):
    def __init__(self):
        super().__init__()
        self._edges = None
        self._nodes = None
        self._source = None
        self._destination = None
        self._node = None
        self._edge = None

    def bind(self, source=None, destination=None, node=None, edge=None, **kwargs):
        out = copy.copy(self)
        if source is not None:
            out._source = source
        if destination is not None:
            out._destination = destination
        if node is not None:
            out._node = node
        if edge is not None:
            out._edge = edge
        return out

    def edges(self, edges, source=None, destination=None, edge=None, **kwargs):
        if callable(edges):
            edges = edges(self)
        out = self.bind(source=source, destination=destination, edge=edge)
        out._edges = edges
        return out

    def nodes(self, nodes, node=None, **kwargs):
        if callable(nodes):
            nodes = nodes(self)
        out = self.bind(node=node)
        out._nodes = nodes
        return out

def _gfql_graph(edges, nodes, source, destination, node):
    return GFQLMiniGraph().edges(edges, source, destination).nodes(nodes, node)
`);

  return new GFQLRuntime(pyodide);
}

export class GFQLRuntime {
  constructor(pyodide) {
    this.pyodide = pyodide;
  }

  async runEdgeWeightAtLeast({
    csv,
    source = "src",
    destination = "dst",
    weightColumn = "weight",
    minWeight = 2,
  }) {
    setGlobals(this.pyodide, {
      _gfql_csv: csv,
      _gfql_source: source,
      _gfql_destination: destination,
      _gfql_weight_column: weightColumn,
      _gfql_min_weight: minWeight,
    });

    const jsonText = await this.pyodide.runPythonAsync(`
import io
import json
import pandas as pd
from graphistry.compute import e, ge

def _records(df):
    if df is None:
        return []
    return json.loads(df.to_json(orient="records"))

_edges = pd.read_csv(io.StringIO(_gfql_csv))
_node_ids = pd.unique(_edges[[_gfql_source, _gfql_destination]].to_numpy().ravel())
_nodes = pd.DataFrame({"id": _node_ids})
_graph = _gfql_graph(_edges, _nodes, _gfql_source, _gfql_destination, "id")
_result = _graph.gfql([e(edge_match={_gfql_weight_column: ge(_gfql_min_weight)})])
json.dumps({
    "edges": _records(getattr(_result, "_edges", None)),
    "nodes": _records(getattr(_result, "_nodes", None)),
})
`);
    return JSON.parse(jsonText);
  }

  async runCypherCsv({
    csv,
    query,
    source = "src",
    destination = "dst",
  }) {
    setGlobals(this.pyodide, {
      _gfql_csv: csv,
      _gfql_query: query,
      _gfql_source: source,
      _gfql_destination: destination,
    });

    const jsonText = await this.pyodide.runPythonAsync(`
import io
import json
import pandas as pd

def _records(df):
    if df is None:
        return []
    return json.loads(df.to_json(orient="records"))

_edges = pd.read_csv(io.StringIO(_gfql_csv))
_node_ids = pd.unique(_edges[[_gfql_source, _gfql_destination]].to_numpy().ravel())
_nodes = pd.DataFrame({"id": _node_ids})
_graph = _gfql_graph(_edges, _nodes, _gfql_source, _gfql_destination, "id")
_result = _graph.gfql(_gfql_query, language="cypher")
json.dumps({
    "edges": _records(getattr(_result, "_edges", None)),
    "nodes": _records(getattr(_result, "_nodes", None)),
})
`);
    return JSON.parse(jsonText);
  }
}
