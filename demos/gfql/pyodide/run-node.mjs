import { readFile } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { createGFQLRuntime } from "./gfql.js";

const wheelPath = process.argv[2];
if (!wheelPath) {
  throw new Error("Usage: node demos/gfql/pyodide/run-node.mjs <graphistry-wheel.whl>");
}

const pyodideModule = process.env.PYODIDE_MODULE || "pyodide";
const pyodideIndexURL = process.env.PYODIDE_INDEX_URL
  || (pyodideModule === "pyodide" ? undefined : dirname(resolve(pyodideModule)));
const { loadPyodide } = await import(pyodideModule);
const csv = await readFile(new URL("./edges.csv", import.meta.url), "utf8");
const wheelBytes = await readFile(resolve(wheelPath));
const requirementWheelPaths = (process.env.GFQL_REQUIREMENT_WHEELS || "")
  .split(":")
  .map((value) => value.trim())
  .filter(Boolean);
const requirements = await Promise.all(requirementWheelPaths.map(async (requirementPath) => ({
  path: `/tmp/${basename(requirementPath)}`,
  data: await readFile(resolve(requirementPath)),
})));

const runtime = await createGFQLRuntime({
  loadPyodide,
  ...(pyodideIndexURL ? { indexURL: pyodideIndexURL } : {}),
  ...(requirements.length > 0 ? { requirements } : {}),
  graphistryWheel: {
    path: `/tmp/${basename(wheelPath)}`,
    data: wheelBytes,
  },
});

const astResult = await runtime.runEdgeWeightAtLeast({ csv, minWeight: 2 });
if (astResult.edges.length !== 2) {
  throw new Error(`Expected 2 AST GFQL edges, got ${astResult.edges.length}`);
}
if (!astResult.edges.every((edge) => edge.weight >= 2)) {
  throw new Error(`Expected AST GFQL weights >= 2: ${JSON.stringify(astResult.edges)}`);
}

const cypherResult = await runtime.runCypherCsv({
  csv,
  query: "MATCH (a)-[e]->(b) WHERE e.weight >= 2 RETURN e",
});
if (cypherResult.nodes.length !== 2) {
  throw new Error(`Expected 2 Cypher rows, got ${cypherResult.nodes.length}`);
}

console.log(JSON.stringify({
  astEdges: astResult.edges,
  cypherRows: cypherResult.nodes,
}, null, 2));
