import { readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { createGFQLRuntime } from "./gfql.js";

const bundleDir = resolve(process.argv[2] || "/tmp/pygraphistry-gfql-pyodide-bundle");
const sizes = (process.env.GFQL_BENCH_SIZES || "10,1000,10000")
  .split(",")
  .map((value) => Number(value.trim()))
  .filter((value) => Number.isFinite(value) && value > 0);
const repeat = Number(process.env.GFQL_BENCH_REPEAT || "3");

function generateCsv(edgeCount) {
  const lines = ["src,dst,weight"];
  for (let i = 0; i < edgeCount; i += 1) {
    lines.push(`n${i},n${i + 1},${i % 5}`);
  }
  return `${lines.join("\n")}\n`;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

async function timed(fn) {
  const start = performance.now();
  const value = await fn();
  return { value, ms: performance.now() - start };
}

function markdownTable(report) {
  const lines = [
    "| edges | AST GFQL median ms | Cypher median ms | returned rows |",
    "| ---: | ---: | ---: | ---: |",
  ];
  for (const row of report.queries) {
    lines.push(
      `| ${row.edges} | ${row.astMedianMs.toFixed(1)} | ${row.cypherMedianMs.toFixed(1)} | ${row.rows} |`,
    );
  }
  return lines.join("\n");
}

async function main() {
  const manifest = JSON.parse(await readFile(join(bundleDir, "manifest.json"), "utf8"));
  const sizeReport = JSON.parse(await readFile(join(bundleDir, "size-report.json"), "utf8"));
  const wheelPath = join(bundleDir, manifest.graphistryWheel.replace("./", ""));
  const wheelData = new Uint8Array(await readFile(wheelPath));
  const pyodideModule = process.env.PYODIDE_MODULE || join(bundleDir, "pyodide/pyodide.mjs");
  if (/^https?:\/\//.test(pyodideModule)) {
    throw new Error("benchmark-node.mjs needs a local Pyodide module. Build with --flavor self-hosted or set PYODIDE_MODULE to a local pyodide.mjs.");
  }
  const requirements = await Promise.all(manifest.requirements.map(async (requirement) => {
    if (!requirement.startsWith("./")) {
      return requirement;
    }
    const path = join(bundleDir, requirement.replace("./", ""));
    return {
      path: `/tmp/${path.split("/").pop()}`,
      data: new Uint8Array(await readFile(path)),
    };
  }));

  const importResult = await timed(() => import(pyodideModule));
  const runtimeResult = await timed(() => createGFQLRuntime({
    loadPyodide: importResult.value.loadPyodide,
    indexURL: manifest.indexURL.startsWith("./")
      ? `${join(bundleDir, manifest.indexURL.replace("./", ""))}/`
      : manifest.indexURL,
    packageBaseUrl: manifest.packageBaseUrl && manifest.packageBaseUrl.startsWith("./")
      ? `${join(bundleDir, manifest.packageBaseUrl.replace("./", ""))}/`
      : manifest.packageBaseUrl,
    pyodidePackages: manifest.pyodidePackages,
    requirements,
    graphistryWheel: {
      path: `/tmp/${wheelPath.split("/").pop()}`,
      data: wheelData,
    },
  }));
  const runtime = runtimeResult.value;

  const warmCsv = generateCsv(10);
  await runtime.runEdgeWeightAtLeast({ csv: warmCsv, minWeight: 3 });
  await runtime.runCypherCsv({
    csv: warmCsv,
    query: "MATCH (a)-[e]->(b) WHERE e.weight >= 3 RETURN e",
  });

  const queries = [];
  for (const edgeCount of sizes) {
    const csv = generateCsv(edgeCount);
    const astTimes = [];
    const cypherTimes = [];
    let rows = 0;
    for (let i = 0; i < repeat; i += 1) {
      const ast = await timed(() => runtime.runEdgeWeightAtLeast({ csv, minWeight: 3 }));
      const cypher = await timed(() => runtime.runCypherCsv({
        csv,
        query: "MATCH (a)-[e]->(b) WHERE e.weight >= 3 RETURN e",
      }));
      astTimes.push(ast.ms);
      cypherTimes.push(cypher.ms);
      rows = ast.value.edges.length;
    }
    queries.push({
      edges: edgeCount,
      rows,
      astMedianMs: median(astTimes),
      cypherMedianMs: median(cypherTimes),
      astMs: astTimes,
      cypherMs: cypherTimes,
    });
  }

  const report = {
    pyodideVersion: manifest.pyodideVersion,
    bundleBytes: sizeReport.totalBytes,
    pyodideBytes: sizeReport.pyodideBytes,
    wheelsBytes: sizeReport.wheelsBytes,
    importPyodideModuleMs: importResult.ms,
    createRuntimeMs: runtimeResult.ms,
    repeat,
    queries,
  };

  console.log(JSON.stringify(report, null, 2));
  console.log("\n" + markdownTable(report));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
