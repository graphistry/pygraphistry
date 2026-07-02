import { copyFile, cp, mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { basename, dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const PYODIDE_VERSION = "314.0.0";
const PYODIDE_CDN_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const BUNDLE_FLAVORS = new Set(["self-hosted", "cdn"]);
const PYODIDE_CORE_PACKAGES = [
  "micropip",
  "pandas",
];
const PYODIDE_REQUIREMENTS = [
  "requests",
  "packaging",
  "typing-extensions",
];
const VENDORED_WHEEL_REQUIREMENTS = [
  "lark>=1.1,<2",
];

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "../../..");
const defaultOutDir = "/tmp/pygraphistry-gfql-pyodide-bundle";
const docsOutDir = join(repoRoot, "docs/source/static/gfql/pyodide");

function parseArgs(argv) {
  const options = {
    flavor: process.env.GFQL_PYODIDE_BUNDLE_FLAVOR || "self-hosted",
    outDir: undefined,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--docs-static") {
      options.outDir = docsOutDir;
    } else if (arg === "--flavor") {
      i += 1;
      options.flavor = argv[i];
    } else if (arg.startsWith("--flavor=")) {
      options.flavor = arg.slice("--flavor=".length);
    } else if (arg === "--help" || arg === "-h") {
      console.log([
        "Usage: node demos/gfql/pyodide/build-bundle.mjs [out-dir] [--docs-static] [--flavor self-hosted|cdn]",
        "",
        "Flavors:",
        "  self-hosted  Copy Pyodide runtime and required package wheels into the bundle.",
        "  cdn          Keep only demo files and wheels; load Pyodide 314 from the pinned CDN.",
      ].join("\n"));
      process.exit(0);
    } else if (arg.startsWith("--")) {
      throw new Error(`Unknown option: ${arg}`);
    } else if (!options.outDir) {
      options.outDir = arg;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }
  if (!BUNDLE_FLAVORS.has(options.flavor)) {
    throw new Error(`Unknown bundle flavor "${options.flavor}". Expected one of: ${[...BUNDLE_FLAVORS].join(", ")}`);
  }
  options.outDir = resolve(options.outDir || defaultOutDir);
  return options;
}

const buildOptions = parseArgs(process.argv.slice(2));
const outDir = buildOptions.outDir;
const bundleFlavor = buildOptions.flavor;
const workDir = join(outDir, ".work");
const srcCopy = join(workDir, "src");
const pyodideNode = join(workDir, "node");
const wheelDir = join(outDir, "wheels");
const pyodideOutDir = join(outDir, "pyodide");

async function directorySize(path) {
  const entry = await stat(path);
  if (!entry.isDirectory()) {
    return entry.size;
  }
  const children = await readdir(path);
  let total = 0;
  for (const child of children) {
    total += await directorySize(join(path, child));
  }
  return total;
}

async function removeIfExists(path) {
  await rm(path, { recursive: true, force: true });
}

async function prunePyodideRuntime(path) {
  const removable = [
    "console.html",
    "console-v2.html",
    "ffi.d.ts",
    "package.json",
    "pyodide.asm.mjs.map",
    "pyodide.d.ts",
    "pyodide.js",
    "pyodide.js.map",
    "pyodide.mjs.map",
    "README.md",
  ];
  await Promise.all(removable.map((filename) => removeIfExists(join(path, filename))));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function collectPyodidePackages(lockFile, packageNames) {
  const packages = lockFile.packages || {};
  const seen = new Set();

  function visit(packageName) {
    if (seen.has(packageName)) {
      return;
    }
    const metadata = packages[packageName];
    if (!metadata) {
      throw new Error(`Pyodide lockfile does not include package: ${packageName}`);
    }
    seen.add(packageName);
    for (const dependency of metadata.depends || []) {
      visit(dependency);
    }
  }

  for (const packageName of packageNames) {
    visit(packageName);
  }

  return [...seen].sort();
}

async function fetchBytesWithRetry(url, attempts = 3) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }
      return Buffer.from(await response.arrayBuffer());
    } catch (error) {
      lastError = error;
      if (attempt < attempts) {
        await new Promise((resolvePromise) => setTimeout(resolvePromise, attempt * 1000));
      }
    }
  }
  throw new Error(`Failed to download ${url}: ${lastError?.message || lastError}`);
}

async function downloadBytes(url, outputPath) {
  try {
    const bytes = await fetchBytesWithRetry(url);
    await writeFile(outputPath, bytes);
    return bytes;
  } catch (fetchError) {
    const result = spawnSync("curl", [
      "-L",
      "--fail",
      "--silent",
      "--show-error",
      "--retry", "3",
      "--output", outputPath,
      url,
    ], {
      encoding: "utf8",
      stdio: "pipe",
    });
    if (result.status !== 0) {
      throw new Error([
        `Failed to download ${url}`,
        `node fetch: ${fetchError?.message || fetchError}`,
        `curl: ${result.stderr || result.stdout}`,
      ].join("\n"));
    }
    return readFile(outputPath);
  }
}

async function downloadPyodidePackages(pyodideDir, packageNames) {
  const lockFile = JSON.parse(await readFile(join(pyodideDir, "pyodide-lock.json"), "utf8"));
  const packages = lockFile.packages || {};
  const resolvedPackages = collectPyodidePackages(lockFile, packageNames);
  const baseURL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full`;

  for (const packageName of resolvedPackages) {
    const metadata = packages[packageName];
    const outputPath = join(pyodideDir, metadata.file_name);
    let bytes;
    try {
      bytes = await readFile(outputPath);
    } catch {
      bytes = await downloadBytes(`${baseURL}/${metadata.file_name}`, outputPath);
    }
    if (sha256(bytes) !== metadata.sha256) {
      throw new Error(`Checksum mismatch for ${metadata.file_name}`);
    }
  }

  return resolvedPackages;
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd || repoRoot,
    env: { ...process.env, ...options.env },
    encoding: "utf8",
    stdio: options.capture ? "pipe" : "inherit",
  });
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed with exit ${result.status}`);
  }
  return result;
}

function sourceCopyExcludes() {
  const excludes = [
    ".git",
    "plans",
    "uv.lock",
    "=2",
  ];
  const relativeOutDir = relative(repoRoot, outDir);
  if (relativeOutDir && relativeOutDir !== "." && !relativeOutDir.startsWith("..")) {
    excludes.push(relativeOutDir, `${relativeOutDir}/***`);
  }
  return excludes.flatMap((pattern) => ["--exclude", pattern]);
}

async function main() {
  await rm(outDir, { recursive: true, force: true });
  await mkdir(wheelDir, { recursive: true });
  await mkdir(workDir, { recursive: true });

  run("rsync", [
    "-a",
    ...sourceCopyExcludes(),
    "./",
    `${srcCopy}/`,
  ]);

  run("uv", [
    "run",
    "--no-project",
    "--with", "build",
    "python",
    "-m", "build",
    "--wheel",
    "--outdir", wheelDir,
    srcCopy,
  ]);

  run("uv", [
    "run",
    "--no-project",
    "--with", "pip",
    "python",
    "-m", "pip",
    "download",
    "--only-binary=:all:",
    "--dest", wheelDir,
    ...VENDORED_WHEEL_REQUIREMENTS,
  ]);

  run("npm", [
    "install",
    "--prefix", pyodideNode,
    "--no-audit",
    "--no-fund",
    "--ignore-scripts",
    `pyodide@${PYODIDE_VERSION}`,
  ]);
  const pyodidePackageDir = join(pyodideNode, "node_modules/pyodide");
  const vendoredPyodidePackages = await downloadPyodidePackages(pyodidePackageDir, [
    ...PYODIDE_CORE_PACKAGES,
    ...PYODIDE_REQUIREMENTS,
  ]);

  const graphistryWheel = run("bash", [
    "-lc",
    `ls ${JSON.stringify(wheelDir)}/graphistry-*.whl | head -1`,
  ], { capture: true }).stdout.trim();
  const requirementWheelPaths = run("bash", [
    "-lc",
    `find ${JSON.stringify(wheelDir)} -maxdepth 1 -name '*.whl' ! -name 'graphistry-*.whl' -print | sort`,
  ], { capture: true }).stdout.trim().split("\n").filter(Boolean);

  run("node", [
    join(scriptDir, "run-node.mjs"),
    graphistryWheel,
  ], {
    env: {
      PYODIDE_MODULE: join(pyodideNode, "node_modules/pyodide/pyodide.mjs"),
      GFQL_REQUIREMENT_WHEELS: requirementWheelPaths.join(":"),
    },
  });

  if (bundleFlavor === "self-hosted") {
    await cp(join(pyodideNode, "node_modules/pyodide"), pyodideOutDir, {
      recursive: true,
    });
    await prunePyodideRuntime(pyodideOutDir);
  }

  for (const filename of ["benchmark-node.mjs", "browser.html", "edges.csv", "gfql.js", "package.json", "run-node.mjs", "test-browser.mjs"]) {
    await copyFile(join(scriptDir, filename), join(outDir, filename));
  }

  const wheelFiles = run("bash", [
    "-lc",
    `find ${JSON.stringify(wheelDir)} -maxdepth 1 -name '*.whl' -printf '%f\\n' | sort`,
  ], { capture: true }).stdout.trim().split("\n").filter(Boolean);
  const graphistryWheelName = basename(graphistryWheel);
  const requirementEntries = wheelFiles
    .filter((filename) => filename !== graphistryWheelName)
    .map((filename) => `./wheels/${filename}`);

  await writeFile(join(outDir, "manifest.json"), `${JSON.stringify({
    pyodideVersion: PYODIDE_VERSION,
    flavor: bundleFlavor,
    pyodideModule: bundleFlavor === "self-hosted" ? "./pyodide/pyodide.mjs" : `${PYODIDE_CDN_URL}pyodide.mjs`,
    indexURL: bundleFlavor === "self-hosted" ? "./pyodide/" : PYODIDE_CDN_URL,
    packageBaseUrl: bundleFlavor === "self-hosted" ? "./pyodide/" : PYODIDE_CDN_URL,
    pyodidePackages: vendoredPyodidePackages,
    graphistryWheel: `./wheels/${graphistryWheelName}`,
    requirements: [
      ...requirementEntries,
      ...PYODIDE_REQUIREMENTS,
    ],
  }, null, 2)}\n`);

  await writeFile(join(outDir, "README.txt"), [
    "GFQL Pyodide bundle",
    "",
    `Flavor: ${bundleFlavor}`,
    `Built from ${relative(process.cwd(), repoRoot) || "."}`,
    "",
    "Serve locally:",
    `  cd ${outDir}`,
    "  python -m http.server 8000",
    "  open http://localhost:8000/browser.html",
    "",
    "Node smoke:",
    bundleFlavor === "self-hosted"
      ? `  PYODIDE_MODULE=${join(outDir, "pyodide/pyodide.mjs")} node ${join(outDir, "run-node.mjs")} ${join(outDir, "wheels", graphistryWheelName)}`
      : "  Build with --flavor self-hosted for an offline Node smoke target, or use the browser smoke.",
    "",
    "Browser smoke:",
    `  node ${join(outDir, "test-browser.mjs")} ${outDir}`,
    "",
    "Benchmark:",
    `  node ${join(outDir, "benchmark-node.mjs")} ${outDir}`,
    "",
  ].join("\n"));

  await rm(workDir, { recursive: true, force: true });

  const sizeReport = {
    totalBytes: 0,
    flavor: bundleFlavor,
    pyodideBytes: bundleFlavor === "self-hosted" ? await directorySize(pyodideOutDir) : 0,
    wheelsBytes: await directorySize(wheelDir),
    generatedAt: new Date().toISOString(),
  };
  await writeFile(join(outDir, "size-report.json"), `${JSON.stringify(sizeReport, null, 2)}\n`);
  sizeReport.totalBytes = await directorySize(outDir);
  await writeFile(join(outDir, "size-report.json"), `${JSON.stringify(sizeReport, null, 2)}\n`);

  const manifest = await readFile(join(outDir, "manifest.json"), "utf8");
  console.log(`\nBundle written to ${outDir}`);
  console.log(manifest);
  console.log(JSON.stringify(sizeReport, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
