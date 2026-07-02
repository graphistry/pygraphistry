import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, normalize, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";

const bundleDir = resolve(process.argv[2] || "/tmp/pygraphistry-gfql-pyodide-bundle");
const timeoutMs = Number(process.env.GFQL_BROWSER_TIMEOUT_MS || "120000");
const screenshotPath = process.env.GFQL_BROWSER_SCREENSHOT;

const contentTypes = new Map([
  [".csv", "text/csv; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".txt", "text/plain; charset=utf-8"],
  [".wasm", "application/wasm"],
  [".whl", "application/octet-stream"],
  [".zip", "application/zip"],
]);

async function fileExists(path) {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
}

function resolveRequestPath(urlPath) {
  const decoded = decodeURIComponent(urlPath.split("?")[0]);
  const relativePath = normalize(decoded === "/" ? "browser.html" : decoded.slice(1));
  const absolutePath = resolve(bundleDir, relativePath);
  if (absolutePath !== bundleDir && !absolutePath.startsWith(`${bundleDir}${sep}`)) {
    return undefined;
  }
  return absolutePath;
}

async function startServer() {
  const server = createServer(async (request, response) => {
    try {
      const requestPath = resolveRequestPath(request.url || "/");
      if (!requestPath || !(await fileExists(requestPath))) {
        response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
        response.end("not found");
        return;
      }
      const body = await readFile(requestPath);
      response.writeHead(200, {
        "content-length": body.length,
        "content-type": contentTypes.get(extname(requestPath)) || "application/octet-stream",
      });
      if (request.method === "HEAD") {
        response.end();
      } else {
        response.end(body);
      }
    } catch (error) {
      response.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
      response.end(String(error.stack || error.message || error));
    }
  });

  await new Promise((resolvePromise) => server.listen(0, "127.0.0.1", resolvePromise));
  const address = server.address();
  return {
    server,
    baseURL: `http://127.0.0.1:${address.port}`,
  };
}

async function importPlaywright() {
  const candidates = [
    process.env.PLAYWRIGHT_MODULE,
    "playwright",
    pathToFileURL(resolve(bundleDir, "node_modules/playwright/index.mjs")).href,
    pathToFileURL(resolve(process.cwd(), "demos/gfql/pyodide/node_modules/playwright/index.mjs")).href,
    pathToFileURL(resolve(process.cwd(), "node_modules/playwright/index.mjs")).href,
  ].filter(Boolean);

  const errors = [];
  for (const candidate of candidates) {
    if (candidate.startsWith("file://") && !(await fileExists(new URL(candidate)))) {
      continue;
    }
    try {
      return await import(candidate);
    } catch (error) {
      errors.push(`${candidate}: ${error.message}`);
    }
  }

  throw new Error([
    "Playwright is not installed or could not be imported.",
    "Run `npm install --prefix demos/gfql/pyodide` or set PLAYWRIGHT_MODULE.",
    ...errors,
  ].join("\n"));
}

async function main() {
  const { chromium } = await importPlaywright();

  const { server, baseURL } = await startServer();
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox"],
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(timeoutMs);

  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      pageErrors.push(message.text());
    }
  });

  try {
    await page.goto(`${baseURL}/browser.html`, { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => {
      const raw = document.querySelector("#jsonOutput")?.textContent || "";
      try {
        const parsed = JSON.parse(raw);
        return parsed.astEdges?.length === 3 && parsed.cypherRows?.length === 3;
      } catch {
        return false;
      }
    });

    const result = await page.evaluate(() => JSON.parse(document.querySelector("#jsonOutput").textContent));
    const status = await page.textContent("#status");
    const metrics = await page.$$eval("#metrics tbody tr", (rows) => rows.map((row) => row.textContent.trim()));

    if (!status.includes("Ready")) {
      throw new Error(`Expected Ready status, got: ${status}`);
    }
    if (!metrics.some((text) => text.includes("Create Pyodide GFQL runtime"))) {
      throw new Error(`Expected runtime creation metric, got: ${JSON.stringify(metrics)}`);
    }
    if (!result.astEdges.every((edge) => edge.weight >= 2)) {
      throw new Error(`Expected AST edge weights >= 2, got: ${JSON.stringify(result.astEdges)}`);
    }
    if (pageErrors.length > 0) {
      throw new Error(`Browser console/page errors:\n${pageErrors.join("\n")}`);
    }
    if (screenshotPath) {
      await page.screenshot({ path: screenshotPath, fullPage: true });
    }
    console.log(JSON.stringify({
      ok: true,
      url: `${baseURL}/browser.html`,
      astEdges: result.astEdges.length,
      cypherRows: result.cypherRows.length,
      metrics,
    }, null, 2));
  } finally {
    await browser.close();
    await new Promise((resolvePromise) => server.close(resolvePromise));
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
