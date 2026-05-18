/**
 * GridDB REST Backend
 *
 * Initialises a real WebGPU device via Dawn (the `webgpu` npm package) and
 * exposes every GridDB operation as an HTTP endpoint so any HTTP client can
 * drive the GPU-accelerated database engine.
 *
 * Endpoints
 * ─────────
 *  POST   /api/tables/:name/csv     – load CSV text into a new table
 *  POST   /api/tables/:name/json    – load a JSON array into a new table
 *  GET    /api/tables               – list all table names
 *  GET    /api/tables/:name         – get schema / row-count for a table
 *  DELETE /api/tables/:name         – drop a table
 *  POST   /api/query                – execute a SELECT query
 *  GET    /api/stats                – database-wide statistics
 *  GET    /api/export               – export the full database as JSON
 *  GET    /api/health               – liveness probe
 *
 *  Static files (griddb_modern.html, griddb.mjs, etc.) are served from the
 *  project root on the same port so the browser UI works out of the box.
 */

// ──────────────────────────────────────────────────────────────────────────────
// 1. Bootstrap WebGPU via Dawn
// ──────────────────────────────────────────────────────────────────────────────

import { globals, create } from "webgpu";
Object.assign(globalThis, globals);

const gpuFeatureFlags = [
  "enable-dawn-features=use_user_defined_labels_in_backend",
];

const navigator = { gpu: create(gpuFeatureFlags) };

async function initGPU() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter found. Is Dawn installed correctly?");
  }

  const hasTimestamp = adapter.features.has("timestamp-query");
  const hasSubgroups = adapter.features.has("subgroups");

  const device = await adapter.requestDevice({
    requiredFeatures: [
      ...(hasTimestamp ? ["timestamp-query"] : []),
      ...(hasSubgroups ? ["subgroups"] : []),
    ],
  });

  if (!device) {
    throw new Error("Failed to create WebGPU device.");
  }

  // adapter.info is a plain object in webgpu >=0.4.0 (no requestAdapterInfo)
  const info = adapter.info ?? {};
  console.log(`[GPU] Adapter: ${info.description || info.vendor || info.device || "unknown"}`);
  return device;
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. Import GridDB (runs entirely on the GPU device we created above)
// ──────────────────────────────────────────────────────────────────────────────

import { GridDB } from "./griddb.mjs";
import { createDataImportRouter } from "./backend/data-import.mjs";

// ──────────────────────────────────────────────────────────────────────────────
// 3. Express server
// ──────────────────────────────────────────────────────────────────────────────

import express from "express";
import cors from "cors";
import multer from "multer";
import { fileURLToPath } from "url";
import path from "path";
import fs from "fs";
import { corsOptions } from "./backend/cors.mjs";

// ── Load .env from backend/ ────────────────────────────────────────────────────
// Use Node's built-in --env-file when available (Node 20.6+), otherwise parse
// the file manually so we work on older Node versions too.
{
  const envPath = new URL("./backend/.env", import.meta.url).pathname;
  if (fs.existsSync(envPath)) {
    const lines = fs.readFileSync(envPath, "utf8").split(/\r?\n/);
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const eq = trimmed.indexOf("=");
      if (eq === -1) continue;
      const key = trimmed.slice(0, eq).trim();
      const val = trimmed.slice(eq + 1).trim();
      // Only set if not already defined in the environment
      if (!(key in process.env)) process.env[key] = val;
    }
  }
}

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
const PORT = process.env.PORT || 3000;

// Upload to memory (we parse it ourselves)
const upload = multer({ storage: multer.memoryStorage() });

// CORS – must be registered before any routes
app.use(cors(corsOptions));
app.options("*", cors(corsOptions)); // pre-flight for all routes

// Parse JSON and plain-text bodies
app.use(express.json({ limit: "500mb" }));
app.use(express.text({ limit: "500mb" }));

// Serve static frontend files from the project root
app.use(express.static(__dirname));

// ──────────────────────────────────────────────────────────────────────────────
// 4. Shared GridDB instance (created after GPU device is ready)
// ──────────────────────────────────────────────────────────────────────────────

let db;
let dataImportRouter; // mounted after db is ready

// ──────────────────────────────────────────────────────────────────────────────
// 5. Helper – uniform error response
// ──────────────────────────────────────────────────────────────────────────────

function sendError(res, err, defaultStatus = 500) {
  const status =
    err.code === "TABLE_NOT_FOUND" ||
    err.code === "JOIN_TABLE_NOT_FOUND" ||
    err.code === "TABLE_EXISTS"
      ? err.code === "TABLE_EXISTS"
        ? 409
        : 404
      : err.code === "UNSUPPORTED_SQL_KEYWORD" ||
        err.code === "INVALID_QUERY" ||
        err.code === "INVALID_SELECT" ||
        err.code === "INVALID_LIMIT" ||
        err.code === "INVALID_JOIN_CONDITION"
      ? 400
      : defaultStatus;

  res.status(status).json({
    error: err.message,
    code: err.code || "INTERNAL_ERROR",
    details: err.details || {},
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// 6. Endpoints
// ──────────────────────────────────────────────────────────────────────────────

// ── Data-import router (mounted lazily after db is ready) ────────────────────
// We use a middleware shim so Express can register the route before `db` exists.
app.use("/api/import", (req, res, next) => {
  if (!dataImportRouter) return res.status(503).json({ error: "Server not ready" });
  dataImportRouter(req, res, next);
});

// ── Health ────────────────────────────────────────────────────────────────────
/**
 * GET /api/health
 * Returns 200 when the server is ready.
 */
app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", webgpu: !!db });
});

// ── Load CSV ──────────────────────────────────────────────────────────────────
/**
 * POST /api/tables/:name/csv
 *
 * Body: raw CSV text (Content-Type: text/plain or text/csv)
 *   – OR –
 * multipart/form-data with a single file field named "file"
 *
 * Returns: { name, rows, columns }
 */
app.post("/api/tables/:name/csv", upload.single("file"), async (req, res) => {
  try {
    const name = req.params.name;
    let csvText;

    if (req.file) {
      csvText = req.file.buffer.toString("utf-8");
    } else if (typeof req.body === "string" && req.body.length > 0) {
      csvText = req.body;
    } else {
      return res
        .status(400)
        .json({ error: "Provide CSV as raw text body or a multipart file field named 'file'" });
    }

    const table = await db.loadCSV(name, csvText);
    res.status(201).json({
      name: table.name,
      rows: table.rowCount,
      columns: table.columns.map((c) => ({ name: c.name, type: c.type })),
    });
  } catch (err) {
    sendError(res, err);
  }
});

// ── Load JSON ─────────────────────────────────────────────────────────────────
/**
 * POST /api/tables/:name/json
 *
 * Body: JSON array of objects  (Content-Type: application/json)
 *
 * Returns: { name, rows, columns }
 */
app.post("/api/tables/:name/json", async (req, res) => {
  try {
    const name = req.params.name;
    const jsonArray = req.body;

    if (!Array.isArray(jsonArray)) {
      return res.status(400).json({ error: "Body must be a JSON array of row objects" });
    }

    const table = await db.loadJSON(name, jsonArray);
    res.status(201).json({
      name: table.name,
      rows: table.rowCount,
      columns: table.columns.map((c) => ({ name: c.name, type: c.type })),
    });
  } catch (err) {
    sendError(res, err);
  }
});

// ── List tables ───────────────────────────────────────────────────────────────
/**
 * GET /api/tables
 *
 * Returns: { tables: string[] }
 */
app.get("/api/tables", (_req, res) => {
  try {
    res.json({ tables: db.listTables() });
  } catch (err) {
    sendError(res, err);
  }
});

// ── Table info ────────────────────────────────────────────────────────────────
/**
 * GET /api/tables/:name
 *
 * Returns: { name, rows, columns, estimatedBytes, estimatedMB }
 */
app.get("/api/tables/:name", (req, res) => {
  try {
    const info = db.getTableInfo(req.params.name);
    res.json(info);
  } catch (err) {
    sendError(res, err);
  }
});

// ── Delete table ──────────────────────────────────────────────────────────────
/**
 * DELETE /api/tables/:name
 *
 * Returns: 204 No Content on success
 */
app.delete("/api/tables/:name", (req, res) => {
  try {
    db.deleteTable(req.params.name);
    res.sendStatus(204);
  } catch (err) {
    sendError(res, err);
  }
});

// ── Run query ─────────────────────────────────────────────────────────────────
/**
 * POST /api/query
 *
 * Body: { "sql": "SELECT ..." }
 *
 * Returns: { sql, rowCount, columns, rows, executionTimeMs }
 */
app.post("/api/query", async (req, res) => {
  try {
    const sql =
      typeof req.body === "string"
        ? req.body
        : typeof req.body?.sql === "string"
        ? req.body.sql
        : null;

    if (!sql) {
      return res
        .status(400)
        .json({ error: "Provide { sql: '...' } in the JSON body or raw SQL as text/plain" });
    }

    const start = Date.now();
    const result = await db.query(sql);
    const executionTimeMs = Date.now() - start;

    res.json({
      sql,
      rowCount: result.rowCount,
      columns: result.columns.map((c) => ({ name: c.name, type: c.type })),
      rows: result.rows,
      executionTimeMs,
    });
  } catch (err) {
    sendError(res, err);
  }
});

// ── Database stats ────────────────────────────────────────────────────────────
/**
 * GET /api/stats
 *
 * Returns comprehensive runtime statistics from GridDB.getStats()
 */
app.get("/api/stats", (_req, res) => {
  try {
    res.json(db.getStats());
  } catch (err) {
    sendError(res, err);
  }
});

// ── Export database ───────────────────────────────────────────────────────────
/**
 * GET /api/export
 *
 * Returns the full database serialised as JSON (tables + stats + version).
 */
app.get("/api/export", (_req, res) => {
  try {
    res.json(db.exportJSON());
  } catch (err) {
    sendError(res, err);
  }
});

// ── Performance metrics ───────────────────────────────────────────────────────
/**
 * GET /api/metrics
 *
 * Returns low-level query performance counters.
 */
app.get("/api/metrics", (_req, res) => {
  try {
    res.json(db.getPerformanceMetrics());
  } catch (err) {
    sendError(res, err);
  }
});

// ── Catch-all: serve index.html for browser navigation ───────────────────────
app.get("*", (_req, res) => {
  const indexPath = path.join(__dirname, "index.html");
  if (fs.existsSync(indexPath)) {
    res.sendFile(indexPath);
  } else {
    res.status(404).json({ error: "Not found" });
  }
});

// ──────────────────────────────────────────────────────────────────────────────
// 7. Boot sequence
// ──────────────────────────────────────────────────────────────────────────────

async function main() {
  console.log("[GridDB] Initialising WebGPU device via Dawn …");
  const device = await initGPU();

  db = new GridDB(device, {
    enableLogging: true,
    enableCache: true,
  });

  // Wire up the data-import router now that db is live
  dataImportRouter = createDataImportRouter(db, upload);

  console.log("[GridDB] WebGPU device ready.");

  app.listen(PORT, () => {
    console.log(`[GridDB] REST backend listening on http://localhost:${PORT}`);
    console.log(`[GridDB] UI available at         http://localhost:${PORT}/`);
    console.log(`[GridDB] API root:               http://localhost:${PORT}/api`);
    console.log("");
    console.log("  Endpoints:");
    console.log("  GET    /api/health");
    console.log("  POST   /api/tables/:name/csv");
    console.log("  POST   /api/tables/:name/json");
    console.log("  GET    /api/tables");
    console.log("  GET    /api/tables/:name");
    console.log("  DELETE /api/tables/:name");
    console.log("  POST   /api/query");
    console.log("  GET    /api/stats");
    console.log("  GET    /api/metrics");
    console.log("  GET    /api/export");
  console.log("");
  console.log("  Data-import endpoints:");
  console.log("  POST   /api/import/csv");
  console.log("  POST   /api/import/json");
  console.log("  POST   /api/import/mysql");
  console.log("  POST   /api/import/test-mysql");
  });
}

main().catch((err) => {
  console.error("[GridDB] Fatal startup error:", err.message);
  process.exit(1);
});
