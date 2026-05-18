/**
 * backend/data-import.mjs
 *
 * Express router – all data-import endpoints.
 *
 * Mounted at /api/import in server.mjs.
 *
 * Endpoints
 * ─────────
 *  POST  /api/import/mysql     – pull a table / custom query from MySQL into GridDB
 *  POST  /api/import/csv       – upload a CSV file and load it into GridDB
 *  POST  /api/import/json      – upload / paste a JSON array and load it into GridDB
 *  GET   /api/import/test-mysql – test a MySQL connection without importing data
 *
 * The `db` (GridDB instance) and `multer upload` middleware are injected via
 * createDataImportRouter(db, upload) so the router always shares the same
 * in-memory database and GPU device as the main server.
 */

import { Router } from "express";
import mysql from "mysql2/promise";

// ──────────────────────────────────────────────────────────────────────────────
// Factory – call with the live GridDB instance and multer upload handler
// ──────────────────────────────────────────────────────────────────────────────

export function createDataImportRouter(db, upload) {
  const router = Router();

  // ── Helpers ────────────────────────────────────────────────────────────────

  function ok(res, body, status = 200) {
    res.status(status).json(body);
  }

  function fail(res, err, defaultStatus = 500) {
    const statusMap = {
      TABLE_NOT_FOUND: 404,
      TABLE_EXISTS: 409,
      INVALID_QUERY: 400,
      INVALID_SELECT: 400,
      UNSUPPORTED_SQL_KEYWORD: 400,
      INVALID_LIMIT: 400,
    };
    const status = statusMap[err.code] ?? defaultStatus;
    res.status(status).json({
      error: err.message,
      code: err.code || "INTERNAL_ERROR",
      details: err.details || {},
    });
  }

  /**
   * Build a mysql2 connection config from the request body.
   * Only whitelisted fields are forwarded to avoid prototype-pollution attacks.
   */
  function buildMysqlConfig(body) {
    // Per-request fields take precedence over .env defaults
    const {
      host     = process.env.MYSQL_HOST     || "127.0.0.1",
      port     = process.env.MYSQL_PORT     || 3306,
      user     = process.env.MYSQL_USER,
      password = process.env.MYSQL_PASSWORD,
      database = process.env.MYSQL_DATABASE,
      ssl      = process.env.MYSQL_SSL === "true",
      connectTimeout = 10000,
    } = body;

    if (!user)     throw Object.assign(new Error("'user' is required (or set MYSQL_USER in .env)"),     { code: "BAD_REQUEST" });
    if (!database) throw Object.assign(new Error("'database' is required (or set MYSQL_DATABASE in .env)"), { code: "BAD_REQUEST" });

    return {
      host: String(host),
      port: Number(port),
      user: String(user),
      password: password != null ? String(password) : undefined,
      database: String(database),
      ssl: ssl === true || ssl === "true" ? {} : undefined,
      connectTimeout: Number(connectTimeout),
      // Always use the promise API; disable multiple statements for security
      multipleStatements: false,
    };
  }

  // ── POST /api/import/mysql ─────────────────────────────────────────────────
  /**
   * Import a MySQL table (or arbitrary SELECT) into GridDB.
   *
   * Body (application/json):
   * {
   *   host:       "localhost",        // optional, default "localhost"
   *   port:       3306,               // optional, default 3306
   *   user:       "root",             // required
   *   password:   "secret",           // optional
   *   database:   "mydb",             // required
   *   table:      "users",            // required unless `query` is given
   *   query:      "SELECT id, name…", // required unless `table` is given
   *   tableName:  "users_imported",   // GridDB table name (defaults to `table`)
   *   limit:      100000,             // row cap to avoid memory exhaustion
   *   ssl:        false               // set true to require TLS
   * }
   *
   * Returns: { tableName, rows, columns, executionTimeMs }
   */
  router.post("/mysql", async (req, res) => {
    let conn;
    try {
      const body = req.body;
      if (!body || typeof body !== "object") {
        return res.status(400).json({ error: "JSON body required" });
      }

      // Validate: need either `table` or `query`
      if (!body.table && !body.query) {
        return res.status(400).json({ error: "Provide either 'table' or 'query' in the body" });
      }

      // Build safe SQL – if a raw query is provided we only allow SELECT statements
      let sql;
      if (body.query) {
        const trimmed = String(body.query).replace(/\s+/g, " ").trim();
        if (!/^SELECT\s/i.test(trimmed)) {
          return res
            .status(400)
            .json({ error: "Only SELECT queries are allowed for security reasons" });
        }
        sql = trimmed;
      } else {
        // Sanitise table name: only letters, digits, underscore, hyphen
        const tbl = String(body.table);
        if (!/^[a-zA-Z0-9_-]+$/.test(tbl)) {
          return res.status(400).json({ error: "Invalid MySQL table name" });
        }
        const envLimit = Number(process.env.MYSQL_ROW_LIMIT) || 200000;
        const rowCap = Math.min(Number(body.limit) || envLimit, 1_000_000);
        sql = `SELECT * FROM \`${tbl}\` LIMIT ${rowCap}`;
      }

      // GridDB table name defaults to MySQL table name
      const griddbName =
        body.tableName && /^[a-zA-Z0-9_-]+$/.test(String(body.tableName))
          ? String(body.tableName)
          : body.table && /^[a-zA-Z0-9_-]+$/.test(String(body.table))
          ? String(body.table)
          : "mysql_import";

      const config = buildMysqlConfig(body);

      const start = Date.now();

      // Connect and query
      conn = await mysql.createConnection(config);
      const [rows] = await conn.execute(sql);
      await conn.end();
      conn = null;

      if (!Array.isArray(rows) || rows.length === 0) {
        return res.status(200).json({
          tableName: griddbName,
          rows: 0,
          columns: [],
          executionTimeMs: Date.now() - start,
          message: "Query returned no rows – no table created in GridDB",
        });
      }

      // Load into GridDB (GPU-accelerated)
      const table = await db.loadJSON(griddbName, rows);
      const executionTimeMs = Date.now() - start;

      ok(
        res,
        {
          tableName: table.name,
          rows: table.rowCount,
          columns: table.columns.map((c) => ({ name: c.name, type: c.type })),
          executionTimeMs,
        },
        201,
      );
    } catch (err) {
      // Ensure connection is always closed on error
      if (conn) {
        try { await conn.end(); } catch (_) {}
      }

      // Map mysql2 errors to friendly messages
      if (err.code === "ECONNREFUSED" || err.code === "ETIMEDOUT") {
        return res
          .status(502)
          .json({ error: `Cannot reach MySQL server: ${err.message}`, code: "MYSQL_UNREACHABLE" });
      }
      if (err.code === "ER_ACCESS_DENIED_ERROR" || err.code === "ER_DBACCESS_DENIED_ERROR") {
        return res
          .status(401)
          .json({ error: "MySQL access denied – check user/password/database", code: err.code });
      }
      if (err.code === "ER_NO_SUCH_TABLE") {
        return res
          .status(404)
          .json({ error: `MySQL table not found: ${err.message}`, code: err.code });
      }

      fail(res, err);
    }
  });

  // ── GET /api/import/test-mysql ─────────────────────────────────────────────
  /**
   * Test a MySQL connection without importing anything.
   * Accepts the same connection fields as POST /mysql (minus table/query).
   * Body: { host, port, user, password, database }
   * Returns: { ok: true, tables: ["t1", "t2", …] }
   */
  router.post("/test-mysql", async (req, res) => {
    let conn;
    try {
      const config = buildMysqlConfig(req.body ?? {});
      conn = await mysql.createConnection(config);
      const [tableRows] = await conn.execute("SHOW TABLES");
      await conn.end();
      conn = null;

      const tables = tableRows.map((r) => Object.values(r)[0]);
      res.json({ ok: true, database: config.database, tables });
    } catch (err) {
      if (conn) { try { await conn.end(); } catch (_) {} }

      if (err.code === "ECONNREFUSED" || err.code === "ETIMEDOUT") {
        return res
          .status(502)
          .json({ ok: false, error: `Cannot reach MySQL server: ${err.message}` });
      }
      if (err.code === "ER_ACCESS_DENIED_ERROR" || err.code === "ER_DBACCESS_DENIED_ERROR") {
        return res.status(401).json({ ok: false, error: "MySQL access denied" });
      }
      res.status(500).json({ ok: false, error: err.message });
    }
  });

  // ── POST /api/import/csv ───────────────────────────────────────────────────
  /**
   * Upload a CSV file and load it into GridDB.
   *
   * multipart/form-data fields:
   *   file      – the CSV file (required)
   *   tableName – GridDB table name (optional, defaults to filename without ext)
   *
   * – OR –
   * raw text body (Content-Type: text/plain) with ?tableName=… query param.
   *
   * Returns: { tableName, rows, columns }
   */
  router.post("/csv", upload.single("file"), async (req, res) => {
    try {
      let csvText;
      let tableName;

      if (req.file) {
        csvText = req.file.buffer.toString("utf-8");
        // strip extension from original name, fall back to "csv_import"
        tableName =
          req.body?.tableName ||
          (req.file.originalname || "csv_import").replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "_") ||
          "csv_import";
      } else if (typeof req.body === "string" && req.body.length > 0) {
        csvText = req.body;
        tableName = req.query.tableName || "csv_import";
      } else {
        return res.status(400).json({
          error: "Provide a multipart file field named 'file', or raw CSV text with ?tableName=…",
        });
      }

      if (!/^[a-zA-Z0-9_-]+$/.test(tableName)) {
        tableName = "csv_import";
      }

      const table = await db.loadCSV(tableName, csvText);
      ok(
        res,
        {
          tableName: table.name,
          rows: table.rowCount,
          columns: table.columns.map((c) => ({ name: c.name, type: c.type })),
        },
        201,
      );
    } catch (err) {
      fail(res, err);
    }
  });

  // ── POST /api/import/json ──────────────────────────────────────────────────
  /**
   * Upload a JSON array and load it into GridDB.
   *
   * multipart/form-data fields:
   *   file      – the .json file (required)
   *   tableName – GridDB table name (optional)
   *
   * – OR –
   * application/json body: { tableName: "…", data: [{…}, …] }
   *
   * Returns: { tableName, rows, columns }
   */
  router.post("/json", upload.single("file"), async (req, res) => {
    try {
      let jsonArray;
      let tableName;

      if (req.file) {
        const text = req.file.buffer.toString("utf-8");
        try {
          jsonArray = JSON.parse(text);
        } catch {
          return res.status(400).json({ error: "Uploaded file is not valid JSON" });
        }
        tableName =
          req.body?.tableName ||
          (req.file.originalname || "json_import").replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "_") ||
          "json_import";
      } else if (req.body && typeof req.body === "object") {
        if (Array.isArray(req.body)) {
          jsonArray = req.body;
          tableName = req.query.tableName || "json_import";
        } else if (Array.isArray(req.body.data)) {
          jsonArray = req.body.data;
          tableName = req.body.tableName || req.query.tableName || "json_import";
        } else {
          return res
            .status(400)
            .json({ error: "Body must be a JSON array, or { tableName, data: […] }" });
        }
      } else {
        return res.status(400).json({ error: "No JSON data provided" });
      }

      if (!Array.isArray(jsonArray)) {
        return res.status(400).json({ error: "Data must be a JSON array of row objects" });
      }

      if (!/^[a-zA-Z0-9_-]+$/.test(tableName)) {
        tableName = "json_import";
      }

      const table = await db.loadJSON(tableName, jsonArray);
      ok(
        res,
        {
          tableName: table.name,
          rows: table.rowCount,
          columns: table.columns.map((c) => ({ name: c.name, type: c.type })),
        },
        201,
      );
    } catch (err) {
      fail(res, err);
    }
  });

  return router;
}
