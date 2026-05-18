/**
 * src/features/data-import/data-import.js
 *
 * Frontend Data Import Module
 *
 * Provides the `DataImport` class which:
 *  - Renders a tabbed import panel (CSV upload, JSON upload, MySQL connection form)
 *  - Calls the backend /api/import/* endpoints
 *  - Returns structured { tableName, rows, columns } on success
 *
 * Usage:
 *   const importer = new DataImport({ apiBase: '/api/import' });
 *   const panel = importer.renderPanel();
 *   document.getElementById('my-container').appendChild(panel);
 *   importer.on('imported', ({ tableName, rows, columns }) => {
 *     console.log(`Loaded ${rows} rows into "${tableName}"`);
 *   });
 */

export class DataImport extends EventTarget {
  /**
   * @param {object}  [opts]
   * @param {string}  [opts.apiBase='/api/import']  – base URL for import endpoints
   * @param {number}  [opts.maxFileSizeMB=500]       – client-side file size guard
   */
  constructor(opts = {}) {
    super();
    this.apiBase = opts.apiBase ?? "/api/import";
    this.maxFileSizeMB = opts.maxFileSizeMB ?? 500;
    this._panel = null;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Import a CSV file (File object or string text) into GridDB.
   * @param {File|string} fileOrText
   * @param {string}      tableName
   * @returns {Promise<{tableName, rows, columns}>}
   */
  async importCSV(fileOrText, tableName) {
    this._assertTableName(tableName);

    let body, headers;

    if (fileOrText instanceof File) {
      this._assertFileSize(fileOrText);
      const fd = new FormData();
      fd.append("file", fileOrText);
      fd.append("tableName", tableName);
      body = fd;
      headers = {};
    } else {
      body = String(fileOrText);
      headers = { "Content-Type": "text/plain" };
    }

    return this._post(
      `${this.apiBase}/csv${fileOrText instanceof File ? "" : `?tableName=${encodeURIComponent(tableName)}`}`,
      body,
      headers,
    );
  }

  /**
   * Import a JSON file (File object) or array into GridDB.
   * @param {File|Array<object>} fileOrArray
   * @param {string}             tableName
   * @returns {Promise<{tableName, rows, columns}>}
   */
  async importJSON(fileOrArray, tableName) {
    this._assertTableName(tableName);

    if (fileOrArray instanceof File) {
      this._assertFileSize(fileOrArray);
      const fd = new FormData();
      fd.append("file", fileOrArray);
      fd.append("tableName", tableName);
      return this._post(`${this.apiBase}/json`, fd, {});
    }

    if (Array.isArray(fileOrArray)) {
      return this._post(
        `${this.apiBase}/json`,
        JSON.stringify({ tableName, data: fileOrArray }),
        { "Content-Type": "application/json" },
      );
    }

    throw new Error("fileOrArray must be a File or an Array of row objects");
  }

  /**
   * Test a MySQL connection and list available tables.
   * @param {MysqlConnectionConfig} config
   * @returns {Promise<{ok, database, tables}>}
   */
  async testMysqlConnection(config) {
    this._assertMysqlConfig(config);
    return this._post(
      `${this.apiBase}/test-mysql`,
      JSON.stringify(config),
      { "Content-Type": "application/json" },
    );
  }

  /**
   * Import a MySQL table (or custom SELECT) into GridDB.
   *
   * @param {MysqlConnectionConfig & {
   *   table?:     string,
   *   query?:     string,
   *   tableName?: string,
   *   limit?:     number,
   * }} config
   * @returns {Promise<{tableName, rows, columns, executionTimeMs}>}
   */
  async importFromMySQL(config) {
    this._assertMysqlConfig(config);
    if (!config.table && !config.query) {
      throw new Error("Provide either 'table' or 'query' in the MySQL import config");
    }
    return this._post(
      `${this.apiBase}/mysql`,
      JSON.stringify(config),
      { "Content-Type": "application/json" },
    );
  }

  // ── UI – render a self-contained import panel ──────────────────────────────

  /**
   * Renders and returns a <div> containing the full import UI.
   * Call `.appendChild(importer.renderPanel())` on any container.
   * @returns {HTMLElement}
   */
  renderPanel() {
    if (this._panel) return this._panel;

    const panel = document.createElement("div");
    panel.className = "data-import-panel";
    panel.innerHTML = this._panelHTML();
    this._panel = panel;

    // Tab switching
    panel.querySelectorAll(".di-tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        panel.querySelectorAll(".di-tab-btn").forEach((b) => b.classList.remove("active"));
        panel.querySelectorAll(".di-tab-pane").forEach((p) => p.classList.add("hidden"));
        btn.classList.add("active");
        panel.querySelector(`#di-pane-${btn.dataset.tab}`).classList.remove("hidden");
      });
    });

    // CSV submit
    panel.querySelector("#di-csv-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const file = panel.querySelector("#di-csv-file").files[0];
      const name = panel.querySelector("#di-csv-tableName").value.trim() || "csv_import";
      if (!file) return this._showStatus(panel, "error", "Please select a CSV file");
      await this._handleImport(panel, () => this.importCSV(file, name));
    });

    // JSON submit
    panel.querySelector("#di-json-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const file = panel.querySelector("#di-json-file").files[0];
      const name = panel.querySelector("#di-json-tableName").value.trim() || "json_import";
      if (!file) return this._showStatus(panel, "error", "Please select a JSON file");
      await this._handleImport(panel, () => this.importJSON(file, name));
    });

    // MySQL test connection
    panel.querySelector("#di-mysql-test").addEventListener("click", async () => {
      const cfg = this._readMysqlForm(panel);
      this._showStatus(panel, "info", "Testing connection…");
      try {
        const result = await this.testMysqlConnection(cfg);
        if (result.ok) {
          this._showStatus(
            panel,
            "success",
            `Connected to "${result.database}". Tables: ${result.tables.join(", ") || "(none)"}`,
          );
          // Populate table dropdown
          const sel = panel.querySelector("#di-mysql-table");
          sel.innerHTML = `<option value="">-- pick a table --</option>` +
            result.tables.map((t) => `<option value="${t}">${t}</option>`).join("");
        } else {
          this._showStatus(panel, "error", result.error || "Connection failed");
        }
      } catch (err) {
        this._showStatus(panel, "error", err.message);
      }
    });

    // MySQL import submit
    panel.querySelector("#di-mysql-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const cfg = this._readMysqlForm(panel);
      const tbl = panel.querySelector("#di-mysql-table").value;
      const qry = panel.querySelector("#di-mysql-query").value.trim();
      const gridName =
        panel.querySelector("#di-mysql-gridName").value.trim() || tbl || "mysql_import";

      if (!tbl && !qry) {
        return this._showStatus(panel, "error", "Select a table or enter a custom SELECT query");
      }

      await this._handleImport(panel, () =>
        this.importFromMySQL({
          ...cfg,
          ...(qry ? { query: qry } : { table: tbl }),
          tableName: gridName,
        }),
      );
    });

    return panel;
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  async _post(url, body, headers) {
    const res = await fetch(url, { method: "POST", headers, body });
    const json = await res.json();
    if (!res.ok) {
      const err = new Error(json.error || `HTTP ${res.status}`);
      err.code = json.code;
      err.details = json.details;
      throw err;
    }
    return json;
  }

  _assertTableName(name) {
    if (!name || !/^[a-zA-Z0-9_-]+$/.test(name)) {
      throw new Error(`Invalid table name "${name}". Use only letters, digits, _ and -`);
    }
  }

  _assertFileSize(file) {
    if (file.size > this.maxFileSizeMB * 1024 * 1024) {
      throw new Error(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max is ${this.maxFileSizeMB} MB`);
    }
  }

  _assertMysqlConfig(cfg) {
    if (!cfg || typeof cfg !== "object") throw new Error("MySQL config must be an object");
    if (!cfg.user) throw new Error("MySQL 'user' is required");
    if (!cfg.database) throw new Error("MySQL 'database' is required");
  }

  _readMysqlForm(panel) {
    return {
      host: panel.querySelector("#di-mysql-host").value.trim() || "localhost",
      port: Number(panel.querySelector("#di-mysql-port").value) || 3306,
      user: panel.querySelector("#di-mysql-user").value.trim(),
      password: panel.querySelector("#di-mysql-password").value,
      database: panel.querySelector("#di-mysql-database").value.trim(),
      ssl: panel.querySelector("#di-mysql-ssl").checked,
    };
  }

  async _handleImport(panel, fn) {
    this._showStatus(panel, "info", "Importing…");
    try {
      const result = await fn();
      this._showStatus(
        panel,
        "success",
        `✓ Loaded "${result.tableName}" — ${result.rows.toLocaleString()} rows, ${result.columns.length} columns${result.executionTimeMs ? ` (${result.executionTimeMs}ms)` : ""}`,
      );
      this.dispatchEvent(new CustomEvent("imported", { detail: result }));
    } catch (err) {
      this._showStatus(panel, "error", `✗ ${err.message}`);
      this.dispatchEvent(new CustomEvent("error", { detail: err }));
    }
  }

  _showStatus(panel, type, msg) {
    const el = panel.querySelector(".di-status");
    el.textContent = msg;
    el.className = `di-status di-status-${type}`;
    el.style.display = "block";
  }

  _panelHTML() {
    return /* html */ `
<style>
  .data-import-panel { font-family: 'Inter', sans-serif; color: #EDEDED; }
  .di-tabs { display:flex; gap:4px; margin-bottom:12px; }
  .di-tab-btn {
    flex:1; padding:6px 10px; border-radius:6px; border:1px solid rgba(255,255,255,0.1);
    background:rgba(255,255,255,0.04); color:rgba(255,255,255,0.5); cursor:pointer;
    font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.05em;
    transition:all .2s;
  }
  .di-tab-btn:hover { background:rgba(52,178,123,.12); color:#34B27B; border-color:rgba(52,178,123,.4); }
  .di-tab-btn.active { background:rgba(52,178,123,.2); color:#34B27B; border-color:rgba(52,178,123,.6); }
  .di-tab-pane.hidden { display:none; }
  .di-form-group { margin-bottom:10px; }
  .di-label { display:block; font-size:10px; text-transform:uppercase; color:rgba(255,255,255,.4);
    margin-bottom:4px; letter-spacing:.06em; }
  .di-input {
    width:100%; padding:7px 10px; background:rgba(0,0,0,.3); border:1px solid rgba(255,255,255,.08);
    border-radius:6px; color:#EDEDED; font-size:13px; box-sizing:border-box;
  }
  .di-input:focus { outline:none; border-color:rgba(52,178,123,.4); box-shadow:0 0 12px rgba(52,178,123,.1); }
  .di-row { display:flex; gap:8px; }
  .di-row .di-form-group { flex:1; }
  .di-btn {
    width:100%; padding:8px; background:#34B27B; color:#000; font-size:12px; font-weight:700;
    text-transform:uppercase; letter-spacing:.05em; border:none; border-radius:6px; cursor:pointer;
    transition:background .2s;
  }
  .di-btn:hover { background:#2d9a68; }
  .di-btn-outline {
    background:transparent; color:#34B27B; border:1px solid rgba(52,178,123,.4);
    margin-bottom:8px;
  }
  .di-btn-outline:hover { background:rgba(52,178,123,.1); }
  .di-status {
    display:none; margin-top:10px; padding:8px 12px; border-radius:6px;
    font-size:12px; word-break:break-word;
  }
  .di-status-success { background:rgba(52,178,123,.15); border:1px solid rgba(52,178,123,.3); color:#34B27B; }
  .di-status-error   { background:rgba(239,68,68,.12);  border:1px solid rgba(239,68,68,.3);  color:#f87171; }
  .di-status-info    { background:rgba(59,130,246,.12);  border:1px solid rgba(59,130,246,.3); color:#93c5fd; }
  .di-divider { margin:10px 0; border:none; border-top:1px solid rgba(255,255,255,.06); }
  .di-hint { font-size:10px; color:rgba(255,255,255,.3); margin-top:4px; }
</style>

<!-- Tabs -->
<div class="di-tabs">
  <button class="di-tab-btn active" data-tab="csv">CSV</button>
  <button class="di-tab-btn" data-tab="json">JSON</button>
  <button class="di-tab-btn" data-tab="mysql">MySQL</button>
</div>

<!-- CSV pane -->
<div id="di-pane-csv" class="di-tab-pane">
  <form id="di-csv-form">
    <div class="di-form-group">
      <label class="di-label">CSV File</label>
      <input id="di-csv-file" class="di-input" type="file" accept=".csv,text/csv,text/plain">
    </div>
    <div class="di-form-group">
      <label class="di-label">GridDB Table Name</label>
      <input id="di-csv-tableName" class="di-input" type="text" placeholder="my_table" pattern="[a-zA-Z0-9_-]+">
      <p class="di-hint">Defaults to the filename without extension</p>
    </div>
    <button type="submit" class="di-btn">⬆ Upload CSV</button>
  </form>
</div>

<!-- JSON pane -->
<div id="di-pane-json" class="di-tab-pane hidden">
  <form id="di-json-form">
    <div class="di-form-group">
      <label class="di-label">JSON File (array of objects)</label>
      <input id="di-json-file" class="di-input" type="file" accept=".json,application/json">
    </div>
    <div class="di-form-group">
      <label class="di-label">GridDB Table Name</label>
      <input id="di-json-tableName" class="di-input" type="text" placeholder="my_table" pattern="[a-zA-Z0-9_-]+">
    </div>
    <button type="submit" class="di-btn">⬆ Upload JSON</button>
  </form>
</div>

<!-- MySQL pane -->
<div id="di-pane-mysql" class="di-tab-pane hidden">
  <form id="di-mysql-form">
    <!-- Connection fields -->
    <div class="di-row">
      <div class="di-form-group">
        <label class="di-label">Host</label>
        <input id="di-mysql-host" class="di-input" type="text" placeholder="localhost" value="localhost">
      </div>
      <div class="di-form-group" style="max-width:90px">
        <label class="di-label">Port</label>
        <input id="di-mysql-port" class="di-input" type="number" placeholder="3306" value="3306" min="1" max="65535">
      </div>
    </div>
    <div class="di-form-group">
      <label class="di-label">Username</label>
      <input id="di-mysql-user" class="di-input" type="text" placeholder="root" autocomplete="username">
    </div>
    <div class="di-form-group">
      <label class="di-label">Password</label>
      <input id="di-mysql-password" class="di-input" type="password" autocomplete="current-password">
    </div>
    <div class="di-form-group">
      <label class="di-label">Database</label>
      <input id="di-mysql-database" class="di-input" type="text" placeholder="mydb">
    </div>
    <div class="di-form-group" style="display:flex;align-items:center;gap:8px">
      <input id="di-mysql-ssl" type="checkbox" style="accent-color:#34B27B">
      <label for="di-mysql-ssl" class="di-label" style="margin:0">Require SSL/TLS</label>
    </div>

    <!-- Test button -->
    <button type="button" id="di-mysql-test" class="di-btn di-btn-outline">🔌 Test Connection</button>
    <hr class="di-divider">

    <!-- Table / query selection -->
    <div class="di-form-group">
      <label class="di-label">Table</label>
      <select id="di-mysql-table" class="di-input">
        <option value="">-- test connection first --</option>
      </select>
    </div>
    <div class="di-form-group">
      <label class="di-label">Custom SELECT (overrides table above)</label>
      <input id="di-mysql-query" class="di-input" type="text" placeholder="SELECT id, name FROM users LIMIT 50000">
      <p class="di-hint">Only SELECT statements are permitted</p>
    </div>
    <div class="di-form-group">
      <label class="di-label">GridDB Table Name</label>
      <input id="di-mysql-gridName" class="di-input" type="text" placeholder="defaults to MySQL table name" pattern="[a-zA-Z0-9_-]+">
    </div>

    <button type="submit" class="di-btn">⚡ Import into GridDB (GPU)</button>
  </form>
</div>

<!-- Status message -->
<div class="di-status"></div>
    `;
  }
}
