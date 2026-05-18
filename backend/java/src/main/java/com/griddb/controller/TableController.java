package com.griddb.controller;

import com.griddb.model.TableData;
import com.griddb.store.TableStore;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * GET    /api/health          – liveness probe
 * GET    /api/tables          – list all table names
 * GET    /api/tables/{name}   – schema + row count for one table
 * DELETE /api/tables/{name}   – drop a table
 * GET    /api/stats           – overall store statistics
 */
@RestController
@RequestMapping("/api")
public class TableController {

    private final TableStore store;

    public TableController(TableStore store) {
        this.store = store;
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        return ResponseEntity.ok(Map.of(
                "status", "ok",
                "tables", store.listNames().size(),
                "totalRows", store.totalRows()
        ));
    }

    @GetMapping("/tables")
    public ResponseEntity<List<String>> listTables() {
        return ResponseEntity.ok(store.listNames());
    }

    @GetMapping("/tables/{name}")
    public ResponseEntity<Map<String, Object>> getTable(@PathVariable String name) {
        TableData t = store.get(name);
        if (t == null) {
            return ResponseEntity.status(404).body(Map.of("error", "Table '" + name + "' not found"));
        }
        Map<String, Object> info = new LinkedHashMap<>();
        info.put("name",           t.name());
        info.put("rows",           t.rowCount());
        info.put("columns",        t.columns());
        info.put("columnCount",    t.columnCount());
        info.put("estimatedBytes", t.estimatedBytes());
        info.put("estimatedMB",    String.format("%.2f", t.estimatedBytes() / 1_048_576.0));
        info.put("createdAt",      t.createdAt().toString());
        return ResponseEntity.ok(info);
    }

    /**
     * GET /api/tables/{name}/data
     *
     * Returns the actual row data so the browser can hydrate a WebGPU GridDB instance.
     * Optionally limited with ?limit=N (default 1 000 000).
     */
    @GetMapping("/tables/{name}/data")
    public ResponseEntity<?> getTableData(
            @PathVariable String name,
            @RequestParam(defaultValue = "1000000") int limit) {

        TableData t = store.get(name);
        if (t == null) {
            return ResponseEntity.status(404).body(Map.of("error", "Table '" + name + "' not found"));
        }

        List<Map<String, Object>> rows = limit >= t.rowCount()
                ? t.rows()
                : t.rows().subList(0, limit);

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("name",    t.name());
        body.put("columns", t.columns());
        body.put("rows",    rows);
        body.put("total",   t.rowCount());
        return ResponseEntity.ok(body);
    }

    @DeleteMapping("/tables/{name}")
    public ResponseEntity<Map<String, Object>> deleteTable(@PathVariable String name) {
        if (!store.exists(name)) {
            return ResponseEntity.status(404).body(Map.of("error", "Table '" + name + "' not found"));
        }
        store.delete(name);
        return ResponseEntity.ok(Map.of("deleted", name));
    }

    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> stats() {
        List<String> names = store.listNames();
        Map<String, Object> tableStats = new LinkedHashMap<>();
        for (String n : names) {
            TableData t = store.get(n);
            if (t != null) {
                tableStats.put(n, Map.of(
                        "rows",    t.rowCount(),
                        "columns", t.columnCount()
                ));
            }
        }
        return ResponseEntity.ok(Map.of(
                "tableCount",         names.size(),
                "totalRows",          store.totalRows(),
                "totalEstimatedBytes", store.totalEstimatedBytes(),
                "tables",             tableStats
        ));
    }
}
