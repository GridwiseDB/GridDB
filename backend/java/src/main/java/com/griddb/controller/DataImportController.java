package com.griddb.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.griddb.model.ImportResult;
import com.griddb.model.MySqlImportRequest;
import com.griddb.service.DataImportService;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

/**
 * POST /api/import/csv        – upload CSV file or raw CSV text
 * POST /api/import/json       – upload JSON file or JSON array body
 * POST /api/import/mysql      – pull a table / SELECT from MySQL into GridDB
 * POST /api/import/test-mysql – verify MySQL credentials without importing
 */
@RestController
@RequestMapping("/api/import")
public class DataImportController {

    private final DataImportService importService;
    private final ObjectMapper      objectMapper;

    public DataImportController(DataImportService importService, ObjectMapper objectMapper) {
        this.importService = importService;
        this.objectMapper  = objectMapper;
    }

    // ── CSV ───────────────────────────────────────────────────────────────────
    /**
     * Accepts two forms:
     *  1. multipart/form-data  – field named "file" (any CSV file)
     *  2. text/plain or text/csv – raw CSV in the request body
     *
     * Query param {@code tableName} sets the GridDB table name.
     * Falls back to the uploaded filename (minus extension) or "csv_import".
     */
    @PostMapping("/csv")
    public ResponseEntity<ImportResult> importCsv(
            @RequestParam(required = false) String tableName,
            @RequestParam(required = false) MultipartFile file,
            HttpServletRequest request
    ) throws Exception {

        String csv;
        String name;

        if (file != null && !file.isEmpty()) {
            csv  = new String(file.getBytes(), StandardCharsets.UTF_8);
            name = tableName != null ? tableName : stripExtension(file.getOriginalFilename());
        } else {
            byte[] body = request.getInputStream().readAllBytes();
            if (body.length == 0) {
                throw new IllegalArgumentException("No CSV data provided. Send a file (multipart) or raw CSV text in the body.");
            }
            csv  = new String(body, StandardCharsets.UTF_8);
            name = tableName != null ? tableName : "csv_import";
        }

        return ResponseEntity.ok(importService.importCsv(name, csv));
    }

    // ── JSON ──────────────────────────────────────────────────────────────────
    /**
     * Accepts three forms:
     *  1. multipart/form-data  – field named "file" (.json file)
     *  2. application/json     – JSON array  [{"col": "val"}, ...]
     *  3. application/json     – object      {"tableName": "...", "data": [...]}
     */
    @PostMapping("/json")
    public ResponseEntity<ImportResult> importJson(
            @RequestParam(required = false) String tableName,
            @RequestParam(required = false) MultipartFile file,
            HttpServletRequest request
    ) throws Exception {

        String json;
        String name;

        if (file != null && !file.isEmpty()) {
            json = new String(file.getBytes(), StandardCharsets.UTF_8);
            name = tableName != null ? tableName : stripExtension(file.getOriginalFilename());
        } else {
            byte[] body = request.getInputStream().readAllBytes();
            if (body.length == 0) {
                throw new IllegalArgumentException("No JSON data provided. Send a .json file (multipart) or a JSON array in the body.");
            }
            json = new String(body, StandardCharsets.UTF_8);
            name = tableName; // may be null – resolved below
        }

        Object parsed = objectMapper.readValue(json, Object.class);

        List<Map<String, Object>> rows;
        if (parsed instanceof List<?> list) {
            //noinspection unchecked
            rows = (List<Map<String, Object>>) list;
            if (name == null) name = "json_import";
        } else if (parsed instanceof Map<?, ?> map) {
            // { "tableName": "...", "data": [...] }
            if (map.containsKey("tableName") && name == null) {
                name = (String) map.get("tableName");
            }
            if (name == null) name = "json_import";
            Object data = map.get("data");
            if (!(data instanceof List<?>)) {
                throw new IllegalArgumentException("Expected 'data' key to be a JSON array");
            }
            //noinspection unchecked
            rows = (List<Map<String, Object>>) data;
        } else {
            throw new IllegalArgumentException("JSON body must be an array or an object with a 'data' array");
        }

        return ResponseEntity.ok(importService.importJson(name, rows));
    }

    // ── MySQL import ──────────────────────────────────────────────────────────

    @PostMapping("/mysql")
    public ResponseEntity<ImportResult> importFromMysql(@RequestBody MySqlImportRequest req) throws Exception {
        return ResponseEntity.ok(importService.importFromMysql(req));
    }

    // ── MySQL connection test ─────────────────────────────────────────────────

    @PostMapping("/test-mysql")
    public ResponseEntity<Map<String, Object>> testMysql(@RequestBody MySqlImportRequest req) throws Exception {
        return ResponseEntity.ok(importService.testMysqlConnection(req));
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static String stripExtension(String filename) {
        if (filename == null || filename.isBlank()) return "import";
        int dot = filename.lastIndexOf('.');
        return (dot > 0) ? filename.substring(0, dot) : filename;
    }
}
