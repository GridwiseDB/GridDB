package com.griddb.service;

import com.griddb.model.ImportResult;
import com.griddb.model.MySqlImportRequest;
import com.griddb.model.TableData;
import com.griddb.store.TableStore;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.StringReader;
import java.sql.*;
import java.time.Instant;
import java.util.*;

@Service
public class DataImportService {

    private final TableStore store;

    @Value("${mysql.host:127.0.0.1}")
    private String defaultHost;

    @Value("${mysql.port:3306}")
    private int defaultPort;

    @Value("${mysql.user:}")
    private String defaultUser;

    @Value("${mysql.password:}")
    private String defaultPassword;

    @Value("${mysql.database:}")
    private String defaultDatabase;

    @Value("${mysql.row-limit:200000}")
    private int defaultRowLimit;

    public DataImportService(TableStore store) {
        this.store = store;
    }

    // ── CSV ───────────────────────────────────────────────────────────────────

    public ImportResult importCsv(String tableName, String csvContent) throws IOException {
        validateName(tableName);
        long start = System.currentTimeMillis();

        CSVFormat format = CSVFormat.DEFAULT.builder()
                .setHeader()
                .setSkipHeaderRecord(true)
                .setTrim(true)
                .setIgnoreEmptyLines(true)
                .build();

        List<String> columns;
        List<Map<String, Object>> rows = new ArrayList<>();

        try (CSVParser parser = new CSVParser(new StringReader(csvContent), format)) {
            columns = parser.getHeaderNames();
            if (columns.isEmpty()) {
                throw new IllegalArgumentException("CSV has no header row");
            }
            for (CSVRecord record : parser) {
                Map<String, Object> row = new LinkedHashMap<>();
                for (String col : columns) {
                    row.put(col, record.isMapped(col) ? record.get(col) : null);
                }
                rows.add(row);
            }
        }

        store.put(tableName, new TableData(tableName, columns, rows, Instant.now()));
        return new ImportResult(tableName, rows.size(), columns, System.currentTimeMillis() - start);
    }

    // ── JSON ──────────────────────────────────────────────────────────────────

    public ImportResult importJson(String tableName, List<Map<String, Object>> jsonRows) {
        validateName(tableName);
        if (jsonRows == null || jsonRows.isEmpty()) {
            throw new IllegalArgumentException("JSON array is empty");
        }
        long start = System.currentTimeMillis();
        List<String> columns = new ArrayList<>(jsonRows.get(0).keySet());
        store.put(tableName, new TableData(tableName, columns, jsonRows, Instant.now()));
        return new ImportResult(tableName, jsonRows.size(), columns, System.currentTimeMillis() - start);
    }

    // ── MySQL import ──────────────────────────────────────────────────────────

    public ImportResult importFromMysql(MySqlImportRequest req) throws SQLException {
        String host     = resolve(req.host(),     defaultHost);
        int    port     = req.port() != null ? req.port() : defaultPort;
        String user     = resolve(req.user(),     defaultUser);
        String password = req.password() != null  ? req.password() : defaultPassword;
        String database = resolve(req.database(), defaultDatabase);

        if (user.isBlank())     throw new IllegalArgumentException("'user' is required (or set MYSQL_USER in .env)");
        if (database.isBlank()) throw new IllegalArgumentException("'database' is required (or set MYSQL_DATABASE in .env)");
        if (req.table() == null && req.query() == null) {
            throw new IllegalArgumentException("Provide either 'table' or 'query'");
        }

        String sql = buildSql(req);
        String griddbName = resolveGriddbName(req);

        long start = System.currentTimeMillis();
        String url = buildJdbcUrl(host, port, database, req.ssl());

        List<String> columns = new ArrayList<>();
        List<Map<String, Object>> rows = new ArrayList<>();

        try (Connection conn = DriverManager.getConnection(url, user, password);
             Statement stmt = conn.createStatement();
             ResultSet rs   = stmt.executeQuery(sql)) {

            ResultSetMetaData meta = rs.getMetaData();
            int colCount = meta.getColumnCount();
            for (int i = 1; i <= colCount; i++) {
                columns.add(meta.getColumnLabel(i));
            }
            while (rs.next()) {
                Map<String, Object> row = new LinkedHashMap<>();
                for (int i = 1; i <= colCount; i++) {
                    row.put(columns.get(i - 1), rs.getObject(i));
                }
                rows.add(row);
            }
        }

        store.put(griddbName, new TableData(griddbName, columns, rows, Instant.now()));
        return new ImportResult(griddbName, rows.size(), columns, System.currentTimeMillis() - start);
    }

    // ── MySQL connection test ─────────────────────────────────────────────────

    public Map<String, Object> testMysqlConnection(MySqlImportRequest req) throws SQLException {
        String host     = resolve(req.host(),     defaultHost);
        int    port     = req.port() != null ? req.port() : defaultPort;
        String user     = resolve(req.user(),     defaultUser);
        String password = req.password() != null  ? req.password() : defaultPassword;
        String database = resolve(req.database(), defaultDatabase);

        if (user.isBlank())     throw new IllegalArgumentException("'user' is required");
        if (database.isBlank()) throw new IllegalArgumentException("'database' is required");

        String url = buildJdbcUrl(host, port, database, req.ssl());
        List<String> tables = new ArrayList<>();

        try (Connection conn  = DriverManager.getConnection(url, user, password);
             Statement  stmt  = conn.createStatement();
             ResultSet  rs    = stmt.executeQuery("SHOW TABLES")) {
            while (rs.next()) {
                tables.add(rs.getString(1));
            }
        }
        return Map.of("ok", true, "database", database, "tables", tables);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private String buildSql(MySqlImportRequest req) {
        if (req.query() != null) {
            String trimmed = req.query().trim().replaceAll("\\s+", " ");
            if (!trimmed.toUpperCase().startsWith("SELECT ")) {
                throw new IllegalArgumentException("Only SELECT queries are allowed");
            }
            return trimmed;
        }
        String tbl = req.table();
        if (!tbl.matches("[a-zA-Z0-9_]+")) {
            throw new IllegalArgumentException("Invalid MySQL table name: only letters, digits, underscore allowed");
        }
        int envLimit = defaultRowLimit > 0 ? defaultRowLimit : 200_000;
        int rowCap   = req.limit() != null ? Math.min(req.limit(), 1_000_000) : Math.min(envLimit, 1_000_000);
        return "SELECT * FROM `" + tbl + "` LIMIT " + rowCap;
    }

    private String buildJdbcUrl(String host, int port, String database, boolean ssl) {
        String sslParam = ssl
                ? "useSSL=true&requireSSL=true"
                : "useSSL=false&allowPublicKeyRetrieval=true";
        return String.format("jdbc:mysql://%s:%d/%s?%s&serverTimezone=UTC&connectTimeout=10000",
                host, port, database, sslParam);
    }

    private String resolveGriddbName(MySqlImportRequest req) {
        if (req.tableName() != null && req.tableName().matches("[a-zA-Z0-9_-]+")) return req.tableName();
        if (req.table()     != null && req.table().matches("[a-zA-Z0-9_-]+"))     return req.table();
        return "mysql_import";
    }

    private static String resolve(String value, String fallback) {
        return (value != null && !value.isBlank()) ? value : (fallback != null ? fallback : "");
    }

    private static void validateName(String name) {
        if (name == null || !name.matches("[a-zA-Z0-9_-]+")) {
            throw new IllegalArgumentException("Invalid table name '" + name +
                    "'. Use letters, digits, underscore, or hyphen only.");
        }
    }
}
