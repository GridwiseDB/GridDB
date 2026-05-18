package com.griddb.model;

/**
 * Request body for POST /api/import/mysql and POST /api/import/test-mysql.
 * All fields are optional – missing ones fall back to application.properties / .env defaults.
 */
public record MySqlImportRequest(
        String host,
        Integer port,
        String user,
        String password,
        String database,
        /** MySQL table name – required unless {@code query} is provided. */
        String table,
        /** Raw SELECT query – required unless {@code table} is provided. */
        String query,
        /** GridDB table name to store results under. Defaults to {@code table}. */
        String tableName,
        /** Row cap (max 1 000 000). Falls back to {@code mysql.row-limit} property. */
        Integer limit,
        boolean ssl
) {}
