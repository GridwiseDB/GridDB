package com.griddb.model;

import java.util.List;

/** Returned by every successful import endpoint. */
public record ImportResult(
        String tableName,
        int rows,
        List<String> columns,
        long executionTimeMs
) {}
