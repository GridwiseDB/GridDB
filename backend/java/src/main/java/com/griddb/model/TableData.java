package com.griddb.model;

import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * Immutable snapshot of a table stored in the in-memory TableStore.
 */
public record TableData(
        String name,
        List<String> columns,
        List<Map<String, Object>> rows,
        Instant createdAt
) {
    public int rowCount()    { return rows.size(); }
    public int columnCount() { return columns.size(); }

    /** Estimated size in bytes (rough: 64 bytes per cell). */
    public long estimatedBytes() {
        return (long) rowCount() * columnCount() * 64;
    }
}
