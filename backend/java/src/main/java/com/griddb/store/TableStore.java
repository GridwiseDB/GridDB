package com.griddb.store;

import com.griddb.model.TableData;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Thread-safe in-memory store that holds all imported tables.
 * Equivalent to the GPU-resident tables in the browser GridDB instance.
 */
@Component
public class TableStore {

    private final ConcurrentHashMap<String, TableData> tables = new ConcurrentHashMap<>();

    public void put(String name, TableData data) {
        tables.put(name, data);
    }

    public TableData get(String name) {
        return tables.get(name);
    }

    public boolean exists(String name) {
        return tables.containsKey(name);
    }

    public void delete(String name) {
        tables.remove(name);
    }

    public List<String> listNames() {
        return new ArrayList<>(tables.keySet());
    }

    public Map<String, TableData> all() {
        return Map.copyOf(tables);
    }

    public long totalRows() {
        return tables.values().stream().mapToLong(TableData::rowCount).sum();
    }

    public long totalEstimatedBytes() {
        return tables.values().stream().mapToLong(TableData::estimatedBytes).sum();
    }
}
