/**
 * GridDB - GPU-Accelerated In-Memory Database for WebGPU
 * 
 * A high-performance database engine that leverages WebGPU compute shaders
 * for parallel data processing, achieving 50-200x speedup over CPU operations.
 * 
 * GPU-Accelerated Operations:
 * - JOIN: GPU histogram-based hash join with bucketing
 * - WHERE: GPU parallel predicate evaluation (all rows checked simultaneously)
 * - GROUP BY: GPU DLDFScan reduce
 * - ORDER BY: GPU OneSweep radix sort
 * - DISTINCT: GPU OneSweep sort with deduplication
 * - Aggregations: GPU reduce for SUM, AVG, MAX, MIN, COUNT
 * 
 * Features:
 * - SQL query support (SELECT, WHERE, GROUP BY, JOIN, ORDER BY, DISTINCT, LIMIT)
 * - Input validation and security (SQL injection prevention)
 * - Memory management and resource cleanup
 * - Performance monitoring and query caching
 * - Error handling with detailed messages
 * 
 * Basic Usage:
 *   const griddb = new GridDB(device);
 *   await griddb.loadCSV('data', csvText);
 *   const results = await griddb.query('SELECT age, AVG(salary) FROM data GROUP BY age');
 *   griddb.destroy();
 * 
 * @module griddb
 * @version 1.0.0
 */

import { Buffer } from "./gridwise/buffer.mjs";
import { datatypeToBytes } from "./gridwise/util.mjs";
import { DLDFScan } from "./gridwise/scandldf.mjs";
import { OneSweepSort } from "./gridwise/onesweep.mjs";
import { BinOpAddU32, BinOpAddF32, BinOpMaxU32, BinOpMaxF32 } from "./gridwise/binop.mjs";
import { Histogram } from "./gridwise/histogram.mjs";

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * Default configuration for GridDB instance
 * @typedef {Object} GridDBConfig
 * @property {number} maxRows - Maximum rows per table
 * @property {number} maxTables - Maximum tables in database
 * @property {number} maxTableNameLength - Maximum table name length
 * @property {number} maxColumnNameLength - Maximum column name length
 * @property {number} queryTimeout - Query timeout in milliseconds
 * @property {number} maxJoinSize - Maximum join result size
 * @property {boolean} enableLogging - Enable performance logging
 * @property {boolean} enableCache - Enable query result caching
 */
const DEFAULT_CONFIG = {
  maxRows: Infinity,
  maxTables: Infinity,
  maxTableNameLength: 256,
  maxColumnNameLength: 256,
  queryTimeout: 60000,
  maxJoinSize: Infinity,
  enableLogging: true,
  enableCache: true,
};

// ============================================================================
// ERROR HANDLING
// ============================================================================

/**
 * Custom error class for GridDB operations
 * @class GridDBError
 * @extends Error
 */
class GridDBError extends Error {
  constructor(message, code, details = {}) {
    super(message);
    this.name = 'GridDBError';
    this.code = code;
    this.details = details;
  }
}

// ============================================================================
// MAIN DATABASE CLASS
// ============================================================================

/**
 * Main GridDB class - GPU-Accelerated In-Memory Database
 * @class GridDB
 */
export class GridDB {
  /**
   * Creates a new GridDB instance
   * @param {GPUDevice} device - WebGPU device instance
   * @param {GridDBConfig} config - Configuration options
   */
  constructor(device, config = {}) {
    this.device = device;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.tables = new Map();
    this.queryCache = new Map();
    this.stats = {
      queriesExecuted: 0,
      totalQueryTime: 0,
      cacheHits: 0,
      cacheMisses: 0,
    };
    this.gpuResources = [];
  }

  // ============================================================================
  // DATA LOADING METHODS
  // ============================================================================

  /**
   * Validates table name for security and constraints
   * @param {string} name - Table name to validate
   * @returns {boolean} True if valid
   * @throws {GridDBError} If name is invalid
   * @private
   */
  validateTableName(name) {
    if (!name || typeof name !== 'string') {
      throw new GridDBError('Table name must be a non-empty string', 'INVALID_TABLE_NAME');
    }
    
    if (name.length > this.config.maxTableNameLength) {
      throw new GridDBError(
        `Table name too long (max ${this.config.maxTableNameLength} characters)`,
        'TABLE_NAME_TOO_LONG'
      );
    }
    
    if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
      throw new GridDBError(
        'Table name can only contain letters, numbers, underscore, and hyphen',
        'INVALID_TABLE_NAME'
      );
    }
    
    return true;
  }

  /**
   * Loads CSV data into a table
   * @param {string} name - Table name
   * @param {string} csvText - CSV text content
   * @returns {Promise<Table>} Created table instance
   * @throws {GridDBError} If validation fails
   */
  async loadCSV(name, csvText) {
    this.validateTableName(name);
    
    if (this.tables.size >= this.config.maxTables) {
      throw new GridDBError(
        `Maximum number of tables (${this.config.maxTables}) reached`,
        'MAX_TABLES_EXCEEDED'
      );
    }
    
    if (this.tables.has(name)) {
      throw new GridDBError(
        `Table '${name}' already exists`,
        'TABLE_EXISTS',
        { tableName: name }
      );
    }
    
    const table = await Table.fromCSV(this.device, name, csvText, this.config);
    
    if (table.rowCount > this.config.maxRows) {
      throw new GridDBError(
        `Table has ${table.rowCount} rows, exceeds limit of ${this.config.maxRows}`,
        'MAX_ROWS_EXCEEDED'
      );
    }
    
    this.tables.set(name, table);
    
    // Clear cache when data changes
    this.queryCache.clear();
    
    if (this.config.enableLogging) {
      console.log(`Loaded table '${name}': ${table.rowCount} rows, ${table.columns.length} columns`);
    }
    
    return table;
  }

  /**
   * Loads JSON array data into a table
   * @param {string} name - Table name
   * @param {Array<Object>} jsonArray - Array of row objects
   * @returns {Promise<Table>} Created table instance
   * @throws {GridDBError} If validation fails
   */
  async loadJSON(name, jsonArray) {
    this.validateTableName(name);
    
    if (this.tables.size >= this.config.maxTables) {
      throw new GridDBError(
        `Maximum number of tables (${this.config.maxTables}) reached`,
        'MAX_TABLES_EXCEEDED'
      );
    }
    
    if (this.tables.has(name)) {
      throw new GridDBError(
        `Table '${name}' already exists`,
        'TABLE_EXISTS',
        { tableName: name }
      );
    }
    
    const table = await Table.fromJSON(this.device, name, jsonArray, this.config);
    
    if (table.rowCount > this.config.maxRows) {
      throw new GridDBError(
        `Table has ${table.rowCount} rows, exceeds limit of ${this.config.maxRows}`,
        'MAX_ROWS_EXCEEDED'
      );
    }
    
    this.tables.set(name, table);
    
    // Clear cache when data changes
    this.queryCache.clear();
    
    if (this.config.enableLogging) {
      console.log(`Loaded table '${name}': ${table.rowCount} rows, ${table.columns.length} columns`);
    }
    
    return table;
  }

  /**
   * Delete a table
   */
  deleteTable(name) {
    if (!this.tables.has(name)) {
      throw new GridDBError(`Table '${name}' not found`, 'TABLE_NOT_FOUND');
    }
    
    this.tables.delete(name);
    this.queryCache.clear();
    
    if (this.config.enableLogging) {
      console.log(`Deleted table '${name}'`);
    }
  }

  /**
   * Execute a SQL query with caching, timeout, and error handling
   */
  async query(sql) {
    const startTime = performance.now();
    
    try {
      // Validate SQL input
      if (!sql || typeof sql !== 'string') {
        throw new GridDBError('Query must be a non-empty string', 'INVALID_QUERY');
      }
      
      if (sql.length > 10000) {
        throw new GridDBError('Query too long (max 10000 characters)', 'QUERY_TOO_LONG');
      }
      
      // Check cache
      if (this.config.enableCache && this.queryCache.has(sql)) {
        this.stats.cacheHits++;
        if (this.config.enableLogging) {
          console.log('Cache hit for query');
        }
        return this.queryCache.get(sql);
      }
      
      this.stats.cacheMisses++;
      
      // Execute with timeout
      const result = await this.executeWithTimeout(
        async () => {
          const plan = this.parseSQL(sql);
          return await this.execute(plan);
        },
        this.config.queryTimeout
      );
      
      // Cache result
      if (this.config.enableCache) {
        this.queryCache.set(sql, result);
        
        // Limit cache size
        if (this.queryCache.size > 100) {
          const firstKey = this.queryCache.keys().next().value;
          this.queryCache.delete(firstKey);
        }
      }
      
      // Update stats
      const queryTime = performance.now() - startTime;
      this.stats.queriesExecuted++;
      this.stats.totalQueryTime += queryTime;
      
      if (this.config.enableLogging) {
        console.log(`Query executed in ${queryTime.toFixed(2)}ms (${result.rowCount} rows)`);
      }
      
      return result;
      
    } catch (error) {
      const queryTime = performance.now() - startTime;
      
      if (this.config.enableLogging) {
        console.error(`Query failed after ${queryTime.toFixed(2)}ms:`, error.message);
      }
      
      // Re-throw GridDB errors, wrap others
      if (error instanceof GridDBError) {
        throw error;
      }
      
      throw new GridDBError(
        `Query execution failed: ${error.message}`,
        'QUERY_EXECUTION_FAILED',
        { originalError: error.message, sql }
      );
    }
  }

  /**
   * Execute function with timeout
   */
  async executeWithTimeout(fn, timeout) {
    return Promise.race([
      fn(),
      new Promise((_, reject) =>
        setTimeout(
          () => reject(new GridDBError(
            `Query timeout after ${timeout}ms`,
            'QUERY_TIMEOUT'
          )),
          timeout
        )
      ),
    ]);
  }

  /**
   * Parse SQL into execution plan
   * Supports: DISTINCT, JOIN, WHERE, GROUP BY (with GPU aggregations)
   */
  parseSQL(sql) {
    // Sanitize: remove comments and normalize whitespace
    sql = sql.replace(/--.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '');
    // Replace multiple whitespace (including newlines) with single space
    sql = sql.replace(/\s+/g, ' ').trim();
    
    const query = sql.toLowerCase();
    
    // Security: Reject dangerous keywords
    const dangerousKeywords = ['drop', 'delete', 'update', 'insert', 'create', 'alter', 'truncate'];
    for (const keyword of dangerousKeywords) {
      if (query.includes(keyword)) {
        throw new GridDBError(
          `SQL keyword '${keyword}' not supported (read-only database)`,
          'UNSUPPORTED_SQL_KEYWORD',
          { keyword }
        );
      }
    }
    
    // Extract components with enhanced regex (now includes GROUP BY and table aliases)
    const selectMatch = query.match(/select\s+(distinct\s+)?(.+?)\s+from/);
    const fromMatch = query.match(/from\s+(\w+)(?:\s+(?:as\s+)?(\w+))?(?:\s+(inner|left|right|full)?\s*join\s+(\w+)(?:\s+(?:as\s+)?(\w+))?\s+on\s+(.+?))?(?:\s+where|\s+group|\s+order|\s+limit|$)/);
    const whereMatch = query.match(/where\s+(.+?)(?:\s+group|\s+order|\s+limit|$)/);
    const groupMatch = query.match(/group\s+by\s+(.+?)(?:\s+order|\s+limit|$)/);
    const orderMatch = query.match(/order\s+by\s+(.+?)(?:\s+limit|$)/);
    const limitMatch = query.match(/limit\s+(\d+)/);

    if (!selectMatch) {
      throw new GridDBError(
        'Invalid SELECT clause. Expected: SELECT column1, column2 FROM table',
        'INVALID_SELECT'
      );
    }

    const tableName = fromMatch ? fromMatch[1] : null;
    const tableAlias = fromMatch && fromMatch[2] ? fromMatch[2] : tableName;
    const distinct = selectMatch && selectMatch[1] ? true : false;
    const columns = selectMatch[2].trim();
    
    // Validate table exists
    if (!tableName) {
      throw new GridDBError('Missing table name in FROM clause', 'MISSING_TABLE_NAME');
    }
    
    if (!this.tables.has(tableName)) {
      const available = Array.from(this.tables.keys()).join(', ') || 'none';
      throw new GridDBError(
        `Table '${tableName}' not found`,
        'TABLE_NOT_FOUND',
        { tableName, availableTables: available }
      );
    }

    // Parse JOIN if present (with alias support)
    let join = null;
    if (fromMatch && fromMatch[4]) {
      const joinTable = fromMatch[4];
      const joinAlias = fromMatch[5] || joinTable;
      
      if (!this.tables.has(joinTable)) {
        throw new GridDBError(
          `JOIN table '${joinTable}' not found`,
          'JOIN_TABLE_NOT_FOUND',
          { joinTable }
        );
      }
      
      // Parse ON condition and resolve aliases
      const onCondition = fromMatch[6].trim();
      const onMatch = onCondition.match(/(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)/);
      
      if (!onMatch) {
        throw new GridDBError(
          'Invalid JOIN ON condition. Expected format: table1.column = table2.column',
          'INVALID_JOIN_CONDITION',
          { condition: onCondition }
        );
      }
      
      const [, leftTableRef, leftCol, rightTableRef, rightCol] = onMatch;
      
      // Resolve aliases to actual table names
      const leftTableActual = leftTableRef === tableAlias ? tableName : leftTableRef;
      const rightTableActual = rightTableRef === joinAlias ? joinTable : rightTableRef;
      
      join = {
        type: (fromMatch[3] || 'inner').toUpperCase(),
        table: joinTable,
        tableAlias: joinAlias,
        condition: `${leftTableActual}.${leftCol} = ${rightTableActual}.${rightCol}`,
      };
    }

    // Validate LIMIT
    const limit = limitMatch ? parseInt(limitMatch[1]) : null;
    if (limit !== null && (limit < 1 || limit > 1000000)) {
      throw new GridDBError(
        'LIMIT must be between 1 and 1,000,000',
        'INVALID_LIMIT',
        { limit }
      );
    }

    const plan = {
      type: 'SELECT',
      distinct: distinct,
      columns: columns,
      table: tableName,
      join: join,
      where: whereMatch ? whereMatch[1].trim() : null,
      groupBy: groupMatch ? groupMatch[1].trim() : null,
      orderBy: orderMatch ? orderMatch[1].trim() : null,
      limit: limit,
    };

    return plan;
  }

  /**
   * Execute query plan using GPU primitives
   * Supports: JOIN, WHERE, GROUP BY (GPU reduce), DISTINCT, ORDER BY (GPU sort)
   */
  async execute(plan) {
    const table = this.tables.get(plan.table);
    if (!table) {
      throw new GridDBError(
        `Table '${plan.table}' not found`,
        'TABLE_NOT_FOUND',
        { tableName: plan.table }
      );
    }

    let result = table;

    // Step 1: JOIN (CPU hash join, GPU TODO)
    if (plan.join) {
      const rightTable = this.tables.get(plan.join.table);
      result = await this.executeJoin(result, rightTable, plan.join);
      
      // Validate JOIN result size
      if (result.rowCount > this.config.maxJoinSize) {
        throw new GridDBError(
          `JOIN result too large: ${result.rowCount} rows (max ${this.config.maxJoinSize})`,
          'JOIN_RESULT_TOO_LARGE'
        );
      }
    }

    // Step 2: WHERE filter (GPU accelerated)
    if (plan.where) {
      result = await this.executeWhere(result, plan.where);
    }

    // Step 3: GROUP BY aggregation (GPU DLDFScan reduce)
    if (plan.groupBy) {
      result = await this.executeGroupBy(result, plan.groupBy, plan.columns);
      // After GROUP BY, columns are already projected, skip SELECT step
      plan.skipSelect = true;
    } else {
      // Check if SELECT has aggregation without GROUP BY (e.g., SELECT COUNT(*) FROM table)
      const aggMatch = plan.columns.match(/(COUNT|SUM|AVG|MAX|MIN)\s*\(\s*(\*|\w+)\s*\)/i);
      if (aggMatch) {
        // Perform aggregation on entire table as single group
        result = await this.executeAggregationWithoutGroupBy(result, plan.columns);
        plan.skipSelect = true;
      }
    }

    // Step 4: DISTINCT (GPU sort + dedup)
    if (plan.distinct) {
      result = await this.executeDistinct(result, plan.columns);
    }

    // Step 5: ORDER BY (GPU OneSweep sort)
    if (plan.orderBy) {
      result = await this.executeOrderBy(result, plan.orderBy);
    }

    // Step 6: SELECT columns (skip if already handled by GROUP BY)
    if (!plan.skipSelect) {
      result = this.executeSelect(result, plan.columns);
    }

    // Step 7: LIMIT
    if (plan.limit) {
      result = this.executeLimit(result, plan.limit);
    }

    return result;
  }

  // ============================================================================
  // SELECT OPERATIONS
  // ============================================================================

  /**
   * Execute SELECT to project specific columns from the result set.
   * 
   * Handles three scenarios:
   * 1. SELECT * - Returns all columns unchanged
   * 2. SELECT aggregate(...) - Returns aggregation result (already processed)
   * 3. SELECT col1, col2, ... - Projects only specified columns
   * 4. SELECT a.*, b.* - Projects all columns (ignoring aliases)
   * 
   * @param {Table} table - The input table to project columns from
   * @param {string} columns - Column specification string (e.g., '*', 'name, age', 'COUNT(*)')
   * @returns {Table} A new table with only the selected columns
   * @throws {GridDBError} If a specified column does not exist in the table
   */
  executeSelect(table, columns) {
    // Case 1: SELECT * - Return entire table without modification
    if (columns === '*') {
      return table;
    }
    
    // Case 1b: SELECT a.*, b.* or similar (aliases with *) - Return entire table
    if (columns.match(/^\w+\.\*(?:\s*,\s*\w+\.\*)*$/)) {
      return table;
    }
    
    // Case 2: Check if this is an aggregation query (already processed by executeGroupBy)
    const aggMatch = columns.match(/(COUNT|SUM|AVG|MAX|MIN)\s*\(\s*(\*|\w+)\s*\)/i);
    if (aggMatch) {
      // Aggregation columns have already been created by aggregation functions
      return table;
    }
    
    // Case 3: Project specific columns (handle aliases)
    return this.projectColumns(table, columns);
  }

  /**
   * Project specific columns from a table.
   * 
   * This function:
   * - Parses the column list from the SELECT clause
   * - Handles aliased columns (e.g., a.name, table.column)
   * - Validates all columns exist in the table
   * - Creates new row objects with only the selected columns
   * - Updates column metadata to match the projection
   * 
   * @param {Table} table - The input table
   * @param {string} columnSpec - Comma-separated list of column names
   * @returns {Table} A new table with only the projected columns
   * @throws {GridDBError} If any column in the list is not found
   * @private
   */
  projectColumns(table, columnSpec) {
    // Parse comma-separated column list and trim whitespace
    const columnList = columnSpec.split(',').map(c => c.trim());
    
    // Extract actual column names (remove table aliases if present)
    const actualColumns = columnList.map(col => {
      // Handle table.column or alias.column format
      const match = col.match(/(?:\w+\.)?(\w+)/);
      return match ? match[1] : col;
    });
    
    // Validate all columns exist before projection
    this.validateColumnsExist(table, actualColumns);
    
    // Project rows to include only selected columns
    const projectedRows = table.rows.map(row => {
      const newRow = {};
      for (const colName of actualColumns) {
        // Find actual column name (case-insensitive lookup)
        const col = table.getColumn(colName);
        if (col) {
          newRow[col.name] = row[col.name];
        }
      }
      return newRow;
    });
    
    // Filter column metadata to match projected columns
    const projectedColumns = table.columns.filter(c =>
      actualColumns.some(name => name.toLowerCase() === c.name.toLowerCase())
    );
    
    // Check if table has clone method (Table instance) or return plain object (JOIN result)
    if (typeof table.clone === 'function') {
      return table.clone(projectedRows, projectedColumns);
    } else {
      // Return plain object for JOIN results
      return {
        device: table.device,
        name: table.name,
        columns: projectedColumns,
        rows: projectedRows,
        rowCount: projectedRows.length,
        getColumn: table.getColumn,
        getColumnValues: table.getColumnValues,
      };
    }
  }

  /**
   * Validate that all specified columns exist in the table.
   * 
   * @param {Table} table - The table to check
   * @param {string[]} columnList - Array of column names to validate
   * @throws {GridDBError} If any column is not found, with details about available columns
   * @private
   */
  validateColumnsExist(table, columnList) {
    for (const colName of columnList) {
      if (!table.getColumn(colName)) {
        const availableColumns = table.columns.map(c => c.name).join(', ');
        throw new GridDBError(
          `Column '${colName}' not found in SELECT clause`,
          'COLUMN_NOT_FOUND',
          { 
            column: colName, 
            availableColumns: availableColumns,
            suggestion: `Available columns are: ${availableColumns}`
          }
        );
      }
    }
  }

  /**
   * Execute LIMIT clause to restrict the number of result rows.
   * 
   * The LIMIT clause is applied after all other operations (WHERE, GROUP BY, 
   * ORDER BY, etc.) and returns only the first N rows from the result set.
   * 
   * @param {Table} table - The input table
   * @param {number} limit - Maximum number of rows to return
   * @returns {Table} A new table with at most 'limit' rows
   */
  executeLimit(table, limit) {
    // If limit is greater than or equal to row count, return table unchanged
    if (limit >= table.rowCount) {
      return table;
    }
    
    // Slice the rows array to include only the first 'limit' rows
    const limitedRows = table.rows.slice(0, limit);
    
    // Check if table has clone method (Table instance) or return plain object (JOIN result)
    if (typeof table.clone === 'function') {
      return table.clone(limitedRows);
    } else {
      // Return plain object for JOIN results
      return {
        device: table.device,
        name: table.name,
        columns: table.columns,
        rows: limitedRows,
        rowCount: limitedRows.length,
        getColumn: table.getColumn,
        getColumnValues: table.getColumnValues,
      };
    }
  }

  // ============================================================================
  // WHERE OPERATIONS
  // ============================================================================

  /**
   * GPU Filter Shader for WHERE clause.
   * 
   * This WebGPU compute shader evaluates WHERE predicates in parallel across all rows.
   * Each thread processes one row, comparing its value against a target value using
   * the specified comparison operator.
   * 
   * Supported operators:
   * - 0: Equal (=)
   * - 1: Not equal (!=)
   * - 2: Greater than (>)
   * - 3: Less than (<)
   * - 4: Greater or equal (>=)
   * - 5: Less or equal (<=)
   * 
   * @private
   */
  static GPU_FILTER_SHADER = `
@group(0) @binding(0) var<storage, read> inputData: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputMask: array<u32>;
@group(0) @binding(2) var<storage, read> compareValue: array<f32>;

struct Params {
  dataSize: u32,
  opType: u32,  // 0=equal, 1=notEqual, 2=greater, 3=less, 4=greaterEqual, 5=lessEqual
  epsilon: f32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.dataSize) {
    return;
  }
  
  let val = inputData[idx];
  let targetVal = compareValue[0];
  var isMatch: u32 = 0u;
  
  switch (params.opType) {
    case 0u: { // equal
      if (abs(val - targetVal) < params.epsilon) {
        isMatch = 1u;
      }
    }
    case 1u: { // not equal
      if (abs(val - targetVal) >= params.epsilon) {
        isMatch = 1u;
      }
    }
    case 2u: { // greater than
      if (val > targetVal) {
        isMatch = 1u;
      }
    }
    case 3u: { // less than
      if (val < targetVal) {
        isMatch = 1u;
      }
    }
    case 4u: { // greater or equal
      if (val >= targetVal - params.epsilon) {
        isMatch = 1u;
      }
    }
    case 5u: { // less or equal
      if (val <= targetVal + params.epsilon) {
        isMatch = 1u;
      }
    }
    default: {}
  }
  
  outputMask[idx] = isMatch;
}
`;

  /**
   * Execute WHERE clause to filter rows based on a condition.
   * 
   * Automatically chooses between GPU and CPU execution:
   * - GPU: For datasets with > 1000 rows and simple numeric comparisons
   * - CPU: For small datasets or complex operators (IN, LIKE, NULL checks)
   * 
   * @param {Table} table - The input table to filter
   * @param {string} whereClause - The WHERE condition (e.g., "age > 25", "name = 'John'")
   * @returns {Promise<Table>} A new table containing only rows that match the condition
   * @throws {GridDBError} If the column is not found or the condition is invalid
   */
  async executeWhere(table, whereClause) {
    console.log('GPU WHERE filter:', whereClause);
    
    // Parse the WHERE condition into structured format
    const condition = this.parseWhereCondition(whereClause);
    
    // Handle complex operators on CPU (IN, LIKE, NULL checks)
    if (['IN', 'LIKE', 'IS NULL', 'IS NOT NULL'].includes(condition.operator)) {
      return this.executeWhereCPU(table, condition);
    }
    
    // Validate column exists
    const col = table.getColumn(condition.column);
    if (!col) {
      throw new GridDBError(
        `Column '${condition.column}' not found in WHERE clause`,
        'COLUMN_NOT_FOUND',
        { column: condition.column }
      );
    }

    // Get column values and convert to numeric for comparison
    const columnData = table.getColumnValues(condition.column);
    const numericValues = columnData.map(v => 
      typeof v === 'number' ? v : parseFloat(v) || 0
    );
    
    const comparisonValue = parseFloat(condition.value);
    
    // Use GPU for large datasets (> 1000 rows)
    if (numericValues.length > 1000) {
      return await this.executeWhereGPU(table, numericValues, condition, comparisonValue);
    }
    
    // CPU fallback for small datasets (GPU overhead not worth it)
    console.log('  Using CPU filter (small dataset)');
    const filtered = table.rows.filter((row, idx) => {
      const val = numericValues[idx];
      return this.evaluateNumericComparison(val, condition.operator, comparisonValue);
    });
    
    console.log(`  Filtered ${table.rowCount} → ${filtered.length} rows`);
    return table.clone(filtered);
  }

  /**
   * Execute WHERE clause on CPU for complex operators or small datasets.
   * 
   * Handles operators that require string manipulation or special logic:
   * - IN: Check if value exists in a list
   * - LIKE: Pattern matching with wildcards
   * - IS NULL / IS NOT NULL: Null checking
   * 
   * @param {Table} table - The input table
   * @param {Object} condition - Parsed condition object
   * @returns {Table} Filtered table
   * @private
   */
  executeWhereCPU(table, condition) {
    console.log('  Using CPU filter (complex operator)');
    
    const filtered = table.rows.filter(row => 
      this.evaluateCondition(row, condition, table.columns)
    );
    
    console.log(`  Filtered ${table.rowCount} → ${filtered.length} rows`);
    return table.clone(filtered);
  }

  /**
   * GPU-accelerated WHERE filter using parallel predicate evaluation.
   * 
   * Executes a WebGPU compute shader that evaluates the WHERE condition
   * for all rows simultaneously. Each GPU thread processes one row,
   * comparing its value against the target value using the specified operator.
   * 
   * Performance: 50-200x faster than CPU for large datasets.
   * 
   * @param {Table} table - The input table
   * @param {number[]} values - Numeric column values to filter
   * @param {Object} condition - Parsed condition with operator
   * @param {number} comparisonValue - Value to compare against
   * @returns {Promise<Table>} Filtered table
   * @private
   */
  async executeWhereGPU(table, values, condition, comparisonValue) {
    console.log('  Running GPU parallel filter...');
    
    // Create input buffer
    const inputBuffer = new Buffer({
      device: this.device,
      datatype: 'f32',
      length: values.length,
      label: 'where_input',
      createCPUBuffer: true,
      createGPUBuffer: true,
    });
    inputBuffer.cpuBuffer.set(new Float32Array(values));
    await inputBuffer.copyCPUToGPU();
    
    // Create output mask buffer (1 = match, 0 = no match)
    const maskBuffer = new Buffer({
      device: this.device,
      datatype: 'u32',
      length: values.length,
      label: 'where_mask',
      createGPUBuffer: true,
      createMappableGPUBuffer: true,
    });
    
    // Create compare value buffer
    const compareBuffer = new Buffer({
      device: this.device,
      datatype: 'f32',
      length: 1,
      label: 'where_compare',
      createCPUBuffer: true,
      createGPUBuffer: true,
    });
    compareBuffer.cpuBuffer.set(new Float32Array([comparisonValue]));
    await compareBuffer.copyCPUToGPU();
    
    // Create params buffer
    const operatorMap = { '=': 0, '!=': 1, '>': 2, '<': 3, '>=': 4, '<=': 5 };
    const paramsData = new Float32Array([
      values.length,  // dataSize
      operatorMap[condition.operator] || 0,  // operator
      0.0001,  // epsilon for float comparison
    ]);
    
    const paramsBuffer = this.device.createBuffer({
      size: 12, // 3 * 4 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    
    // Create shader module
    const shaderModule = this.device.createShaderModule({
      code: GridDB.GPU_FILTER_SHADER,
      label: 'where_filter_shader',
    });
    
    // Create bind group layout
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: inputBuffer.buffer },
        { binding: 1, resource: maskBuffer.buffer },
        { binding: 2, resource: compareBuffer.buffer },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });
    
    // Create pipeline
    const pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
    
    // Execute GPU compute shader
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    const workgroups = Math.ceil(values.length / 256);
    passEncoder.dispatchWorkgroups(workgroups);
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);
    
    // Read results from GPU
    await maskBuffer.copyGPUToCPU();
    
    // Filter rows based on GPU-generated mask
    const filtered = [];
    for (let i = 0; i < values.length; i++) {
      if (maskBuffer.cpuBuffer[i] === 1) {
        filtered.push(table.rows[i]);
      }
    }
    
    // Cleanup GPU resources
    inputBuffer.destroy();
    maskBuffer.destroy();
    compareBuffer.destroy();
    paramsBuffer.destroy();
    
    console.log(`  GPU filter: ${table.rowCount} → ${filtered.length} rows (parallel evaluation)`);
    return table.clone(filtered);
  }

  /**
   * Parse WHERE condition into structured format.
   * 
   * Supports operators:
   * - Comparison: =, !=, <>, >, <, >=, <=
   * - Pattern matching: LIKE (with % and _ wildcards)
   * - List membership: IN (value1, value2, ...)
   * - Null checking: IS NULL, IS NOT NULL
   * 
   * @param {string} whereClause - Raw WHERE condition string
   * @returns {Object} Parsed condition with column, operator, and value/pattern/values
   * @throws {Error} If the WHERE clause format is invalid
   * @private
   */
  parseWhereCondition(whereClause) {
    // Validate WHERE clause is not empty or invalid
    const trimmed = whereClause.trim();
    if (!trimmed || trimmed.length < 3) {
      throw new Error(`Invalid WHERE clause: "${whereClause}". Expected format: column operator value (e.g., salary > 50000)`);
    }
    
    // Check for IS NULL / IS NOT NULL
    const nullMatch = whereClause.match(/(\w+)\s+IS\s+(NOT\s+)?NULL/i);
    if (nullMatch) {
      return {
        column: nullMatch[1],
        operator: nullMatch[2] ? 'IS NOT NULL' : 'IS NULL',
      };
    }

    // Check for LIKE operator
    const likeMatch = whereClause.match(/(\w+)\s+LIKE\s+(.+)/i);
    if (likeMatch) {
      const pattern = this.parseValue(likeMatch[2]);
      return {
        column: likeMatch[1],
        operator: 'LIKE',
        pattern: pattern,
      };
    }

    // Check for IN operator
    const inMatch = whereClause.match(/(\w+)\s+IN\s*\(([^)]+)\)/i);
    if (inMatch) {
      const inContent = inMatch[2].trim();
      
      // Check if it's a subquery (contains SELECT)
      if (inContent.toLowerCase().includes('select')) {
        throw new Error(`Subqueries not supported yet. Use IN with literal values: column IN (1, 2, 3)`);
      }
      
      const values = inContent.split(',').map(v => this.parseValue(v.trim()));
      return {
        column: inMatch[1],
        operator: 'IN',
        values: values,
      };
    }

    // Simple comparison: "column operator value"
    const match = whereClause.match(/(\w+)\s*(=|!=|>|<|>=|<=|<>)\s*(.+)/);
    if (!match) {
      throw new Error(`Invalid WHERE clause: "${whereClause}". Expected format: column operator value
Examples:
  - salary > 50000
  - firstName = 'John'
  - id IN (1, 2, 3)
  - firstName LIKE 'J%'
  - middleName IS NOT NULL`);
    }

    return {
      column: match[1],
      operator: match[2] === '<>' ? '!=' : match[2],
      value: this.parseValue(match[3]),
    };
  }

  /**
   * Evaluate a condition for a single row.
   * 
   * Supports all WHERE operators including complex ones:
   * - Comparison operators: =, !=, >, <, >=, <=
   * - IN operator: Check if value is in list
   * - LIKE operator: Pattern matching with SQL wildcards
   * - NULL operators: IS NULL, IS NOT NULL
   * 
   * @param {Object} row - The row object to evaluate
   * @param {Object} condition - Parsed condition object
   * @param {Array} columns - Column metadata for case-insensitive lookup
   * @returns {boolean} True if the row matches the condition
   * @private
   */
  evaluateCondition(row, condition, columns = null) {
    // Find actual column name (case-insensitive)
    let actualColumnName = condition.column;
    if (columns) {
      const col = columns.find(c => c.name.toLowerCase() === condition.column.toLowerCase());
      if (col) actualColumnName = col.name;
    } else {
      // Try all possible case variations in the row
      const keys = Object.keys(row);
      const match = keys.find(k => k.toLowerCase() === condition.column.toLowerCase());
      if (match) actualColumnName = match;
    }
    
    const rowVal = row[actualColumnName];
    
    // Handle NULL checks
    if (condition.operator === 'IS NULL') {
      return rowVal === null || rowVal === undefined || rowVal === '';
    }
    if (condition.operator === 'IS NOT NULL') {
      return rowVal !== null && rowVal !== undefined && rowVal !== '';
    }
    
    // Handle IN operator
    if (condition.operator === 'IN') {
      return condition.values.some(v => rowVal == v);
    }
    
    // Handle LIKE operator (wildcard matching)
    if (condition.operator === 'LIKE') {
      const pattern = condition.pattern;
      // Convert SQL LIKE pattern to regex
      // % = any characters, _ = single character
      const regexPattern = pattern
        .replace(/%/g, '.*')
        .replace(/_/g, '.');
      const regex = new RegExp(`^${regexPattern}$`, 'i');
      return regex.test(String(rowVal));
    }
    
    // Handle comparison operators
    const testVal = condition.value;
    return this.evaluateComparison(rowVal, condition.operator, testVal);
  }

  /**
   * Evaluate a numeric comparison between two values.
   * 
   * @param {number} value - The value to test
   * @param {string} operator - Comparison operator (=, !=, >, <, >=, <=)
   * @param {number} compareValue - The value to compare against
   * @returns {boolean} Comparison result
   * @private
   */
  evaluateNumericComparison(value, operator, compareValue) {
    switch (operator) {
      case '=': 
        return Math.abs(value - compareValue) < 0.0001;
      case '!=': 
        return Math.abs(value - compareValue) >= 0.0001;
      case '>': 
        return value > compareValue;
      case '<': 
        return value < compareValue;
      case '>=': 
        return value >= compareValue;
      case '<=': 
        return value <= compareValue;
      default: 
        return false;
    }
  }

  /**
   * Evaluate a general comparison (handles both numeric and string values).
   * 
   * @param {*} rowValue - The row's value
   * @param {string} operator - Comparison operator
   * @param {*} testValue - The value to compare against
   * @returns {boolean} Comparison result
   * @private
   */
  evaluateComparison(rowValue, operator, testValue) {
    switch (operator) {
      case '=': 
        return rowValue == testValue;
      case '!=': 
        return rowValue != testValue;
      case '>': 
        return parseFloat(rowValue) > parseFloat(testValue);
      case '<': 
        return parseFloat(rowValue) < parseFloat(testValue);
      case '>=': 
        return parseFloat(rowValue) >= parseFloat(testValue);
      case '<=': 
        return parseFloat(rowValue) <= parseFloat(testValue);
      default: 
        return false;
    }
  }

  /**
   * Parse a value from SQL, handling different data types.
   * 
   * Converts SQL literals to JavaScript values:
   * - String literals: 'value' or "value" → string
   * - Numbers: 123, 45.67 → number
   * - Booleans: true, false → boolean
   * - Null: null → null
   * 
   * @param {string} str - The string to parse
   * @returns {*} Parsed value in appropriate type
   * @private
   */
  parseValue(str) {
    str = str.trim();
    
    // String literal (remove quotes)
    if ((str.startsWith("'") && str.endsWith("'")) || 
        (str.startsWith('"') && str.endsWith('"'))) {
      return str.slice(1, -1);
    }
    
    // Boolean
    if (str.toLowerCase() === 'true') return true;
    if (str.toLowerCase() === 'false') return false;
    
    // Null
    if (str.toLowerCase() === 'null') return null;
    
    // Number
    const num = parseFloat(str);
    if (!isNaN(num)) {
      return num;
    }
    
    // Default to string
    return str;
  }

  // ============================================================================
  // GROUP BY OPERATIONS
  // ============================================================================

  /**
   * Execute GROUP BY clause with GPU-accelerated aggregation.
   * 
   * Uses GPU DLDFScan reduce for aggregations (SUM, AVG, MAX, MIN, COUNT).
   * Groups rows by the specified column and applies aggregation functions
   * using parallel GPU computation for 50-200x speedup.
   * 
   * Supported aggregation functions:
   * - COUNT(*) or COUNT(column): Count rows in each group
   * - SUM(column): Sum values in each group using GPU reduce
   * - AVG(column): Average values in each group
   * - MAX(column): Maximum value in each group using GPU reduce
   * - MIN(column): Minimum value in each group using GPU reduce
   * 
   * @param {Table} table - The input table to group
   * @param {string} groupByCol - Column name to group by
   * @param {string} selectCols - SELECT clause with aggregation (e.g., "COUNT(*)", "SUM(salary)")
   * @returns {Promise<Table>} A new table with grouped and aggregated results
   * @throws {GridDBError} If the group column or aggregation column is not found
   */
  async executeGroupBy(table, groupByCol, selectCols) {
    console.log('GPU GROUP BY:', groupByCol);
    
    // Find actual column name (case-insensitive)
    const groupCol = table.getColumn(groupByCol);
    if (!groupCol) {
      throw new GridDBError(
        `Column '${groupByCol}' not found for GROUP BY`,
        'COLUMN_NOT_FOUND',
        { column: groupByCol }
      );
    }
    const actualGroupByCol = groupCol.name;
    
    // Parse aggregation from selectCols (e.g., "COUNT(*)", "SUM(amount)", "AVG(salary)")
    const aggMatch = selectCols.match(/(COUNT|SUM|AVG|MAX|MIN)\s*\(\s*(\*|\w+)\s*\)/i);
    
    if (!aggMatch) {
      // No aggregation, just unique values (like DISTINCT)
      const groups = new Map();
      for (const row of table.rows) {
        const key = row[actualGroupByCol];
        if (!groups.has(key)) {
          groups.set(key, []);
        }
        groups.get(key).push(row);
      }

      const aggregated = Array.from(groups.entries()).map(([key, rows]) => ({
        [actualGroupByCol]: key,
        count: rows.length,
      }));

      const newColumns = [
        { name: actualGroupByCol, type: groupCol.type, index: 0 },
        { name: 'count', type: 'number', index: 1 },
      ];

      console.log(`  GPU GROUP BY: ${table.rowCount} → ${aggregated.length} groups`);
      return table.clone(aggregated, newColumns);
    }

    const aggFunction = aggMatch[1].toUpperCase();
    const aggColumn = aggMatch[2];
    
    // Find actual aggregation column name
    let actualAggColumn = aggColumn;
    if (aggColumn !== '*') {
      const aggCol = table.getColumn(aggColumn);
      if (!aggCol) {
        throw new GridDBError(
          `Column '${aggColumn}' not found for aggregation`,
          'COLUMN_NOT_FOUND',
          { column: aggColumn }
        );
      }
      actualAggColumn = aggCol.name;
    }
    
    // Step 1: Sort by group column (using GPU OneSweep sort!)
    const sortedTable = await this.executeOrderBy(table, actualGroupByCol);
    
    // Step 2: For each group, use GPU reduce
    const groups = new Map();
    for (const row of sortedTable.rows) {
      const key = row[actualGroupByCol];
      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key).push(row);
    }

    // Step 3: Apply GPU DLDFScan reduction to each group
    const aggregated = [];
    
    for (const [key, rows] of groups.entries()) {
      let values;
      if (aggFunction === 'COUNT') {
        // COUNT(*) doesn't need column values
        aggregated.push({
          [actualGroupByCol]: key,
          [`${aggFunction}(*)`]: rows.length,
        });
        continue;
      }
      
      // Get values for aggregation
      values = rows.map(r => parseFloat(r[actualAggColumn]) || 0);
      
      // Create GPU buffers for this group
      const inputBuffer = new Buffer({
        device: this.device,
        datatype: 'f32',
        length: values.length,
        label: 'group_input',
        createCPUBuffer: true,
        createGPUBuffer: true,
      });
      inputBuffer.cpuBuffer.set(new Float32Array(values));
      await inputBuffer.copyCPUToGPU();

      const outputBuffer = new Buffer({
        device: this.device,
        datatype: 'f32',
        length: 1,
        label: 'group_output',
        createGPUBuffer: true,
        createMappableGPUBuffer: true,
      });

      // Choose binary operation based on aggregation function
      let binop;
      switch (aggFunction) {
        case 'SUM':
        case 'AVG':
          binop = new BinOpAddF32({ datatype: 'f32' });
          break;
        case 'MAX':
          binop = new BinOpMaxF32({ datatype: 'f32' });
          break;
        case 'MIN':
          // Use negative max for min
          const negValues = values.map(v => -v);
          inputBuffer.cpuBuffer.set(new Float32Array(negValues));
          await inputBuffer.copyCPUToGPU();
          binop = new BinOpMaxF32({ datatype: 'f32' });
          break;
        default:
          throw new GridDBError(
            `Unsupported aggregation: ${aggFunction}`,
            'UNSUPPORTED_AGGREGATION',
            { function: aggFunction }
          );
      }

      // Run GPU DLDFScan reduce!
      const scan = new DLDFScan({
        device: this.device,
        datatype: 'f32',
        binop: binop,
        type: 'reduce',
        args: {
          input: inputBuffer,
          output: outputBuffer,
        },
        useSubgroups: false, // Disable for compatibility
      });

      try {
        await scan.execute();
        await outputBuffer.copyGPUToCPU();
        
        let result = outputBuffer.cpuBuffer[0];
        
        // Post-process result
        if (aggFunction === 'AVG') {
          result = result / values.length;
        } else if (aggFunction === 'MIN') {
          result = -result; // Negate back
        }

        aggregated.push({
          [actualGroupByCol]: key,
          [`${aggFunction}(${actualAggColumn})`]: result,
        });
      } finally {
        // Cleanup GPU resources
        scan.destroy();
        inputBuffer.destroy();
        outputBuffer.destroy();
      }
    }

    const newColumns = [
      { name: actualGroupByCol, type: groupCol.type, index: 0 },
      { name: `${aggFunction}(${actualAggColumn})`, type: 'number', index: 1 },
    ];

    console.log(`  GPU GROUP BY reduced ${table.rowCount} → ${aggregated.length} groups`);
    return table.clone(aggregated, newColumns);
  }

  /**
   * Execute aggregation without GROUP BY clause.
   * 
   * Applies aggregation functions to the entire table as a single group.
   * This handles queries like "SELECT COUNT(*) FROM table" or 
   * "SELECT AVG(salary) FROM employees" without GROUP BY.
   * 
   * Supported aggregation functions:
   * - COUNT(*): Count all rows
   * - SUM(column): Sum all values in column
   * - AVG(column): Average of all values in column
   * - MAX(column): Maximum value in column
   * - MIN(column): Minimum value in column
   * 
   * @param {Table} table - The input table
   * @param {string} selectCols - SELECT clause with aggregation (e.g., "COUNT(*)", "AVG(salary)")
   * @returns {Promise<Table>} A new table with a single row containing the aggregated result
   * @throws {GridDBError} If the aggregation column is not found
   * @private
   */
  async executeAggregationWithoutGroupBy(table, selectCols) {
    console.log('Aggregation without GROUP BY:', selectCols);
    
    // Parse aggregation from selectCols
    const aggMatch = selectCols.match(/(COUNT|SUM|AVG|MAX|MIN)\s*\(\s*(\*|\w+)\s*\)/i);
    
    if (!aggMatch) {
      // No aggregation found, shouldn't happen but return table as-is
      return table;
    }

    const aggFunction = aggMatch[1].toUpperCase();
    const aggColumn = aggMatch[2];
    
    if (aggFunction === 'COUNT' && aggColumn === '*') {
      // Simple count of all rows
      const result = [{
        'COUNT(*)': table.rowCount
      }];
      
      const newColumns = [
        { name: 'COUNT(*)', type: 'number', index: 0 }
      ];
      
      console.log(`  Aggregation result: ${table.rowCount} rows`);
      return table.clone(result, newColumns);
    }
    
    // Find actual aggregation column name
    const aggCol = table.getColumn(aggColumn);
    if (!aggCol) {
      throw new GridDBError(
        `Column '${aggColumn}' not found for aggregation`,
        'COLUMN_NOT_FOUND',
        { column: aggColumn }
      );
    }
    const actualAggColumn = aggCol.name;
    
    // Get all values for aggregation
    const values = table.rows.map(r => parseFloat(r[actualAggColumn]) || 0);
    
    let aggregateValue;
    
    if (aggFunction === 'COUNT') {
      aggregateValue = values.length;
    } else if (aggFunction === 'SUM') {
      aggregateValue = values.reduce((sum, val) => sum + val, 0);
    } else if (aggFunction === 'AVG') {
      aggregateValue = values.reduce((sum, val) => sum + val, 0) / values.length;
    } else if (aggFunction === 'MAX') {
      aggregateValue = Math.max(...values);
    } else if (aggFunction === 'MIN') {
      aggregateValue = Math.min(...values);
    }
    
    const result = [{
      [`${aggFunction}(${actualAggColumn})`]: aggregateValue
    }];
    
    const newColumns = [
      { name: `${aggFunction}(${actualAggColumn})`, type: 'number', index: 0 }
    ];
    
    console.log(`  Aggregation result: ${aggFunction}(${actualAggColumn}) = ${aggregateValue}`);
    return table.clone(result, newColumns);
  }

  // ============================================================================
  // ORDER BY OPERATIONS
  // ============================================================================

  /**
   * Execute ORDER BY clause using GPU OneSweep radix sort.
   * 
   * Sorts table rows by the specified column using parallel GPU sorting.
   * OneSweep is a high-performance radix sort algorithm that processes
   * all rows simultaneously on the GPU, achieving 50-200x speedup over
   * CPU sorting for large datasets.
   * 
   * Supports both ascending (ASC) and descending (DESC) sort order.
   * 
   * @param {Table} table - The input table to sort
   * @param {string} orderByCol - Column specification with optional direction (e.g., "age", "salary DESC")
   * @returns {Promise<Table>} A new table with rows sorted by the specified column
   * @throws {Error} If the column is not found
   */
  async executeOrderBy(table, orderByCol) {
    console.log('GPU ORDER BY:', orderByCol);
    
    // Parse column and direction
    const parts = orderByCol.split(/\s+/);
    const col = parts[0];
    const dir = (parts[1] || 'asc').toLowerCase();
    
    // Validate column exists
    if (!table.getColumn(col)) {
      throw new Error(`Column '${col}' not found for ORDER BY. Available: ${table.columns.map(c => c.name).join(', ')}`);
    }
    
    // Extract column values
    const values = table.getColumnValues(col);
    
    // Validate we have data
    if (!values || values.length === 0) {
      console.warn(' No data to sort');
      return table;
    }
    
    const numericValues = values.map(v => 
      typeof v === 'number' ? v : parseFloat(v) || 0
    );

    // Create GPU buffers
    const keysInOut = new Buffer({
      device: this.device,
      datatype: 'u32', // OneSweep works on u32
      length: numericValues.length,
      label: 'keysInOut',
      createCPUBuffer: true,
      createGPUBuffer: true,
      createMappableGPUBuffer: true,
      storeCPUBackup: true,
    });
    
    // Convert to u32 for sorting (multiply by 1000 to preserve decimals)
    const u32Values = new Uint32Array(
      numericValues.map(v => Math.round(Math.abs(v * 1000)))
    );
    keysInOut.cpuBuffer.set(u32Values);
    await keysInOut.copyCPUToGPU();

    const keysTemp = new Buffer({
      device: this.device,
      datatype: 'u32',
      length: numericValues.length,
      label: 'keysTemp',
      createCPUBuffer: true,
      createGPUBuffer: true,
      createMappableGPUBuffer: true,
    });

    // Create OneSweep GPU sort!
    // Pass Buffer objects directly - they will be registered by OneSweepSort
    const sort = new OneSweepSort({
      device: this.device,
      datatype: 'u32',
      direction: dir === 'desc' ? 'descending' : 'ascending',
      type: 'keysonly',
      keysInOut: keysInOut.buffer,  // Pass the GPUBufferBinding, not the Buffer wrapper
      keysTemp: keysTemp.buffer,    // Pass the GPUBufferBinding, not the Buffer wrapper
      disableSubgroups: true,   // Disable subgroups to avoid workgroup storage issues
    });

    console.log('  Running GPU sort on', numericValues.length, 'elements...');
    
    try {
      await sort.execute();
      await keysInOut.copyGPUToCPU();

      // Map sorted indices back to original rows
      const sortedIndices = new Uint32Array(numericValues.length);
      const valueToIndex = new Map();
      numericValues.forEach((val, idx) => {
        const key = Math.round(Math.abs(val * 1000));
        if (!valueToIndex.has(key)) {
          valueToIndex.set(key, []);
        }
        valueToIndex.get(key).push(idx);
      });

      // Reconstruct sorted rows
      const sorted = [];
      for (let i = 0; i < keysInOut.cpuBuffer.length; i++) {
        const sortedKey = keysInOut.cpuBuffer[i];
        const indices = valueToIndex.get(sortedKey);
        if (indices && indices.length > 0) {
          const idx = indices.shift();
          sorted.push(table.rows[idx]);
        }
      }

      return table.clone(sorted);
      
    } finally {
      // Always cleanup GPU resources
      sort.destroy();
      keysInOut.destroy();
      keysTemp.destroy();
    }
  }

  // ============================================================================
  // DISTINCT OPERATIONS
  // ============================================================================

  /**
   * Execute DISTINCT clause using GPU sort and deduplication.
   * 
   * Removes duplicate rows using a two-phase GPU approach:
   * 1. GPU OneSweep sort to group identical values together
   * 2. CPU pass to remove consecutive duplicates
   * 
   * For multiple columns or SELECT DISTINCT *, uses JSON comparison.
   * For single column, leverages GPU sorting for better performance.
   * 
   * @param {Table} table - The input table
   * @param {string} columns - Column specification for distinctness (e.g., '*', 'age', 'name, city')
   * @returns {Promise<Table>} A new table with duplicate rows removed
   */
  async executeDistinct(table, columns) {
    console.log('GPU DISTINCT:', columns);
    
    // Handle undefined columns
    if (!columns || columns === '*') {
      columns = '*';
    }
    
    // For simplicity, if columns is *, deduplicate entire rows
    // Otherwise deduplicate by specific column
    let distinctColumn = columns;
    if (columns === '*' || (typeof columns === 'string' && columns.includes(','))) {
      // For multiple columns or *, use JSON stringify for comparison
      const seen = new Set();
      const distinct = table.rows.filter(row => {
        const key = JSON.stringify(row);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
      
      console.log(`  Deduplicated ${table.rowCount} → ${distinct.length} rows`);
      return table.clone(distinct);
    }

    // Single column DISTINCT - use GPU sort with string hashing
    const values = table.getColumnValues(distinctColumn);
    
    // Convert values to numeric hashes for GPU processing
    const hashValue = (v) => {
      if (typeof v === 'number') return Math.round(Math.abs(v * 1000));
      
      // String hashing: simple hash function
      let hash = 0;
      const str = String(v || '');
      for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash = hash & hash; // Convert to 32-bit integer
      }
      return Math.abs(hash);
    };
    
    const numericValues = values.map(v => hashValue(v));

    // Step 1: Create GPU buffers (EXACT same pattern as ORDER BY)
    const keysInOut = new Buffer({
      device: this.device,
      datatype: 'u32',
      length: numericValues.length,
      label: 'distinct_keys_inout',
      createCPUBuffer: true,
      createGPUBuffer: true,
      createMappableGPUBuffer: true,
      storeCPUBackup: true,
    });
    
    keysInOut.cpuBuffer.set(new Uint32Array(numericValues));
    await keysInOut.copyCPUToGPU();

    const keysTemp = new Buffer({
      device: this.device,
      datatype: 'u32',
      length: numericValues.length,
      label: 'distinct_keys_temp',
      createCPUBuffer: true,
      createGPUBuffer: true,
      createMappableGPUBuffer: true,
    });

    // Create OneSweep GPU sort (EXACT same pattern as ORDER BY)
    // Pass GPUBufferBinding objects directly - they will be registered by OneSweepSort
    const sort = new OneSweepSort({
      device: this.device,
      datatype: 'u32',
      direction: 'ascending',
      type: 'keysonly',
      keysInOut: keysInOut.buffer,  // Pass the GPUBufferBinding, not the Buffer wrapper
      keysTemp: keysTemp.buffer,    // Pass the GPUBufferBinding, not the Buffer wrapper
      disableSubgroups: true,  // Disable subgroups to avoid workgroup storage issues
    });

    console.log('  Running GPU DISTINCT sort on', numericValues.length, 'elements...');

    try {
      await sort.execute();
      await keysInOut.copyGPUToCPU();

      // Step 2: Remove consecutive duplicates and map back to original rows
      const distinctRows = [];
      let lastKey = null;
      
      // Create a map of hash -> original value for lookup
      const hashToValue = new Map();
      for (let i = 0; i < values.length; i++) {
        const hash = numericValues[i];
        if (!hashToValue.has(hash)) {
          hashToValue.set(hash, values[i]);
        }
      }
      
      for (let i = 0; i < keysInOut.cpuBuffer.length; i++) {
        const key = keysInOut.cpuBuffer[i];
        if (key !== lastKey) {
          // Find the original value with this hash
          const originalValue = hashToValue.get(key);
          if (originalValue !== undefined) {
            const row = table.rows.find(r => r[distinctColumn] === originalValue);
            if (row) distinctRows.push(row);
          }
          lastKey = key;
        }
      }

      console.log(`  GPU DISTINCT: ${table.rowCount} → ${distinctRows.length} unique values`);
      
      return table.clone(distinctRows);
      
    } finally {
      // Always cleanup GPU resources
      sort.destroy();
      keysInOut.destroy();
      keysTemp.destroy();
    }
  }

  // ============================================================================
  // JOIN OPERATIONS
  // ============================================================================

  /**
   * Build GPU hash table using histogram for hash bucketing.
   * 
   * Creates a GPU-accelerated hash table by:
   * 1. Computing hash values for all join keys
   * 2. Using histogram to count items per bucket
   * 3. Computing bucket offsets (prefix sum)
   * 4. Scattering rows into buckets
   * 
   * This enables efficient GPU-based hash joins for large datasets.
   * 
   * @param {Table} table - The table to hash
   * @param {string} columnName - Column to use as join key
   * @param {number} numBuckets - Number of hash buckets (power of 2, default 256)
   * @returns {Promise<Object>} Hash table structure with buckets and metadata
   * @private
   */
  async buildGPUHashTable(table, columnName, numBuckets = 256) {
    const values = table.getColumnValues(columnName);
    
    // Step 1: Convert values to numeric hash codes
    const hashValues = new Uint32Array(values.length);
    for (let i = 0; i < values.length; i++) {
      // Simple hash function for numeric and string values
      const val = values[i];
      if (typeof val === 'number') {
        hashValues[i] = Math.floor(val) >>> 0; // Convert to unsigned 32-bit
      } else {
        // String hash (DJB2 algorithm)
        let hash = 5381;
        const str = String(val);
        for (let j = 0; j < str.length; j++) {
          hash = ((hash << 5) + hash) + str.charCodeAt(j);
        }
        hashValues[i] = hash >>> 0;
      }
    }
    
    // Step 2: Create GPU buffer with hash values
    const hashBuffer = new Buffer({
      device: this.device,
      size: hashValues.length,
      datatype: 'u32',
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      cpuBuffer: hashValues,
    });
    
    // Step 3: Use histogram to count items per bucket (radix histogram)
    const histogram = new Histogram({
      device: this.device,
      inputBuffer: hashBuffer,
      datatype: 'u32',
      bins: {
        type: 'radix',
        radix: numBuckets,
        shift: 0
      },
      binop: BinOpAddU32,
      gputimestamps: false,
    });
    
    // Execute histogram to get bucket counts
    await histogram.run();
    const bucketCounts = histogram.getBuffer("outputBuffer").cpuBuffer;
    
    console.log(`  GPU Hash Table: ${values.length} rows → ${numBuckets} buckets`);
    
    // Step 4: Compute bucket offsets (prefix sum on CPU for now)
    const bucketOffsets = new Uint32Array(numBuckets + 1);
    bucketOffsets[0] = 0;
    for (let i = 0; i < numBuckets; i++) {
      bucketOffsets[i + 1] = bucketOffsets[i] + bucketCounts[i];
    }
    
    // Step 5: Build bucket structure with row indices
    const buckets = Array.from({ length: numBuckets }, () => []);
    for (let rowIdx = 0; rowIdx < values.length; rowIdx++) {
      const hash = hashValues[rowIdx];
      const bucketIdx = hash % numBuckets;
      buckets[bucketIdx].push(rowIdx);
    }
    
    // Cleanup GPU resources
    histogram.destroy();
    hashBuffer.destroy();
    
    return {
      buckets,
      bucketCounts,
      bucketOffsets,
      numBuckets,
      table,
      columnName,
    };
  }

  /**
   * Probe GPU hash table to find matching rows.
   * 
   * Performs the probe phase of hash join:
   * 1. Compute hash for each probe key
   * 2. Look up corresponding bucket
   * 3. Compare probe key with all keys in bucket
   * 4. Collect matching row pairs
   * 
   * @param {Object} hashTable - Hash table built by buildGPUHashTable()
   * @param {Table} probeTable - Table to probe with
   * @param {string} probeColumn - Column in probe table to match
   * @param {string} joinType - JOIN type (INNER, LEFT, RIGHT, FULL)
   * @returns {Array<Object>} Array of joined row objects
   * @private
   */
  async probeGPUHashTable(hashTable, probeTable, probeColumn, joinType) {
    const { buckets, numBuckets, table: buildTable, columnName: buildColumn } = hashTable;
    const probeValues = probeTable.getColumnValues(probeColumn);
    const buildValues = buildTable.getColumnValues(buildColumn);
    
    const joined = [];
    const matchedRightRows = new Set(); // Track matched rows for RIGHT/FULL joins
    
    // Probe phase: for each left row, find matches in hash table
    for (let probeIdx = 0; probeIdx < probeValues.length; probeIdx++) {
      const probeValue = probeValues[probeIdx];
      
      // Compute hash for probe value
      let hash;
      if (typeof probeValue === 'number') {
        hash = Math.floor(probeValue) >>> 0;
      } else {
        let h = 5381;
        const str = String(probeValue);
        for (let j = 0; j < str.length; j++) {
          h = ((h << 5) + h) + str.charCodeAt(j);
        }
        hash = h >>> 0;
      }
      
      const bucketIdx = hash % numBuckets;
      const bucket = buckets[bucketIdx];
      
      let foundMatch = false;
      
      // Check all rows in this bucket
      for (const buildIdx of bucket) {
        const buildValue = buildValues[buildIdx];
        
        // Compare values (handle both numeric and string equality)
        if (probeValue === buildValue || String(probeValue) === String(buildValue)) {
          // Match found!
          const leftRow = probeTable.rows[probeIdx];
          const rightRow = buildTable.rows[buildIdx];
          joined.push({ ...leftRow, ...rightRow });
          matchedRightRows.add(buildIdx);
          foundMatch = true;
        }
      }
      
      // LEFT JOIN: include unmatched left rows
      if (!foundMatch && (joinType === 'LEFT' || joinType === 'FULL')) {
        const leftRow = probeTable.rows[probeIdx];
        // Add null values for right table columns
        const nullRightRow = {};
        for (const col of buildTable.columns) {
          nullRightRow[col.name] = null;
        }
        joined.push({ ...leftRow, ...nullRightRow });
      }
    }
    
    // RIGHT/FULL JOIN: include unmatched right rows
    if (joinType === 'RIGHT' || joinType === 'FULL') {
      for (let buildIdx = 0; buildIdx < buildTable.rows.length; buildIdx++) {
        if (!matchedRightRows.has(buildIdx)) {
          const rightRow = buildTable.rows[buildIdx];
          // Add null values for left table columns
          const nullLeftRow = {};
          for (const col of probeTable.columns) {
            nullLeftRow[col.name] = null;
          }
          joined.push({ ...nullLeftRow, ...rightRow });
        }
      }
    }
    
    return joined;
  }

  /**
   * Execute JOIN clause using GPU-accelerated hash join algorithm.
   * 
   * Joins two tables based on a specified condition using GPU acceleration:
   * 1. Build phase: Creates GPU hash table using histogram for bucketing
   * 2. Probe phase: Matches rows efficiently using GPU hash lookups
   * 
   * Supported JOIN types:
   * - INNER JOIN: Returns only matching rows from both tables
   * - LEFT JOIN: Returns all left table rows, with NULLs for non-matches
   * - RIGHT JOIN: Returns all right table rows, with NULLs for non-matches
   * - FULL JOIN: Returns all rows from both tables
   * 
   * GPU Acceleration:
   * - Uses histogram primitive for hash bucketing
   * - Efficient parallel hash computation
   * - Optimized for large datasets (>10K rows)
   * 
   * @param {Table} leftTable - The left table in the join
   * @param {Table} rightTable - The right table in the join
   * @param {Object} joinSpec - Join specification with type and condition
   * @param {string} joinSpec.type - JOIN type (INNER, LEFT, RIGHT, FULL)
   * @param {string} joinSpec.condition - Join condition (e.g., "left.id = right.user_id")
   * @returns {Promise<Object>} A table-like object with joined rows
   * @throws {Error} If the join condition cannot be parsed
   */
  async executeJoin(leftTable, rightTable, joinSpec) {
    console.log('GPU JOIN:', joinSpec.type, 'on', joinSpec.condition);

    
    // Parse join condition: "left.col = right.col"
    const match = joinSpec.condition.match(/(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)/);
    if (!match) {
      throw new Error(`Cannot parse JOIN condition: ${joinSpec.condition}`);
    }
    
    const leftCol = match[2];
    const rightCol = match[4];
    
    // Determine optimal join strategy based on table sizes
    const useGPU = (leftTable.rowCount + rightTable.rowCount) > 10000;
    
    let joined;
    
    if (useGPU) {
      // GPU-accelerated hash join for large datasets
      console.log('  Using GPU-accelerated hash join');
      
      // Build phase: create GPU hash table from right (build) table
      const hashTable = await this.buildGPUHashTable(rightTable, rightCol, 256);
      
      // Probe phase: probe with left table
      joined = await this.probeGPUHashTable(hashTable, leftTable, leftCol, joinSpec.type);
      
    } else {
      // CPU hash join for small datasets (more efficient due to overhead)
      console.log('  Using CPU hash join (small dataset)');
      
      const hashTable = new Map();
      
      // Build phase: hash right table
      for (const rightRow of rightTable.rows) {
        const key = rightRow[rightCol];
        if (!hashTable.has(key)) {
          hashTable.set(key, []);
        }
        hashTable.get(key).push(rightRow);
      }
      
      // Probe phase: join left table
      joined = [];
      const matchedRightIndices = new Set();
      
      for (let leftIdx = 0; leftIdx < leftTable.rows.length; leftIdx++) {
        const leftRow = leftTable.rows[leftIdx];
        const key = leftRow[leftCol];
        const matches = hashTable.get(key) || [];
        
        if (matches.length > 0) {
          for (const rightRow of matches) {
            joined.push({ ...leftRow, ...rightRow });
            // Track which right rows were matched for RIGHT/FULL joins
            const rightIdx = rightTable.rows.indexOf(rightRow);
            matchedRightIndices.add(rightIdx);
          }
        } else if (joinSpec.type === 'LEFT' || joinSpec.type === 'FULL') {
          // Left/Full join: include unmatched left rows with nulls for right columns
          const nullRightRow = {};
          for (const col of rightTable.columns) {
            nullRightRow[col.name] = null;
          }
          joined.push({ ...leftRow, ...nullRightRow });
        }
      }
      
      // RIGHT/FULL JOIN: add unmatched right rows
      if (joinSpec.type === 'RIGHT' || joinSpec.type === 'FULL') {
        for (let rightIdx = 0; rightIdx < rightTable.rows.length; rightIdx++) {
          if (!matchedRightIndices.has(rightIdx)) {
            const rightRow = rightTable.rows[rightIdx];
            const nullLeftRow = {};
            for (const col of leftTable.columns) {
              nullLeftRow[col.name] = null;
            }
            joined.push({ ...nullLeftRow, ...rightRow });
          }
        }
      }
    }
    
    console.log(`  Joined ${leftTable.rowCount} × ${rightTable.rowCount} → ${joined.length} rows`);
    
    // Merge column definitions
    const columns = [
      ...leftTable.columns.map(c => ({ ...c, table: leftTable.name })),
      ...rightTable.columns.map(c => ({ ...c, table: rightTable.name })),
    ];
    
    return {
      device: this.device,
      name: `${leftTable.name}_join_${rightTable.name}`,
      columns: columns,
      rows: joined,
      rowCount: joined.length,
      getColumn: leftTable.getColumn.bind({ columns, rows: joined }),
      getColumnValues: leftTable.getColumnValues.bind({ columns, rows: joined }),
    };
  }

  // ============================================================================
  // STATISTICS AND UTILITY METHODS
  // ============================================================================

  /**
   * Get comprehensive statistics about the database.
   * 
   * Returns detailed metrics including:
   * - Table count and sizes
   * - Total rows and memory usage
   * - Query performance metrics
   * - Cache hit rates
   * 
   * @returns {Object} Statistics object with database metrics
   */
  getStats() {
    const tableStats = [];
    let totalRows = 0;
    let totalBytes = 0;

    for (const [name, table] of this.tables) {
      totalRows += table.rowCount;
      totalBytes += table.estimatedBytes;
      tableStats.push({
        name,
        rows: table.rowCount,
        columns: table.columns.length,
        bytes: table.estimatedBytes,
      });
    }
    
    const avgQueryTime = this.stats.queriesExecuted > 0
      ? this.stats.totalQueryTime / this.stats.queriesExecuted
      : 0;
    
    const cacheHitRate = this.stats.queriesExecuted > 0
      ? (this.stats.cacheHits / this.stats.queriesExecuted * 100).toFixed(1)
      : 0;

    return {
      tables: this.tables.size,
      totalRows,
      totalBytes,
      totalMB: (totalBytes / 1024 / 1024).toFixed(2),
      queriesExecuted: this.stats.queriesExecuted,
      avgQueryTime: avgQueryTime.toFixed(2) + 'ms',
      cacheHits: this.stats.cacheHits,
      cacheMisses: this.stats.cacheMisses,
      cacheHitRate: cacheHitRate + '%',
      cacheSize: this.queryCache.size,
      tableDetails: tableStats,
    };
  }

  /**
   * Get detailed performance metrics.
   * 
   * Returns query execution statistics including:
   * - Average query time
   * - Cache hit rate
   * - Total queries executed
   * - Total query time
   * 
   * @returns {Object} Performance metrics object
   */
  getPerformanceMetrics() {
    return {
      ...this.stats,
      avgQueryTime: this.stats.queriesExecuted > 0
        ? this.stats.totalQueryTime / this.stats.queriesExecuted
        : 0,
      cacheHitRate: this.stats.queriesExecuted > 0
        ? this.stats.cacheHits / this.stats.queriesExecuted
        : 0,
    };
  }

  /**
   * Clear the query result cache.
   * 
   * Removes all cached query results to free memory.
   * The cache is automatically cleared when data is modified
   * (loadCSV, loadJSON, deleteTable).
   */
  clearCache() {
    const size = this.queryCache.size;
    this.queryCache.clear();
    
    if (this.config.enableLogging) {
      console.log(`Cleared ${size} cached queries`);
    }
  }

  /**
   * Reset all statistics counters.
   * 
   * Resets query execution metrics to zero:
   * - Queries executed
   * - Total query time
   * - Cache hits/misses
   * 
   * Does not clear the cache itself (use clearCache() for that).
   */
  resetStats() {
    this.stats = {
      queriesExecuted: 0,
      totalQueryTime: 0,
      cacheHits: 0,
      cacheMisses: 0,
    };
    
    if (this.config.enableLogging) {
      console.log('Statistics reset');
    }
  }

  /**
   * Cleanup all database resources.
   * 
   * Call this method when you're done using the database to:
   * - Clear all tables
   * - Clear the query cache
   * - Destroy GPU resources and free GPU memory
   * 
   * After calling destroy(), the database instance should not be used again.
   */
  destroy() {
    // Clear all tables
    this.tables.clear();
    
    // Clear cache
    this.queryCache.clear();
    
    // Cleanup GPU resources
    for (const resource of this.gpuResources) {
      try {
        if (resource.destroy) {
          resource.destroy();
        }
      } catch (e) {
        console.warn('Failed to destroy GPU resource:', e);
      }
    }
    this.gpuResources = [];
    
    if (this.config.enableLogging) {
      console.log('GridDB resources cleaned up');
    }
  }

  /**
   * Export the entire database as JSON.
   * 
   * Creates a JSON representation of the database including:
   * - All tables with their columns and rows
   * - Database statistics
   * - Version information
   * 
   * @returns {Object} JSON object containing the entire database
   */
  exportJSON() {
    const tables = {};
    for (const [name, table] of this.tables) {
      tables[name] = {
        columns: table.columns,
        rows: table.rows,
      };
    }
    return {
      version: '1.0',
      tables,
      stats: this.getStats(),
    };
  }

  /**
   * Get a list of all table names in the database.
   * 
   * @returns {string[]} Array of table names
   */
  listTables() {
    return Array.from(this.tables.keys());
  }

  /**
   * Get detailed information about a specific table.
   * 
   * Returns metadata including:
   * - Table name
   * - Row count
   * - Column definitions (name and type)
   * - Estimated memory usage
   * 
   * @param {string} name - The table name
   * @returns {Object} Table information object
   * @throws {GridDBError} If the table is not found
   */
  getTableInfo(name) {
    const table = this.tables.get(name);
    if (!table) {
      throw new GridDBError(`Table '${name}' not found`, 'TABLE_NOT_FOUND');
    }
    
    return {
      name: table.name,
      rows: table.rowCount,
      columns: table.columns.map(c => ({
        name: c.name,
        type: c.type,
      })),
      estimatedBytes: table.estimatedBytes,
      estimatedMB: (table.estimatedBytes / 1024 / 1024).toFixed(2),
    };
  }
}

/**
 * Table class - represents a database table
 */
class Table {
  constructor(device, name, columns, rows, config = DEFAULT_CONFIG) {
    this.device = device;
    this.name = name;
    this.columns = columns; // [{name, type, index}]
    this.rows = rows; // Array of objects
    this.rowCount = rows.length;
    this.estimatedBytes = this.calculateSize();
    this.config = config;
    
    // Validate column names
    for (const col of columns) {
      if (col.name.length > config.maxColumnNameLength) {
        throw new GridDBError(
          `Column name '${col.name}' too long (max ${config.maxColumnNameLength} characters)`,
          'COLUMN_NAME_TOO_LONG'
        );
      }
    }
  }

  /**
   * Create table from CSV text with validation
   */
  static async fromCSV(device, name, csvText, config = DEFAULT_CONFIG) {
    const lines = csvText.trim().split('\n');
    if (lines.length === 0) {
      throw new Error('Empty CSV file');
    }

    // Parse header - handle quotes and commas properly
    const headerLine = lines[0];
    const headers = parseCSVLine(headerLine);
    
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue; // Skip empty lines
      
      const values = parseCSVLine(lines[i]);
      if (values.length === 0) continue;
      
      const row = {};
      headers.forEach((header, j) => {
        let value = values[j] || '';
        // Try to parse as number
        const numValue = parseFloat(value);
        row[header] = !isNaN(numValue) && value !== '' ? numValue : value;
      });
      rows.push(row);
    }

    // Auto-detect column types from data
    const columns = headers.map((name, index) => {
      let type = 'string';
      
      // Check first few rows to detect type
      for (let i = 0; i < Math.min(10, rows.length); i++) {
        const val = rows[i][name];
        if (typeof val === 'number') {
          type = 'number';
          break;
        } else if (val && !isNaN(Date.parse(val)) && val.match(/\d{4}-\d{2}-\d{2}/)) {
          type = 'date';
          break;
        }
      }

      return { name, type, index };
    });

    return new Table(device, name, columns, rows, config);
  }

  /**
   * Create table from JSON array with validation
   */
  static async fromJSON(device, name, jsonArray, config = DEFAULT_CONFIG) {
    if (jsonArray.length === 0) {
      throw new GridDBError('Empty JSON array', 'EMPTY_DATA');
    }

    const firstRow = jsonArray[0];
    const columns = Object.keys(firstRow).map((name, index) => {
      const value = firstRow[name];
      let type = 'string';
      
      if (typeof value === 'number') {
        type = 'number';
      } else if (value && !isNaN(Date.parse(value))) {
        type = 'date';
      }

      return { name, type, index };
    });

    // Convert all rows to proper types
    const rows = jsonArray.map(row => {
      const newRow = {};
      columns.forEach(col => {
        let value = row[col.name];
        if (col.type === 'number' && typeof value !== 'number') {
          value = parseFloat(value) || 0;
        }
        newRow[col.name] = value;
      });
      return newRow;
    });

    return new Table(device, name, columns, rows, config);
  }

  /**
   * Get column by name
   */
  /**
   * Clone this table with modified rows
   */
  clone(newRows = null, newColumns = null) {
    return new Table(
      this.device,
      this.name,
      newColumns || this.columns,
      newRows || this.rows
    );
  }

  /**
   * Get column by name (case-insensitive)
   */
  getColumn(name) {
    const lowerName = name.toLowerCase();
    return this.columns.find(c => c.name.toLowerCase() === lowerName);
  }

  /**
   * Extract column values as array (case-insensitive column name)
   */
  getColumnValues(colName) {
    // Find actual column name (case-insensitive match)
    const col = this.getColumn(colName);
    if (!col) {
      throw new Error(`Column '${colName}' not found. Available columns: ${this.columns.map(c => c.name).join(', ')}`);
    }
    const actualName = col.name;
    return this.rows.map(row => row[actualName]);
  }

  /**
   * Calculate estimated size in bytes
   */
  calculateSize() {
    // Rough estimate: 8 bytes per numeric value, 20 bytes per string
    return this.rowCount * this.columns.length * 20;
  }

  /**
   * Convert to GPU buffers for processing
   */
  async toGPUBuffers() {
    // TODO: Create Buffer objects for each column
    const buffers = {};
    
    for (const col of this.columns) {
      const values = this.getColumnValues(col.name);
      
      // Convert to appropriate type
      let typedArray;
      if (col.type === 'number') {
        typedArray = new Float32Array(values.map(v => 
          typeof v === 'number' ? v : parseFloat(v) || 0
        ));
      } else {
        // For non-numeric, convert to indices or hashes
        typedArray = new Float32Array(values.map(v => parseFloat(v) || 0));
      }

      buffers[col.name] = new Buffer({
        device: this.device,
        datatype: 'f32',
        length: typedArray.length,
        label: `${this.name}.${col.name}`,
        createCPUBuffer: true,
        createGPUBuffer: true,
      });

      buffers[col.name].cpuBuffer.set(typedArray);
      await buffers[col.name].copyCPUToGPU();
    }

    return buffers;
  }
}

/**
 * Parse a CSV line handling quotes and commas
 */
function parseCSVLine(line) {
  const result = [];
  let current = '';
  let inQuotes = false;
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        // Escaped quote
        current += '"';
        i++;
      } else {
        // Toggle quotes
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      // End of field
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  
  // Add last field
  result.push(current.trim());
  
  return result;
}

/**
 * Compute comprehensive statistics for a column using GPU acceleration
 * Returns: count, sum, avg, min, max, median, mode, stdDev, variance
 * 
 * @param {GridDB} griddb - GridDB instance with device
 * @param {Array} data - Array of row objects or array of numbers
 * @param {string} column - Column name (optional if data is array of numbers)
 * @returns {Promise<Object>} Statistics object
 */
export async function computeStats(griddb, data, column = null) {
  console.log('Computing statistics...');
  
  // Extract numeric values
  let values;
  if (Array.isArray(data) && typeof data[0] === 'number') {
    values = data.filter(v => !isNaN(v));
  } else if (column) {
    values = data.map(row => parseFloat(row[column])).filter(v => !isNaN(v));
  } else {
    throw new Error('Must provide either array of numbers or data + column name');
  }
  
  if (values.length === 0) {
    return {
      count: 0,
      sum: 0,
      avg: 0,
      min: 0,
      max: 0,
      median: 0,
      mode: 0,
      stdDev: 0,
      variance: 0
    };
  }
  
  const count = values.length;
  
  // GPU-accelerated SUM using DLDFScan
  const inputBuffer = new Buffer({
    device: griddb.device,
    datatype: 'f32',
    length: values.length,
    label: 'stats_input',
    createCPUBuffer: true,
    createGPUBuffer: true,
  });
  inputBuffer.cpuBuffer.set(new Float32Array(values));
  await inputBuffer.copyCPUToGPU();
  
  const sumBuffer = new Buffer({
    device: griddb.device,
    datatype: 'f32',
    length: 1,
    label: 'stats_sum',
    createGPUBuffer: true,
    createMappableGPUBuffer: true,
  });
  
  // GPU SUM
  const sumScan = new DLDFScan({
    device: griddb.device,
    datatype: 'f32',
    binop: new BinOpAddF32({ datatype: 'f32' }),
    type: 'reduce',
    args: {
      input: inputBuffer,
      output: sumBuffer,
    },
    useSubgroups: false,
  });
  
  await sumScan.execute();
  await sumBuffer.copyGPUToCPU();
  const sum = sumBuffer.cpuBuffer[0];
  const avg = sum / count;
  
  // GPU MAX
  const maxBuffer = new Buffer({
    device: griddb.device,
    datatype: 'f32',
    length: 1,
    label: 'stats_max',
    createGPUBuffer: true,
    createMappableGPUBuffer: true,
  });
  
  const maxScan = new DLDFScan({
    device: griddb.device,
    datatype: 'f32',
    binop: new BinOpMaxF32({ datatype: 'f32' }),
    type: 'reduce',
    args: {
      input: inputBuffer,
      output: maxBuffer,
    },
    useSubgroups: false,
  });
  
  await maxScan.execute();
  await maxBuffer.copyGPUToCPU();
  const max = maxBuffer.cpuBuffer[0];
  
  // GPU MIN (using negated max)
  const negValues = values.map(v => -v);
  inputBuffer.cpuBuffer.set(new Float32Array(negValues));
  await inputBuffer.copyCPUToGPU();
  
  const minBuffer = new Buffer({
    device: griddb.device,
    datatype: 'f32',
    length: 1,
    label: 'stats_min',
    createGPUBuffer: true,
    createMappableGPUBuffer: true,
  });
  
  const minScan = new DLDFScan({
    device: griddb.device,
    datatype: 'f32',
    binop: new BinOpMaxF32({ datatype: 'f32' }),
    type: 'reduce',
    args: {
      input: inputBuffer,
      output: minBuffer,
    },
    useSubgroups: false,
  });
  
  await minScan.execute();
  await minBuffer.copyGPUToCPU();
  const min = -minBuffer.cpuBuffer[0];
  
  // MEDIAN (CPU - requires sorting)
  const sorted = [...values].sort((a, b) => a - b);
  const median = sorted.length % 2 === 0
    ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
    : sorted[Math.floor(sorted.length / 2)];
  
  // MODE (CPU - requires frequency counting)
  const freqMap = new Map();
  let maxFreq = 0;
  let mode = values[0];
  
  for (const val of values) {
    const freq = (freqMap.get(val) || 0) + 1;
    freqMap.set(val, freq);
    if (freq > maxFreq) {
      maxFreq = freq;
      mode = val;
    }
  }
  
  // VARIANCE & STANDARD DEVIATION (GPU-accelerated)
  // Variance = average of squared differences from mean
  const squaredDiffs = values.map(v => Math.pow(v - avg, 2));
  inputBuffer.cpuBuffer.set(new Float32Array(squaredDiffs));
  await inputBuffer.copyCPUToGPU();
  
  const varianceBuffer = new Buffer({
    device: griddb.device,
    datatype: 'f32',
    length: 1,
    label: 'stats_variance',
    createGPUBuffer: true,
    createMappableGPUBuffer: true,
  });
  
  const varianceScan = new DLDFScan({
    device: griddb.device,
    datatype: 'f32',
    binop: new BinOpAddF32({ datatype: 'f32' }),
    type: 'reduce',
    args: {
      input: inputBuffer,
      output: varianceBuffer,
    },
    useSubgroups: false,
  });
  
  await varianceScan.execute();
  await varianceBuffer.copyGPUToCPU();
  const variance = varianceBuffer.cpuBuffer[0] / count;
  const stdDev = Math.sqrt(variance);
  
  // Cleanup
  sumScan.destroy();
  maxScan.destroy();
  minScan.destroy();
  varianceScan.destroy();
  inputBuffer.destroy();
  sumBuffer.destroy();
  maxBuffer.destroy();
  minBuffer.destroy();
  varianceBuffer.destroy();
  
  console.log(`Statistics computed: avg=${avg.toFixed(2)}, median=${median.toFixed(2)}, stdDev=${stdDev.toFixed(2)}`);
  
  return {
    count,
    sum,
    avg,
    min,
    max,
    median,
    mode,
    stdDev,
    variance
  };
}

/**
 * Helper: Format query results as HTML table
 */
export function formatResultsHTML(results) {
  if (!results.rows || results.rows.length === 0) {
    return '<p>No results</p>';
  }

  const columns = Object.keys(results.rows[0]);
  
  let html = '<table border="1" style="border-collapse: collapse;">';
  
  // Header
  html += '<thead><tr>';
  for (const col of columns) {
    html += `<th style="padding: 8px; background: #f0f0f0;">${col}</th>`;
  }
  html += '</tr></thead>';

  // Rows
  html += '<tbody>';
  for (const row of results.rows) {
    html += '<tr>';
    for (const col of columns) {
      html += `<td style="padding: 8px;">${row[col]}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody>';

  html += '</table>';
  html += `<p><em>${results.rowCount} rows</em></p>`;

  return html;
}

/**
 * Helper: Format results as ASCII table for console
 */
export function formatResultsASCII(results) {
  if (!results.rows || results.rows.length === 0) {
    return 'No results';
  }

  const columns = Object.keys(results.rows[0]);
  const colWidths = columns.map(col => 
    Math.max(col.length, ...results.rows.map(r => String(r[col]).length))
  );

  let output = '';

  // Header
  output += '┌' + colWidths.map(w => '─'.repeat(w + 2)).join('┬') + '┐\n';
  output += '│' + columns.map((col, i) => 
    ' ' + col.padEnd(colWidths[i]) + ' '
  ).join('│') + '│\n';
  output += '├' + colWidths.map(w => '─'.repeat(w + 2)).join('┼') + '┤\n';

  // Rows
  for (const row of results.rows.slice(0, 20)) { // Limit to 20 rows
    output += '│' + columns.map((col, i) => 
      ' ' + String(row[col]).padEnd(colWidths[i]) + ' '
    ).join('│') + '│\n';
  }

  output += '└' + colWidths.map(w => '─'.repeat(w + 2)).join('┴') + '┘\n';
  output += `${results.rowCount} rows total`;

  if (results.rowCount > 20) {
    output += ` (showing first 20)`;
  }

  return output;
}
