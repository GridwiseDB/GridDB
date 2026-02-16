/**
 * Visual Table Relationship Designer - Like draw.io
 * 
 * Features:
 * - Drag and drop tables on canvas
 * - Click columns to connect them
 * - Visual connection lines
 * - Auto-execute operations on file upload
 * - Graph visualization from connected data
 */

export class TableRelationshipDesignerV2 {
    constructor() {
        this.tables = new Map();
        this.connections = [];
        this.db = null;
        this.modal = null;
        this.canvas = null;
        this.ctx = null;
        
        // Input A and B for structured operations
        this.inputA = { table: null, columns: [] };
        this.inputB = { table: null, columns: [] };
        this.selectedColumns = []; // Array of {table, column}
        this.selectedTables = new Set(); // Tables selected on canvas
        this.selectedCanvasColumns = new Set(); // Columns selected by clicking in canvas (stored as "table.column")
        this.activeColumnButtons = []; // Track button positions for click detection
        this.activeConnectionButtons = []; // Track connection execute button positions
        this.expandedColumns = new Map(); // Track expanded columns showing data rows: "table.column" -> true
        this.lastClickTime = 0; // For detecting double-click
        this.lastClickColumn = null; // Track last clicked column for double-click detection
        this.currentConnection = null; // Current connection for operations
        this.lastResultTable = null; // Track last created result table for saving
        
        // Table rename state
        this.renamingTable = null; // Currently renaming table
        this.originalTableName = null; // Original table name before rename
        this.renamingTableFirstKey = true; // Flag to clear name on first key
        this.lastHeaderClickTime = 0; // For double-click detection on header
        this.lastHeaderClickTable = null; // Track last clicked header
        
        // Column rename state
        this.renamingColumn = null; // { table, column, index }
        this.originalColumnName = null; // Original column name before rename
        this.renamingColumnFirstKey = true; // Flag to clear name on first key
        this.lastColumnClickTime = 0; // For double-click detection on column
        this.lastColumnClickKey = null; // Track last clicked column
        
        // Interaction state
        this.isDragging = false;
        this.draggedTable = null;
        this.dragOffset = { x: 0, y: 0 };
        this.isDraggingConnection = false; // Track if dragging a connection line
        
        // Connection state
        this.isConnecting = false;
        this.connectionStart = null; // { table, column }
        this.mousePos = { x: 0, y: 0 };
        
        // Drag operation state
        this.isDraggingOperation = false;
        this.draggedOperation = null;
        this.hoveredConnection = null;
        
        // View state
        this.zoom = 1;
        this.panOffset = { x: 0, y: 0 };
        
        // Constants
        this.TABLE_WIDTH = 250;
        this.TABLE_HEADER_HEIGHT = 50;
        this.COLUMN_HEIGHT = 35;
        this.COLUMN_PADDING = 10;
    }

    /**
     * Open the designer with loaded tables
     */
    open(dbTables, dbInstance) {
        this.db = dbInstance;
        const tableList = Array.from(dbTables.entries());
        
        if (tableList.length === 0) {
            console.error('No tables loaded. Please load at least one table first.');
            return;
        }

        console.log(`Opening designer with ${tableList.length} table(s)`);
        this.loadTables(tableList);
        this.showDesignerModal();
        this.startRendering();
    }

    /**
     * Load tables into the designer
     */
    loadTables(tableList) {
        this.tables.clear();
        
        // Position tables in a grid
        const spacing = 350;
        const startX = 100;
        const startY = 100;
        const cols = Math.ceil(Math.sqrt(tableList.length));
        
        tableList.forEach(([tableName, table], index) => {
            // Handle different table structures
            let columns = [];
            let rowCount = 0;
            let data = [];
            
            if (table && typeof table === 'object') {
                if (table.columns && Array.isArray(table.columns)) {
                    columns = table.columns.map(col => col.name || col);
                    rowCount = table.rowCount || 0;
                    data = table.rows || [];
                } else if (Array.isArray(table)) {
                    data = table;
                    rowCount = table.length;
                    columns = table.length > 0 ? Object.keys(table[0]) : [];
                } else if (table.rows && Array.isArray(table.rows)) {
                    data = table.rows;
                    rowCount = table.rows.length;
                    columns = table.rows.length > 0 ? Object.keys(table.rows[0]) : [];
                }
            }
            
            const row = Math.floor(index / cols);
            const col = index % cols;
            
            this.tables.set(tableName, {
                name: tableName,
                columns: columns,
                rowCount: rowCount,
                data: data,
                position: {
                    x: startX + col * spacing,
                    y: startY + row * spacing
                }
            });
            
            console.log(`📋 Loaded table "${tableName}": ${rowCount} rows, ${columns.length} columns`);
        });
        
        // Populate dropdowns after loading tables
        this.updateColumnSelectors();
    }

    /**
     * Show the designer modal with canvas
     */
    showDesignerModal() {
        // Remove existing modal
        const existing = document.getElementById('relationship-designer-v2');
        if (existing) existing.remove();
        
        // Hide the center panel (SQL queries/results) but keep left sidebar visible
        const centerPanel = document.getElementById('centerPanel');
        const leftResizer = document.getElementById('leftResizer');
        if (centerPanel) {
            centerPanel.style.display = 'none';
        }
        if (leftResizer) {
            leftResizer.style.display = 'none';
        }

        this.modal = document.createElement('div');
        this.modal.id = 'relationship-designer-v2';
        this.modal.className = 'fixed right-0 bottom-0 bg-[#050505] z-[9999] flex flex-col';
        // Position below the main header (h-14 = 56px)
        this.modal.style.top = '56px';
        // Calculate width: leave space for the left sidebar (approximately 320px + padding)
        this.modal.style.left = '360px';
        this.modal.style.width = 'calc(100vw - 360px)';
        
        this.modal.innerHTML = `
            <!-- Designer Tools Panel (matches left sidebar style) -->
            <section class="mx-4 mt-4 bg-[#11181C] rounded-xl border border-white/5 glow-border overflow-visible">
                <div class="panel-header p-4 rounded-t-xl flex items-center justify-between cursor-pointer" onclick="window.tableDesignerV2.toggleToolbarPanel()">
                    <h2 class="text-sm font-bold uppercase text-white/40 tracking-widest">Designer Tools</h2>
                    <div class="flex items-center gap-4">
                        <div class="px-3 py-1.5 bg-white/5 rounded-lg text-xs text-white/50">
                            <span class="text-[#34B27B] font-bold">${this.tables.size}</span> tables • 
                            <span class="text-[#4ECDC4] font-bold">${this.connections.length}</span> connections
                        </div>
                        <span class="collapse-btn text-white/40 text-sm transition-all" id="toolbarCollapseBtn">▼</span>
                        <button onclick="event.stopPropagation(); window.tableDesignerV2.close()" 
                                class="text-white/50 hover:text-white text-2xl leading-none transition-colors px-2">×</button>
                    </div>
                </div>
                <div id="designer-toolbar-panel" class="panel-content p-4 pt-0 space-y-3 overflow-visible">
                    <div class="flex flex-wrap items-center gap-3 overflow-visible">
                        <!-- New Table Button -->
                        <button onclick="window.tableDesignerV2.createQuickTable()" 
                                class="px-4 py-2 bg-[#34B27B] hover:bg-[#2d9a68] text-black text-sm font-bold uppercase rounded-lg transition-all">
                            New Table
                        </button>
                        
                        <!-- JOIN Dropdown Button -->
                        <div class="relative">
                            <button onclick="window.tableDesignerV2.toggleJoinMenu()" id="designerJoinBtn" 
                                    class="bg-white/5 hover:bg-[#34B27B]/20 text-[#34B27B] text-sm font-bold uppercase px-4 py-2 rounded-lg transition-all border border-[#34B27B]/30 flex items-center gap-2">
                                <span>JOIN</span>
                                <span>▼</span>
                            </button>
                            
                            <!-- Dropdown Menu -->
                            <div id="designerJoinMenu" class="hidden absolute top-full left-0 mt-1 bg-[#1a1a1a] border border-[#34B27B]/30 rounded-lg shadow-lg z-[9999] min-w-[280px]">
                                <div class="p-3 border-b border-white/10">
                                    <div class="text-[10px] text-white/40 uppercase mb-2">Select Tables</div>
                                    <div class="space-y-2">
                                        <div class="flex items-center gap-2">
                                            <span class="text-xs text-white/50 w-16">Input A:</span>
                                            <select id="inputTableA" onchange="window.tableDesignerV2.setInputTable('A', this.value)"
                                                    class="flex-1 px-2 py-1 bg-white/5 text-white text-xs rounded border border-white/10">
                                                <option value="">Select...</option>
                                            </select>
                                        </div>
                                        <div class="flex items-center gap-2">
                                            <span class="text-xs text-white/50 w-16">Input B:</span>
                                            <select id="inputTableB" onchange="window.tableDesignerV2.setInputTable('B', this.value)"
                                                    class="flex-1 px-2 py-1 bg-white/5 text-white text-xs rounded border border-white/10">
                                                <option value="">Select...</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                                <div class="p-1">
                                    <button onclick="window.tableDesignerV2.executeInputJoin('inner_join')" 
                                            class="w-full text-left px-3 py-2 text-sm text-white hover:bg-[#34B27B]/20 rounded transition-all flex items-center gap-2">
                                        <span class="text-[#34B27B]">⋈</span>
                                        <span>INNER JOIN</span>
                                    </button>
                                    <button onclick="window.tableDesignerV2.executeInputJoin('left_join')" 
                                            class="w-full text-left px-3 py-2 text-sm text-white hover:bg-[#34B27B]/20 rounded transition-all flex items-center gap-2">
                                        <span class="text-[#34B27B]">⟕</span>
                                        <span>LEFT JOIN</span>
                                    </button>
                                    <button onclick="window.tableDesignerV2.executeInputJoin('right_join')" 
                                            class="w-full text-left px-3 py-2 text-sm text-white hover:bg-[#34B27B]/20 rounded transition-all flex items-center gap-2">
                                        <span class="text-[#34B27B]">⟖</span>
                                        <span>RIGHT JOIN</span>
                                    </button>
                                    <button onclick="window.tableDesignerV2.executeInputJoin('full_join')" 
                                            class="w-full text-left px-3 py-2 text-sm text-white hover:bg-[#34B27B]/20 rounded transition-all flex items-center gap-2">
                                        <span class="text-[#34B27B]">⟗</span>
                                        <span>FULL JOIN</span>
                                    </button>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Execute Button -->
                        <button onclick="window.tableDesignerV2.executeAllConnections()" 
                                class="px-4 py-2 bg-[#34B27B] hover:bg-[#2d9a68] text-black text-sm font-bold uppercase rounded-lg transition-all">
                            Execute
                        </button>
                        
                        <!-- Download Dropdown Button -->
                        <div class="relative">
                            <button onclick="window.tableDesignerV2.toggleDownloadMenu()" id="designerDownloadBtn" 
                                    class="bg-white/5 hover:bg-white/10 text-white/70 text-sm font-bold uppercase px-4 py-2 rounded-lg transition-all flex items-center gap-2">
                                <span>Download</span>
                                <span>▼</span>
                            </button>
                            
                            <!-- Dropdown Menu -->
                            <div id="designerDownloadMenu" class="hidden absolute top-full left-0 mt-1 bg-[#1a1a1a] border border-white/10 rounded-lg shadow-lg z-[9999] min-w-[200px]">
                                <div class="p-1">
                                    <button onclick="window.tableDesignerV2.exportAllTablesToCSV()" 
                                            class="w-full text-left px-3 py-2 text-sm text-white hover:bg-white/10 rounded transition-all">
                                        All Tables as CSV
                                    </button>
                                    <button onclick="window.tableDesignerV2.exportSelectedTableToCSV()" 
                                            class="w-full text-left px-3 py-2 text-sm text-white hover:bg-white/10 rounded transition-all">
                                        Selected Table
                                    </button>
                                    <button onclick="window.tableDesignerV2.exportConnectionsAsJSON()" 
                                            class="w-full text-left px-3 py-2 text-sm text-white hover:bg-white/10 rounded transition-all">
                                        Connections (JSON)
                                    </button>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Connection Operations Dropdown Button -->
                        <div class="relative">
                            <button onclick="window.tableDesignerV2.toggleOperationsMenu()" id="designerOperationsBtn"
                                    class="bg-white/5 hover:bg-[#34B27B]/20 text-[#34B27B] text-sm font-bold uppercase px-4 py-2 rounded-lg transition-all border border-[#34B27B]/30 flex items-center gap-2">
                                <span>Operations</span>
                                <span>▼</span>
                            </button>
                            
                            <!-- Dropdown Menu -->
                            <div id="designerOperationsMenu" class="hidden absolute top-full left-0 mt-1 bg-[#1a1a1a] border border-[#34B27B]/30 rounded-lg shadow-lg z-[9999] min-w-[280px]">
                                <div class="p-3 border-b border-white/10">
                                    <div class="text-[10px] text-white/40 uppercase tracking-wide">Row-by-Row Operations</div>
                                    <div class="text-[9px] text-white/30 mt-1">WebGPU compute shaders • Drag onto connections</div>
                                </div>
                                <div class="p-1">
                                    <button draggable="true" data-operation="sum"
                                            ondragstart="window.tableDesignerV2.onOperationDragStart(event, 'sum')" 
                                            onclick="window.tableDesignerV2.quickCalc('sum')" 
                                            class="w-full text-left px-3 py-2 text-sm text-white hover:bg-[#34B27B]/20 rounded transition-all flex items-center justify-between cursor-move">
                                        <span>Sum</span>
                                        <span class="text-white/30 text-xs">+</span>
                                    </button>
                                    <button draggable="true" data-operation="subtract"
                                            ondragstart="window.tableDesignerV2.onOperationDragStart(event, 'subtract')" 
                                            onclick="window.tableDesignerV2.quickCalc('subtract')" 
                                            class="w-full text-left px-3 py-2 text-sm text-white hover:bg-[#34B27B]/20 rounded transition-all flex items-center justify-between cursor-move">
                                        <span>Subtract</span>
                                        <span class="text-white/30 text-xs">−</span>
                                    </button>
                                    <button draggable="true" data-operation="multiply"
                                            ondragstart="window.tableDesignerV2.onOperationDragStart(event, 'multiply')" 
                                            onclick="window.tableDesignerV2.quickCalc('multiply')" 
                                            class="w-full text-left px-3 py-2 text-sm text-white hover:bg-[#34B27B]/20 rounded transition-all flex items-center justify-between cursor-move">
                                        <span>Multiply</span>
                                        <span class="text-white/30 text-xs">×</span>
                                    </button>
                                    <button draggable="true" data-operation="divide"
                                            ondragstart="window.tableDesignerV2.onOperationDragStart(event, 'divide')" 
                                            onclick="window.tableDesignerV2.quickCalc('divide')" 
                                            class="w-full text-left px-3 py-2 text-sm text-white hover:bg-[#34B27B]/20 rounded transition-all flex items-center justify-between cursor-move">
                                        <span>Divide</span>
                                        <span class="text-white/30 text-xs">÷</span>
                                    </button>
                                    <button draggable="true" data-operation="average"
                                            ondragstart="window.tableDesignerV2.onOperationDragStart(event, 'average')" 
                                            onclick="window.tableDesignerV2.quickCalc('average')" 
                                            class="w-full text-left px-3 py-2 text-sm text-white hover:bg-[#34B27B]/20 rounded transition-all flex items-center justify-between cursor-move">
                                        <span>Average</span>
                                        <span class="text-white/30 text-xs">AVG</span>
                                    </button>
                                </div>
                                <div class="p-3 border-t border-white/10 bg-white/5">
                                    <div class="text-[9px] text-white/50 space-y-1">
                                        <div class="flex items-start gap-1">
                                            <span class="text-[#34B27B]">•</span>
                                            <span>Single: Table1.X → Table2.Y + Op</span>
                                        </div>
                                        <div class="flex items-start gap-1">
                                            <span class="text-[#4ECDC4]">•</span>
                                            <span>Multi: Table1.X + Table2.Y → Table3</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Remove Table Button -->
                        <button onclick="window.tableDesignerV2.removeTable()" 
                                class="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 text-sm font-bold uppercase rounded-lg transition-all">
                            Remove
                        </button>
                    </div>
                </div>
            </section>

            <!-- Connection Operation Toolbar -->
            <div id="connectionOperationBar" class="hidden border-b border-white/10 bg-gradient-to-r from-[#0d1117] to-[#161b22] p-4">
                <div class="flex items-center gap-3">
                    <div class="text-sm text-white/70 font-bold">
                        <span id="connectionInfo">Select operation for connected columns:</span>
                    </div>
                    
                    <div class="flex-1"></div>
                    
                    <!-- Row-by-row operations -->
                    <div class="flex gap-2">
                        <button onclick="window.tableDesignerV2.executeColumnOperation('sum')" 
                                class="px-3 py-2 bg-[#34B27B] hover:bg-[#2a9463] text-black text-xs font-bold uppercase rounded transition">
                            Add Rows
                        </button>
                        <button onclick="window.tableDesignerV2.executeColumnOperation('subtract')" 
                                class="px-3 py-2 bg-[#4ECDC4] hover:bg-[#3db3a6] text-white text-xs font-bold rounded transition">
                            Subtract
                        </button>
                        <button onclick="window.tableDesignerV2.executeColumnOperation('multiply')" 
                                class="px-3 py-2 bg-[#8b5cf6] hover:bg-[#7c3aed] text-white text-xs font-bold rounded transition">
                            ✕ Multiply
                        </button>
                        <button onclick="window.tableDesignerV2.executeColumnOperation('divide')" 
                                class="px-3 py-2 bg-[#f59e0b] hover:bg-[#d97706] text-white text-xs font-bold rounded transition">
                            Divide
                        </button>
                    </div>
                    
                    <div class="h-8 w-px bg-white/20"></div>
                    
                    <!-- Aggregate operations -->
                    <div class="flex gap-2">
                        <button onclick="window.tableDesignerV2.executeColumnOperation('sum_all')" 
                                class="px-3 py-2 bg-gradient-to-r from-[#34B27B] to-[#2a9463] text-white text-xs font-bold rounded transition">
                            Σ SUM ALL
                        </button>
                        <button onclick="window.tableDesignerV2.executeColumnOperation('avg_all')" 
                                class="px-3 py-2 bg-gradient-to-r from-[#4ECDC4] to-[#3db3a6] text-white text-xs font-bold rounded transition">
                            μ AVG ALL
                        </button>
                        <button onclick="window.tableDesignerV2.executeColumnOperation('count_all')" 
                                class="px-3 py-2 bg-gradient-to-r from-[#8b5cf6] to-[#7c3aed] text-white text-xs font-bold rounded transition">
                            # COUNT ALL
                        </button>
                    </div>
                    
                    <div class="h-8 w-px bg-white/20"></div>
                    
                    <!-- Join operations -->
                    <div class="flex gap-2">
                        <button onclick="window.tableDesignerV2.executeColumnOperation('inner_join')" 
                                class="px-3 py-2 bg-[#06b6d4] hover:bg-[#0891b2] text-white text-xs font-bold rounded transition">
                            ⋈ INNER JOIN
                        </button>
                        <button onclick="window.tableDesignerV2.executeColumnOperation('left_join')" 
                                class="px-3 py-2 bg-[#3b82f6] hover:bg-[#2563eb] text-white text-xs font-bold rounded transition">
                            ⟕ LEFT JOIN
                        </button>
                        <button onclick="window.tableDesignerV2.executeColumnOperation('right_join')" 
                                class="px-3 py-2 bg-[#6366f1] hover:bg-[#4f46e5] text-white text-xs font-bold rounded transition">
                            ⟖ RIGHT JOIN
                        </button>
                        <button onclick="window.tableDesignerV2.executeColumnOperation('full_join')" 
                                class="px-3 py-2 bg-[#8b5cf6] hover:bg-[#7c3aed] text-white text-xs font-bold rounded transition">
                            ⟗ FULL JOIN
                        </button>
                    </div>
                    
                    <div class="h-8 w-px bg-white/20"></div>
                    
                    <!-- Other operations -->
                    <div class="flex gap-2">
                        <button onclick="window.tableDesignerV2.executeColumnOperation('concat')" 
                                class="px-3 py-2 bg-[#ec4899] hover:bg-[#db2777] text-white text-xs font-bold rounded transition">
                            || CONCAT
                        </button>
                    </div>
                    
                    <div class="h-8 w-px bg-white/20"></div>
                    
                    <button onclick="window.tableDesignerV2.dragColumnToNewTable()" 
                            class="px-4 py-2 bg-gradient-to-r from-[#f97316] to-[#ea580c] text-white text-xs font-bold rounded transition shadow-lg">
                        Create New Table
                    </button>
                    
                    <button onclick="window.tableDesignerV2.saveLastResultToDatabase()" 
                            class="px-4 py-2 bg-gradient-to-r from-[#10b981] to-[#059669] text-white text-xs font-bold rounded transition shadow-lg">
                        Save to DB
                    </button>
                    
                    <button onclick="window.tableDesignerV2.closeOperationBar()" 
                            class="px-3 py-2 bg-white/5 hover:bg-white/10 text-white/70 text-xs rounded transition">
                        ✕
                    </button>
                </div>
            </div>

            <!-- Operation Panel (Hidden by default) -->
            <div id="operationPanel" class="hidden border-b border-white/10 bg-[#0d1117] p-6">
                <div class="max-w-6xl mx-auto">
                    <h3 class="text-white font-bold mb-4">Build Operation - Select Columns from Any Table</h3>
                    
                    <div class="grid grid-cols-3 gap-4 mb-6">
                        <!-- All tables and their columns -->
                        <div id="allTablesColumns" class="col-span-3 grid grid-cols-3 gap-4">
                            <div class="text-xs text-white/30">No tables loaded</div>
                        </div>
                    </div>
                    
                    <div class="border-t border-white/10 pt-4">
                        <div class="text-sm text-white/70 font-bold mb-2">Selected Columns:</div>
                        <div id="selectedColumns" class="bg-black/30 p-3 rounded text-xs text-white/50 min-h-12">
                            No columns selected
                        </div>
                    </div>
                    
                    <div class="mt-6 flex items-center gap-3">
                        <span class="text-sm text-white/50">Operation:</span>
                        <select id="operationType" class="px-3 py-1.5 bg-white/5 text-white text-sm rounded border border-white/10">
                            <option value="select">SELECT (Custom Query)</option>
                            <option value="join">JOIN (Combine Tables)</option>
                            <option value="add">ADD (A + B)</option>
                            <option value="subtract">SUBTRACT (A - B)</option>
                            <option value="multiply">MULTIPLY (A × B)</option>
                            <option value="divide">DIVIDE (A ÷ B)</option>
                            <option value="concat">CONCAT (A | B)</option>
                        </select>
                        
                        <input id="whereClause" type="text" placeholder="WHERE condition (optional)..." 
                               class="px-3 py-1.5 bg-white/5 text-white text-sm rounded border border-white/10 flex-1">
                        
                        <button onclick="window.tableDesignerV2.executeOperation()" 
                                class="px-6 py-1.5 bg-gradient-to-r from-[#34B27B] to-[#4ECDC4] text-white text-sm font-bold rounded">
                            Execute
                        </button>
                        <button onclick="window.tableDesignerV2.hideOperationPanel()" 
                                class="px-4 py-1.5 bg-white/5 text-white/70 text-sm rounded">
                            Cancel
                        </button>
                    </div>
                    
                    <div id="operationResult" class="mt-4 hidden">
                        <div class="text-xs text-white/50 font-bold mb-2">Operation Result:</div>
                        <div id="operationOutput" class="bg-black/30 p-3 rounded text-xs text-white/70 font-mono max-h-32 overflow-y-auto"></div>
                    </div>
                </div>
            </div>

            <!-- Canvas -->
            <div class="flex-1 relative overflow-hidden bg-[#0a0a0a]">
                <canvas id="designer-canvas" class="absolute inset-0 w-full h-full cursor-move"></canvas>
            </div>
            
            <!-- Create Table Modal -->
            <div id="createTableModalDesigner" class="hidden fixed inset-0 bg-black/80 backdrop-blur-sm z-[10000] flex items-center justify-center">
                <div class="bg-[#11181C] rounded-xl border border-white/5 p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
                    <div class="flex justify-between items-center mb-6">
                        <h2 class="text-xl font-bold text-white">Create New Table</h2>
                        <button onclick="window.tableDesignerV2.closeCreateTableModal()" 
                                class="text-white/40 hover:text-white text-2xl leading-none">&times;</button>
                    </div>
                    
                    <div class="space-y-4">
                        <div>
                            <label class="text-sm text-white/70 block mb-2">Table Name</label>
                            <input type="text" id="newTableNameDesigner" placeholder="customers"
                                   class="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white placeholder-white/30 focus:outline-none focus:border-[#34B27B]" />
                        </div>
                        
                        <div>
                            <label class="text-sm text-white/70 block mb-2">Columns</label>
                            <div id="columnsListDesigner" class="space-y-2 mb-2">
                                <!-- Column rows added here -->
                            </div>
                            <button onclick="window.tableDesignerV2.addColumnRow()" 
                                    class="text-xs bg-white/5 hover:bg-white/10 text-white/70 px-3 py-1.5 rounded font-bold transition-all">
                                Add Column
                            </button>
                        </div>
                        
                        <div>
                            <label class="text-sm text-white/70 block mb-2">Initial Data (Optional - JSON Array)</label>
                            <textarea id="initialDataDesigner" placeholder='[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'
                                      class="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-white/30 focus:outline-none focus:border-[#34B27B] mono h-32 resize-none"></textarea>
                        </div>
                        
                        <div class="flex gap-3">
                            <button onclick="window.tableDesignerV2.createNewTable()" 
                                    class="flex-1 bg-[#34B27B] hover:bg-[#2d9a68] text-black text-xs font-bold uppercase py-3 rounded-lg transition-all">
                                Create Table
                            </button>
                            <button onclick="window.tableDesignerV2.closeCreateTableModal()" 
                                    class="flex-1 bg-white/5 hover:bg-white/10 text-white/70 text-xs font-bold py-3 rounded-lg transition-all">
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Status Bar -->
            <div class="h-10 border-t border-white/10 flex items-center px-6 bg-[#0a0a0a] text-xs text-white/50">
                <span id="status-text">Ready • Connect tables • Drag operations • Execute uses WebGPU</span>
            </div>
        `;

        document.body.appendChild(this.modal);
        
        // Store reference globally
        window.tableDesignerV2 = this;
        
        // Setup canvas after DOM is ready
        requestAnimationFrame(() => {
            this.canvas = document.getElementById('designer-canvas');
            if (!this.canvas) {
                console.error('Canvas element not found');
                return;
            }
            
            this.ctx = this.canvas.getContext('2d');
            this.resizeCanvas();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Set default connection mode
            this.connectionMode = 'add';
        });
    }

    /**
     * Resize canvas to fill container
     */
    resizeCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }

    /**
     * Setup event listeners for canvas interaction
     */
    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.onMouseUp.bind(this));
        
        // Allow drag-drop on canvas
        this.canvas.addEventListener('dragover', (e) => {
            e.preventDefault();
            if (this.isDraggingOperation) {
                const rect = this.canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                this.mousePos = { x, y };
                this.hoveredConnection = this.getConnectionAtPosition(x, y);
                this.render();
            }
        });
        
        this.canvas.addEventListener('drop', (e) => {
            e.preventDefault();
            if (this.isDraggingOperation && this.hoveredConnection && this.draggedOperation) {
                this.executeOperationOnConnection(this.hoveredConnection, this.draggedOperation);
            }
            this.isDraggingOperation = false;
            this.draggedOperation = null;
            this.hoveredConnection = null;
            this.render();
        });
        
        this.canvas.addEventListener('dragend', (e) => {
            this.isDraggingOperation = false;
            this.draggedOperation = null;
            this.hoveredConnection = null;
            this.render();
        });
        
        // Keyboard events for rename
        document.addEventListener('keydown', this.onKeyDown.bind(this));
        
        // Window resize
        window.addEventListener('resize', () => {
            this.resizeCanvas();
            this.render();
        });
        
        // Close dropdown menus when clicking outside
        document.addEventListener('click', (event) => {
            const joinBtn = document.getElementById('designerJoinBtn');
            const joinMenu = document.getElementById('designerJoinMenu');
            const downloadBtn = document.getElementById('designerDownloadBtn');
            const downloadMenu = document.getElementById('designerDownloadMenu');
            
            if (joinBtn && joinMenu && !joinBtn.contains(event.target) && !joinMenu.contains(event.target)) {
                joinMenu.classList.add('hidden');
            }
            
            if (downloadBtn && downloadMenu && !downloadBtn.contains(event.target) && !downloadMenu.contains(event.target)) {
                downloadMenu.classList.add('hidden');
            }
        });
    }
    
    /**
     * Handle keyboard input for rename
     */
    onKeyDown(e) {
        // Handle table rename
        if (this.renamingTable) {
            const table = this.tables.get(this.renamingTable);
            if (!table) return;
            
            if (e.key === 'Enter') {
                // Finish rename - if empty, restore original name
                const finalName = table.name.trim();
                if (!finalName || finalName.length === 0) {
                    // Restore original name
                    const oldName = table.name;
                    table.name = this.originalTableName;
                    this.tables.delete(oldName);
                    this.tables.set(table.name, table);
                    this.updateConnectionReferences(oldName, table.name);
                }
                
                this.renamingTable = null;
                this.originalTableName = null;
                this.renamingTableFirstKey = true;
                
                // Update database if table exists
                if (this.db && this.db.tables.has(table.name)) {
                    // Note: actual DB rename would need more complex handling
                    console.log(`Renamed table to ${table.name}`);
                }
                
                this.updateStatus(`Renamed to ${table.name}`);
                this.updateConnectionCount();
                this.render();
            } else if (e.key === 'Escape') {
                // Cancel rename - restore original name
                const oldName = table.name;
                table.name = this.originalTableName;
                this.tables.delete(oldName);
                this.tables.set(table.name, table);
                this.updateConnectionReferences(oldName, table.name);
                
                this.renamingTable = null;
                this.originalTableName = null;
                this.renamingTableFirstKey = true;
                this.updateStatus('Rename cancelled');
                this.render();
            } else if (e.key === 'Backspace') {
                // Delete last character
                e.preventDefault();
                if (table.name.length > 1) {
                    const oldName = table.name;
                    table.name = table.name.slice(0, -1);
                    
                    // Update map key
                    this.tables.delete(oldName);
                    this.tables.set(table.name, table);
                    
                    // Update renamingTable to track the new name
                    this.renamingTable = table.name;
                    
                    // Update connections that reference this table
                    this.updateConnectionReferences(oldName, table.name);
                    
                    this.render();
                }
            } else if (e.key.length === 1 && /[a-zA-Z0-9_]/.test(e.key)) {
                // Add character to name
                e.preventDefault();
                const oldName = table.name;
                
                // On first key, clear the old name
                if (this.renamingTableFirstKey) {
                    table.name = e.key;
                    this.renamingTableFirstKey = false;
                } else {
                    table.name += e.key;
                }
                
                // Update map key
                this.tables.delete(oldName);
                this.tables.set(table.name, table);
                
                // Update renamingTable to track the new name
                this.renamingTable = table.name;
                
                // Update connections that reference this table
                this.updateConnectionReferences(oldName, table.name);
                
                this.render();
            }
            return;
        }
        
        // Handle column rename
        if (this.renamingColumn) {
            const table = this.tables.get(this.renamingColumn.table);
            if (!table) return;
            
            if (e.key === 'Enter') {
                // Finish rename - if empty, restore original name
                const currentColumn = table.columns[this.renamingColumn.index];
                if (!currentColumn || currentColumn.trim().length === 0) {
                    // Restore original name
                    table.columns[this.renamingColumn.index] = this.originalColumnName;
                    
                    // Update data keys back to original
                    table.data.forEach(row => {
                        if (currentColumn in row) {
                            row[this.originalColumnName] = row[currentColumn];
                            delete row[currentColumn];
                        }
                    });
                    
                    // Restore connections
                    this.connections.forEach(conn => {
                        if (conn.from.table === this.renamingColumn.table && conn.from.column === currentColumn) {
                            conn.from.column = this.originalColumnName;
                        }
                        if (conn.to.table === this.renamingColumn.table && conn.to.column === currentColumn) {
                            conn.to.column = this.originalColumnName;
                        }
                    });
                }
                
                this.renamingColumn = null;
                this.originalColumnName = null;
                this.renamingColumnFirstKey = true;
                this.updateStatus('Column renamed');
                this.render();
            } else if (e.key === 'Escape') {
                // Cancel rename - restore original name
                const currentColumn = table.columns[this.renamingColumn.index];
                table.columns[this.renamingColumn.index] = this.originalColumnName;
                
                // Update data keys back to original
                table.data.forEach(row => {
                    if (currentColumn in row) {
                        row[this.originalColumnName] = row[currentColumn];
                        delete row[currentColumn];
                    }
                });
                
                // Restore connections
                this.connections.forEach(conn => {
                    if (conn.from.table === this.renamingColumn.table && conn.from.column === currentColumn) {
                        conn.from.column = this.originalColumnName;
                    }
                    if (conn.to.table === this.renamingColumn.table && conn.to.column === currentColumn) {
                        conn.to.column = this.originalColumnName;
                    }
                });
                
                this.renamingColumn = null;
                this.originalColumnName = null;
                this.renamingColumnFirstKey = true;
                this.updateStatus('Rename cancelled');
                this.render();
            } else if (e.key === 'Backspace') {
                // Delete last character
                e.preventDefault();
                const currentColumn = table.columns[this.renamingColumn.index];
                if (currentColumn.length > 1) {
                    const oldColumn = currentColumn;
                    const newColumn = currentColumn.slice(0, -1);
                    table.columns[this.renamingColumn.index] = newColumn;
                    
                    // Update data keys
                    table.data.forEach(row => {
                        if (oldColumn in row) {
                            row[newColumn] = row[oldColumn];
                            delete row[oldColumn];
                        }
                    });
                    
                    // Update renaming tracker
                    this.renamingColumn.column = newColumn;
                    
                    // Update connections
                    this.connections.forEach(conn => {
                        if (conn.from.table === this.renamingColumn.table && conn.from.column === oldColumn) {
                            conn.from.column = newColumn;
                        }
                        if (conn.to.table === this.renamingColumn.table && conn.to.column === oldColumn) {
                            conn.to.column = newColumn;
                        }
                    });
                    
                    this.render();
                }
            } else if (e.key.length === 1 && /[a-zA-Z0-9_]/.test(e.key)) {
                // Add character
                e.preventDefault();
                const oldColumn = table.columns[this.renamingColumn.index];
                let newColumn;
                
                // On first key, clear the old name
                if (this.renamingColumnFirstKey) {
                    newColumn = e.key;
                    this.renamingColumnFirstKey = false;
                } else {
                    newColumn = oldColumn + e.key;
                }
                
                table.columns[this.renamingColumn.index] = newColumn;
                
                // Update data keys
                table.data.forEach(row => {
                    if (oldColumn in row) {
                        row[newColumn] = row[oldColumn];
                        delete row[oldColumn];
                    }
                });
                
                // Update renaming tracker
                this.renamingColumn.column = newColumn;
                
                // Update connections
                this.connections.forEach(conn => {
                    if (conn.from.table === this.renamingColumn.table && conn.from.column === oldColumn) {
                        conn.from.column = newColumn;
                    }
                    if (conn.to.table === this.renamingColumn.table && conn.to.column === oldColumn) {
                        conn.to.column = newColumn;
                    }
                });
                
                this.render();
            }
        }
    }
    
    /**
     * Update all connection references when a table is renamed
     */
    updateConnectionReferences(oldName, newName) {
        this.connections.forEach(conn => {
            if (conn.from.table === oldName) {
                conn.from.table = newName;
            }
            if (conn.to.table === oldName) {
                conn.to.table = newName;
            }
        });
        
        // Update selected tables
        if (this.selectedTables.has(oldName)) {
            this.selectedTables.delete(oldName);
            this.selectedTables.add(newName);
        }
        
        // Update selected columns
        const newSelectedColumns = new Set();
        this.selectedCanvasColumns.forEach(colKey => {
            const [table, column] = colKey.split('.');
            if (table === oldName) {
                newSelectedColumns.add(`${newName}.${column}`);
            } else {
                newSelectedColumns.add(colKey);
            }
        });
        this.selectedCanvasColumns = newSelectedColumns;
    }

    /**
     * Mouse down - start drag or connection
     */
    onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Check if clicked on a column button
        for (const btn of this.activeColumnButtons) {
            if (x >= btn.x && x <= btn.x + btn.width &&
                y >= btn.y && y <= btn.y + btn.height) {
                // Button clicked
                if (btn.action === 'inputA') {
                    this.setInputTable('A', btn.table);
                    this.updateStatus(`Set Input A to ${btn.table}`);
                } else if (btn.action === 'inputB') {
                    this.setInputTable('B', btn.table);
                    this.updateStatus(`Set Input B to ${btn.table}`);
                } else if (btn.action === 'addColumn') {
                    // Add new column to table
                    this.addColumnToTable(btn.table);
                }
                return;
            }
        }

        // Check if clicked on a column - start drag to connect OR double-click to rename
        const columnClick = this.getColumnAtPosition(x, y);
        if (columnClick) {
            const colKey = `${columnClick.table}.${columnClick.column}`;
            
            // Shift+Click to start external drag to visualization
            if (e.shiftKey) {
                this.startColumnDrag(columnClick.table, columnClick.column, e);
                this.updateStatus(`Hold Shift and click ${colKey} to select column`);
                return;
            }
            
            // Double-click detection for column rename
            if (!e.altKey && !e.ctrlKey && !e.metaKey) {
                const now = Date.now();
                const clickKey = `${columnClick.table}_${columnClick.index}`;
                if (this.lastColumnClickKey === clickKey && now - this.lastColumnClickTime < 500) {
                    // Double-click detected - start rename
                    this.renamingColumn = {
                        table: columnClick.table,
                        column: columnClick.column,
                        index: columnClick.index
                    };
                    this.originalColumnName = columnClick.column; // Store original name
                    this.renamingColumnFirstKey = true; // Reset first key flag
                    this.renamingTable = null; // Cancel table rename
                    this.lastColumnClickKey = null;
                    this.lastColumnClickTime = 0;
                    this.render();
                    return;
                }
                this.lastColumnClickKey = clickKey;
                this.lastColumnClickTime = now;
            }
            
            // Alt+Click or Ctrl+Click to select column for analytics
            if (e.altKey || e.ctrlKey || e.metaKey) {
                if (this.selectedCanvasColumns.has(colKey)) {
                    this.selectedCanvasColumns.delete(colKey);
                    this.selectedColumns = this.selectedColumns.filter(c => c.key !== colKey);
                    this.updateStatus(`Deselected: ${colKey}`);
                } else {
                    this.selectedCanvasColumns.add(colKey);
                    this.selectedColumns.push({ 
                        table: columnClick.table, 
                        column: columnClick.column, 
                        key: colKey 
                    });
                    this.updateStatus(`Selected: ${colKey} • Use Operations menu to calculate`);
                }
                this.render();
                return;
            }
            
            // Normal click: start dragging a connection line from this column
            this.isDraggingConnection = true;
            this.isConnecting = true;
            this.connectionStart = columnClick;
            this.updateStatus(`Drag from ${colKey} to another column to connect...`);
            this.canvas.style.cursor = 'crosshair';
            return;
        }

        // Check if clicked on a table
        const table = this.getTableAtPosition(x, y);
        if (table) {
            // Check if clicked on table header (for selection)
            const clickedOnHeader = y < table.position.y + this.TABLE_HEADER_HEIGHT;
            
            // Double-click detection on header for rename
            if (clickedOnHeader && !e.shiftKey) {
                const now = Date.now();
                if (this.lastHeaderClickTable === table.name && now - this.lastHeaderClickTime < 500) {
                    // Double-click detected - start rename
                    this.renamingTable = table.name;
                    this.originalTableName = table.name; // Store original name
                    this.renamingTableFirstKey = true; // Reset first key flag
                    this.lastHeaderClickTable = null;
                    this.lastHeaderClickTime = 0;
                    this.render();
                    return;
                }
                this.lastHeaderClickTable = table.name;
                this.lastHeaderClickTime = now;
            }
            
            if (clickedOnHeader && e.shiftKey) {
                // Shift+Click on header to toggle selection
                if (this.selectedTables.has(table.name)) {
                    this.selectedTables.delete(table.name);
                } else {
                    this.selectedTables.add(table.name);
                }
                this.updateColumnSelectors(); // Update UI
                this.render();
                return; // Don't start dragging
            }
            
            this.isDragging = true;
            this.draggedTable = table;
            this.dragOffset = {
                x: x - table.position.x,
                y: y - table.position.y
            };
            this.canvas.style.cursor = 'grabbing';
        }
    }

    /**
     * Mouse move - drag table or show connection preview
     */
    onMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        this.mousePos = { x, y };

        if (this.isDragging && this.draggedTable) {
            this.draggedTable.position.x = x - this.dragOffset.x;
            this.draggedTable.position.y = y - this.dragOffset.y;
            this.render();
        } else if (this.isDraggingOperation) {
            // Check which connection is being hovered
            this.hoveredConnection = this.getConnectionAtPosition(x, y);
            this.render();
        } else if (this.isDraggingConnection || this.isConnecting) {
            // Update mouse position for connection preview
            this.render();
        } else {
            // Update cursor based on hover
            let overButton = false;
            
            // Check column buttons
            for (const btn of this.activeColumnButtons) {
                if (x >= btn.x && x <= btn.x + btn.width &&
                    y >= btn.y && y <= btn.y + btn.height) {
                    overButton = true;
                    break;
                }
            }
            
            if (overButton) {
                this.canvas.style.cursor = 'pointer';
            } else {
                const column = this.getColumnAtPosition(x, y);
                const table = this.getTableAtPosition(x, y);
                
                if (column) {
                    this.canvas.style.cursor = 'pointer';
                } else if (table) {
                    this.canvas.style.cursor = 'grab';
                } else {
                    this.canvas.style.cursor = 'move';
                }
            }
        }
    }

    /**
     * Mouse up - finish drag or create connection
     */
    onMouseUp(e) {
        if (this.isDraggingConnection && this.connectionStart) {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const columnClick = this.getColumnAtPosition(x, y);
            if (columnClick && 
                (columnClick.table !== this.connectionStart.table || 
                 columnClick.column !== this.connectionStart.column)) {
                
                // Create connection
                this.createConnection(this.connectionStart, columnClick);
            } else {
                this.updateStatus('Connection cancelled');
            }

            this.isDraggingConnection = false;
            this.isConnecting = false;
            this.connectionStart = null;
            this.canvas.style.cursor = 'move';
        }

        if (this.isConnecting && this.connectionStart) {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const columnClick = this.getColumnAtPosition(x, y);
            if (columnClick && 
                (columnClick.table !== this.connectionStart.table || 
                 columnClick.column !== this.connectionStart.column)) {
                
                // Create connection
                this.createConnection(this.connectionStart, columnClick);
            }
            
            this.isConnecting = false;
            this.connectionStart = null;
            this.updateStatus('Ready • Drag columns to connect • Click Execute on connections to create result tables');
        }

        if (this.isDragging) {
            this.isDragging = false;
            this.draggedTable = null;
            this.canvas.style.cursor = 'move';
        }

        this.render();
    }

    /**
     * Create quick table on canvas
     */
    createQuickTable() {
        // Find a unique name
        let counter = 1;
        let tableName = 'new_table';
        while (this.tables.has(tableName)) {
            tableName = `new_table_${counter}`;
            counter++;
        }
        
        // Position tables vertically on the left side
        const leftMargin = 50;
        const topMargin = 100;
        const verticalSpacing = 200;
        
        // Create table with default column
        const newTable = {
            name: tableName,
            columns: ['id'],
            data: [{ id: 1 }],
            rowCount: 1,
            position: {
                x: leftMargin,
                y: topMargin + (this.tables.size * verticalSpacing)
            }
        };
        
        // Add to canvas
        this.tables.set(tableName, newTable);
        
        // Don't auto-rename - let user double-click to rename
        this.updateStatus(`Created ${tableName} • Double-click header to rename • Drag to move`);
        this.updateColumnSelectors();
        this.render();
    }
    
    /**
     * Add column to existing table
     */
    async addColumnToTable(tableName) {
        const table = this.tables.get(tableName);
        if (!table) return;
        
        // Find unique column name
        let counter = 1;
        let columnName = 'column';
        while (table.columns.includes(columnName)) {
            columnName = `column_${counter}`;
            counter++;
        }
        
        // Add column to table
        table.columns.push(columnName);
        
        // Update data rows to include new column
        if (table.data) {
            table.data.forEach(row => {
                row[columnName] = null;
            });
        }
        
        // If this is a real DB table, we need to update it
        if (this.db && this.db.tables.has(tableName)) {
            try {
                // Reload table data with new column
                const updatedData = table.data || [{ [columnName]: null }];
                await this.db.loadJSON(tableName, updatedData);
                console.log(`Added column ${columnName} to ${tableName}`);
            } catch (err) {
                console.error(`Failed to add column:`, err);
            }
        }
        
        this.updateStatus(`Added column: ${columnName} to ${tableName}`);
        this.updateColumnSelectors();
        this.render();
    }

    /**
     * Get table at position
     */
    getTableAtPosition(x, y) {
        for (const [name, table] of this.tables.entries()) {
            const height = this.TABLE_HEADER_HEIGHT + (table.columns.length * this.COLUMN_HEIGHT) + 30; // +30 for Add Column button
            
            if (x >= table.position.x && 
                x <= table.position.x + this.TABLE_WIDTH &&
                y >= table.position.y && 
                y <= table.position.y + height) {
                table.name = name; // Ensure name is set
                return table;
            }
        }
        return null;
    }

    /**
     * Get column at position
     */
    getColumnAtPosition(x, y) {
        for (const [tableName, table] of this.tables.entries()) {
            const headerBottom = table.position.y + this.TABLE_HEADER_HEIGHT;
            
            for (let index = 0; index < table.columns.length; index++) {
                const column = table.columns[index];
                const colY = headerBottom + (index * this.COLUMN_HEIGHT);
                
                if (x >= table.position.x && 
                    x <= table.position.x + this.TABLE_WIDTH &&
                    y >= colY && 
                    y <= colY + this.COLUMN_HEIGHT) {
                    return { table: tableName, column: column, index: index };
                }
            }
        }
        return null;
    }

    /**
     * Create connection between columns
     */
    createConnection(from, to) {
        const connection = {
            from: from,
            to: to,
            type: this.connectionMode || 'copy',  // Default to 'copy' which shows = symbol
            operation: null  // Operation can be set later by dragging
        };
        
        this.connections.push(connection);
        console.log(`Connected: ${from.table}.${from.column} → ${to.table}.${to.column}`);
        this.updateStatus(`${to.table}.${to.column} = ${from.table}.${from.column} • Drag operation to change`);
        this.updateConnectionCount();
    }

    /**
     * Get operation symbol
     */
    getOperationSymbol(type) {
        const symbols = {
            'add': '+',
            'subtract': '-',
            'copy': '📋',
            'graph': 'GRAPH'
        };
        return symbols[type] || '→';
    }

    /**
     * Start rendering loop
     */
    startRendering() {
        this.render();
    }

    /**
     * Render canvas
     */
    render() {
        if (!this.ctx) return;

        // Clear canvas
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Clear button tracking
        this.activeColumnButtons = [];
        this.activeConnectionButtons = [];
        
        // Clear column drag overlays for re-creation
        const container = document.getElementById('column-drag-overlays');
        if (container) {
            container.innerHTML = '';
        }

        // Draw grid
        this.drawGrid();

        // Draw connections
        this.drawConnections();

        // Draw connection preview
        if (this.isConnecting && this.connectionStart) {
            this.drawConnectionPreview();
        }

        // Draw tables
        for (const [name, table] of this.tables.entries()) {
            this.drawTable(table);
        }
    }

    /**
     * Draw background grid
     */
    drawGrid() {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
        this.ctx.lineWidth = 1;

        const gridSize = 50;
        for (let x = 0; x < this.canvas.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        for (let y = 0; y < this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
    }

    /**
     * Draw table card
     */
    drawTable(table) {
        const x = table.position.x;
        const y = table.position.y;
        const addBtnHeight = 30;
        const height = this.TABLE_HEADER_HEIGHT + (table.columns.length * this.COLUMN_HEIGHT) + addBtnHeight;

        // Shadow
        this.ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
        this.ctx.shadowBlur = 20;
        this.ctx.shadowOffsetX = 0;
        this.ctx.shadowOffsetY = 4;

        // Table background
        this.ctx.fillStyle = 'rgba(17, 24, 28, 0.95)';
        this.ctx.fillRect(x, y, this.TABLE_WIDTH, height);

        // Reset shadow
        this.ctx.shadowColor = 'transparent';
        this.ctx.shadowBlur = 0;

        // Table border (highlight if selected)
        const isSelected = this.selectedTables.has(table.name);
        this.ctx.strokeStyle = isSelected ? '#34B27B' : 'rgba(255, 255, 255, 0.1)';
        this.ctx.lineWidth = isSelected ? 3 : 1;
        this.ctx.strokeRect(x, y, this.TABLE_WIDTH, height);

        // Header background
        const gradient = this.ctx.createLinearGradient(x, y, x + this.TABLE_WIDTH, y);
        gradient.addColorStop(0, 'rgba(52, 178, 123, 0.2)');
        gradient.addColorStop(1, 'rgba(78, 205, 196, 0.2)');
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(x, y, this.TABLE_WIDTH, this.TABLE_HEADER_HEIGHT);

        // Header border
        this.ctx.strokeStyle = 'rgba(52, 178, 123, 0.3)';
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(x, y, this.TABLE_WIDTH, this.TABLE_HEADER_HEIGHT);

        // Table name (with rename UI if renaming)
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = 'bold 14px Inter, system-ui';
        this.ctx.textAlign = 'left';
        
        if (this.renamingTable === table.name) {
            // Show cursor and hint for renaming
            this.ctx.fillText(table.name + '|', x + 15, y + 25);
            this.ctx.fillStyle = 'rgba(52, 178, 123, 0.7)';
            this.ctx.font = '10px Inter, system-ui';
            this.ctx.fillText('Type new name, press Enter', x + 15, y + 42);
        } else {
            this.ctx.fillText(table.name, x + 15, y + 25);
            // Row count (only show when not renaming)
            this.ctx.fillStyle = 'rgba(52, 178, 123, 0.8)';
            this.ctx.font = '11px Inter, system-ui';
            const rowCount = table.rowCount || table.data?.length || 0;
            this.ctx.fillText(`${rowCount.toLocaleString()} rows`, x + 15, y + 40);
        }

        // Columns
        table.columns.forEach((column, index) => {
            const colY = y + this.TABLE_HEADER_HEIGHT + (index * this.COLUMN_HEIGHT);
            const colKey = `${table.name}.${column}`;
            const isColumnSelected = this.selectedCanvasColumns.has(colKey);

            // Column background (highlight if selected)
            if (isColumnSelected) {
                this.ctx.fillStyle = 'rgba(52, 178, 123, 0.3)';
                this.ctx.fillRect(x, colY, this.TABLE_WIDTH, this.COLUMN_HEIGHT);
            } else if (index % 2 === 0) {
                this.ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
                this.ctx.fillRect(x, colY, this.TABLE_WIDTH, this.COLUMN_HEIGHT);
            }

            // Column divider
            this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.moveTo(x, colY);
            this.ctx.lineTo(x + this.TABLE_WIDTH, colY);
            this.ctx.stroke();

            // Column icon
            this.ctx.fillStyle = 'rgba(78, 205, 196, 0.6)';
            this.ctx.beginPath();
            this.ctx.arc(x + 15, colY + 17, 4, 0, Math.PI * 2);
            this.ctx.fill();

            // Column name
            const isRenamingColumn = this.renamingColumn && 
                                    this.renamingColumn.table === table.name && 
                                    this.renamingColumn.index === index;
            
            if (isRenamingColumn) {
                // Show rename indicator
                this.ctx.fillStyle = 'rgba(252, 211, 77, 0.3)';
                this.ctx.fillRect(x + 25, colY + 6, this.TABLE_WIDTH - 30, 20);
                this.ctx.strokeStyle = '#FCD34D';
                this.ctx.lineWidth = 2;
                this.ctx.strokeRect(x + 25, colY + 6, this.TABLE_WIDTH - 30, 20);
                
                this.ctx.fillStyle = '#FCD34D';
                this.ctx.font = 'bold 12px "Courier New", monospace';
            } else {
                this.ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                this.ctx.font = '12px "Courier New", monospace';
            }
            
            this.ctx.textAlign = 'left';
            this.ctx.fillText(column, x + 30, colY + 22);
            
            // Show buttons below selected column
            if (isColumnSelected) {
                const btnY = colY + this.COLUMN_HEIGHT - 23;
                const btnWidth = 40;
                const btnHeight = 18;
                const btnSpacing = 3;
                const btnStartX = x + this.TABLE_WIDTH - (btnWidth * 2 + btnSpacing + 5);
                
                // Input A button
                this.ctx.fillStyle = 'rgba(52, 178, 123, 0.9)';
                this.ctx.fillRect(btnStartX, btnY, btnWidth, btnHeight);
                this.ctx.strokeStyle = '#34B27B';
                this.ctx.lineWidth = 1;
                this.ctx.strokeRect(btnStartX, btnY, btnWidth, btnHeight);
                this.ctx.fillStyle = '#ffffff';
                this.ctx.font = 'bold 9px Inter, system-ui';
                this.ctx.textAlign = 'center';
                this.ctx.fillText('A', btnStartX + btnWidth/2, btnY + 12);
                
                // Input B button
                const btn2X = btnStartX + btnWidth + btnSpacing;
                this.ctx.fillStyle = 'rgba(78, 205, 196, 0.9)';
                this.ctx.fillRect(btn2X, btnY, btnWidth, btnHeight);
                this.ctx.strokeStyle = '#4ECDC4';
                this.ctx.lineWidth = 1;
                this.ctx.strokeRect(btn2X, btnY, btnWidth, btnHeight);
                this.ctx.fillStyle = '#ffffff';
                this.ctx.fillText('B', btn2X + btnWidth/2, btnY + 12);
                
                // Store button positions for click detection
                this.activeColumnButtons.push({
                    x: btnStartX, y: btnY, width: btnWidth, height: btnHeight,
                    action: 'inputA', table: table.name, column: column
                });
                this.activeColumnButtons.push({
                    x: btn2X, y: btnY, width: btnWidth, height: btnHeight,
                    action: 'inputB', table: table.name, column: column
                });
            }
        });
        
        // Draw "Add Column" button at bottom of table
        const addBtnY = y + this.TABLE_HEADER_HEIGHT + (table.columns.length * this.COLUMN_HEIGHT);
        
        // Button background
        this.ctx.fillStyle = 'rgba(52, 178, 123, 0.15)';
        this.ctx.fillRect(x, addBtnY, this.TABLE_WIDTH, addBtnHeight);
        
        // Button border
        this.ctx.strokeStyle = 'rgba(52, 178, 123, 0.3)';
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(x, addBtnY, this.TABLE_WIDTH, addBtnHeight);
        
        // Button text
        this.ctx.fillStyle = 'rgba(52, 178, 123, 0.9)';
        this.ctx.font = 'bold 11px Inter, system-ui';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Add Column', x + this.TABLE_WIDTH / 2, addBtnY + 19);
        
        // Store button position for click detection
        this.activeColumnButtons.push({
            x: x, y: addBtnY, width: this.TABLE_WIDTH, height: addBtnHeight,
            action: 'addColumn', table: table.name
        });
    }

    /**
     * Draw all connections
     */
    drawConnections() {
        this.connections.forEach(conn => {
            this.drawConnection(conn);
        });
    }

    /**
     * Draw single connection
     */
    drawConnection(conn) {
        const fromTable = this.tables.get(conn.from.table);
        const toTable = this.tables.get(conn.to.table);
        
        if (!fromTable || !toTable) return;

        // Calculate positions
        const fromY = fromTable.position.y + this.TABLE_HEADER_HEIGHT + (conn.from.index * this.COLUMN_HEIGHT) + (this.COLUMN_HEIGHT / 2);
        const toY = toTable.position.y + this.TABLE_HEADER_HEIGHT + (conn.to.index * this.COLUMN_HEIGHT) + (this.COLUMN_HEIGHT / 2);

        const fromX = fromTable.position.x + this.TABLE_WIDTH;
        const toX = toTable.position.x;

        // Highlight if hovered during drag
        const isHovered = this.isDraggingOperation && this.hoveredConnection === conn;
        
        // Draw curved line
        this.ctx.strokeStyle = isHovered ? '#FCD34D' : this.getConnectionColor(conn.operation || conn.type);
        this.ctx.lineWidth = isHovered ? 6 : 3;
        this.ctx.globalAlpha = 1.0;  // Ensure full opacity
        this.ctx.beginPath();
        this.ctx.moveTo(fromX, fromY);

        const midX = (fromX + toX) / 2;
        this.ctx.bezierCurveTo(midX, fromY, midX, toY, toX, toY);
        this.ctx.stroke();
        this.ctx.globalAlpha = 1.0;  // Reset alpha
        
        // Store connection path for hit detection
        conn._path = { fromX, fromY, toX, toY, midX };

        // Draw arrow
        this.drawArrow(toX, toY, conn.type);
        
        // Calculate midpoint and draw operation label
        const midY = (fromY + toY) / 2;
        const operation = conn.operation;
        
        // If no operation set, show = (assignment)
        let operationSymbol, operationText;
        if (!operation || operation === 'copy') {
            operationSymbol = '=';
            operationText = 'COPY';
        } else {
            operationSymbol = {
                'sum': '+',
                'subtract': '-',
                'multiply': 'x',
                'divide': '/',
                'average': 'AVG'
            }[operation] || '=';
            operationText = operation.toUpperCase();
        }
        
        // Draw operation label background
        const labelWidth = 80;
        const labelHeight = 22;
        const labelX = midX - labelWidth / 2;
        const labelY = midY - labelHeight / 2;
        
        this.ctx.fillStyle = 'rgba(17, 24, 28, 0.95)';
        this.ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
        
        this.ctx.strokeStyle = isHovered ? '#FCD34D' : '#34B27B';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(labelX, labelY, labelWidth, labelHeight);
        
        // Draw operation text
        this.ctx.fillStyle = isHovered ? '#FCD34D' : '#34B27B';
        this.ctx.font = 'bold 11px Inter';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(`${operationSymbol} ${operationText}`, midX, midY);
    }

    /**
     * Draw connection preview while connecting
     */
    drawConnectionPreview() {
        const fromTable = this.tables.get(this.connectionStart.table);
        if (!fromTable) return;

        const fromY = fromTable.position.y + this.TABLE_HEADER_HEIGHT + 
                     (this.connectionStart.index * this.COLUMN_HEIGHT) + (this.COLUMN_HEIGHT / 2);
        const fromX = fromTable.position.x + this.TABLE_WIDTH;

        this.ctx.strokeStyle = this.getConnectionColor(this.connectionMode);
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([10, 5]);
        this.ctx.beginPath();
        this.ctx.moveTo(fromX, fromY);
        this.ctx.lineTo(this.mousePos.x, this.mousePos.y);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
    }

    /**
     * Draw arrow at end of connection
     */
    drawArrow(x, y, type) {
        const symbol = this.getOperationSymbol(type);
        this.ctx.font = '16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillStyle = this.getConnectionColor(type);
        this.ctx.fillText(symbol, x - 20, y);
    }

    /**
     * Get color for connection type
     */
    getConnectionColor(type) {
        const colors = {
            'add': '#10b981',        // green
            'sum': '#10b981',        // green (same as add)
            'subtract': '#ef4444',   // red
            'multiply': '#f59e0b',   // amber/orange
            'divide': '#8b5cf6',     // violet
            'average': '#06b6d4',    // cyan
            'copy': '#3b82f6',       // blue
            'graph': '#a855f7'       // purple
        };
        return colors[type] || 'rgba(255, 255, 255, 0.5)'; // White semi-transparent default
    }

    /**
     * Set connection mode (add, subtract, copy, graph)
     */
    setConnectionMode(mode) {
        this.connectionMode = mode;
        
        // Update button states
        ['add', 'subtract', 'copy', 'graph'].forEach(m => {
            const btn = document.getElementById(`mode-${m}`);
            if (btn) {
                if (m === mode) {
                    btn.style.opacity = '1';
                    btn.style.transform = 'scale(1.05)';
                } else {
                    btn.style.opacity = '0.6';
                    btn.style.transform = 'scale(1)';
                }
            }
        });
        
        this.updateStatus(`Mode: ${mode.toUpperCase()} • Click columns to connect`);
    }

    /**
     * Clear all connections
     */
    clearConnections() {
        if (this.connections.length === 0) return;
        
        const count = this.connections.length;
        this.connections = [];
        this.render();
        this.updateConnectionCount();
        this.updateStatus(`Cleared ${count} connection(s)`);
    }

    /**
     * Export selected table to CSV
     */
    exportSelectedTableToCSV() {
        // Close the download menu
        const downloadMenu = document.getElementById('designerDownloadMenu');
        if (downloadMenu) downloadMenu.classList.add('hidden');
        
        if (this.selectedTables.size === 0) {
            this.updateStatus('Select a table first (Shift+Click on header)');
            return;
        }

        // Export the first selected table
        const tableName = Array.from(this.selectedTables)[0];
        const table = this.tables.get(tableName);
        
        if (!table) {
            this.updateStatus('Table not found');
            return;
        }

        if (table.data.length === 0) {
            this.updateStatus('Table is empty');
            return;
        }

        // Convert table to CSV
        const columns = table.columns;
        const rows = table.data;
        
        // CSV header
        let csv = columns.join(',') + '\n';
        
        // CSV rows
        rows.forEach(row => {
            const values = columns.map(col => {
                const value = row[col];
                if (value === null || value === undefined) {
                    return '';
                }
                // Escape commas and quotes
                if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\n'))) {
                    return '"' + value.replace(/"/g, '""') + '"';
                }
                return value;
            });
            csv += values.join(',') + '\n';
        });

        // Download CSV
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${tableName}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.updateStatus(`Exported ${tableName} to CSV (${rows.length} rows)`);
    }

    /**
     * Execute all connections (perform operations)
     */
    async executeAllConnections() {
        if (this.connections.length === 0) {
            this.updateStatus('No connections to execute');
            return;
        }

        console.log(`Executing ${this.connections.length} connection(s)...`);
        this.updateStatus(`Executing ${this.connections.length} operation(s)...`);

        // Track modified tables for export
        this.modifiedTables = new Set();
        
        // Track operations for creating result tables
        this.resultTables = [];

        // Group connections by target table.column
        const targetGroups = new Map();
        
        for (const conn of this.connections) {
            const targetKey = `${conn.to.table}.${conn.to.column}`;
            if (!targetGroups.has(targetKey)) {
                targetGroups.set(targetKey, []);
            }
            targetGroups.get(targetKey).push(conn);
        }

        let successCount = 0;
        let errorCount = 0;

        // Execute each target group
        for (const [targetKey, connections] of targetGroups) {
            try {
                await this.executeConnectionGroup(connections);
                successCount += connections.length;
            } catch (error) {
                console.error(`Connection group failed:`, targetKey, error);
                errorCount += connections.length;
            }
        }

        // Create result tables for operations
        if (this.resultTables.length > 0) {
            const createdTables = [];
            for (const resultInfo of this.resultTables) {
                const resultTableName = `result_${resultInfo.operation}_${Date.now()}`;
                const resultTable = {
                    name: resultTableName,
                    columns: [resultInfo.column],
                    data: resultInfo.data,
                    position: {
                        x: resultInfo.targetTable.position.x + 350,
                        y: resultInfo.targetTable.position.y
                    },
                    rowCount: resultInfo.data.length
                };
                
                this.tables.set(resultTableName, resultTable);
                await this.db.loadJSON(resultTableName, resultInfo.data);
                this.modifiedTables.add(resultTableName);
                createdTables.push(resultTableName);
                console.log(`Created new result table: ${resultTableName}`);
            }
            
            if (createdTables.length > 0) {
                this.updateStatus(`Created ${createdTables.length} result table(s) • Results saved to GridDB`);
            }
        }
        
        const message = `Executed ${successCount}/${this.connections.length} operations`;
        console.log(message);
        this.updateStatus(message + (errorCount > 0 ? ` • ${errorCount} failed` : ' • Results saved to GridDB'));
        
        // Update column selectors and re-render to show updated data
        this.updateColumnSelectors();
        this.render();
        
        // Auto-export modified tables
        if (this.modifiedTables.size > 0) {
            console.log(`Auto-downloading ${this.modifiedTables.size} modified table(s)...`);
            for (const tableName of this.modifiedTables) {
                const table = this.tables.get(tableName);
                if (table) {
                    this.exportTableToCSV(tableName, table);
                }
            }
            this.updateStatus(message + ` • Downloaded ${this.modifiedTables.size} CSV files`);
        }
    }

    /**
     * Execute a group of connections that target the same column
     * Example: Table1.A → Table3.C and Table2.B → Table3.C
     * Result: Table3.C = operation(Table1.A, Table2.B)
     * 
     * Uses WebGPU compute shaders for binary operations (add, subtract, multiply, divide, average)
     * and GridDB SQL SELECT for copy operations.
     */
    async executeConnectionGroup(connections) {
        if (connections.length === 0) return;

        // Get target table and column
        const firstConn = connections[0];
        const targetTable = this.tables.get(firstConn.to.table);
        
        if (!targetTable) {
            throw new Error(`Target table ${firstConn.to.table} not found`);
        }

        const targetColumn = firstConn.to.column;
        
        // Get all source tables
        const sourceTables = connections.map(conn => {
            const table = this.tables.get(conn.from.table);
            if (!table) {
                throw new Error(`Source table ${conn.from.table} not found`);
            }
            return {
                table: table,
                tableName: conn.from.table,
                column: conn.from.column,
                connection: conn
            };
        });

        // Get operation from connection (or use 'copy' as default = assignment)
        const operation = firstConn.operation || 'copy';
        
        try {
            // Build SQL query based on operation type
            let sql;
            let resultTableName = `${targetTable.name}_updated`;
            
            // Store source column name for later mapping (GridDB doesn't support AS aliases well)
            let sourceColumnName = null;
            
            if (operation === 'copy' || !operation) {
                // Simple copy: SELECT first source column
                const source = sourceTables[0];
                sourceColumnName = source.column;
                sql = `
                    SELECT 
                        ${source.column}
                    FROM ${source.tableName}
                `;
            } else if (sourceTables.length === 1 && (operation === 'copy' || !operation)) {
                // Single source with copy - just copy the value
                const source = sourceTables[0];
                sourceColumnName = source.column;
                sql = `
                    SELECT 
                        ${source.column}
                    FROM ${source.tableName}
                `;
            } else if (sourceTables.length === 1 && operation !== 'copy') {
                // Single source with operation - use TARGET column as second operand
                // Example: source.X + target.Y → writes back into target.Y
                const source = sourceTables[0];
                
                console.log(`Executing GPU operation: ${source.tableName}.${source.column} ${operation} ${targetTable.name}.${targetColumn}`);
                
                const result = await this.db.applyBinaryOperation(
                    source.tableName,
                    source.column,
                    targetTable.name,
                    targetColumn,
                    operation
                );
                
                // Convert result array directly to target table
                targetTable.data = [];
                const resultData = [];
                for (let i = 0; i < result.length; i++) {
                    const newRow = {};
                    targetTable.columns.forEach(col => {
                        newRow[col] = null;
                    });
                    newRow[targetColumn] = result[i];
                    targetTable.data.push(newRow);
                    resultData.push({ [targetColumn]: result[i] });
                }
                
                // Store result for creating new table
                if (!this.resultTables) this.resultTables = [];
                this.resultTables.push({
                    operation: operation,
                    column: targetColumn,
                    data: resultData,
                    targetTable: targetTable,
                    sources: [source.tableName]
                });
                
                // Add target column if it doesn't exist
                if (!targetTable.columns.includes(targetColumn)) {
                    targetTable.columns.push(targetColumn);
                }
                
                // Save to GridDB
                if (this.db.tables.has(targetTable.name)) {
                    this.db.deleteTable(targetTable.name);
                }
                
                await this.db.loadJSON(targetTable.name, targetTable.data);
                if (this.modifiedTables) this.modifiedTables.add(targetTable.name);
                console.log(`Saved ${targetTable.name} to GridDB using GPU ${operation}`);
                
                const operationDesc = operation.toUpperCase();
                console.log(`${operationDesc}: ${source.tableName}.${source.column} ${operation} ${targetTable.name}.${targetColumn} → ${targetTable.name}.${targetColumn}`);
                
                // Log sample results
                const sampleSize = Math.min(3, targetTable.data.length);
                if (sampleSize > 0) {
                    console.log(`Sample results (first ${sampleSize} rows):`);
                    for (let i = 0; i < sampleSize; i++) {
                        const row = targetTable.data[i];
                        console.log(`   Row ${i+1}: ${targetColumn} = ${row[targetColumn]}`);
                    }
                }
                
                this.updateStatus(`Executed GPU ${operationDesc} → ${targetTable.name}.${targetColumn} • Saved to GridDB`);
                return; // Early return for GPU path
            } else if (sourceTables.length === 2) {
                // Two-table operation using WebGPU!
                const src1 = sourceTables[0];
                const src2 = sourceTables[1];
                
                // Use GridDB's GPU binary operation
                console.log(`Executing GPU operation: ${src1.tableName}.${src1.column} ${operation} ${src2.tableName}.${src2.column}`);
                
                const result = await this.db.applyBinaryOperation(
                    src1.tableName,
                    src1.column,
                    src2.tableName,
                    src2.column,
                    operation
                );
                
                // Convert result array directly to target table
                targetTable.data = [];
                const resultData = [];
                for (let i = 0; i < result.length; i++) {
                    const newRow = {};
                    targetTable.columns.forEach(col => {
                        newRow[col] = null;
                    });
                    newRow[targetColumn] = result[i];
                    targetTable.data.push(newRow);
                    resultData.push({ [targetColumn]: result[i] });
                }
                
                // Store result for creating new table
                if (!this.resultTables) this.resultTables = [];
                this.resultTables.push({
                    operation: operation,
                    column: targetColumn,
                    data: resultData,
                    targetTable: targetTable,
                    sources: [src1.tableName, src2.tableName]
                });
                
                // Add target column if it doesn't exist
                if (!targetTable.columns.includes(targetColumn)) {
                    targetTable.columns.push(targetColumn);
                }
                
                // Save to GridDB
                if (this.db.tables.has(targetTable.name)) {
                    this.db.deleteTable(targetTable.name);
                }
                
                await this.db.loadJSON(targetTable.name, targetTable.data);
                if (this.modifiedTables) this.modifiedTables.add(targetTable.name);
                console.log(`Saved ${targetTable.name} to GridDB using GPU ${operation}`);
                
                const sourceNames = sourceTables.map(src => `${src.tableName}.${src.column}`).join(` ${operation} `);
                const operationDesc = operation.toUpperCase();
                console.log(`${operationDesc}: ${sourceNames} → ${targetTable.name}.${targetColumn}`);
                
                // Log sample results
                const sampleSize = Math.min(3, targetTable.data.length);
                if (sampleSize > 0) {
                    console.log(`Sample results (first ${sampleSize} rows):`);
                    for (let i = 0; i < sampleSize; i++) {
                        const row = targetTable.data[i];
                        console.log(`   Row ${i+1}: ${targetColumn} = ${row[targetColumn]}`);
                    }
                }
                
                this.updateStatus(`Executed GPU ${operationDesc} → ${targetTable.name}.${targetColumn} • Saved to GridDB`);
                return; // Early return for GPU path
            } else if (sourceTables.length > 2) {
                // Multiple sources - just copy first source
                // (Full multi-table operations would need complex JOINs)
                const source = sourceTables[0];
                sourceColumnName = source.column;
                sql = `
                    SELECT 
                        ${source.column}
                    FROM ${source.tableName}
                `;
            }

            // For single-source operations (copy), use SQL
            console.log(`🔍 Executing SQL query for ${operation}:`, sql.trim());

            // Execute query using GridDB
            const result = await this.db.query(sql);
            
            // Update target table with results
            if (result && result.rows) {
                // Clear target table data
                targetTable.data = [];
                
                // Fill with new results - single source copy
                result.rows.forEach((row, index) => {
                    const newRow = {};
                    
                    // Copy all existing columns from target table
                    targetTable.columns.forEach(col => {
                        newRow[col] = null;
                    });
                    
                    // Get value from source column
                    // GridDB returns rows with original column names
                    let value = row[sourceColumnName];
                    
                    // If not found, try first column
                    if (value === undefined) {
                        const firstKey = Object.keys(row)[0];
                        value = row[firstKey];
                    }
                    
                    newRow[targetColumn] = value;
                    
                    // Add row number if not exists
                    if (!targetTable.columns.includes('row_num')) {
                        newRow.row_num = index + 1;
                    }
                    
                    targetTable.data.push(newRow);
                });
                
                // Add target column if it doesn't exist
                if (!targetTable.columns.includes(targetColumn)) {
                    targetTable.columns.push(targetColumn);
                }
                
                // Update the table in GridDB
                if (this.db.tables.has(targetTable.name)) {
                    this.db.deleteTable(targetTable.name);
                }
                
                await this.db.loadJSON(targetTable.name, targetTable.data);
                if (this.modifiedTables) this.modifiedTables.add(targetTable.name);
                console.log(`Saved ${targetTable.name} to GridDB using SQL SELECT`);
            }

            const sourceNames = sourceTables.map(src => `${src.tableName}.${src.column}`).join(' + ');
            const operationDesc = operation === 'copy' || !operation ? 'Copied' : operation.toUpperCase();
            console.log(`${operationDesc}: ${sourceNames} → ${targetTable.name}.${targetColumn}`);
            
            // Log sample results
            const sampleSize = Math.min(3, targetTable.data.length);
            if (sampleSize > 0) {
                console.log(`Sample results (first ${sampleSize} rows):`);
                for (let i = 0; i < sampleSize; i++) {
                    const row = targetTable.data[i];
                    console.log(`   Row ${i+1}: ${targetColumn} = ${row[targetColumn]}`);
                }
            }
            
            this.updateStatus(`Executed ${operationDesc} → ${targetTable.name}.${targetColumn} • Saved to GridDB`);
            
        } catch (error) {
            console.error(`SQL operation failed:`, error);
            throw error;
        }
    }

    /**
     * Execute single connection
     */
    async executeConnection(conn) {
        const sourceTable = this.tables.get(conn.from.table);
        const targetTable = this.tables.get(conn.to.table);

        if (!sourceTable || !targetTable) {
            throw new Error(`Table not found`);
        }

        switch (conn.type) {
            case 'add':
                await this.executeAdd(sourceTable, conn.from.column, targetTable, conn.to.column);
                break;
            case 'subtract':
                await this.executeSubtract(sourceTable, conn.from.column, targetTable, conn.to.column);
                break;
            case 'copy':
                await this.executeCopy(sourceTable, conn.from.column, targetTable, conn.to.column);
                break;
            case 'graph':
                await this.executeGraph(sourceTable, conn.from.column, targetTable, conn.to.column);
                break;
        }

        console.log(`${conn.type}: ${conn.from.table}.${conn.from.column} → ${conn.to.table}.${conn.to.column}`);
    }

    /**
     * Execute add operation
     */
    async executeAdd(sourceTable, sourceCol, targetTable, targetCol) {
        for (let i = 0; i < Math.min(sourceTable.data.length, targetTable.data.length); i++) {
            const srcVal = parseFloat(sourceTable.data[i][sourceCol]);
            const tgtVal = parseFloat(targetTable.data[i][targetCol]);
            
            if (!isNaN(srcVal) && !isNaN(tgtVal)) {
                targetTable.data[i][targetCol] = tgtVal + srcVal;
            }
        }
    }

    /**
     * Execute subtract operation
     */
    async executeSubtract(sourceTable, sourceCol, targetTable, targetCol) {
        for (let i = 0; i < Math.min(sourceTable.data.length, targetTable.data.length); i++) {
            const srcVal = parseFloat(sourceTable.data[i][sourceCol]);
            const tgtVal = parseFloat(targetTable.data[i][targetCol]);
            
            if (!isNaN(srcVal) && !isNaN(tgtVal)) {
                targetTable.data[i][targetCol] = tgtVal - srcVal;
            }
        }
    }

    /**
     * Execute copy operation
     */
    async executeCopy(sourceTable, sourceCol, targetTable, targetCol) {
        const newColName = `${targetCol}_from_${sourceTable.name}`;
        
        for (let i = 0; i < Math.min(sourceTable.data.length, targetTable.data.length); i++) {
            targetTable.data[i][newColName] = sourceTable.data[i][sourceCol];
        }
        
        if (!targetTable.columns.includes(newColName)) {
            targetTable.columns.push(newColName);
        }
    }

    /**
     * Execute graph operation (open visualization)
     */
    async executeGraph(sourceTable, sourceCol, targetTable, targetCol) {
        // Close designer and trigger visualization
        console.log(`GRAPH:APH: Opening graph: ${sourceTable.name}.${sourceCol} vs ${targetTable.name}.${targetCol}`);
        // This would integrate with your existing visualization system
        // Graph feature placeholder
        this.updateStatus(`GRAPH: ${sourceTable.name}.${sourceCol} vs ${targetTable.name}.${targetCol}`);
        console.log(`GRAPH:APH:APH:APHAPHAPH: Would visualize: ${sourceTable.name}.${sourceCol} vs ${targetTable.name}.${targetCol}`);
    }

    /**
     * Update status text
     */
    updateStatus(text) {
        const statusEl = document.getElementById('status-text');
        if (statusEl) {
            statusEl.textContent = text;
        }
    }

    /**
     * Update connection count in header
     */
    updateConnectionCount() {
        const modal = document.getElementById('relationship-designer-v2');
        if (modal) {
            const existing = modal.querySelector('.flex.items-center.gap-3');
            if (existing) {
                this.showDesignerModal(); // Refresh
            }
        }
    }

    /**
     * Upload more tables from the designer
     */
    uploadMoreTables() {
        // Trigger the main file input
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.click();
        } else {
            console.error('File input not found');
            this.updateStatus('File input not found');
        }
    }

    /**
     * Reload tables after new upload
     */
    reloadTables() {
        const existingTableNames = new Set(Array.from(this.tables.keys()));
        
        const newTables = [];
        for (const [tableName, tableData] of this.db.tables) {
            if (!existingTableNames.has(tableName)) {
                newTables.push(tableName);
            }
        }
        
        if (newTables.length === 0) return;

        
        // Find next available position (stack them to the right)
        let maxX = 50;
        for (const table of this.tables.values()) {
            maxX = Math.max(maxX, table.position.x + 320);
        }
        
        // Add new tables
        let offsetY = 50;
        for (const tableName of newTables) {
            const tableData = this.db.tables.get(tableName);
            if (!tableData) continue;
            
            // Handle different table structures (same logic as loadTables)
            let columns = [];
            let rowCount = 0;
            let data = [];
            
            if (tableData && typeof tableData === 'object') {
                if (tableData.columns && Array.isArray(tableData.columns)) {
                    columns = tableData.columns.map(col => col.name || col);
                    rowCount = tableData.rowCount || 0;
                    data = tableData.rows || [];
                } else if (Array.isArray(tableData)) {
                    data = tableData;
                    rowCount = tableData.length;
                    columns = tableData.length > 0 ? Object.keys(tableData[0]) : [];
                } else if (tableData.rows && Array.isArray(tableData.rows)) {
                    data = tableData.rows;
                    rowCount = tableData.rows.length;
                    columns = tableData.rows.length > 0 ? Object.keys(tableData.rows[0]) : [];
                }
            }
            
            this.tables.set(tableName, {
                name: tableName,
                columns: columns,
                rowCount: rowCount,
                position: { x: maxX, y: offsetY },
                data: data
            });
            
            offsetY += 100;
            if (offsetY > 500) {
                offsetY = 50;
                maxX += 320;
            }
        }
        
        // Update the table count in header
        const countEl = this.modal.querySelector('.text-\\[\\#34B27B\\]');
        if (countEl) {
            countEl.textContent = this.tables.size;
        }
        
        this.render();
        this.updateColumnSelectors(); // Update Input A/B dropdowns
        this.updateStatus(`Added ${newTables.length} new table(s) • ${this.tables.size} total`);
    }

    /**
     * Close the designer
     */
    close() {
        if (this.modal) {
            this.modal.remove();
            this.modal = null;
            this.canvas = null;
            this.ctx = null;
        }
        
        // Show the center panel again when closing
        const centerPanel = document.getElementById('centerPanel');
        const leftResizer = document.getElementById('leftResizer');
        if (centerPanel) {
            centerPanel.style.display = 'flex';
        }
        if (leftResizer) {
            leftResizer.style.display = 'block';
        }
        
        window.tableDesignerV2 = null;
    }
    
    setInputTable(input, tableName) {
        console.log(`setInputTable called: Input ${input} = ${tableName}`);
        if (input === 'A') {
            this.inputA.table = tableName;
            this.inputA.columns = [];
        } else {
            this.inputB.table = tableName;
            this.inputB.columns = [];
        }
        this.updateColumnSelectors();
        console.log(`Current state - Input A: ${this.inputA.table}, Input B: ${this.inputB.table}`);
    }
    
    addColumnToTable(tableName) {
        const table = this.tables.get(tableName);
        if (!table) return;
        
        // Auto-generate column name
        const columnName = `new_column_${table.columns.length + 1}`;
        
        // Add column to table
        table.columns.push(columnName);
        
        // Add empty values to existing data rows
        if (table.data && Array.isArray(table.data)) {
            table.data.forEach(row => {
                row[columnName] = null;
            });
        }
        
        this.updateStatus(`Added column "${columnName}" to ${tableName}`);
        this.updateColumnSelectors(); // Refresh UI
        this.render();
    }
    
    showOperationMenu(connection) {
        this.currentConnection = connection;
        
        // Show operation toolbar
        const bar = document.getElementById('connectionOperationBar');
        const info = document.getElementById('connectionInfo');
        
        if (bar && info) {
            info.textContent = `${connection.from.table}.${connection.from.column} ↔ ${connection.to.table}.${connection.to.column}`;
            bar.classList.remove('hidden');
        }
    }
    
    closeOperationBar() {
        const bar = document.getElementById('connectionOperationBar');
        if (bar) bar.classList.add('hidden');
        this.currentConnection = null;
    }
    
    async executeQuickJoin(joinType) {
        if (!this.currentConnection) {
            this.updateStatus('Please connect two columns first: Click & drag from one column to another');
            return;
        }
        
        this.currentConnection.type = joinType;
        await this.executeConnectionOperation(this.currentConnection);
        this.closeOperationBar();
    }
    
    async executeInputJoin(joinType) {
        // Close the JOIN menu
        const joinMenu = document.getElementById('designerJoinMenu');
        if (joinMenu) joinMenu.classList.add('hidden');
        
        console.log(`cuteInputJoin called with: ${joinType}`);
        console.log(`   Input A: ${this.inputA.table}, Input B: ${this.inputB.table}`);
        
        // Check if Input A and B are set
        if (!this.inputA.table || !this.inputB.table) {
            this.updateStatus('ase select both Input A and Input B tables first');
            return;
        }
        
        const tableA = this.tables.get(this.inputA.table);
        const tableB = this.tables.get(this.inputB.table);
        
        console.log(`   Table A found: ${!!tableA}, Table B found: ${!!tableB}`);
        
        if (!tableA || !tableB) {
            this.updateStatus('Selected tables not found');
            return;
        }
        
        // Find common columns between the two tables
        const commonColumns = tableA.columns.filter(col => tableB.columns.includes(col));
        
        console.log(`   Common columns:`, commonColumns);
        console.log(`   Table A columns:`, tableA.columns);
        console.log(`   Table B columns:`, tableB.columns);
        
        let joinColumnA, joinColumnB;
        if (commonColumns.length === 0) {
            // No common columns - show error
            this.updateStatus(`No common columns found between ${this.inputA.table} and ${this.inputB.table}. Use drag-to-connect for manual joins.`);
            console.warn('No common columns. Available columns:', {
                tableA: tableA.columns,
                tableB: tableB.columns
            });
            return;
        } else {
            // Use first common column automatically
            joinColumnA = commonColumns[0];
            joinColumnB = commonColumns[0];
            this.updateStatus(`Using join column: ${joinColumnA}`);
            console.log(`   Using join column: ${joinColumnA}`);
        }
        
        // Auto-generate result name
        const resultName = `${joinType}_${this.inputA.table}_${this.inputB.table}`;
        
        try {
            // Ensure tables are in database
            console.log(`   Ensuring tables in DB...`);
            await this.ensureTableInDB(this.inputA.table, tableA);
            await this.ensureTableInDB(this.inputB.table, tableB);
            
            // Build and execute query
            let query = '';
            switch(joinType) {
                case 'inner_join':
                    query = `SELECT * FROM ${this.inputA.table} INNER JOIN ${this.inputB.table} ON ${this.inputA.table}.${joinColumnA} = ${this.inputB.table}.${joinColumnB}`;
                    break;
                case 'left_join':
                    query = `SELECT * FROM ${this.inputA.table} LEFT JOIN ${this.inputB.table} ON ${this.inputA.table}.${joinColumnA} = ${this.inputB.table}.${joinColumnB}`;
                    break;
                case 'right_join':
                    query = `SELECT * FROM ${this.inputA.table} RIGHT JOIN ${this.inputB.table} ON ${this.inputA.table}.${joinColumnA} = ${this.inputB.table}.${joinColumnB}`;
                    break;
            }
            
            console.log(`   Executing query: ${query}`);
            const result = await this.db.query(query);
            console.log(`   Query result:`, result);
            console.log(`   Query result:`, result);
            const newColumns = result.columns || [];
            const newData = result.rows || [];
            
            console.log(`   Result: ${newData.length} rows, ${newColumns.length} columns`);
            
            // Create result table
            const maxX = Math.max(...Array.from(this.tables.values()).map(t => t.position.x), 0) + 350;
            const newTable = {
                name: resultName,
                columns: newColumns,
                rowCount: newData.length,
                position: { x: maxX, y: 100 },
                data: newData
            };
            this.tables.set(resultName, newTable);
            this.lastResultTable = resultName;
            
            console.log(`   Saving to database...`);
            // Save to database
            await this.saveTableToDatabase(resultName, newTable, true);
            
            console.log(`   Downloading CSV...`);
            // Download as CSV with custom name
            this.downloadTableAsCSV(resultName, newTable);
            
            console.log(`   Join complete!`);
            this.updateStatus(`${joinType.toUpperCase()} complete! Saved & downloaded: ${resultName} (${newData.length} rows)`);
            this.render();
            
        } catch (error) {
            console.error('Join operation failed:', error);
            console.error('Error stack:', error.stack);
            this.updateStatus(`Join failed: ${error.message}`);
        }
    }
    
    removeTable() {
        if (this.tables.size === 0) {
            this.updateStatus('WARNING tables available to remove.');
            return;
        }
        
        // Remove the last selected table (Input B, Input A, or most recent)
        let tableName = this.inputB.table || this.inputA.table;
        
        if (!tableName) {
            // Remove the last table added
            const tableList = Array.from(this.tables.keys());
            tableName = tableList[tableList.length - 1];
        }
        
        if (!tableName) {
            this.updateStatus('WARNING table selected to remove.');
            return;
        }
        
        // Remove table
        this.tables.delete(tableName);
        
        // Remove connections involving this table
        this.connections = this.connections.filter(conn => 
            conn.from.table !== tableName && conn.to.table !== tableName
        );
        
        // Clear selection states
        this.selectedTables.delete(tableName);
        if (this.inputA.table === tableName) {
            this.inputA = { table: null, columns: [] };
        }
        if (this.inputB.table === tableName) {
            this.inputB = { table: null, columns: [] };
        }
        
        this.updateStatus(`SUCCESSemoved table: ${tableName}`);
        this.updateColumnSelectors();
        this.render();
    }
    
    async executeColumnOperation(operation) {
        if (!this.currentConnection) return;
        
        this.currentConnection.type = operation;
        await this.executeConnectionOperation(this.currentConnection);
        this.closeOperationBar();
    }
    
    async dragColumnToNewTable() {
        if (!this.currentConnection) return;
        
        const fromTable = this.tables.get(this.currentConnection.from.table);
        const toTable = this.tables.get(this.currentConnection.to.table);
        
        if (!fromTable || !toTable) return;
        
        // Auto-generate table name
        const newTableName = `Combined_${fromTable.name}_${toTable.name}`;
        
        // Extract all data from both columns
        const fromData = fromTable.data || [];
        const toData = toTable.data || [];
        
        const fromColName = this.currentConnection.from.column;
        const toColName = this.currentConnection.to.column;
        
        // Create new table with both columns
        const newColumns = [fromColName, toColName];
        const newData = [];
        
        // Combine data (use longer dataset length)
        const maxLength = Math.max(fromData.length, toData.length);
        for (let i = 0; i < maxLength; i++) {
            const row = {};
            row[fromColName] = fromData[i] ? fromData[i][fromColName] : null;
            row[toColName] = toData[i] ? toData[i][toColName] : null;
            newData.push(row);
        }
        
        // Add new table to canvas
        const maxX = Math.max(...Array.from(this.tables.values()).map(t => t.position.x), 0) + 350;
        const newTable = {
            name: newTableName,
            columns: newColumns,
            rowCount: newData.length,
            position: { x: maxX, y: 100 },
            data: newData
        };
        this.tables.set(newTableName, newTable);
        
        // Track this as the last result for saving
        this.lastResultTable = newTableName;
        
        // Auto-save to GridDB
        await this.saveTableToDatabase(newTableName, newTable, true);
        
        // Auto-download result as CSV
        this.downloadTableAsCSV(newTableName, newTable);
        
        this.updateStatus(`SUCCESSreated, saved & downloaded: ${newTableName} with ${newData.length} rows`);
        this.closeOperationBar();
        this.render();
    }
    
    async executeConnectionOperation(connection) {
        const fromTable = this.tables.get(connection.from.table);
        const toTable = this.tables.get(connection.to.table);
        
        if (!fromTable || !toTable || !this.db) return;
        
        const fromCol = connection.from.column;
        const toCol = connection.to.column;
        const resultName = `Result_${connection.type}`;
        
        let query = '';
        
        try {
            switch(connection.type) {
                // Aggregate operations - sum entire columns
                case 'sum_all':
                    query = `SELECT SUM(${fromCol}) AS ${fromCol}_sum, SUM(${toCol}) AS ${toCol}_sum FROM ${connection.from.table}, ${connection.to.table}`;
                    break;
                case 'avg_all':
                    query = `SELECT AVG(${fromCol}) AS ${fromCol}_avg, AVG(${toCol}) AS ${toCol}_avg FROM ${connection.from.table}, ${connection.to.table}`;
                    break;
                case 'count_all':
                    query = `SELECT COUNT(${fromCol}) AS ${fromCol}_count, COUNT(${toCol}) AS ${toCol}_count FROM ${connection.from.table}, ${connection.to.table}`;
                    break;
                    
                // Row-by-row operations
                case 'sum':
                    query = `SELECT (${connection.from.table}.${fromCol} + ${connection.to.table}.${toCol}) AS ${resultName} FROM ${connection.from.table}, ${connection.to.table}`;
                    break;
                case 'subtract':
                    query = `SELECT (${connection.from.table}.${fromCol} - ${connection.to.table}.${toCol}) AS ${resultName} FROM ${connection.from.table}, ${connection.to.table}`;
                    break;
                case 'multiply':
                    query = `SELECT (${connection.from.table}.${fromCol} * ${connection.to.table}.${toCol}) AS ${resultName} FROM ${connection.from.table}, ${connection.to.table}`;
                    break;
                case 'divide':
                    query = `SELECT (${connection.from.table}.${fromCol} / ${connection.to.table}.${toCol}) AS ${resultName} FROM ${connection.from.table}, ${connection.to.table}`;
                    break;
                case 'average':
                    query = `SELECT ((${connection.from.table}.${fromCol} + ${connection.to.table}.${toCol}) / 2) AS ${resultName} FROM ${connection.from.table}, ${connection.to.table}`;
                    break;
                case 'concat':
                    query = `SELECT (${connection.from.table}.${fromCol} || ${connection.to.table}.${toCol}) AS ${resultName} FROM ${connection.from.table}, ${connection.to.table}`;
                    break;
                case 'transfer':
                    query = `SELECT ${connection.from.table}.${fromCol} AS ${resultName} FROM ${connection.from.table}`;
                    break;
                    
                // Join operations
                case 'inner_join':
                    query = `SELECT * FROM ${connection.from.table} INNER JOIN ${connection.to.table} ON ${connection.from.table}.${fromCol} = ${connection.to.table}.${toCol}`;
                    break;
                case 'left_join':
                    query = `SELECT * FROM ${connection.from.table} LEFT JOIN ${connection.to.table} ON ${connection.from.table}.${fromCol} = ${connection.to.table}.${toCol}`;
                    break;
                case 'right_join':
                    query = `SELECT * FROM ${connection.from.table} RIGHT JOIN ${connection.to.table} ON ${connection.from.table}.${fromCol} = ${connection.to.table}.${toCol}`;
                    break;
                case 'full_join':
                    query = `SELECT * FROM ${connection.from.table} FULL OUTER JOIN ${connection.to.table} ON ${connection.from.table}.${fromCol} = ${connection.to.table}.${toCol}`;
                    break;
                case 'join':
                    query = `SELECT * FROM ${connection.from.table} INNER JOIN ${connection.to.table} ON ${connection.from.table}.${fromCol} = ${connection.to.table}.${toCol}`;
                    break;
            }
            
            const result = await this.db.query(query);
            
            // Auto-generate result name
            const newTableName = `${connection.type}_${connection.from.table}_${connection.to.table}`;
            
            // Create new table with results
            const newColumns = result.columns || [resultName];
            const newData = result.rows || [];
            
            // Add new table to canvas
            const maxX = Math.max(...Array.from(this.tables.values()).map(t => t.position.x)) + 350;
            const newTable = {
                name: newTableName,
                columns: newColumns,
                rowCount: newData.length,
                position: { x: maxX, y: 100 },
                data: newData
            };
            this.tables.set(newTableName, newTable);
            
            // Track this as the last result for saving
            this.lastResultTable = newTableName;
            
            // Auto-save to GridDB
            await this.saveTableToDatabase(newTableName, newTable, true);
            
            // Auto-download result as CSV
            this.downloadTableAsCSV(newTableName, newTable);
            
            this.updateStatus(`SUCCESS ${connection.type.toUpperCase()} complete! Saved & downloaded: ${newTableName} (${newData.length} rows)`);
            this.render();
            
        } catch (error) {
            console.error('Operation failed:', error);
            this.updateStatus(`ERROROperation failed: ${error.message}`);
        }
    }
    
    async ensureTableInDB(tableName, tableData) {
        if (!this.db || !tableData) return;
        
        try {
            const existingTables = await this.db.listTables();
            if (!existingTables || !existingTables.includes(tableName)) {
                console.log(`SAVING:VING-saving source table: ${tableName}`);
                await this.db.createTable(tableName, tableData.columns);
                
                if (tableData.data && tableData.data.length > 0) {
                    for (const row of tableData.data) {
                        await this.db.insert(tableName, row);
                    }
                }
            }
        } catch (error) {
            console.error(`Failed to ensure table ${tableName} in DB:`, error);
        }
    }
    
    async saveTableToDatabase(tableName, tableData, isAutoSave = false) {
        if (!this.db || !tableData || !tableData.data) return;
        
        try {
            const existingTables = await this.db.listTables();
            
            // Check if table exists - always overwrite
            if (existingTables && existingTables.includes(tableName)) {
                try {
                    await this.db.query(`DROP TABLE ${tableName}`);
                } catch (e) {
                    // Table might not exist, continue
                }
            }
            
            // Create and populate table
            await this.db.createTable(tableName, tableData.columns);
            
            for (const row of tableData.data) {
                await this.db.insert(tableName, row);
            }
            
            console.log(`SAVED: Saved table to database: ${tableName} (${tableData.rowCount} rows)`);
            
        } catch (error) {
            console.error('Save failed:', error);
            if (!isAutoSave) {
                this.updateStatus(`ERROR:RRORiled to save: ${error.message}`);
            }
        }
    }
    
    downloadTableAsCSV(tableName, tableData) {
        if (!tableData || !tableData.data || tableData.data.length === 0) {
            console.log('No data to download');
            return;
        }
        
        try {
            // Create CSV content
            const headers = tableData.columns.join(',');
            const rows = tableData.data.map(row => {
                return tableData.columns.map(col => {
                    const value = row[col];
                    // Escape values containing commas or quotes
                    if (value === null || value === undefined) return '';
                    const str = String(value);
                    if (str.includes(',') || str.includes('"') || str.includes('\n')) {
                        return `"${str.replace(/"/g, '""')}"`;
                    }
                    return str;
                }).join(',');
            });
            
            const csv = [headers, ...rows].join('\n');
            
            // Create download link
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            
            // Use provided table name for filename
            link.setAttribute('href', url);
            link.setAttribute('download', `${tableName}.csv`);
            link.style.visibility = 'hidden';
            
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            console.log(`DOWNLOADED Downloaded: ${tableName}.csv`);
            
        } catch (error) {
            console.error('Download failed:', error);
            this.updateStatus(`ERRORnload failed: ${error.message}`);
        }
    }
    
    async saveLastResultToDatabase() {
        if (!this.lastResultTable) {
            this.updateStatus('WARNING No result table to save. Please perform an operation first.');
            return;
        }
        
        const table = this.tables.get(this.lastResultTable);
        if (!table || !table.data) {
            this.updateStatus('WARNINGesult table not found or has no data.');
            return;
        }
        
        try {
            // Auto-save with current name (no prompt)
            await this.saveTableToDatabase(this.lastResultTable, table, false);
            this.downloadTableAsCSV(this.lastResultTable, table);
            this.updateStatus(`SUCCESS Saved & downloaded: ${this.lastResultTable} (${table.rowCount} rows)`);
            
        } catch (error) {
            console.error('Save failed:', error);
            this.updateStatus(`ERRORailed to save: ${error.message}`);
        }
    }
    
    updateColumnSelectors() {
        const tableASelect = document.getElementById('inputTableA');
        const tableBSelect = document.getElementById('inputTableB');
        
        if (tableASelect && tableBSelect) {
            const options = '<option value="">Select table...</option>' + 
                Array.from(this.tables.keys()).map(name => `<option value="${name}">${name}</option>`).join('');
            tableASelect.innerHTML = options;
            tableBSelect.innerHTML = options;
            
            if (this.inputA.table) tableASelect.value = this.inputA.table;
            if (this.inputB.table) tableBSelect.value = this.inputB.table;
        }
        // Update all tables columns view - show only selected tables or all if none selected
        const allTablesDiv = document.getElementById('allTablesColumns');
        if (allTablesDiv && this.tables.size > 0) {
            const tablesToShow = this.selectedTables.size > 0 
                ? Array.from(this.selectedTables).map(name => [name, this.tables.get(name)]).filter(([, t]) => t)
                : Array.from(this.tables.entries());
            
            if (tablesToShow.length === 0) {
                allTablesDiv.innerHTML = '<div class="text-xs text-white/30 col-span-3">No tables selected. Shift+Click table headers on canvas to select them.</div>';
            } else {
                allTablesDiv.innerHTML = tablesToShow.map(([tableName, table]) => `
                    <div class="border border-white/10 rounded p-3">
                        <div class="text-sm text-[#34B27B] font-bold mb-2">${tableName}</div>
                        <div class="space-y-1 max-h-48 overflow-y-auto">
                            ${table.columns.map(col => `
                                <label class="flex items-center gap-2 px-2 py-1 hover:bg-white/5 rounded cursor-pointer">
                                    <input type="checkbox" value="${tableName}.${col}" 
                                           onchange="window.tableDesignerV2.toggleColumn('${tableName}', '${col}', this.checked)">
                                    <span class="text-xs text-white/70">${col}</span>
                                </label>
                            `).join('')}
                        </div>
                    </div>
                `).join('');
            }
        }
        
        this.updateSelectedColumnsList();
    }
    
    toggleColumn(tableName, columnName, checked) {
        const key = `${tableName}.${columnName}`;
        if (checked) {
            this.selectedColumns.push({ table: tableName, column: columnName, key: key });
            this.selectedCanvasColumns.add(key);
        } else {
            this.selectedColumns = this.selectedColumns.filter(c => c.key !== key);
            this.selectedCanvasColumns.delete(key);
        }
        this.updateSelectedColumnsList();
        this.render(); // Update canvas highlighting
    }
    
    syncCheckboxesWithCanvas() {
        // Update checkboxes to match canvas selection
        const checkboxes = document.querySelectorAll('#allTablesColumns input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            const key = checkbox.value;
            checkbox.checked = this.selectedCanvasColumns.has(key);
        });
    }
    
    updateSelectedColumnsList() {
        const selectedDiv = document.getElementById('selectedColumns');
        if (selectedDiv) {
            if (this.selectedColumns.length === 0) {
                selectedDiv.textContent = 'No columns selected';
            } else {
                selectedDiv.textContent = this.selectedColumns.map(c => c.key).join(', ');
            }
        }
    }
    
    async executeOperation() {
        if (this.selectedColumns.length === 0) {
            this.updateStatus('WARNING:RNING:RNING:RNINGRNING: Please select at least one column');
            return;
        }
        
        const operationType = document.getElementById('operationType').value;
        const whereClause = document.getElementById('whereClause').value;
        
        let query = '';
        
        if (operationType === 'select') {
            // Custom SELECT with selected columns
            const columns = this.selectedColumns.map(c => c.key).join(', ');
            const tables = [...new Set(this.selectedColumns.map(c => c.table))].join(', ');
            query = `SELECT ${columns} FROM ${tables}`;
            if (whereClause) query += ` WHERE ${whereClause}`;
        } else if (operationType === 'join' && this.selectedColumns.length >= 2) {
            // JOIN on selected columns
            const col1 = this.selectedColumns[0];
            const col2 = this.selectedColumns[1];
            query = `SELECT * FROM ${col1.table} INNER JOIN ${col2.table} ON ${col1.key} = ${col2.key}`;
            if (whereClause) query += ` WHERE ${whereClause}`;
        } else if (this.selectedColumns.length >= 2) {
            // Arithmetic operations
            const col1 = this.selectedColumns[0];
            const col2 = this.selectedColumns[1];
            const op = operationType === 'add' ? '+' : operationType === 'subtract' ? '-' : 
                       operationType === 'multiply' ? '*' : operationType === 'divide' ? '/' : '||';
            const resultCol = `result_${operationType}`;
            query = `SELECT ${col1.table}.*, (${col1.key} ${op} ${col2.key}) AS ${resultCol} FROM ${col1.table}, ${col2.table}`;
            if (whereClause) query += ` WHERE ${whereClause}`;
        }
        
        try {
            const result = await this.db.query(query);
            document.getElementById('operationOutput').textContent = `SUCCESS:UCCESS: Success!\nRows: ${result.rowCount}\nQuery: ${query}`;
            document.getElementById('operationResult').classList.remove('hidden');
            this.updateStatus(`SUCCESS:UCCESS: Operation completed! ${result.rowCount} rows generated.`);
        } catch (error) {
            console.error('Operation failed:', error);
            this.updateStatus(`ERROR:RROR:RROR:RRORRROR: Operation failed: ${error.message}`);
        }
    }
    
    toggleJoinMenu() {
        const menu = document.getElementById('designerJoinMenu');
        if (menu) {
            menu.classList.toggle('hidden');
        }
    }
    
    toggleDownloadMenu() {
        const menu = document.getElementById('designerDownloadMenu');
        if (menu) {
            menu.classList.toggle('hidden');
        }
    }
    
    toggleOperationsMenu() {
        const menu = document.getElementById('designerOperationsMenu');
        if (menu) {
            menu.classList.toggle('hidden');
        }
    }
    
    exportConnectionsAsJSON() {
        // Close the download menu
        const menu = document.getElementById('designerDownloadMenu');
        if (menu) menu.classList.add('hidden');
        
        const connectionsData = {
            tables: Array.from(this.tables.keys()),
            connections: this.connections.map(conn => ({
                from: conn.from,
                to: conn.to,
                type: conn.type || 'relationship'
            })),
            timestamp: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(connectionsData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `griddb-connections-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.updateStatus('SUCCESS Connections exported as JSON');
    }
    
    async quickCalc(operation) {
        const columns = Array.from(this.selectedCanvasColumns);
        
        if (columns.length === 0) {
            this.updateStatus('WARNINGlease select at least one column');
            return;
        }
        
        try {
            let result;
            
            if (operation === 'percentage' && columns.length === 2) {
                // Calculate percentage: col1/col2 * 100
                const [col1, col2] = columns.map(c => {
                    const [table, column] = c.split('.');
                    return { table, column };
                });
                
                const tableData1 = this.tables.get(col1.table);
                const tableData2 = this.tables.get(col2.table);
                
                if (!tableData1 || !tableData2) {
                    throw new Error('Table not found');
                }
                
                // Calculate percentage for each row
                const percentages = tableData1.data.map((row, idx) => {
                    const val1 = parseFloat(row[col1.column]) || 0;
                    const val2 = parseFloat(tableData2.data[idx]?.[col2.column]) || 1;
                    return (val1 / val2 * 100).toFixed(2);
                });
                
                const avgPercentage = percentages.reduce((a, b) => a + parseFloat(b), 0) / percentages.length;
                result = `${avgPercentage.toFixed(2)}% (avg)`;
                
            } else if (operation === 'add_two' && columns.length === 2) {
                // Add two columns row by row
                const [col1, col2] = columns.map(c => {
                    const [table, column] = c.split('.');
                    return { table, column };
                });
                
                const tableData1 = this.tables.get(col1.table);
                const tableData2 = this.tables.get(col2.table);
                
                if (!tableData1 || !tableData2) {
                    throw new Error('Table not found');
                }
                
                const sums = tableData1.data.map((row, idx) => {
                    const val1 = parseFloat(row[col1.column]) || 0;
                    const val2 = parseFloat(tableData2.data[idx]?.[col2.column]) || 0;
                    return val1 + val2;
                });
                
                const totalSum = sums.reduce((a, b) => a + b, 0);
                result = `${totalSum.toLocaleString()} (total)`;
                
            } else if ((operation === 'subtract' || operation === 'multiply' || operation === 'divide') && columns.length === 2) {
                // Two-column operations
                const [col1, col2] = columns.map(c => {
                    const [table, column] = c.split('.');
                    return { table, column };
                });
                
                const tableData1 = this.tables.get(col1.table);
                const tableData2 = this.tables.get(col2.table);
                
                if (!tableData1 || !tableData2) {
                    throw new Error('Table not found');
                }
                
                const results = tableData1.data.map((row, idx) => {
                    const val1 = parseFloat(row[col1.column]) || 0;
                    const val2 = parseFloat(tableData2.data[idx]?.[col2.column]) || 0;
                    switch (operation) {
                        case 'subtract': return val1 - val2;
                        case 'multiply': return val1 * val2;
                        case 'divide': return val2 !== 0 ? val1 / val2 : 0;
                        default: return 0;
                    }
                });
                
                const total = results.reduce((a, b) => a + b, 0);
                result = `${total.toFixed(2)} (total)`;
                
            } else {
                // Single column operations
                const [table, column] = columns[0].split('.');
                const tableData = this.tables.get(table);
                
                if (!tableData) {
                    throw new Error('Table not found');
                }
                
                const values = tableData.data.map(row => parseFloat(row[column]) || 0);
                
                switch (operation) {
                    case 'sum':
                        result = values.reduce((a, b) => a + b, 0).toLocaleString();
                        break;
                    case 'average':
                        result = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2);
                        break;
                    case 'count':
                        result = values.length.toLocaleString();
                        break;
                    case 'min':
                        result = Math.min(...values).toLocaleString();
                        break;
                    case 'max':
                        result = Math.max(...values).toLocaleString();
                        break;
                    default:
                        result = 'N/A';
                }
            }
            
            this.updateStatus(`SUCCESS Calculation complete: ${result}`);
            
        } catch (error) {
            console.error('Calculation failed:', error);
            this.updateStatus(`ERRORCalculation failed: ${error.message}`);
        }
    }
    
    /**
     * Start dragging an operation from Quick Analytics panel
     */
    onOperationDragStart(event, operation) {
        this.isDraggingOperation = true;
        this.draggedOperation = operation;
        event.dataTransfer.effectAllowed = 'copy';
        event.dataTransfer.setData('operation', operation);
        this.updateStatus(`Drag ${operation} operation onto a connection line...`);
    }
    
    /**
     * Get connection at mouse position
     */
    getConnectionAtPosition(x, y) {
        const threshold = 15; // Distance threshold for hit detection
        
        for (const conn of this.connections) {
            if (!conn._path) continue;
            
            const { fromX, fromY, toX, toY, midX } = conn._path;
            
            // Check distance to bezier curve (simplified check at multiple points)
            for (let t = 0; t <= 1; t += 0.05) {
                const t2 = t * t;
                const t3 = t2 * t;
                const mt = 1 - t;
                const mt2 = mt * mt;
                const mt3 = mt2 * mt;
                
                // Bezier curve formula
                const px = fromX * mt3 + 3 * midX * mt2 * t + 3 * midX * mt * t2 + toX * t3;
                const py = fromY * mt3 + 3 * fromY * mt2 * t + 3 * toY * mt * t2 + toY * t3;
                
                const distance = Math.sqrt(Math.pow(x - px, 2) + Math.pow(y - py, 2));
                if (distance < threshold) {
                    return conn;
                }
            }
        }
        
        return null;
    }
    
    /**
     * Execute operation on connected columns and create result table
     */
    async executeOperationOnConnection(conn, operation) {
        // Just assign the operation to the connection
        // Don't execute it immediately - wait for Execute All button
        conn.operation = operation;
        
        this.updateStatus(`SUCCESS: Set ${operation.toUpperCase()} operation on ${conn.from.table}.${conn.from.column} → ${conn.to.table}.${conn.to.column}`);
        this.render();
    }
    
    /**
     * Execute connection and create new result table (called by clicking execute button)
     */
    async executeConnectionToNewTable(conn) {
        try {
            const fromTable = this.tables.get(conn.from.table);
            const toTable = this.tables.get(conn.to.table);
            
            if (!fromTable || !toTable) {
                throw new Error('Table not found');
            }
            
            // Get operation (default to copy if not set)
            const operation = conn.operation || 'copy';
            
            // Get column data
            const fromData = fromTable.data || [];
            const toData = toTable.data || [];
            const maxRows = Math.max(fromData.length, toData.length);
            
            // Perform operation row by row
            const resultData = [];
            
            if (operation === 'copy') {
                // Just copy both columns without operation
                for (let i = 0; i < maxRows; i++) {
                    const val1 = fromData[i]?.[conn.from.column];
                    const val2 = toData[i]?.[conn.to.column];
                    
                    resultData.push({
                        row: i + 1,
                        [`${conn.from.table}_${conn.from.column}`]: val1,
                        [`${conn.to.table}_${conn.to.column}`]: val2
                    });
                }
            } else {
                // Perform mathematical operation
                for (let i = 0; i < maxRows; i++) {
                    const val1 = parseFloat(fromData[i]?.[conn.from.column]) || 0;
                    const val2 = parseFloat(toData[i]?.[conn.to.column]) || 0;
                    let resultValue;
                    
                    switch (operation) {
                        case 'sum':
                            resultValue = val1 + val2;
                            break;
                        case 'subtract':
                            resultValue = val1 - val2;
                            break;
                        case 'multiply':
                            resultValue = val1 * val2;
                            break;
                        case 'divide':
                            resultValue = val2 !== 0 ? val1 / val2 : 0;
                            break;
                        case 'average':
                            resultValue = (val1 + val2) / 2;
                            break;
                        default:
                            resultValue = val1;
                    }
                    
                    resultData.push({
                        row: i + 1,
                        [`${conn.from.table}_${conn.from.column}`]: val1,
                        [`${conn.to.table}_${conn.to.column}`]: val2,
                        result: resultValue
                    });
                }
            }
            
            // Create result table name
            let counter = 1;
            let tableName = operation === 'copy' ? 'merged_table' : `${operation}_result`;
            while (this.tables.has(tableName)) {
                tableName = operation === 'copy' ? `merged_table_${counter}` : `${operation}_result_${counter}`;
                counter++;
            }
            
            // Determine columns based on operation
            let columns;
            if (operation === 'copy') {
                columns = ['row', `${conn.from.table}_${conn.from.column}`, `${conn.to.table}_${conn.to.column}`];
            } else {
                columns = ['row', `${conn.from.table}_${conn.from.column}`, `${conn.to.table}_${conn.to.column}`, 'result'];
            }
            
            // Create new table with results
            const newTable = {
                name: tableName,
                columns: columns,
                data: resultData,
                rowCount: resultData.length,
                position: {
                    x: Math.max(fromTable.position.x, toTable.position.x) + 320,
                    y: Math.min(fromTable.position.y, toTable.position.y)
                }
            };
            
            this.tables.set(tableName, newTable);
            
            // Add to database if available
            if (this.db) {
                await this.db.loadJSON(tableName, resultData);
            }
            
            const operationDesc = operation === 'copy' ? 'Merged columns' : `${operation.toUpperCase()} operation (row-by-row)`;
            this.updateStatus(`SUCCESS ${operationDesc} complete! Created ${tableName} with ${resultData.length} rows`);
            this.updateColumnSelectors();
            this.render();
            
        } catch (error) {
            console.error('Failed to create result table:', error);
            this.updateStatus(`ERRORFailed to create result table: ${error.message}`);
        }
    }
    
    // ========================================
    // CREATE TABLE FUNCTIONALITY
    // ========================================
    
    showCreateTableModal() {
        const modal = document.getElementById('createTableModalDesigner');
        if (!modal) return;
        
        modal.classList.remove('hidden');
        
        // Clear previous inputs
        document.getElementById('newTableNameDesigner').value = '';
        document.getElementById('initialDataDesigner').value = '';
        document.getElementById('columnsListDesigner').innerHTML = '';
        
        // Add initial column rows
        this.addColumnRow();
        this.addColumnRow();
        
        this.updateStatus('Creating new table...');
    }
    
    closeCreateTableModal() {
        const modal = document.getElementById('createTableModalDesigner');
        if (modal) {
            modal.classList.add('hidden');
        }
        this.updateStatus('Ready');
    }
    
    addColumnRow() {
        const columnsList = document.getElementById('columnsListDesigner');
        if (!columnsList) return;
        
        const rowId = `col-designer-${Date.now()}-${Math.random()}`;
        
        const row = document.createElement('div');
        row.id = rowId;
        row.className = 'flex gap-2';
        row.innerHTML = `
            <input 
                type="text" 
                placeholder="Column name"
                class="flex-1 bg-white/5 border border-white/10 rounded px-2 py-1.5 text-xs text-white placeholder-white/30 focus:outline-none focus:border-[#34B27B]"
                data-column-name
            />
            <select 
                class="bg-white/5 border border-white/10 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-[#34B27B]"
                data-column-type
            >
                <option value="string">Text</option>
                <option value="number">Number</option>
            </select>
            <button 
                onclick="document.getElementById('${rowId}').remove()"
                class="bg-red-500/20 hover:bg-red-500/30 text-red-400 px-2 py-1.5 rounded text-xs font-bold transition-all"
            >
                ✕
            </button>
        `;
        
        columnsList.appendChild(row);
    }
    
    async createNewTable() {
        if (!this.db) {
            this.updateStatus('ERROR Database not initialized');
            return;
        }

        const tableName = document.getElementById('newTableNameDesigner').value.trim();
        if (!tableName) {
            this.updateStatus('WARNINGlease enter a table name');
            return;
        }

        // Validate table name
        if (!/^[a-zA-Z0-9_-]+$/.test(tableName)) {
            this.updateStatus('ERROR Table name can only contain letters, numbers, underscore, and hyphen');
            return;
        }

        if (this.db.tables.has(tableName)) {
            this.updateStatus(`WARNINGable "${tableName}" already exists`);
            return;
        }

        // Get columns
        const columnRows = document.querySelectorAll('#columnsListDesigner > div');
        const columns = [];
        
        for (const row of columnRows) {
            const nameInput = row.querySelector('[data-column-name]');
            const typeSelect = row.querySelector('[data-column-type]');
            
            const colName = nameInput.value.trim();
            const colType = typeSelect.value;
            
            if (colName) {
                columns.push({ name: colName, type: colType });
            }
        }

        if (columns.length === 0) {
            this.updateStatus('WARNING Please add at least one column');
            return;
        }

        // Get initial data
        const initialDataText = document.getElementById('initialDataDesigner').value.trim();
        let data = [];
        
        if (initialDataText) {
            try {
                data = JSON.parse(initialDataText);
                if (!Array.isArray(data)) {
                    throw new Error('Data must be a JSON array');
                }
            } catch (error) {
                this.updateStatus(`ERRORnvalid JSON data: ${error.message}`);
                return;
            }
        } else {
            // Create empty table with one sample row
            data = [{}];
            columns.forEach(col => {
                data[0][col.name] = col.type === 'number' ? 0 : '';
            });
        }

        // Create table
        try {
            this.updateStatus(`Creating table "${tableName}"...`);
            
            await this.db.loadJSON(tableName, data);
            
            const table = this.db.tables.get(tableName);
            
            // Add table to designer canvas
            const x = 100 + (this.tables.size * 50);
            const y = 100 + (this.tables.size * 50);
            
            this.tables.set(tableName, {
                name: tableName,
                columns: columns.map(c => c.name),
                data: data,
                rowCount: data.length,
                position: { x, y }
            });
            
            // Update input dropdowns
            this.updateColumnSelectors();
            
            // Close modal
            this.closeCreateTableModal();
            
            // Render canvas
            this.render();
            
            this.updateStatus(`SUCCESS Table "${tableName}" created with ${data.length} rows and ${columns.length} columns`);
            
            console.log(`SUCCESS Created table "${tableName}": ${table.rowCount} rows, ${columns.length} columns`);
            
        } catch (error) {
            console.error('Table creation error:', error);
            this.updateStatus(`ERRORError creating table: ${error.message}`);
        }
    }

    /**
     * Export all tables to CSV files
     */
    exportAllTablesToCSV() {
        // Close the download menu
        const downloadMenu = document.getElementById('designerDownloadMenu');
        if (downloadMenu) downloadMenu.classList.add('hidden');
        
        let exportedCount = 0;
        
        for (const [name, table] of this.tables) {
            try {
                this.exportTableToCSV(name, table);
                exportedCount++;
            } catch (error) {
                console.error(`Failed to export ${name}:`, error);
            }
        }
        
        this.updateStatus(`SUCCESSxported ${exportedCount} tables as CSV files`);
    }

    /**
     * Export a single table to CSV
     */
    exportTableToCSV(tableName, table) {
        if (!table || !table.data || table.data.length === 0) {
            console.warn(`Table ${tableName} has no data to export`);
            return;
        }

        // Get columns
        const columns = table.columns || Object.keys(table.data[0]);
        
        // Build CSV content
        let csvContent = '';
        
        // Header row
        csvContent += columns.join(',') + '\n';
        
        // Data rows
        for (const row of table.data) {
            const values = columns.map(col => {
                let value = row[col];
                
                // Handle null/undefined
                if (value === null || value === undefined) {
                    return '';
                }
                
                // Convert to string and escape if needed
                value = String(value);
                
                // Quote if contains comma, newline, or quote
                if (value.includes(',') || value.includes('\n') || value.includes('"')) {
                    value = '"' + value.replace(/"/g, '""') + '"';
                }
                
                return value;
            });
            
            csvContent += values.join(',') + '\n';
        }
        
        // Create blob and download
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        link.setAttribute('href', url);
        link.setAttribute('download', `${tableName}.csv`);
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        console.log(`DOWNLOADEDownloaded ${tableName}.csv (${table.data.length} rows)`);
    }
}
