/**
 * Enhanced Table Relationship Designer with Operations
 * 
 * Features:
 * - Visual canvas for connecting tables
 * - Left sidebar showing all loaded tables
 * - Operations: Add, Subtract, Merge, Copy columns
 * - Automatic data transformation and updates
 * - Time series support
 */

export class TableRelationshipDesigner {
    constructor() {
        this.tables = new Map();
        this.connections = [];
        this.operations = []; // Defined operations to execute
        this.canvas = null;
        this.ctx = null;
        this.isDragging = false;
        this.draggedTable = null;
        this.isConnecting = false;
        this.connectionStart = null;
        this.currentMousePos = { x: 0, y: 0 };
        this.db = null; // Reference to the database
        this.mode = 'move';
    }

    /**
     * Open the relationship designer with loaded tables
     */
    open(dbTables, dbInstance) {
        this.db = dbInstance;
        const tableList = Array.from(dbTables.entries());
        
        if (tableList.length === 0) {
            throw new Error('No tables loaded. Please load at least one table first.');
        }

        this.loadTables(tableList);
        this.showDesignerModal();
        this.render();
    }

    /**
     * Load tables and auto-position them
     */
    loadTables(tableList) {
        this.tables.clear();
        
        // Auto-position tables in a grid
        const spacing = 350;
        const startX = 100;
        const startY = 100;
        const perRow = Math.ceil(Math.sqrt(tableList.length));
        
        tableList.forEach(([tableName, table], index) => {
            const row = Math.floor(index / perRow);
            const col = index % perRow;
            
            this.tables.set(tableName, {
                name: tableName,
                columns: table.columns.map(col => ({
                    name: col.name,
                    type: col.type
                })),
                rowCount: table.rowCount,
                position: {
                    x: startX + (col * spacing),
                    y: startY + (row * spacing)
                }
            });
        });
    }

    /**
     * Show the designer modal
     */
    showDesignerModal() {
        const modal = document.createElement('div');
        modal.id = 'relationship-designer-modal';
        modal.className = 'fixed inset-0 bg-black/95 backdrop-blur-sm z-[9999] flex flex-col';
        
        modal.innerHTML = `
            <div class="flex-1 flex flex-col overflow-hidden">
                <!-- Header -->
                <div class="h-16 border-b border-white/10 flex items-center justify-between px-6 bg-[#0a0a0a]">
                    <div class="flex items-center gap-4">
                        <div class="w-10 h-10 bg-gradient-to-br from-[#34B27B] to-[#4ECDC4] rounded-xl flex items-center justify-center text-xl">
                            🔗
                        </div>
                        <div>
                            <h2 class="text-xl font-black outfit text-white">Table Relationship Designer</h2>
                            <p class="text-xs text-white/50">Connect tables and define data flows</p>
                        </div>
                    </div>
                    
                    <div class="flex items-center gap-3">
                        <button onclick="window.relationshipDesigner.clearConnections()" 
                                class="px-4 py-2 bg-white/5 hover:bg-white/10 text-white/70 text-sm font-bold rounded-lg transition-all">
                            Clear Connections
                        </button>
                        <button onclick="window.relationshipDesigner.exportSchema()" 
                                class="px-4 py-2 bg-gradient-to-r from-[#34B27B] to-[#4ECDC4] hover:from-[#2a9463] hover:to-[#3ba89f] text-white text-sm font-bold rounded-lg transition-all">
                            Export Schema
                        </button>
                        <button onclick="window.relationshipDesigner.close()" 
                                class="text-white/50 hover:text-white text-3xl leading-none transition-colors">×</button>
                    </div>
                </div>

                <!-- Toolbar -->
                <div class="h-14 border-b border-white/10 flex items-center px-6 gap-4 bg-[#11181C]">
                    <div class="flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-lg">
                        <span class="text-xs text-white/50">Mode:</span>
                        <button id="mode-move" onclick="window.relationshipDesigner.setMode('move')" 
                                class="px-3 py-1 text-xs font-bold bg-[#34B27B] text-black rounded transition-all">
                            ✋ Move
                        </button>
                        <button id="mode-connect" onclick="window.relationshipDesigner.setMode('connect')" 
                                class="px-3 py-1 text-xs font-bold bg-white/10 text-white/70 rounded transition-all">
                            🔗 Connect
                        </button>
                    </div>
                    
                    <div class="h-8 w-px bg-white/10"></div>
                    
                    <div class="text-xs text-white/40">
                        <span class="font-bold text-white/60">Instructions:</span>
                        <span id="mode-instructions">Drag tables to reposition</span>
                    </div>
                    
                    <div class="ml-auto text-xs text-white/50">
                        Tables: <span class="text-[#34B27B] font-bold">${this.tables.size}</span> | 
                        Connections: <span id="connection-count" class="text-[#34B27B] font-bold">0</span>
                    </div>
                </div>

                <!-- Canvas -->
                <div class="flex-1 overflow-hidden relative bg-[#050505]">
                    <canvas id="relationship-canvas" class="absolute inset-0 w-full h-full cursor-move"></canvas>
                </div>

                <!-- Legend -->
                <div class="h-12 border-t border-white/10 flex items-center justify-between px-6 bg-[#0a0a0a] text-xs">
                    <div class="flex items-center gap-6">
                        <div class="flex items-center gap-2">
                            <div class="w-3 h-3 rounded-full bg-[#34B27B]"></div>
                            <span class="text-white/50">Data Flow</span>
                        </div>
                        <div class="flex items-center gap-2">
                            <div class="w-3 h-3 rounded-full bg-blue-500"></div>
                            <span class="text-white/50">Foreign Key</span>
                        </div>
                        <div class="flex items-center gap-2">
                            <div class="w-3 h-3 rounded-full bg-purple-500"></div>
                            <span class="text-white/50">Calculation</span>
                        </div>
                    </div>
                    <div class="text-white/40 italic">
                        💡 Click columns in Connect mode to create relationships
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        
        // Initialize canvas
        this.canvas = document.getElementById('relationship-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Set canvas size
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        
        // Add event listeners
        this.attachEventListeners();
        
        // Store reference globally
        window.relationshipDesigner = this;
    }

    /**
     * Resize canvas to fill container
     */
    resizeCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
        this.render();
    }

    /**
     * Attach mouse event listeners
     */
    attachEventListeners() {
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('click', (e) => this.handleClick(e));
    }

    /**
     * Set interaction mode
     */
    setMode(mode) {
        this.mode = mode;
        
        // Update UI
        document.getElementById('mode-move').className = 
            mode === 'move' 
                ? 'px-3 py-1 text-xs font-bold bg-[#34B27B] text-black rounded transition-all'
                : 'px-3 py-1 text-xs font-bold bg-white/10 text-white/70 rounded transition-all';
        
        document.getElementById('mode-connect').className = 
            mode === 'connect' 
                ? 'px-3 py-1 text-xs font-bold bg-[#34B27B] text-black rounded transition-all'
                : 'px-3 py-1 text-xs font-bold bg-white/10 text-white/70 rounded transition-all';
        
        // Update instructions
        const instructions = mode === 'move' 
            ? 'Drag tables to reposition them on canvas'
            : 'Click on a column, then click another column to connect';
        document.getElementById('mode-instructions').textContent = instructions;
        
        // Update cursor
        this.canvas.style.cursor = mode === 'move' ? 'move' : 'crosshair';
    }

    /**
     * Handle mouse down
     */
    handleMouseDown(e) {
        if (this.mode !== 'move') return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Check if clicking on a table
        for (const [tableName, table] of this.tables) {
            const tableBox = this.getTableBounds(table);
            
            if (x >= tableBox.x && x <= tableBox.x + tableBox.width &&
                y >= tableBox.y && y <= tableBox.y + tableBox.height) {
                this.isDragging = true;
                this.draggedTable = tableName;
                this.dragOffset = {
                    x: x - table.position.x,
                    y: y - table.position.y
                };
                break;
            }
        }
    }

    /**
     * Handle mouse move
     */
    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        this.currentMousePos = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
        
        if (this.isDragging && this.draggedTable) {
            const table = this.tables.get(this.draggedTable);
            table.position.x = this.currentMousePos.x - this.dragOffset.x;
            table.position.y = this.currentMousePos.y - this.dragOffset.y;
            this.render();
        }
        
        // Show preview line when connecting
        if (this.isConnecting) {
            this.render();
        }
    }

    /**
     * Handle mouse up
     */
    handleMouseUp(e) {
        this.isDragging = false;
        this.draggedTable = null;
    }

    /**
     * Handle click (for connections)
     */
    handleClick(e) {
        if (this.mode !== 'connect') return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Check if clicking on a column
        const clickedColumn = this.getColumnAtPosition(x, y);
        
        if (clickedColumn) {
            if (!this.isConnecting) {
                // Start connection
                this.isConnecting = true;
                this.connectionStart = clickedColumn;
                console.log('Connection started:', clickedColumn);
            } else {
                // Complete connection
                if (this.connectionStart.tableName !== clickedColumn.tableName) {
                    this.addConnection(
                        this.connectionStart.tableName,
                        this.connectionStart.columnName,
                        clickedColumn.tableName,
                        clickedColumn.columnName
                    );
                }
                this.isConnecting = false;
                this.connectionStart = null;
                this.render();
            }
        }
    }

    /**
     * Get column at mouse position
     */
    getColumnAtPosition(x, y) {
        for (const [tableName, table] of this.tables) {
            const tableBox = this.getTableBounds(table);
            const headerHeight = 50;
            const columnHeight = 28;
            
            if (x >= tableBox.x && x <= tableBox.x + tableBox.width) {
                table.columns.forEach((col, index) => {
                    const colY = tableBox.y + headerHeight + (index * columnHeight);
                    
                    if (y >= colY && y < colY + columnHeight) {
                        return { tableName, columnName: col.name };
                    }
                });
            }
        }
        return null;
    }

    /**
     * Add a connection between columns
     */
    addConnection(fromTable, fromColumn, toTable, toColumn) {
        // Check if connection already exists
        const exists = this.connections.some(conn => 
            conn.fromTable === fromTable && conn.fromColumn === fromColumn &&
            conn.toTable === toTable && conn.toColumn === toColumn
        );
        
        if (exists) {
            alert('Connection already exists!');
            return;
        }
        
        // Prompt for operation type
        const operation = prompt(
            'Define the relationship:\n\n' +
            '1. "copy" - Copy values from source to destination\n' +
            '2. "sum" - Add source values to destination\n' +
            '3. "lookup" - Use as foreign key lookup\n' +
            '4. "calculate" - Custom calculation\n\n' +
            'Enter operation type:',
            'lookup'
        );
        
        if (!operation) return;
        
        this.connections.push({
            fromTable,
            fromColumn,
            toTable,
            toColumn,
            operation: operation.toLowerCase(),
            color: this.getConnectionColor(operation)
        });
        
        // Update connection count
        document.getElementById('connection-count').textContent = this.connections.length;
        
        console.log('Connection added:', { fromTable, fromColumn, toTable, toColumn, operation });
    }

    /**
     * Get connection color based on operation
     */
    getConnectionColor(operation) {
        const colors = {
            'copy': '#34B27B',
            'sum': '#FFD700',
            'lookup': '#4ECDC4',
            'calculate': '#9B59B6',
            'default': '#34B27B'
        };
        return colors[operation] || colors.default;
    }

    /**
     * Clear all connections
     */
    clearConnections() {
        if (confirm('Clear all connections?')) {
            this.connections = [];
            document.getElementById('connection-count').textContent = '0';
            this.render();
        }
    }

    /**
     * Get table bounding box
     */
    getTableBounds(table) {
        const width = 250;
        const headerHeight = 50;
        const columnHeight = 28;
        const height = headerHeight + (table.columns.length * columnHeight) + 20;
        
        return {
            x: table.position.x,
            y: table.position.y,
            width,
            height
        };
    }

    /**
     * Render the entire canvas
     */
    render() {
        if (!this.ctx) return;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid pattern
        this.drawGrid();
        
        // Draw connections first (behind tables)
        this.drawConnections();
        
        // Draw preview connection line
        if (this.isConnecting && this.connectionStart) {
            this.drawPreviewConnection();
        }
        
        // Draw tables
        for (const [tableName, table] of this.tables) {
            this.drawTable(table);
        }
    }

    /**
     * Draw grid background
     */
    drawGrid() {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        this.ctx.lineWidth = 1;
        
        const gridSize = 50;
        
        // Vertical lines
        for (let x = 0; x < this.canvas.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y < this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
    }

    /**
     * Draw a table card
     */
    drawTable(table) {
        const bounds = this.getTableBounds(table);
        const ctx = this.ctx;
        
        // Shadow
        ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
        ctx.shadowBlur = 15;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 4;
        
        // Table background
        ctx.fillStyle = '#11181C';
        ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
        
        // Border
        ctx.strokeStyle = 'rgba(52, 178, 123, 0.3)';
        ctx.lineWidth = 2;
        ctx.strokeRect(bounds.x, bounds.y, bounds.width, bounds.height);
        
        ctx.shadowColor = 'transparent';
        
        // Header
        ctx.fillStyle = 'rgba(52, 178, 123, 0.1)';
        ctx.fillRect(bounds.x, bounds.y, bounds.width, 50);
        
        // Table name
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 16px Inter';
        ctx.textAlign = 'left';
        ctx.fillText(table.name, bounds.x + 15, bounds.y + 25);
        
        // Row count
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.font = '11px Inter';
        ctx.fillText(`${table.rowCount.toLocaleString()} rows`, bounds.x + 15, bounds.y + 40);
        
        // Columns
        table.columns.forEach((col, index) => {
            const colY = bounds.y + 50 + (index * 28);
            
            // Column background (alternating)
            if (index % 2 === 0) {
                ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
                ctx.fillRect(bounds.x, colY, bounds.width, 28);
            }
            
            // Column name
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.font = '13px Inter';
            ctx.textAlign = 'left';
            ctx.fillText(col.name, bounds.x + 15, colY + 18);
            
            // Column type badge
            ctx.fillStyle = col.type === 'number' ? 'rgba(52, 178, 123, 0.2)' : 'rgba(78, 205, 196, 0.2)';
            ctx.fillRect(bounds.x + bounds.width - 65, colY + 6, 50, 16);
            
            ctx.fillStyle = col.type === 'number' ? '#34B27B' : '#4ECDC4';
            ctx.font = 'bold 10px Inter';
            ctx.textAlign = 'center';
            ctx.fillText(col.type, bounds.x + bounds.width - 40, colY + 17);
        });
    }

    /**
     * Draw connections between tables
     */
    drawConnections() {
        this.connections.forEach(conn => {
            const fromTable = this.tables.get(conn.fromTable);
            const toTable = this.tables.get(conn.toTable);
            
            if (!fromTable || !toTable) return;
            
            // Find column positions
            const fromPos = this.getColumnPosition(fromTable, conn.fromColumn);
            const toPos = this.getColumnPosition(toTable, conn.toColumn);
            
            // Draw curved line
            this.drawConnection(fromPos, toPos, conn.color || '#34B27B', conn.operation);
        });
    }

    /**
     * Get column center position
     */
    getColumnPosition(table, columnName) {
        const bounds = this.getTableBounds(table);
        const columnIndex = table.columns.findIndex(col => col.name === columnName);
        
        const x = bounds.x + bounds.width;
        const y = bounds.y + 50 + (columnIndex * 28) + 14;
        
        return { x, y };
    }

    /**
     * Draw a connection line
     */
    drawConnection(from, to, color, operation) {
        const ctx = this.ctx;
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.setLineDash([]);
        
        // Draw curved line
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        
        const controlPointOffset = Math.abs(to.x - from.x) / 2;
        ctx.bezierCurveTo(
            from.x + controlPointOffset, from.y,
            to.x - controlPointOffset, to.y,
            to.x, to.y
        );
        
        ctx.stroke();
        
        // Draw arrowhead
        const angle = Math.atan2(to.y - from.y, to.x - from.x);
        const arrowSize = 10;
        
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(to.x, to.y);
        ctx.lineTo(
            to.x - arrowSize * Math.cos(angle - Math.PI / 6),
            to.y - arrowSize * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
            to.x - arrowSize * Math.cos(angle + Math.PI / 6),
            to.y - arrowSize * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();
        
        // Draw operation label
        const midX = (from.x + to.x) / 2;
        const midY = (from.y + to.y) / 2;
        
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(midX - 30, midY - 10, 60, 20);
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(midX - 30, midY - 10, 60, 20);
        
        ctx.fillStyle = color;
        ctx.font = 'bold 11px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(operation, midX, midY + 4);
    }

    /**
     * Draw preview connection while connecting
     */
    drawPreviewConnection() {
        if (!this.connectionStart) return;
        
        const fromTable = this.tables.get(this.connectionStart.tableName);
        if (!fromTable) return;
        
        const fromPos = this.getColumnPosition(fromTable, this.connectionStart.columnName);
        
        this.ctx.strokeStyle = 'rgba(52, 178, 123, 0.5)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        
        this.ctx.beginPath();
        this.ctx.moveTo(fromPos.x, fromPos.y);
        this.ctx.lineTo(this.currentMousePos.x, this.currentMousePos.y);
        this.ctx.stroke();
        
        this.ctx.setLineDash([]);
    }

    /**
     * Export schema as JSON
     */
    exportSchema() {
        const schema = {
            tables: Array.from(this.tables.entries()).map(([name, table]) => ({
                name,
                columns: table.columns,
                rowCount: table.rowCount,
                position: table.position
            })),
            connections: this.connections,
            exportedAt: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(schema, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `griddb-schema-${Date.now()}.json`;
        link.click();
        
        console.log('📄 Schema exported:', schema);
    }

    /**
     * Close the designer
     */
    close() {
        const modal = document.getElementById('relationship-designer-modal');
        if (modal) {
            modal.remove();
        }
        window.relationshipDesigner = null;
    }
}
