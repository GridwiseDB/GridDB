/**
 * CSV Export Feature
 * 
 * Exports query results to CSV format with proper escaping
 */

export class CSVExporter {
    /**
     * Export data to CSV file
     * @param {Object} results - Query results with rows array
     */
    export(results) {
        if (!results || !results.rows || results.rows.length === 0) {
            throw new Error('No results to export. Run a query first.');
        }

        try {
            const csvContent = this.generateCSV(results.rows);
            this.downloadFile(csvContent, 'text/csv', 'csv');
            
            console.log(`✅ Exported ${results.rows.length} rows to CSV`);
            return true;
        } catch (error) {
            console.error('CSV export error:', error);
            throw new Error('Export failed: ' + error.message);
        }
    }

    /**
     * Generate CSV content from rows
     */
    generateCSV(rows) {
        const columns = Object.keys(rows[0]);
        const csvRows = [columns.join(',')];
        
        rows.forEach(row => {
            csvRows.push(columns.map(col => {
                const val = row[col];
                // Escape values that contain commas or quotes
                if (typeof val === 'string' && (val.includes(',') || val.includes('"'))) {
                    return `"${val.replace(/"/g, '""')}"`;
                }
                return val;
            }).join(','));
        });
        
        return csvRows.join('\n');
    }

    /**
     * Download file to browser
     */
    downloadFile(content, mimeType, extension) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `query_results_${Date.now()}.${extension}`;
        a.click();
        URL.revokeObjectURL(url);
    }
}
