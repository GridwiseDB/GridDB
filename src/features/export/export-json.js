/**
 * JSON Export Feature
 * 
 * Exports query results to JSON format
 */

export class JSONExporter {
    /**
     * Export data to JSON file
     * @param {Object} results - Query results with rows array
     */
    export(results) {
        if (!results || !results.rows || results.rows.length === 0) {
            throw new Error('No results to export.');
        }

        try {
            const content = JSON.stringify(results.rows, null, 2);
            this.downloadFile(content, 'application/json', 'json');
            
            console.log(`Exported ${results.rows.length} rows to JSON`);
            return true;
        } catch (error) {
            console.error('JSON export error:', error);
            throw new Error('Export failed: ' + error.message);
        }
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
