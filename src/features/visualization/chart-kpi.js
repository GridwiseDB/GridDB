/**
 * KPI Card Renderer
 * 
 * Displays key performance indicators and statistical summaries
 * Shows min, max, average, median, standard deviation, and count
 */

export class KPICard {
    /**
     * Render KPI card
     * @param {HTMLElement} modal - Modal container element
     * @param {Array} data - Dataset to visualize
     * @param {Array} columns - Column configuration
     */
    async render(modal, data, columns) {
        const canvas = modal.querySelector('#viz-canvas');
        
        // Extract column name
        const column = columns[0]?.name || columns[0];
        
        console.log('💹 KPI Card for column:', column);
        
        // Get numeric values
        const values = data.map(d => parseFloat(d[column])).filter(v => !isNaN(v));
        
        if (values.length === 0) {
            throw new Error(`No numeric values found in column ${column}`);
        }
        
        // Compute statistics
        const stats = this.calculateStats(values);
        
        // Display as HTML
        canvas.style.display = 'none';
        const container = canvas.parentElement;
        
        container.innerHTML = this.generateKPIHTML(stats);
        
        // Update controls
        modal.querySelector('#viz-controls').innerHTML = `
            <div class="text-sm text-white/50">
                Analyzed ${column} • ${stats.count.toLocaleString()} values
            </div>
        `;
        
        console.log('✅ KPI card rendered');
    }

    /**
     * Calculate all statistics
     */
    calculateStats(values) {
        const min = Math.min(...values);
        const max = Math.max(...values);
        const sum = values.reduce((a, b) => a + b, 0);
        const avg = sum / values.length;
        const sortedValues = [...values].sort((a, b) => a - b);
        const median = sortedValues[Math.floor(sortedValues.length / 2)];
        
        // Calculate standard deviation
        const variance = values.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);

        return {
            min,
            max,
            avg,
            median,
            stdDev,
            count: values.length,
            range: max - min
        };
    }

    /**
     * Generate KPI card HTML
     */
    generateKPIHTML(stats) {
        return `
            <div class="grid grid-cols-3 gap-6 max-w-6xl w-full">
                <div class="bg-[#11181C] rounded-xl border border-white/10 p-8 hover:border-[#34B27B]/30 transition-all">
                    <div class="text-sm text-white/50 uppercase tracking-wider mb-2">Minimum</div>
                    <div class="text-4xl font-bold outfit text-[#34B27B] glow-text">${stats.min.toLocaleString()}</div>
                </div>
                <div class="bg-[#11181C] rounded-xl border border-white/10 p-8 hover:border-[#34B27B]/30 transition-all">
                    <div class="text-sm text-white/50 uppercase tracking-wider mb-2">Maximum</div>
                    <div class="text-4xl font-bold outfit text-[#34B27B] glow-text">${stats.max.toLocaleString()}</div>
                </div>
                <div class="bg-[#11181C] rounded-xl border border-white/10 p-8 hover:border-[#34B27B]/30 transition-all">
                    <div class="text-sm text-white/50 uppercase tracking-wider mb-2">Average</div>
                    <div class="text-4xl font-bold outfit text-[#34B27B] glow-text">${stats.avg.toFixed(2)}</div>
                </div>
                <div class="bg-[#11181C] rounded-xl border border-white/10 p-8 hover:border-[#34B27B]/30 transition-all">
                    <div class="text-sm text-white/50 uppercase tracking-wider mb-2">Median</div>
                    <div class="text-4xl font-bold outfit text-[#34B27B] glow-text">${stats.median.toFixed(2)}</div>
                </div>
                <div class="bg-[#11181C] rounded-xl border border-white/10 p-8 hover:border-[#34B27B]/30 transition-all">
                    <div class="text-sm text-white/50 uppercase tracking-wider mb-2">Std Dev</div>
                    <div class="text-4xl font-bold outfit text-[#34B27B] glow-text">${stats.stdDev.toFixed(2)}</div>
                </div>
                <div class="bg-[#11181C] rounded-xl border border-white/10 p-8 hover:border-[#34B27B]/30 transition-all">
                    <div class="text-sm text-white/50 uppercase tracking-wider mb-2">Count</div>
                    <div class="text-4xl font-bold outfit text-[#34B27B] glow-text">${stats.count.toLocaleString()}</div>
                </div>
                <div class="col-span-3 bg-[#11181C] rounded-xl border border-white/10 p-8">
                    <div class="text-sm text-white/50 uppercase tracking-wider mb-4">Range</div>
                    <div class="flex items-center gap-4">
                        <div class="flex-1">
                            <div class="h-3 bg-gradient-to-r from-[#34B27B]/20 via-[#34B27B]/40 to-[#34B27B]/20 rounded-full relative">
                                <div class="absolute left-0 top-1/2 -translate-y-1/2 w-4 h-4 bg-[#34B27B] rounded-full shadow-lg shadow-[#34B27B]/50"></div>
                                <div class="absolute right-0 top-1/2 -translate-y-1/2 w-4 h-4 bg-[#34B27B] rounded-full shadow-lg shadow-[#34B27B]/50"></div>
                            </div>
                        </div>
                        <div class="text-2xl font-bold text-[#34B27B] mono">${stats.range.toLocaleString()}</div>
                    </div>
                </div>
            </div>
        `;
    }
}
