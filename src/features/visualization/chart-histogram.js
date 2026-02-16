/**
 * Histogram Chart Renderer
 * 
 * Renders histogram visualizations for numeric data distribution
 * Uses HTML/CSS for smooth animations and interactive bars
 */

export class HistogramChart {
    constructor() {
        this.defaultBins = 30;
    }

    /**
     * Render histogram chart
     * @param {HTMLElement} modal - Modal container element
     * @param {Array} data - Dataset to visualize
     * @param {Array} columns - Column configuration
     */
    async render(modal, data, columns) {
        const canvas = modal.querySelector('#viz-canvas');
        
        // Extract column name properly
        const column = columns[0]?.name || columns[0];
        
        console.log(' Histogram for column:', column);
        
        // Get numeric values
        const values = data.map(d => parseFloat(d[column])).filter(v => !isNaN(v));
        
        if (values.length === 0) {
            throw new Error(`No numeric values found in column ${column}`);
        }

        const min = Math.min(...values);
        const max = Math.max(...values);
        const numBins = this.defaultBins;
        const binWidth = (max - min) / numBins;

        // Create bins
        const bins = Array(numBins).fill(0);
        values.forEach(v => {
            const binIndex = Math.min(Math.floor((v - min) / binWidth), numBins - 1);
            bins[binIndex]++;
        });

        const maxBinCount = Math.max(...bins);

        // Display as HTML bars
        canvas.style.display = 'none';
        const container = canvas.parentElement;
        
        container.innerHTML = this.generateHistogramHTML(bins, min, max, binWidth, maxBinCount);

        // Calculate and display stats
        this.displayStats(modal, values, min, max, numBins, column);
        
        console.log(' Histogram rendered');
    }

    /**
     * Generate HTML for histogram bars
     */
    generateHistogramHTML(bins, min, max, binWidth, maxBinCount) {
        return `
            <div class="w-full max-w-6xl">
                <div class="flex items-end gap-1 h-96 px-8">
                    ${bins.map((count, i) => {
                        const height = (count / maxBinCount) * 100;
                        const binStart = min + i * binWidth;
                        const binEnd = binStart + binWidth;
                        return `
                        <div class="flex-1 flex flex-col items-center justify-end group cursor-pointer">
                            <div class="w-full bg-gradient-to-t from-[#34B27B] to-[#4ecdc4] rounded-t transition-all hover:opacity-80"
                                 style="height: ${height}%"
                                 title="${binStart.toFixed(1)} - ${binEnd.toFixed(1)}: ${count} items">
                            </div>
                            <div class="text-[9px] text-white/30 mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                ${count}
                            </div>
                        </div>
                        `;
                    }).join('')}
                </div>
                <div class="flex justify-between text-xs text-white/50 px-8 mt-4">
                    <span>${min.toFixed(1)}</span>
                    <span>${((min + max) / 2).toFixed(1)}</span>
                    <span>${max.toFixed(1)}</span>
                </div>
            </div>
        `;
    }

    /**
     * Calculate and display statistics
     */
    displayStats(modal, values, min, max, numBins, column) {
        const controlsDiv = modal.querySelector('#viz-controls');
        
        // Calculate stats locally
        const sum = values.reduce((a, b) => a + b, 0);
        const avg = sum / values.length;
        const sortedValues = [...values].sort((a, b) => a - b);
        const median = sortedValues[Math.floor(sortedValues.length / 2)];
        
        controlsDiv.innerHTML = `
            <div class="text-sm flex gap-6">
                <div><span class="text-white/50">Range:</span> <span class="text-[#34B27B]">${min.toFixed(1)} - ${max.toFixed(1)}</span></div>
                <div><span class="text-white/50">Mean:</span> <span class="text-[#34B27B]">${avg.toFixed(2)}</span></div>
                <div><span class="text-white/50">Median:</span> <span class="text-[#34B27B]">${median.toFixed(2)}</span></div>
                <div><span class="text-white/50">Bins:</span> <span class="text-[#34B27B]">${numBins}</span></div>
                <div><span class="text-white/50">Count:</span> <span class="text-[#34B27B]">${values.length.toLocaleString()}</span></div>
            </div>
        `;
    }
}
