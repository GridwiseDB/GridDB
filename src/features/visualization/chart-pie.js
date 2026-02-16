/**
 * Pie Chart Renderer
 * 
 * Renders pie chart visualizations for categorical data distribution
 * Uses Canvas 2D for high-quality rendering with colors and legends
 */

export class PieChart {
    constructor() {
        this.maxCategories = 10;
    }

    /**
     * Render pie chart
     * @param {HTMLElement} modal - Modal container element
     * @param {Array} data - Dataset to visualize
     * @param {Array} columns - Column configuration
     */
    async render(modal, data, columns) {
        const canvas = modal.querySelector('#viz-canvas');
        const ctx = canvas.getContext('2d');
        
        // Extract column name
        const column = columns[0]?.name || columns[0];
        
        console.log('🥧 Pie Chart for column:', column);
        
        // Aggregate data by category
        const categoryMap = this.aggregateData(data, column);
        
        // Sort and limit to top categories
        const sortedCategories = Array.from(categoryMap.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, this.maxCategories);
        
        if (sortedCategories.length === 0) {
            throw new Error(`No data found in column ${column}`);
        }

        const total = sortedCategories.reduce((sum, [_, count]) => sum + count, 0);
        
        // Setup canvas
        canvas.width = 1200;
        canvas.height = 700;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Generate colors
        const colors = this.generateColors(sortedCategories.length);
        
        // Draw chart
        this.drawPieSlices(ctx, sortedCategories, total, colors);
        this.drawLegend(ctx, sortedCategories, total, colors);
        this.drawTitle(ctx, canvas, column);
        
        // Display stats
        this.displayStats(modal, sortedCategories, total);
        
        console.log('✅ Pie chart rendered');
    }

    /**
     * Aggregate data by category
     */
    aggregateData(data, column) {
        const categoryMap = new Map();
        data.forEach(row => {
            const category = String(row[column]);
            if (categoryMap.has(category)) {
                categoryMap.set(category, categoryMap.get(category) + 1);
            } else {
                categoryMap.set(category, 1);
            }
        });
        return categoryMap;
    }

    /**
     * Generate color palette
     */
    generateColors(count) {
        return Array.from({ length: count }, (_, i) => {
            const hue = (i / count) * 360;
            return `hsl(${hue}, 70%, 60%)`;
        });
    }

    /**
     * Draw pie slices
     */
    drawPieSlices(ctx, categories, total, colors) {
        const centerX = 400;
        const centerY = 350;
        const radius = 250;
        let currentAngle = -Math.PI / 2;
        
        categories.forEach(([category, count], i) => {
            const sliceAngle = (count / total) * Math.PI * 2;
            
            // Draw slice
            ctx.fillStyle = colors[i];
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
            ctx.closePath();
            ctx.fill();
            
            // Draw border
            ctx.strokeStyle = '#0a0a0a';
            ctx.lineWidth = 3;
            ctx.stroke();
            
            // Add percentage label
            const percentage = ((count / total) * 100).toFixed(1);
            if (parseFloat(percentage) > 3) {
                const midAngle = currentAngle + sliceAngle / 2;
                const labelRadius = radius * 0.7;
                const labelX = centerX + Math.cos(midAngle) * labelRadius;
                const labelY = centerY + Math.sin(midAngle) * labelRadius;
                
                ctx.fillStyle = '#fff';
                ctx.font = 'bold 16px Inter';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(`${percentage}%`, labelX, labelY);
            }
            
            currentAngle += sliceAngle;
        });
    }

    /**
     * Draw legend
     */
    drawLegend(ctx, categories, total, colors) {
        const legendX = 750;
        const legendY = 100;
        const legendItemHeight = 45;
        
        // Legend title
        ctx.font = 'bold 14px Inter';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.textAlign = 'left';
        ctx.fillText('Categories', legendX, legendY - 30);
        
        // Legend items
        categories.forEach(([category, count], i) => {
            const y = legendY + i * legendItemHeight;
            
            // Color box
            ctx.fillStyle = colors[i];
            ctx.fillRect(legendX, y, 30, 30);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1;
            ctx.strokeRect(legendX, y, 30, 30);
            
            // Category name
            ctx.fillStyle = '#fff';
            ctx.font = '14px Inter';
            const displayCategory = category.length > 25 ? category.substring(0, 22) + '...' : category;
            ctx.fillText(displayCategory, legendX + 40, y + 12);
            
            // Count and percentage
            ctx.fillStyle = '#34B27B';
            ctx.font = 'bold 14px monospace';
            const percentage = ((count / total) * 100).toFixed(1);
            ctx.fillText(`${count.toLocaleString()} (${percentage}%)`, legendX + 40, y + 27);
        });
    }

    /**
     * Draw title
     */
    drawTitle(ctx, canvas, column) {
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 24px Outfit';
        ctx.textAlign = 'center';
        ctx.fillText(`Distribution: ${column}`, canvas.width / 2, 40);
    }

    /**
     * Display statistics
     */
    displayStats(modal, categories, total) {
        const controlsDiv = modal.querySelector('#viz-controls');
        controlsDiv.innerHTML = `
            <div class="text-sm flex gap-6">
                <div><span class="text-white/50">Total Items:</span> <span class="text-[#34B27B]">${total.toLocaleString()}</span></div>
                <div><span class="text-white/50">Categories:</span> <span class="text-[#34B27B]">${categories.length}</span></div>
                ${categories.length === this.maxCategories ? '<div class="text-white/40 text-xs italic">(Showing top 10 categories)</div>' : ''}
            </div>
        `;
    }
}
