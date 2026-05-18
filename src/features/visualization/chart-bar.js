/**
 * Bar / Column Chart Renderer
 *
 * Renders SVG-based vertical column charts:
 *  - Single column       → frequency-count columns
 *  - Category × Numeric  → aggregated columns (avg / sum / count / min / max)
 *  - Two categoricals    → stacked cross-tab (horizontal segments)
 */

export class BarChart {
    constructor() {
        this._crossTabColors = [
            '#34B27B', '#4ECDC4', '#FF6B6B', '#FFA07A',
            '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9',
            '#A29BFE', '#FD79A8',
        ];
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /**
     * Main entry point — mirrors the call signature of the other chart modules.
     * @param {HTMLElement} modal
     * @param {Array}       data
     * @param {Array}       columns          column names or {name} objects
     * @param {Map}         columnDataTypes  colName → { isNumeric: bool }
     */
    async render(modal, data, columns, columnDataTypes = new Map()) {
        if (!columns || columns.length === 0) {
            throw new Error('Need at least 1 column for bar chart');
        }

        const xCol = columns[0]?.name || columns[0];
        const yCol = columns[1]?.name || columns[1];

        if (yCol && yCol !== xCol) {
            await this._renderGrouped(modal, data, xCol, yCol, columnDataTypes);
        } else {
            await this._renderFrequency(modal, data, xCol);
        }
    }

    // ── Single column: frequency count ──────────────────────────────────────

    async _renderFrequency(modal, data, xCol) {
        const counts = {};
        data.forEach(row => {
            const v = String(row[xCol] ?? '');
            if (v) counts[v] = (counts[v] || 0) + 1;
        });

        const entries = Object.entries(counts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 30);
        const maxC = entries[0]?.[1] || 1;

        const canvas = modal.querySelector('#viz-canvas');
        canvas.style.display = 'none';
        const container = canvas.parentElement;
        container.style.cssText = 'background:#0a0a0a;display:flex;flex-direction:column;overflow:hidden;padding:0;align-items:stretch;justify-content:flex-start;';

        const svg = this._buildFrequencySVG(entries, maxC, xCol);

        container.innerHTML =
            '<div style="display:flex;flex-direction:column;width:100%;height:100%;overflow:hidden">'
            + '<div style="flex-shrink:0;display:flex;align-items:center;padding:10px 20px;border-bottom:1px solid rgba(255,255,255,0.06)">'
            + '<span style="font-size:10px;color:rgba(255,255,255,0.3);text-transform:uppercase;letter-spacing:0.1em">'
            + xCol + ' &nbsp;\u00b7&nbsp; Count'
            + '</span></div>'
            + '<div style="flex:1;overflow:auto;padding:12px 16px">' + svg + '</div>'
            + '</div>';

        modal.querySelector('#viz-controls').innerHTML =
            '<div style="font-size:13px;display:flex;gap:24px">'
            + '<div><span style="color:rgba(255,255,255,0.4)">Column: </span><span style="color:#34B27B">' + xCol + '</span></div>'
            + '<div><span style="color:rgba(255,255,255,0.4)">Unique: </span><span style="color:#34B27B">' + entries.length + '</span></div>'
            + '<div><span style="color:rgba(255,255,255,0.4)">Rows: </span><span style="color:#34B27B">' + data.length.toLocaleString() + '</span></div>'
            + '</div>';
    }

    _buildFrequencySVG(entries, maxC, xCol) {
        const mT = 28, mR = 20, mB = 90, mL = 62;
        const svgW = Math.max(700, entries.length * 60);
        const svgH = 380;
        const cW = svgW - mL - mR;
        const cH = svgH - mT - mB;
        const barSpacing = cW / entries.length;
        const barW = Math.max(8, barSpacing * 0.62);
        const barPad = (barSpacing - barW) / 2;

        const ticks = [0, 0.25, 0.5, 0.75, 1].map(t => Math.round(t * maxC));
        const gridLines = ticks.map(tick => {
            const ty = mT + cH - (tick / maxC) * cH;
            // Y-axis tick mark + gridline + value label
            return `<line x1="${mL}" y1="${ty}" x2="${mL + cW}" y2="${ty}" stroke="rgba(255,255,255,${tick === 0 ? '0.15' : '0.06'})" stroke-width="1"/>`
                + `<line x1="${mL - 5}" y1="${ty}" x2="${mL}" y2="${ty}" stroke="rgba(255,255,255,0.5)" stroke-width="1"/>`
                + `<text x="${mL - 9}" y="${ty + 4}" text-anchor="end" font-size="10" fill="rgba(255,255,255,0.55)" font-family="monospace">${tick.toLocaleString()}</text>`;
        }).join('');

        const bars = entries.map((e, i) => {
            const [value, count] = e;
            const bH = Math.max(1, (count / maxC) * cH);
            const bX = mL + i * barSpacing + barPad;
            const bY = mT + cH - bH;
            const cx = bX + barW / 2;
            const lY = mT + cH + 12;
            const safe = String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
            const catShort = safe.length > 13 ? safe.slice(0, 12) + '\u2026' : safe;
            // X-axis tick mark at each bar's centre
            return `<rect x="${bX}" y="${bY}" width="${barW}" height="${bH}" fill="url(#freqGrad)" rx="2"/>`
                + `<text x="${cx}" y="${bY - 5}" text-anchor="middle" font-size="10" font-weight="700" fill="#34B27B" font-family="monospace">${count.toLocaleString()}</text>`
                + `<line x1="${cx}" y1="${mT + cH}" x2="${cx}" y2="${mT + cH + 5}" stroke="rgba(255,255,255,0.35)" stroke-width="1"/>`
                + `<text x="${cx}" y="${lY}" text-anchor="end" font-size="10" fill="rgba(255,255,255,0.55)" font-family="monospace" transform="rotate(-42,${cx},${lY})">${catShort}</text>`;
        }).join('');

        return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${svgW} ${svgH}" preserveAspectRatio="xMinYMid meet" style="width:100%;min-width:${svgW}px;height:100%;display:block">
            <defs><linearGradient id="freqGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#34B27B"/><stop offset="100%" stop-color="#1a5e3a"/>
            </linearGradient></defs>
            ${gridLines}${bars}
            <line x1="${mL}" y1="${mT}" x2="${mL}" y2="${mT + cH}" stroke="rgba(255,255,255,0.5)" stroke-width="1.5"/>
            <line x1="${mL}" y1="${mT + cH}" x2="${mL + cW}" y2="${mT + cH}" stroke="rgba(255,255,255,0.5)" stroke-width="1.5"/>
            <text x="${mL - 44}" y="${mT + cH / 2}" text-anchor="middle" font-size="11" font-weight="600" fill="rgba(255,255,255,0.6)" font-family="monospace" transform="rotate(-90,${mL - 44},${mT + cH / 2})">Count</text>
            <text x="${mL + cW / 2}" y="${svgH - 2}" text-anchor="middle" font-size="11" font-weight="600" fill="rgba(255,255,255,0.6)" font-family="monospace">${xCol}</text>
        </svg>`;
    }

    // ── Two columns: route by data type ─────────────────────────────────────

    async _renderGrouped(modal, data, xColumn, yColumn, columnDataTypes) {
        const xNum = columnDataTypes.get(xColumn)?.isNumeric || false;
        const yNum = columnDataTypes.get(yColumn)?.isNumeric || false;

        if (!xNum && yNum) {
            await this._renderCategoryVsNumeric(modal, data, xColumn, yColumn);
        } else if (xNum && !yNum) {
            await this._renderCategoryVsNumeric(modal, data, yColumn, xColumn);
        } else if (xNum && yNum) {
            await this._renderCategoryVsNumeric(modal, data, xColumn, yColumn);
        } else {
            await this._renderCrossTab(modal, data, xColumn, yColumn);
        }
    }

    // ── Category × Numeric: SVG vertical column chart ───────────────────────

    async _renderCategoryVsNumeric(modal, data, xColumn, yColumn) {
        // Aggregate Y values per X category
        const catStats = new Map();
        data.forEach(row => {
            const cat = String(row[xColumn] ?? '');
            const val = parseFloat(row[yColumn]);
            if (cat && !isNaN(val)) {
                if (!catStats.has(cat)) catStats.set(cat, { sum: 0, count: 0, min: Infinity, max: -Infinity });
                const s = catStats.get(cat);
                s.sum += val; s.count++;
                if (val < s.min) s.min = val;
                if (val > s.max) s.max = val;
            }
        });

        const aggregated = Array.from(catStats.entries())
            .map(([category, s]) => ({
                category,
                count: s.count,
                sum: s.sum,
                avg: s.sum / s.count,
                min: s.min,
                max: s.max,
            }))
            .sort((a, b) => b.avg - a.avg)
            .slice(0, 30);

        // Expose on window so switchBarAggregation (inline onclick) can reach them
        window.currentBarData = aggregated;
        window.currentBarXColumn = xColumn;
        window.currentBarYColumn = yColumn;
        window._barBuildColChart = (agg, type) => this._buildColChartSVG(agg, type, xColumn, yColumn);

        const canvas = modal.querySelector('#viz-canvas');
        canvas.style.display = 'none';
        const container = canvas.parentElement;
        container.style.cssText = 'background:#0a0a0a;display:flex;flex-direction:column;overflow:hidden;padding:0;align-items:stretch;justify-content:flex-start;';

        const aggBtn = (t, active) => active
            ? `style="padding:3px 10px;border-radius:4px;font-size:11px;font-weight:700;background:#34B27B;color:#000;border:none;cursor:pointer"`
            : `style="padding:3px 10px;border-radius:4px;font-size:11px;font-weight:700;background:rgba(255,255,255,0.07);color:rgba(255,255,255,0.45);border:none;cursor:pointer"`;

        container.innerHTML = `
            <div style="display:flex;flex-direction:column;width:100%;height:100%;overflow:hidden">
                <div style="flex-shrink:0;display:flex;align-items:center;gap:8px;padding:10px 16px;border-bottom:1px solid rgba(255,255,255,0.06)">
                    <span style="font-size:10px;color:rgba(255,255,255,0.3);text-transform:uppercase;letter-spacing:0.08em;margin-right:4px">Aggregate:</span>
                    ${['avg', 'sum', 'count', 'min', 'max'].map(t =>
                        `<button id="agg-${t}" onclick="switchBarAggregation('${t}')" ${aggBtn(t, t === 'avg')}>${t.charAt(0).toUpperCase() + t.slice(1)}</button>`
                    ).join('')}
                    <div style="flex:1"></div>
                    <span style="font-size:10px;color:rgba(255,255,255,0.3)">${xColumn} &nbsp;\u00b7&nbsp; ${yColumn}</span>
                </div>
                <div id="bar-chart-container" style="flex:1;overflow:auto;padding:12px 16px">
                    ${this._buildColChartSVG(aggregated, 'avg', xColumn, yColumn)}
                </div>
            </div>`;

        modal.querySelector('#viz-controls').innerHTML = `
            <div style="font-size:13px;display:flex;gap:24px;flex-wrap:wrap">
                <div><span style="color:rgba(255,255,255,0.4)">X: </span><span style="color:#34B27B">${xColumn}</span></div>
                <div><span style="color:rgba(255,255,255,0.4)">Y: </span><span style="color:#34B27B">${yColumn}</span></div>
                <div><span style="color:rgba(255,255,255,0.4)">Categories: </span><span style="color:#34B27B">${aggregated.length}</span></div>
                <div><span style="color:rgba(255,255,255,0.4)">Rows: </span><span style="color:#34B27B">${data.length.toLocaleString()}</span></div>
            </div>`;
    }

    _buildColChartSVG(agg, type, xColumn, yColumn) {
        const display = [...agg].sort((a, b) => b[type] - a[type]);
        const maxV = Math.max(...display.map(d => d[type])) || 1;
        const fmtV = v => type === 'count'
            ? Math.round(v).toLocaleString()
            : parseFloat(v.toFixed(2)).toLocaleString();

        const mT = 28, mR = 20, mB = 90, mL = 62;
        const svgW = Math.max(700, display.length * 60);
        const svgH = 380;
        const cW = svgW - mL - mR;
        const cH = svgH - mT - mB;
        const barSpacing = cW / display.length;
        const barW = Math.max(8, barSpacing * 0.62);
        const barPad = (barSpacing - barW) / 2;

        const ticks = [0, 0.25, 0.5, 0.75, 1].map(t => t * maxV);
        const gridLines = ticks.map(tick => {
            const ty = mT + cH - (tick / maxV) * cH;
            // Y-axis tick mark + gridline + value label
            return `<line x1="${mL}" y1="${ty}" x2="${mL + cW}" y2="${ty}" stroke="rgba(255,255,255,${tick === 0 ? '0.15' : '0.06'})" stroke-width="1"/>`
                + `<line x1="${mL - 5}" y1="${ty}" x2="${mL}" y2="${ty}" stroke="rgba(255,255,255,0.5)" stroke-width="1"/>`
                + `<text x="${mL - 9}" y="${ty + 4}" text-anchor="end" font-size="10" fill="rgba(255,255,255,0.55)" font-family="monospace">${fmtV(tick)}</text>`;
        }).join('');

        const bars = display.map((item, i) => {
            const val = item[type];
            const bH = Math.max(1, (val / maxV) * cH);
            const bX = mL + i * barSpacing + barPad;
            const bY = mT + cH - bH;
            const cx = bX + barW / 2;
            const lY = mT + cH + 12;
            const cat = String(item.category)
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
            const catShort = cat.length > 13 ? cat.slice(0, 12) + '\u2026' : cat;
            // X-axis tick mark at each bar's centre
            return `<rect x="${bX}" y="${bY}" width="${barW}" height="${bH}" fill="url(#colGrad)" rx="2"/>`
                + `<text x="${cx}" y="${bY - 5}" text-anchor="middle" font-size="10" font-weight="700" fill="#34B27B" font-family="monospace">${fmtV(val)}</text>`
                + `<line x1="${cx}" y1="${mT + cH}" x2="${cx}" y2="${mT + cH + 5}" stroke="rgba(255,255,255,0.35)" stroke-width="1"/>`
                + `<text x="${cx}" y="${lY}" text-anchor="end" font-size="10" fill="rgba(255,255,255,0.55)" font-family="monospace" transform="rotate(-42,${cx},${lY})">${catShort}</text>`;
        }).join('');

        return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${svgW} ${svgH}" preserveAspectRatio="xMinYMid meet" style="width:100%;min-width:${svgW}px;height:100%;display:block">
            <defs><linearGradient id="colGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#34B27B"/><stop offset="100%" stop-color="#1a5e3a"/>
            </linearGradient></defs>
            ${gridLines}${bars}
            <line x1="${mL}" y1="${mT}" x2="${mL}" y2="${mT + cH}" stroke="rgba(255,255,255,0.5)" stroke-width="1.5"/>
            <line x1="${mL}" y1="${mT + cH}" x2="${mL + cW}" y2="${mT + cH}" stroke="rgba(255,255,255,0.5)" stroke-width="1.5"/>
            <text x="${mL - 44}" y="${mT + cH / 2}" text-anchor="middle" font-size="11" font-weight="600" fill="rgba(255,255,255,0.6)" font-family="monospace" transform="rotate(-90,${mL - 44},${mT + cH / 2})">${yColumn}</text>
            <text x="${mL + cW / 2}" y="${svgH - 2}" text-anchor="middle" font-size="11" font-weight="600" fill="rgba(255,255,255,0.6)" font-family="monospace">${xColumn}</text>
        </svg>`;
    }

    // ── Categorical × Categorical: stacked vertical column chart ───────────

    async _renderCrossTab(modal, data, xColumn, yColumn) {
        const colors = this._crossTabColors;
        const canvas = modal.querySelector('#viz-canvas');

        const crosstab = new Map();
        const yCategories = new Set();

        data.forEach(row => {
            const x = row[xColumn];
            const y = row[yColumn];
            if (x != null && y != null) {
                yCategories.add(y);
                if (!crosstab.has(x)) crosstab.set(x, new Map());
                const yMap = crosstab.get(x);
                yMap.set(y, (yMap.get(y) || 0) + 1);
            }
        });

        const xSorted = Array.from(crosstab.entries())
            .map(([x, yMap]) => ({
                x,
                yMap,
                total: Array.from(yMap.values()).reduce((s, c) => s + c, 0),
            }))
            .sort((a, b) => b.total - a.total)
            .slice(0, 30);

        const yCounts = new Map();
        yCategories.forEach(y => {
            let total = 0;
            xSorted.forEach(({ yMap }) => { total += yMap.get(y) || 0; });
            yCounts.set(y, total);
        });

        const yArray = Array.from(yCounts.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10)
            .map(([y]) => y);

        canvas.style.display = 'none';
        const container = canvas.parentElement;
        container.style.cssText = 'background:#0a0a0a;display:flex;flex-direction:column;overflow:hidden;padding:0;align-items:stretch;justify-content:flex-start;';

        const legendHTML = yArray.map((y, idx) => {
            const safe = String(y).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return `<div style="display:flex;align-items:center;gap:4px;flex-shrink:0">
                <div style="width:10px;height:10px;border-radius:2px;background:${colors[idx % colors.length]};flex-shrink:0"></div>
                <span style="font-size:9px;color:rgba(255,255,255,0.6);font-family:monospace;white-space:nowrap">${safe}</span>
            </div>`;
        }).join('');

        container.innerHTML = `
            <div style="display:flex;flex-direction:column;width:100%;height:100%;overflow:hidden">
                <div style="flex-shrink:0;display:flex;align-items:center;gap:8px;padding:8px 16px;border-bottom:1px solid rgba(255,255,255,0.06);flex-wrap:wrap;row-gap:6px">
                    ${legendHTML}
                    <div style="flex:1;min-width:8px"></div>
                    <span style="font-size:10px;color:rgba(255,255,255,0.3);white-space:nowrap">${xColumn} &nbsp;\u00b7&nbsp; ${yColumn}</span>
                </div>
                <div style="flex:1;overflow:auto;padding:12px 16px">
                    ${this._buildCrossTabSVG(xSorted, yArray, colors, xColumn, yColumn)}
                </div>
            </div>`;

        modal.querySelector('#viz-controls').innerHTML = `
            <div style="font-size:13px;display:flex;gap:24px;flex-wrap:wrap">
                <div><span style="color:rgba(255,255,255,0.4)">X: </span><span style="color:#34B27B">${xColumn}</span></div>
                <div><span style="color:rgba(255,255,255,0.4)">Y: </span><span style="color:#34B27B">${yColumn}</span></div>
                <div><span style="color:rgba(255,255,255,0.4)">Categories: </span><span style="color:#34B27B">${xSorted.length}</span></div>
                <div><span style="color:rgba(255,255,255,0.4)">Groups: </span><span style="color:#34B27B">${yArray.length}</span></div>
                <div><span style="color:rgba(255,255,255,0.4)">Rows: </span><span style="color:#34B27B">${data.length.toLocaleString()}</span></div>
            </div>`;
    }

    _buildCrossTabSVG(xSorted, yArray, colors, xColumn, yColumn) {
        const mT = 28, mR = 20, mB = 90, mL = 62;
        const svgW = Math.max(700, xSorted.length * 60);
        const svgH = 380;
        const cW = svgW - mL - mR;
        const cH = svgH - mT - mB;
        const barSpacing = cW / xSorted.length;
        const barW = Math.max(8, barSpacing * 0.7);
        const barPad = (barSpacing - barW) / 2;

        const maxTotal = Math.max(...xSorted.map(d => d.total)) || 1;

        const ticks = [0, 0.25, 0.5, 0.75, 1].map(t => Math.round(t * maxTotal));
        const gridLines = ticks.map(tick => {
            const ty = mT + cH - (tick / maxTotal) * cH;
            return `<line x1="${mL}" y1="${ty}" x2="${mL + cW}" y2="${ty}" stroke="rgba(255,255,255,${tick === 0 ? '0.15' : '0.06'})" stroke-width="1"/>`
                + `<line x1="${mL - 5}" y1="${ty}" x2="${mL}" y2="${ty}" stroke="rgba(255,255,255,0.5)" stroke-width="1"/>`
                + `<text x="${mL - 9}" y="${ty + 4}" text-anchor="end" font-size="10" fill="rgba(255,255,255,0.55)" font-family="monospace">${tick.toLocaleString()}</text>`;
        }).join('');

        const bars = xSorted.map(({ x, yMap, total }, i) => {
            const bX = mL + i * barSpacing + barPad;
            const cx = bX + barW / 2;
            const lY = mT + cH + 12;
            const xStr = String(x).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
            const catShort = xStr.length > 13 ? xStr.slice(0, 12) + '\u2026' : xStr;

            let stackY = mT + cH;
            const segments = yArray.map((y, idx) => {
                const count = yMap.get(y) || 0;
                if (count === 0) return '';
                const segH = Math.max(1, (count / maxTotal) * cH);
                const segY = stackY - segH;
                stackY -= segH;
                return `<rect x="${bX}" y="${segY}" width="${barW}" height="${segH}" fill="${colors[idx % colors.length]}" rx="1"/>`;
            }).join('');

            const topY = mT + cH - (total / maxTotal) * cH;
            return segments
                + `<text x="${cx}" y="${topY - 4}" text-anchor="middle" font-size="9" font-weight="700" fill="rgba(255,255,255,0.65)" font-family="monospace">${total.toLocaleString()}</text>`
                + `<line x1="${cx}" y1="${mT + cH}" x2="${cx}" y2="${mT + cH + 5}" stroke="rgba(255,255,255,0.35)" stroke-width="1"/>`
                + `<text x="${cx}" y="${lY}" text-anchor="end" font-size="10" fill="rgba(255,255,255,0.55)" font-family="monospace" transform="rotate(-42,${cx},${lY})">${catShort}</text>`;
        }).join('');

        return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${svgW} ${svgH}" preserveAspectRatio="xMinYMid meet" style="width:100%;min-width:${svgW}px;height:100%;display:block">
            ${gridLines}${bars}
            <line x1="${mL}" y1="${mT}" x2="${mL}" y2="${mT + cH}" stroke="rgba(255,255,255,0.5)" stroke-width="1.5"/>
            <line x1="${mL}" y1="${mT + cH}" x2="${mL + cW}" y2="${mT + cH}" stroke="rgba(255,255,255,0.5)" stroke-width="1.5"/>
            <text x="${mL - 44}" y="${mT + cH / 2}" text-anchor="middle" font-size="11" font-weight="600" fill="rgba(255,255,255,0.6)" font-family="monospace" transform="rotate(-90,${mL - 44},${mT + cH / 2})">Count</text>
            <text x="${mL + cW / 2}" y="${svgH - 2}" text-anchor="middle" font-size="11" font-weight="600" fill="rgba(255,255,255,0.6)" font-family="monospace">${xColumn}</text>
        </svg>`;
    }
}
