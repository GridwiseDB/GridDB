/**
 * Root Cause Analysis (RCA) & CAPA Effectiveness Engine
 *
 *
 * Works entirely on data already loaded into the WebGPU GridDB engine —
 * no server round-trips required for analysis.
 */

export class RCAAnalyzer {

    // ──────────────────────────────────────────────────────────────────────────
    // Statistics helpers
    // ──────────────────────────────────────────────────────────────────────────

    /** Pearson correlation coefficient r ∈ [-1, 1] */
    pearson(x, y) {
        const n = Math.min(x.length, y.length);
        if (n < 5) return 0;
        const mx = x.slice(0, n).reduce((a, b) => a + b, 0) / n;
        const my = y.slice(0, n).reduce((a, b) => a + b, 0) / n;
        let num = 0, dx2 = 0, dy2 = 0;
        for (let i = 0; i < n; i++) {
            const dx = x[i] - mx, dy = y[i] - my;
            num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
        }
        const denom = Math.sqrt(dx2 * dy2);
        return denom === 0 ? 0 : num / denom;
    }

    /** Z-score anomaly detection — returns entries where z > threshold */
    zScoreAnomalies(arr, threshold = 1.8) {
        if (arr.length < 3) return [];
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const std = Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length) || 1;
        return arr.map((v, i) => ({ i, z: (v - mean) / std, v })).filter(e => e.z > threshold);
    }

    /**
     * Two-tailed p-value approximation for Pearson r.
     * Uses t = r√(n-2)/√(1-r²) and maps to rough p-value bracket.
     * Returns: { pValue: number, significant: boolean, confidence: 'HIGH'|'MEDIUM'|'LOW' }
     */
    _pearsonSignificance(r, n) {
        if (n < 5) return { pValue: 1, significant: false, confidence: 'INSUFFICIENT DATA' };
        const t = Math.abs(r) * Math.sqrt(n - 2) / Math.sqrt(Math.max(1e-10, 1 - r * r));
        let pValue;
        if      (t > 4.5) pValue = 0.001;
        else if (t > 3.2) pValue = 0.005;
        else if (t > 2.5) pValue = 0.02;
        else if (t > 2.0) pValue = 0.05;
        else if (t > 1.5) pValue = 0.15;
        else              pValue = 0.30;
        const significant = pValue <= 0.05;
        const confidence = pValue <= 0.01 ? 'HIGH' : pValue <= 0.05 ? 'MEDIUM' : 'LOW';
        return { pValue, significant, confidence };
    }

    /** Simple linear regression → { slope, intercept, r2 } */
    linearRegression(x, y) {
        const n = x.length;
        if (n < 2) return { slope: 0, intercept: 0, r2: 0 };
        const mx = x.reduce((a, b) => a + b, 0) / n;
        const my = y.reduce((a, b) => a + b, 0) / n;
        let num = 0, den = 0;
        for (let i = 0; i < n; i++) { num += (x[i] - mx) * (y[i] - my); den += (x[i] - mx) ** 2; }
        const slope = den === 0 ? 0 : num / den;
        const intercept = my - slope * mx;
        const yhat = x.map(xi => slope * xi + intercept);
        const ssTot = y.reduce((a, b) => a + (b - my) ** 2, 0) || 1;
        const ssRes = y.reduce((a, b, i) => a + (b - yhat[i]) ** 2, 0);
        return { slope, intercept, r2: Math.max(0, 1 - ssRes / ssTot) };
    }

    /** Mean ± SD summary */
    stats(arr) {
        if (!arr.length) return { mean: 0, std: 0, min: 0, max: 0, median: 0 };
        const sorted = [...arr].sort((a, b) => a - b);
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const std = Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length);
        return { mean, std, min: sorted[0], max: sorted[sorted.length - 1], median: sorted[Math.floor(sorted.length / 2)] };
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Time-series helpers
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Bucket rows by hour or day, summing countCol (or counting rows if null).
     * Returns [{time: string, count: number}] sorted chronologically.
     */
    _bucketByTime(rows, tsCol, countCol, bucket = 'hour') {
        const map = new Map();
        for (const row of rows) {
            const ts = row[tsCol];
            if (!ts) continue;
            const d = new Date(ts);
            if (isNaN(d.getTime())) continue;
            let key;
            if (bucket === 'hour') key = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')} ${String(d.getHours()).padStart(2,'0')}:00`;
            else                   key = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
            const cnt = countCol ? (parseFloat(row[countCol]) || 0) : 1;
            map.set(key, (map.get(key) || 0) + cnt);
        }
        return [...map.entries()].sort((a, b) => a[0] < b[0] ? -1 : 1).map(([time, count]) => ({ time, count }));
    }

    /** Detect defect spike windows using Z-score */
    _detectDefectWindows(timeSeries) {
        const counts = timeSeries.map(t => t.count);
        return this.zScoreAnomalies(counts, 1.8).map(a => ({ ...timeSeries[a.i], zscore: +a.z.toFixed(2) }));
    }

    /**
     * Find events that occurred up to `lookbackMs` before each defect window.
     * These are "change events" that may have caused the defect spike.
     */
    _findPrecedingEvents(windows, factorRows, tsCol, lookbackMs = 7200000 /* 2h */) {
        const events = [];
        for (const w of windows) {
            const wt = new Date(w.time).getTime();
            if (isNaN(wt)) continue;
            const prior = factorRows.filter(r => {
                const t = new Date(r[tsCol]).getTime();
                return !isNaN(t) && t < wt && t > wt - lookbackMs;
            });
            if (prior.length > 0) events.push({ window: w, precedingRows: prior.slice(0, 10) });
        }
        return events;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Correlation analysis
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * For each numeric column in factorRows, compute Pearson r vs defect counts,
     * aligned by the same hourly time bucket.
     */
    _correlateNumericFactors(defectSeries, factorRows, factorTsCol, factorCols) {
        const results = [];
        for (const col of factorCols) {
            // Build hourly average map for this factor column
            const buckets = new Map();
            for (const row of factorRows) {
                const ts = row[factorTsCol];
                if (!ts) continue;
                const d = new Date(ts);
                if (isNaN(d.getTime())) continue;
                const key = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')} ${String(d.getHours()).padStart(2,'0')}:00`;
                const v = parseFloat(row[col]);
                if (!isNaN(v)) {
                    if (!buckets.has(key)) buckets.set(key, []);
                    buckets.get(key).push(v);
                }
            }
            // Align with defect time series (inner join on time bucket)
            const defX = [], factY = [];
            for (const { time, count } of defectSeries) {
                const vals = buckets.get(time);
                if (vals && vals.length) {
                    defX.push(count);
                    factY.push(vals.reduce((a, b) => a + b, 0) / vals.length);
                }
            }
            if (defX.length < 5) continue;
            const r = this.pearson(defX, factY);
            const reg = this.linearRegression(factY, defX);
            const fStats = this.stats(factY);
            const sig = this._pearsonSignificance(r, defX.length);
            results.push({
                column: col,
                pearsonR: +r.toFixed(4),
                r2: +reg.r2.toFixed(4),
                slope: +reg.slope.toFixed(4),
                sampleSize: defX.length,
                absR: Math.abs(r),
                factorMean: +fStats.mean.toFixed(2),
                factorStd: +fStats.std.toFixed(2),
                pValue: sig.pValue,
                significant: sig.significant,
                confidence: sig.confidence,
            });
        }
        return results.sort((a, b) => b.absR - a.absR);
    }

    /**
     * Categorical correlation: for each unique value of a categorical column,
     * compute what % of defect windows had that value present in the preceding
     * 2-hour window.
     */
    _correlateCategoricalFactors(windows, factorRows, factorTsCol, catCols) {
        const results = [];
        for (const col of catCols) {
            const valueCounts = new Map();
            for (const w of windows) {
                const wt = new Date(w.time).getTime();
                if (isNaN(wt)) continue;
                const prior = factorRows.filter(r => {
                    const t = new Date(r[factorTsCol]).getTime();
                    return !isNaN(t) && t < wt && t > wt - 7200000;
                });
                const vals = prior.map(r => r[col]).filter(Boolean);
                for (const v of vals) valueCounts.set(v, (valueCounts.get(v) || 0) + 1);
            }
            if (!valueCounts.size) continue;
            const top = [...valueCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3);
            results.push({ column: col, topValues: top, windowCount: windows.length });
        }
        return results;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // CAPA KPIs
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Calculate the three core CAPA effectiveness KPIs:
     *
     *   Recurrence Rate  = (CAPAs where defect recurred within 30d) / totalClosed × 100%
     *   Effectiveness    = (pre-rate − post-rate) / pre-rate × 100%
     *   Escape Rate      = defects that reached customer / total defects × 100%
     */
    _calculateCapaKPIs(defectRows, capaRows, opts) {
        const {
            defectTsCol, defectTypeCol, defectCountCol,
            capaTsCol, capaStatusCol, capaTypeCol,
            capaClosedVal = 'closed', windowDays = 30,
        } = opts;

        // --- Recurrence Rate ---
        const closedCapas = capaRows.filter(r =>
            (r[capaStatusCol] || '').toLowerCase().includes(capaClosedVal.toLowerCase())
        );
        let recurred = 0;
        for (const capa of closedCapas) {
            const closeTime = new Date(capa[capaTsCol]).getTime();
            if (isNaN(closeTime)) continue;
            const capaType = capaTypeCol ? (capa[capaTypeCol] || '') : '';
            const windowEnd = closeTime + windowDays * 86400000;
            const recurrence = defectRows.find(d => {
                const dt = new Date(d[defectTsCol]).getTime();
                if (isNaN(dt)) return false;
                if (dt <= closeTime || dt >= windowEnd) return false;
                if (capaType && defectTypeCol) return (d[defectTypeCol] || '') === capaType;
                return true;
            });
            if (recurrence) recurred++;
        }
        const recurrenceRate = closedCapas.length ? (recurred / closedCapas.length) * 100 : null;

        // --- Effectiveness (pre vs post defect rate) ---
        let effectiveness = null;
        if (closedCapas.length > 0 && defectTsCol) {
            // Use the median-dated CAPA as the reference point
            const sorted = closedCapas
                .map(c => new Date(c[capaTsCol]).getTime())
                .filter(t => !isNaN(t))
                .sort((a, b) => a - b);
            if (sorted.length > 0) {
                const refTime = sorted[Math.floor(sorted.length / 2)];
                const winMs = windowDays * 86400000;
                const preRows = defectRows.filter(d => {
                    const t = new Date(d[defectTsCol]).getTime();
                    return !isNaN(t) && t < refTime && t > refTime - winMs;
                });
                const postRows = defectRows.filter(d => {
                    const t = new Date(d[defectTsCol]).getTime();
                    return !isNaN(t) && t > refTime && t < refTime + winMs;
                });
                const sum = (rows) => rows.reduce((a, r) => a + (defectCountCol ? (parseFloat(r[defectCountCol]) || 1) : 1), 0);
                const preRate = preRows.length ? sum(preRows) / preRows.length : 0;
                const postRate = postRows.length ? sum(postRows) / postRows.length : 0;
                if (preRate > 0) effectiveness = ((preRate - postRate) / preRate) * 100;
            }
        }

        // --- Escape Rate ---
        const escapeKeywords = ['escaped', 'customer', 'field', 'external', 'warranty', 'return', 'recall'];
        const escaped = defectRows.filter(r => {
            const combined = Object.values(r).join(' ').toLowerCase();
            return escapeKeywords.some(k => combined.includes(k));
        });
        const escapeRate = defectRows.length ? (escaped.length / defectRows.length) * 100 : 0;

        // --- Time to Containment (avg hours from defect to CAPA open) ---
        let avgContainmentHours = null;
        if (capaRows.length > 0 && capaTsCol && defectTsCol && defectRows.length > 0) {
            const firstDefect = Math.min(...defectRows.map(r => new Date(r[defectTsCol]).getTime()).filter(t => !isNaN(t)));
            const firstCapa = Math.min(...capaRows.map(r => new Date(r[capaTsCol]).getTime()).filter(t => !isNaN(t)));
            if (isFinite(firstDefect) && isFinite(firstCapa) && firstCapa > firstDefect) {
                avgContainmentHours = (firstCapa - firstDefect) / 3600000;
            }
        }

        // ── Threshold pass/fail (industry benchmarks) ─────────────────────
        const rr  = recurrenceRate !== null ? +recurrenceRate.toFixed(1) : null;
        const eff = effectiveness  !== null ? +effectiveness.toFixed(1)  : null;
        const er  = +escapeRate.toFixed(1);
        const ch  = avgContainmentHours !== null ? +avgContainmentHours.toFixed(1) : null;

        return {
            recurrenceRate:       rr,
            recurrenceStatus:     rr === null ? 'N/A' : rr < 10 ? 'PASS' : rr < 25 ? 'WARNING' : 'FAIL',
            recurrenceBenchmark:  '< 10 %',
            effectiveness:        eff,
            effectivenessStatus:  eff === null ? 'N/A' : eff > 60 ? 'PASS' : eff > 30 ? 'WARNING' : 'FAIL',
            effectivenessBenchmark: '> 60 %',
            escapeRate:           er,
            escapeStatus:         er < 1 ? 'PASS' : er < 5 ? 'WARNING' : 'FAIL',
            escapeBenchmark:      '< 1 %',
            avgContainmentHours:  ch,
            containmentStatus:    ch === null ? 'N/A' : ch < 24 ? 'PASS' : ch < 72 ? 'WARNING' : 'FAIL',
            containmentBenchmark: '< 24 h',
            totalClosed:          closedCapas.length,
            recurred,
            escapedCount:         escaped.length,
        };
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Auto-detection
    // ──────────────────────────────────────────────────────────────────────────

    /** Guess column roles from column names */
    autoDetect(columns) {
        const lc = c => (c == null ? '' : String(c).toLowerCase());
        return {
            timestampCols: columns.filter(c => ['timestamp','date','time','datetime','created_at','occurred_at','event_time','report_date','defect_date','logged_at'].some(k => lc(c).includes(k))),
            countCols:     columns.filter(c => ['count','defect_count','qty','quantity','defects','rejects','failures','errors','scrap','ncr'].some(k => lc(c).includes(k))),
            typeCols:      columns.filter(c => ['type','defect_type','category','failure_mode','failure_type','code','reason','root_cause','classification','mode'].some(k => lc(c).includes(k))),
            statusCols:    columns.filter(c => ['status','state','closed','open','resolution','result','outcome','disposition'].some(k => lc(c).includes(k))),
            numericCols:   columns.filter(c => ['temp','temperature','speed','pressure','torque','vibration','voltage','current','humidity','rpm','feed','flow','rate','load','force','weight','thickness','diameter','gap'].some(k => lc(c).includes(k))),
        };
    }

    /** Detect whether a column is mostly numeric (>70% parseable) */
    _isNumericCol(rows, col) {
        const sample = rows.slice(0, 50).map(r => r[col]).filter(v => v !== null && v !== '');
        if (!sample.length) return false;
        return sample.filter(v => !isNaN(parseFloat(v))).length / sample.length > 0.7;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Top categories
    // ──────────────────────────────────────────────────────────────────────────

    _topCategories(rows, col, n = 8) {
        if (!col) return [];
        const cnt = {};
        for (const r of rows) { const v = r[col] || '(unknown)'; cnt[v] = (cnt[v] || 0) + 1; }
        return Object.entries(cnt).sort((a, b) => b[1] - a[1]).slice(0, n).map(([name, count]) => ({ name, count }));
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Causal chain builder
    // ──────────────────────────────────────────────────────────────────────────

    _buildCausalChain(results) {
        const chain = [];

        if (results.defectWindows.length > 0) {
            const peak = Math.max(...results.defectWindows.map(w => w.count));
            const mean = results.defectSummary.timeSeries.length
                ? results.defectSummary.timeSeries.reduce((a, b) => a + b.count, 0) / results.defectSummary.timeSeries.length
                : 0;
            chain.push({
                step: 1,
                severity: 'high',
                finding: `${results.defectWindows.length} defect spike window(s) detected`,
                details: `Peak: ${peak.toLocaleString()} events (${mean > 0 ? ((peak/mean)*100).toFixed(0) : '?'}% above baseline)`,
                recommendation: 'Investigate what changed in the 2-hour window before each spike.',
            });
        }

        let step = 2;
        const top3 = results.correlations.slice(0, 3);
        for (const c of top3) {
            const strength = c.absR > 0.7 ? 'STRONG' : c.absR > 0.4 ? 'MODERATE' : 'WEAK';
            const dir = c.pearsonR > 0 ? '▲ increases' : '▼ decreases';
            const sev = c.absR > 0.7 ? 'high' : c.absR > 0.4 ? 'medium' : 'low';
            chain.push({
                step: step++,
                severity: sev,
                finding: `${strength} signal: "${c.column}" from [${c.sourceTable}] (r = ${c.pearsonR.toFixed(2)})`,
                details: `Defect count ${dir} as ${c.column} ${dir === '▲ increases' ? 'rises' : 'falls'}. Explains ${(c.r2*100).toFixed(0)}% of variance.`,
                recommendation: c.absR > 0.5
                    ? `Investigate ${c.column} control limits — set up an SPC chart.`
                    : `Monitor ${c.column} alongside defect rate for 30 days.`,
            });
        }

        if (results.precedingEvents && results.precedingEvents.length > 0) {
            const tables = [...new Set(results.precedingEvents.map(e => e.sourceTable))];
            chain.push({
                step: step++,
                severity: 'medium',
                finding: `Change events detected before ${results.precedingEvents.length} defect spike(s)`,
                details: `Preceding events found in: ${tables.join(', ')}`,
                recommendation: 'Cross-reference these events with change logs (ECO/ECR, maintenance, supplier lot changes).',
            });
        }

        if (results.capaKPIs) {
            const k = results.capaKPIs;
            if (k.recurrenceRate !== null) {
                const sev = k.recurrenceRate > 25 ? 'high' : k.recurrenceRate > 10 ? 'medium' : 'low';
                chain.push({
                    step: step++,
                    severity: sev,
                    finding: `CAPA Recurrence Rate: ${k.recurrenceRate}%`,
                    details: k.recurrenceRate > 25
                        ? `⚠️ High recurrence — corrective actions are not addressing root cause.`
                        : k.recurrenceRate > 10
                        ? `⚠️ Moderate recurrence — verify containment actions are sustaining.`
                        : `✓ Recurrence within acceptable range (<10%).`,
                    recommendation: k.recurrenceRate > 10
                        ? 'Re-evaluate root cause depth. Use 5-Why or Fishbone to go deeper.'
                        : 'Continue monitoring for 90 days to confirm sustained improvement.',
                });
            }
            if (k.effectiveness !== null) {
                const sev = k.effectiveness < 30 ? 'high' : k.effectiveness < 60 ? 'medium' : 'low';
                chain.push({
                    step: step++,
                    severity: sev,
                    finding: `CAPA Effectiveness: ${k.effectiveness}% defect rate reduction`,
                    details: k.effectiveness < 30
                        ? `⚠️ Low effectiveness — defect rate barely improved after corrective actions.`
                        : k.effectiveness < 60
                        ? `⚠️ Partial effectiveness — some improvement but room to do better.`
                        : `✓ Strong effectiveness — actions have significantly reduced defect rate.`,
                    recommendation: k.effectiveness < 50
                        ? 'Review corrective action quality. Ensure actions target verified root causes, not symptoms.'
                        : 'Good. Standardize this fix as a Best Practice and deploy to similar process lines.',
                });
            }
            if (k.escapeRate > 1) {
                chain.push({
                    step: step++,
                    severity: k.escapeRate > 5 ? 'high' : 'medium',
                    finding: `Escape Rate: ${k.escapeRate}% (${k.escapedCount} escapes)`,
                    details: `Defects with keywords matching customer/field/warranty found.`,
                    recommendation: 'Strengthen containment detection. Add automated inspection gates at process exit points.',
                });
            }
        }

        return chain;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Main entry point
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Run the full RCA pipeline.
     *
     * @param {object} db          - GridDB WebGPU instance
     * @param {object} config      - {
     *   defectTable, defectTsCol, defectCountCol, defectTypeCol,
     *   capaTable, capaTsCol, capaStatusCol, capaTypeCol,
     *   factorTables: [{ name, tsCol }]
     * }
     * @returns {Promise<object>}  - Full analysis results
     */
    async analyze(db, config) {
        const {
            defectTable, defectTsCol, defectCountCol, defectTypeCol,
            capaTable,   capaTsCol,   capaStatusCol,  capaTypeCol,
            factorTables = [],
        } = config;

        if (!defectTable) throw new Error('defectTable is required');

        const results = {
            defectSummary: null,
            defectWindows: [],
            correlations: [],
            categoricalCorrelations: [],
            precedingEvents: [],
            capaKPIs: null,
            causalChain: [],
            config,
        };

        // ── 1. Load defect table ──────────────────────────────────────────────
        const defRes = await db.query(`SELECT * FROM ${defectTable}`);
        const defectRows = defRes.rows || [];
        if (defectRows.length === 0) throw new Error(`Table "${defectTable}" is empty.`);
        const defCols = Object.keys(defectRows[0]);

        // ── 2. Build defect time series & detect spikes ───────────────────────
        let defSeries = [];
        if (defectTsCol) {
            const bucket = defectRows.length > 5000 ? 'day' : 'hour';
            defSeries = this._bucketByTime(defectRows, defectTsCol, defectCountCol, bucket);
            results.defectWindows = this._detectDefectWindows(defSeries);
        }

        const totalDefects = defectCountCol
            ? defectRows.reduce((a, r) => a + (parseFloat(r[defectCountCol]) || 0), 0)
            : defectRows.length;

        results.defectSummary = {
            table: defectTable,
            totalRows: defectRows.length,
            totalDefects,
            timeSeries: defSeries,
            columns: defCols,
            topDefectTypes: this._topCategories(defectRows, defectTypeCol, 8),
            columnStats: this._columnStatsSummary(defectRows, defCols),
        };

        // ── 3. Factor correlation analysis ────────────────────────────────────
        for (const ft of factorTables) {
            if (!ft.name || !ft.tsCol) continue;
            const fRes = await db.query(`SELECT * FROM ${ft.name}`);
            const fRows = fRes.rows || [];
            if (!fRows.length) continue;

            const fCols = Object.keys(fRows[0]);
            const numCols = fCols.filter(c => c !== ft.tsCol && this._isNumericCol(fRows, c));
            const catCols = fCols.filter(c => c !== ft.tsCol && !this._isNumericCol(fRows, c)).slice(0, 5);

            if (defSeries.length > 0 && numCols.length > 0) {
                const corrs = this._correlateNumericFactors(defSeries, fRows, ft.tsCol, numCols);
                corrs.forEach(c => { c.sourceTable = ft.name; });
                results.correlations.push(...corrs);
            }

            if (results.defectWindows.length > 0 && catCols.length > 0) {
                const catCorrs = this._correlateCategoricalFactors(results.defectWindows, fRows, ft.tsCol, catCols);
                catCorrs.forEach(c => { c.sourceTable = ft.name; });
                results.categoricalCorrelations.push(...catCorrs);
            }

            if (results.defectWindows.length > 0 && ft.tsCol) {
                const preceding = this._findPrecedingEvents(results.defectWindows, fRows, ft.tsCol);
                preceding.forEach(p => { p.sourceTable = ft.name; });
                results.precedingEvents.push(...preceding);
            }
        }

        results.correlations.sort((a, b) => b.absR - a.absR);

        // ── 4. CAPA KPIs ──────────────────────────────────────────────────────
        if (capaTable && capaTsCol && capaStatusCol && defectTsCol) {
            const capaRes = await db.query(`SELECT * FROM ${capaTable}`);
            const capaRows = capaRes.rows || [];
            if (capaRows.length > 0) {
                results.capaKPIs = this._calculateCapaKPIs(defectRows, capaRows, {
                    defectTsCol, defectTypeCol, defectCountCol,
                    capaTsCol, capaStatusCol, capaTypeCol,
                    capaClosedVal: 'closed', windowDays: 30,
                });
            }
        }

        // ── 5. Causal chain ───────────────────────────────────────────────────
        results.causalChain = this._buildCausalChain(results);

        return results;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────────────────

    _columnStatsSummary(rows, cols) {
        const summary = {};
        for (const col of cols) {
            const vals = rows.map(r => r[col]).filter(v => v !== null && v !== '');
            const numVals = vals.map(v => parseFloat(v)).filter(v => !isNaN(v));
            if (numVals.length / (vals.length || 1) > 0.7) {
                summary[col] = { type: 'numeric', ...this.stats(numVals), count: numVals.length };
            } else {
                const dist = {};
                vals.forEach(v => { dist[v] = (dist[v] || 0) + 1; });
                summary[col] = {
                    type: 'categorical',
                    uniqueCount: Object.keys(dist).length,
                    topValues: Object.entries(dist).sort((a, b) => b[1] - a[1]).slice(0, 5),
                };
            }
        }
        return summary;
    }
}
