/**
 * AI-Powered Insights Analyzer
 * 
 * Analyzes query results to generate actionable insights about:
 * - Quality metrics (defect rates, yield)
 * - Cost savings potential
 * - Data quality issues
 * - Actionable recommendations
 */

export class InsightsAnalyzer {
    constructor() {
        this.currentInsights = null;
    }

    /**
     * Generate insights from query results
     * @param {Object} results - Query results with rows array
     */
    async generate(results) {
        if (!results || !results.rows || results.rows.length === 0) {
            throw new Error('No results to analyze. Run a query first.');
        }

        const data = results.rows;
        const columns = Object.keys(data[0]);
        
        // Analyze dataset
        this.currentInsights = await this.analyzeDataset(data, columns);
        
        // Show insights modal
        this.showModal(this.currentInsights, data, columns);
        
        console.log('✨ Insights generated:', this.currentInsights);
        
        return this.currentInsights;
    }

    /**
     * Analyze dataset for insights
     */
    async analyzeDataset(data, columns) {
        const insights = {
            overview: {
                totalRows: data.length,
                totalColumns: columns.length,
                completeness: 0
            },
            numericColumns: [],
            categoricalColumns: [],
            qualityMetrics: {},
            costSavings: {},
            recommendations: [],
            keyFindings: []
        };

        // Analyze each column
        columns.forEach(col => {
            const values = data.map(row => row[col]);
            const nonNull = values.filter(v => v !== null && v !== undefined && v !== '');
            const completeness = (nonNull.length / values.length) * 100;
            
            insights.overview.completeness += completeness / columns.length;
            
            // Check if numeric
            const numericValues = nonNull.filter(v => !isNaN(parseFloat(v))).map(v => parseFloat(v));
            const isNumeric = numericValues.length / nonNull.length > 0.8;
            
            if (isNumeric && numericValues.length > 0) {
                const min = Math.min(...numericValues);
                const max = Math.max(...numericValues);
                const avg = numericValues.reduce((a, b) => a + b, 0) / numericValues.length;
                const sorted = [...numericValues].sort((a, b) => a - b);
                const median = sorted[Math.floor(sorted.length / 2)];
                
                insights.numericColumns.push({
                    name: col,
                    min, max, avg, median,
                    count: numericValues.length,
                    completeness
                });
                
                // Quality-related metrics
                if (col.toLowerCase().includes('defect') || col.toLowerCase().includes('error')) {
                    const defectRate = (avg / max) * 100;
                    insights.qualityMetrics.defectRate = defectRate;
                    insights.qualityMetrics.defectColumn = col;
                }
                
                if (col.toLowerCase().includes('yield') || col.toLowerCase().includes('fpy')) {
                    insights.qualityMetrics.yield = avg;
                    insights.qualityMetrics.yieldColumn = col;
                }
                
            } else {
                // Categorical
                const unique = new Set(nonNull);
                const distribution = {};
                nonNull.forEach(v => {
                    distribution[v] = (distribution[v] || 0) + 1;
                });
                
                insights.categoricalColumns.push({
                    name: col,
                    uniqueCount: unique.size,
                    topValues: Object.entries(distribution)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 5),
                    completeness
                });
            }
        });

        // Calculate cost savings potential
        this.calculateCostSavings(insights, data);
        
        // Generate recommendations
        this.generateRecommendations(insights, data);
        
        // Key findings
        this.generateKeyFindings(insights, data);
        
        return insights;
    }

    /**
     * Calculate cost savings based on quality metrics
     */
    calculateCostSavings(insights, data) {
        const savings = {
            scrapReduction: 0,
            warrantyReduction: 0,
            yieldImprovement: 0,
            totalPotential: 0,
            description: ''
        };

        // Defect rate impact
        if (insights.qualityMetrics.defectRate) {
            const currentDefectRate = insights.qualityMetrics.defectRate;
            const targetDefectRate = Math.max(0.5, currentDefectRate * 0.7); // 30% reduction target
            const improvement = currentDefectRate - targetDefectRate;
            
            // Assume $500 per defect (scrap, rework, warranty)
            const annualDefects = (currentDefectRate / 100) * data.length * 52; // Weekly to annual
            const potentialSavings = annualDefects * improvement / currentDefectRate * 500;
            
            savings.scrapReduction = potentialSavings * 0.6;
            savings.warrantyReduction = potentialSavings * 0.3;
            savings.totalPotential += potentialSavings;
        }

        // Yield improvement
        if (insights.qualityMetrics.yield) {
            const currentYield = insights.qualityMetrics.yield;
            const targetYield = Math.min(99.5, currentYield + 2); // +2% yield target
            const improvement = targetYield - currentYield;
            
            // Each 1% yield improvement = ~$10k annually per production line
            const yieldSavings = improvement * 10000;
            
            savings.yieldImprovement = yieldSavings;
            savings.totalPotential += yieldSavings;
        }

        // If no specific metrics, estimate based on data size
        if (savings.totalPotential === 0) {
            savings.totalPotential = data.length * 25; // $25 per record improvement
            savings.description = 'Estimated based on process optimization';
        }

        insights.costSavings = savings;
    }

    /**
     * Generate actionable recommendations
     */
    generateRecommendations(insights, data) {
        const recommendations = [];

        // Quality recommendations
        if (insights.qualityMetrics.defectRate > 5) {
            recommendations.push({
                priority: 'HIGH',
                category: 'Quality Control',
                title: 'Reduce Defect Rate',
                description: `Current defect rate of ${insights.qualityMetrics.defectRate.toFixed(1)}% is above industry standard (3-5%). Implement early defect detection.`,
                impact: 'High',
                effort: 'Medium',
                savings: `$${(insights.costSavings.scrapReduction).toLocaleString()}/year`
            });
        }

        if (insights.qualityMetrics.yield && insights.qualityMetrics.yield < 95) {
            recommendations.push({
                priority: 'HIGH',
                category: 'Process Efficiency',
                title: 'Improve First Pass Yield',
                description: `Current FPY of ${insights.qualityMetrics.yield.toFixed(1)}% has room for improvement. Target 97%+.`,
                impact: 'High',
                effort: 'Medium',
                savings: `$${(insights.costSavings.yieldImprovement).toLocaleString()}/year`
            });
        }

        // Data quality recommendations
        if (insights.overview.completeness < 90) {
            recommendations.push({
                priority: 'MEDIUM',
                category: 'Data Quality',
                title: 'Improve Data Completeness',
                description: `Only ${insights.overview.completeness.toFixed(0)}% of data is complete. Missing data impacts analysis accuracy.`,
                impact: 'Medium',
                effort: 'Low',
                savings: 'Enables better insights'
            });
        }

        // Statistical recommendations
        const highVarianceColumns = insights.numericColumns.filter(col => {
            const range = col.max - col.min;
            const cv = range / col.avg;
            return cv > 0.5;
        });

        if (highVarianceColumns.length > 0) {
            recommendations.push({
                priority: 'MEDIUM',
                category: 'Process Consistency',
                title: 'Reduce Process Variation',
                description: `High variance detected in ${highVarianceColumns[0].name}. Standardize process parameters.`,
                impact: 'Medium',
                effort: 'Medium',
                savings: 'Improved quality consistency'
            });
        }

        insights.recommendations = recommendations;
    }

    /**
     * Generate key findings
     */
    generateKeyFindings(insights, data) {
        const findings = [];

        // Total potential savings
        findings.push({
            icon: '💰',
            title: 'Annual Savings Potential',
            value: `$${insights.costSavings.totalPotential.toLocaleString()}`,
            description: 'Through quality improvements and waste reduction'
        });

        // Defect rate
        if (insights.qualityMetrics.defectRate) {
            findings.push({
                icon: '🎯',
                title: 'Current Defect Rate',
                value: `${insights.qualityMetrics.defectRate.toFixed(2)}%`,
                description: `Target: <3% | Detected in ${insights.qualityMetrics.defectColumn}`
            });
        }

        // Yield
        if (insights.qualityMetrics.yield) {
            findings.push({
                icon: '📈',
                title: 'First Pass Yield',
                value: `${insights.qualityMetrics.yield.toFixed(1)}%`,
                description: `Target: 97%+ | From ${insights.qualityMetrics.yieldColumn}`
            });
        }

        // Data coverage
        findings.push({
            icon: '📊',
            title: 'Data Completeness',
            value: `${insights.overview.completeness.toFixed(0)}%`,
            description: `${insights.overview.totalRows.toLocaleString()} rows analyzed across ${insights.overview.totalColumns} columns`
        });

        insights.keyFindings = findings;
    }

    /**
     * Show insights modal
     */
    showModal(insights, data, columns) {
        const modal = document.createElement('div');
        modal.className = 'insights-modal';
        modal.id = 'insights-modal';
        
        modal.innerHTML = `
            <div class="insights-content" style="width: 1400px; max-width: 95vw; height: 90vh; display: flex; flex-direction: column;">
                <!-- Header -->
                <div class="p-8 border-b border-white/10 flex justify-between items-start bg-gradient-to-br from-[#11181C] to-[#1a2632]">
                    <div class="flex-1">
                        <div class="flex items-center gap-3 mb-2">
                            <div class="w-12 h-12 bg-gradient-to-br from-[#34B27B] to-[#4ECDC4] rounded-xl flex items-center justify-center text-2xl">
                                ✨
                            </div>
                            <div>
                                <h2 class="text-3xl font-black outfit text-white">AI-Powered Insights</h2>
                                <p class="text-sm text-white/50 mt-1">Quality & Cost Analysis Report</p>
                            </div>
                        </div>
                    </div>
                    <button onclick="document.getElementById('insights-modal').remove()" 
                            class="text-white/50 hover:text-white text-3xl leading-none transition-colors">×</button>
                </div>

                <!-- Content -->
                <div class="flex-1 overflow-y-auto p-8" style="background: #0a0a0a;">
                    
                    <!-- Key Findings Grid -->
                    <div class="grid grid-cols-4 gap-4 mb-8">
                        ${insights.keyFindings.map(finding => `
                            <div class="metric-card">
                                <div class="text-3xl mb-3">${finding.icon}</div>
                                <div class="text-sm text-white/50 uppercase tracking-wider mb-2">${finding.title}</div>
                                <div class="metric-value mb-2">${finding.value}</div>
                                <div class="text-xs text-white/40">${finding.description}</div>
                            </div>
                        `).join('')}
                    </div>

                    <!-- Cost Savings Section -->
                    <div class="savings-highlight mb-8 premium-glow">
                        <div class="flex items-start gap-4">
                            <div class="text-4xl">💵</div>
                            <div class="flex-1">
                                <h3 class="text-xl font-bold text-yellow-400 mb-3">Why This Saves Money</h3>
                                <div class="grid grid-cols-2 gap-6 text-sm text-white/80">
                                    <div>
                                        <div class="font-bold text-white mb-2">📉 Cost Reduction Areas:</div>
                                        <ul class="space-y-1 ml-4">
                                            <li>• Reduces scrap and rework by detecting defects early</li>
                                            <li>• Lowers warranty claims and returns</li>
                                            <li>• Improves first pass yield → fewer materials wasted</li>
                                            <li>• Minimizes production downtime</li>
                                        </ul>
                                    </div>
                                    <div>
                                        <div class="font-bold text-white mb-2">📊 Key Metrics to Track:</div>
                                        <ul class="space-y-1 ml-4">
                                            <li>• Defect rates by process/product</li>
                                            <li>• First Pass Yield (FPY)</li>
                                            <li>• CAPA status & closure</li>
                                            <li>• Customer complaints</li>
                                        </ul>
                                    </div>
                                </div>
                                <div class="mt-4 p-3 bg-black/30 rounded-lg border border-yellow-500/30">
                                    <div class="flex items-center gap-2 text-yellow-300 font-bold">
                                        <span>💡</span>
                                        <span>Impact:</span>
                                        <span class="text-yellow-400">Every percentage point increase in FPY can save thousands in material, labor, and logistics.</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Recommendations -->
                    <div class="insight-card mb-8">
                        <h3 class="text-xl font-bold text-white mb-4 flex items-center gap-2">
                            <span>🎯</span>
                            <span>Actionable Recommendations</span>
                        </h3>
                        <div class="space-y-3">
                            ${insights.recommendations.map(rec => `
                                <div class="recommendation-item">
                                    <div class="flex items-start justify-between mb-2">
                                        <div class="flex items-center gap-3">
                                            <span class="px-2 py-1 rounded text-xs font-bold ${
                                                rec.priority === 'HIGH' ? 'bg-red-500/20 text-red-400' :
                                                rec.priority === 'MEDIUM' ? 'bg-yellow-500/20 text-yellow-400' :
                                                'bg-green-500/20 text-green-400'
                                            }">${rec.priority}</span>
                                            <span class="text-sm font-semibold text-white/70">${rec.category}</span>
                                        </div>
                                        <span class="text-xs text-[#34B27B] font-bold">${rec.savings}</span>
                                    </div>
                                    <div class="text-base font-bold text-white mb-1">${rec.title}</div>
                                    <div class="text-sm text-white/60">${rec.description}</div>
                                    <div class="flex gap-4 mt-2 text-xs">
                                        <span class="text-white/40">Impact: <span class="text-white/70">${rec.impact}</span></span>
                                        <span class="text-white/40">Effort: <span class="text-white/70">${rec.effort}</span></span>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>

                    <!-- Statistical Summary -->
                    <div class="grid grid-cols-2 gap-6">
                        <!-- Numeric Columns -->
                        ${insights.numericColumns.length > 0 ? `
                        <div class="insight-card">
                            <h3 class="text-lg font-bold text-white mb-3 flex items-center gap-2">
                                <span>🔢</span>
                                <span>Numeric Analysis</span>
                            </h3>
                            <div class="space-y-3">
                                ${insights.numericColumns.slice(0, 5).map(col => `
                                    <div class="p-3 bg-white/3 rounded-lg">
                                        <div class="font-bold text-sm text-white/90 mb-2">${col.name}</div>
                                        <div class="grid grid-cols-2 gap-2 text-xs">
                                            <div><span class="text-white/40">Min:</span> <span class="text-[#34B27B]">${col.min.toFixed(2)}</span></div>
                                            <div><span class="text-white/40">Max:</span> <span class="text-[#34B27B]">${col.max.toFixed(2)}</span></div>
                                            <div><span class="text-white/40">Avg:</span> <span class="text-[#34B27B]">${col.avg.toFixed(2)}</span></div>
                                            <div><span class="text-white/40">Median:</span> <span class="text-[#34B27B]">${col.median.toFixed(2)}</span></div>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        ` : ''}

                        <!-- Categorical Columns -->
                        ${insights.categoricalColumns.length > 0 ? `
                        <div class="insight-card">
                            <h3 class="text-lg font-bold text-white mb-3 flex items-center gap-2">
                                <span>🏷️</span>
                                <span>Categorical Analysis</span>
                            </h3>
                            <div class="space-y-3">
                                ${insights.categoricalColumns.slice(0, 5).map(col => `
                                    <div class="p-3 bg-white/3 rounded-lg">
                                        <div class="font-bold text-sm text-white/90 mb-2">${col.name}</div>
                                        <div class="text-xs text-white/40 mb-2">${col.uniqueCount} unique values</div>
                                        <div class="space-y-1">
                                            ${col.topValues.slice(0, 3).map(([value, count]) => `
                                                <div class="flex justify-between items-center text-xs">
                                                    <span class="text-white/60 truncate max-w-[180px]">${value}</span>
                                                    <span class="text-[#34B27B] font-bold">${count}</span>
                                                </div>
                                            `).join('')}
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        ` : ''}
                    </div>
                </div>

                <!-- Footer -->
                <div class="p-6 border-t border-white/10 flex justify-between items-center bg-[#11181C]">
                    <div class="text-sm text-white/50">
                        Analysis generated on ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}
                    </div>
                    <button onclick="alert('📄 Insights report export coming soon!')" class="px-6 py-2 bg-gradient-to-r from-[#34B27B] to-[#4ECDC4] hover:from-[#2a9463] hover:to-[#3ba89f] rounded-lg font-bold text-white transition-all shadow-lg">
                        Export Report
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
    }

    /**
     * Export insights report
     */
    exportReport() {
        if (!this.currentInsights) {
            throw new Error('No insights to export');
        }
        
        // TODO: Implement report export (PDF/Excel)
        console.log('📄 Exporting insights report...');
        return this.currentInsights;
    }
}
