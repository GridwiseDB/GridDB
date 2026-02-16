# 🎯 AI-Powered Insights Analyzer

## Overview

The **Insights Analyzer** module provides AI-powered data analysis for GridDB query results, generating actionable insights about quality metrics, cost savings, data quality, and process optimization.

## Features

### 📊 Analysis Capabilities

1. **Statistical Analysis**
   - Numeric column statistics (min, max, average, median)
   - Categorical distribution analysis
   - Data completeness metrics
   - Variance and consistency checks

2. **Quality Metrics Detection**
   - Defect rate identification
   - First Pass Yield (FPY) analysis
   - Error pattern detection
   - Quality trend analysis

3. **Cost Savings Calculation**
   - Scrap reduction potential
   - Warranty cost reduction
   - Yield improvement value
   - ROI projections

4. **Smart Recommendations**
   - Priority-based action items
   - Impact & effort estimation
   - Category-specific guidance
   - Implementation roadmap

## Usage

### Basic Usage

```javascript
import { InsightsAnalyzer } from './src/features/insights/insights-analyzer.js';

// Initialize
const insightsAnalyzer = new InsightsAnalyzer();

// Generate insights from query results
const results = await db.query('SELECT * FROM production_data');
await insightsAnalyzer.generate(results);
```

### HTML Integration

The INSIGHTS button in the UI automatically triggers analysis:

```html
<button onclick="generateInsights()">✨ INSIGHTS</button>
```

### Manual Export

```javascript
// Export insights data
const insights = insightsAnalyzer.exportReport();
console.log(insights);
```

## Insights Output Structure

```javascript
{
  overview: {
    totalRows: 1000,
    totalColumns: 15,
    completeness: 95.5
  },
  numericColumns: [
    {
      name: "yield_rate",
      min: 85.2,
      max: 99.8,
      avg: 94.3,
      median: 95.1,
      count: 1000,
      completeness: 98.5
    }
  ],
  categoricalColumns: [
    {
      name: "product_line",
      uniqueCount: 5,
      topValues: [["Product A", 450], ["Product B", 320]],
      completeness: 100
    }
  ],
  qualityMetrics: {
    defectRate: 2.3,
    defectColumn: "defect_count",
    yield: 94.3,
    yieldColumn: "fpy"
  },
  costSavings: {
    scrapReduction: 45000,
    warrantyReduction: 22500,
    yieldImprovement: 20000,
    totalPotential: 87500
  },
  recommendations: [
    {
      priority: "HIGH",
      category: "Quality Control",
      title: "Reduce Defect Rate",
      description: "Current defect rate of 2.3% needs improvement",
      impact: "High",
      effort: "Medium",
      savings: "$45,000/year"
    }
  ],
  keyFindings: [
    {
      icon: "💰",
      title: "Annual Savings Potential",
      value: "$87,500",
      description: "Through quality improvements"
    }
  ]
}
```

## Column Recognition

The analyzer automatically detects quality-related columns:

- **Defect Detection**: Columns containing "defect", "error", "failure"
- **Yield Detection**: Columns containing "yield", "fpy", "pass"
- **Quality Metrics**: Columns with "quality", "qc", "inspection"

## Visual Modal Features

### Key Metrics Cards
- 💰 Annual savings potential
- 🎯 Current defect rate
- 📈 First pass yield
- 📊 Data completeness

### Cost Savings Breakdown
- 📉 Scrap & rework reduction
- 🔧 Warranty claim reduction  
- ⚡ Yield improvement value
- 💡 Process optimization tips

### Recommendations
- Priority ranking (HIGH/MEDIUM/LOW)
- Impact & effort assessment
- Category-based grouping
- Projected savings per action

### Statistical Summary
- Numeric column deep-dive
- Categorical distribution analysis
- Top values & frequencies
- Completeness tracking

## Customization

### Adding Custom Metrics

```javascript
// Extend the analyzer
class CustomInsightsAnalyzer extends InsightsAnalyzer {
  analyzeDataset(data, columns) {
    const insights = super.analyzeDataset(data, columns);
    
    // Add custom metrics
    insights.customMetrics = {
      // Your custom analysis
    };
    
    return insights;
  }
}
```

### Custom Recommendation Rules

Modify `generateRecommendations()` to add domain-specific rules:

```javascript
// In insights-analyzer.js
if (insights.customMetric > threshold) {
  recommendations.push({
    priority: 'HIGH',
    category: 'Custom Category',
    title: 'Your Title',
    description: 'Your description',
    impact: 'High',
    effort: 'Low',
    savings: '$X/year'
  });
}
```

## Industry Applications

### Manufacturing
- Defect rate optimization
- First Pass Yield tracking
- Scrap reduction strategies
- Process consistency monitoring

### Quality Control
- CAPA effectiveness
- Customer complaint trends
- Warranty claim analysis
- Inspection failure patterns

### Process Improvement
- Cycle time reduction
- Resource utilization
- Bottleneck identification
- Efficiency metrics

## Cost Savings Methodology

### Defect Rate Impact
```
Annual Savings = (Current Defects × Improvement %) × Cost per Defect
Default: $500 per defect (scrap + rework + warranty)
```

### Yield Improvement
```
Yield Savings = Improvement % × $10,000 per production line
Industry standard: 1% yield improvement = $10k annually
```

### Default Estimation
If no specific metrics detected:
```
Estimated Savings = Row Count × $25 per record
Based on general process optimization potential
```

## Best Practices

1. **Run on Representative Data**: Use recent production data for accurate insights
2. **Check Data Quality**: Ensure >90% completeness for reliable analysis
3. **Review Recommendations**: Prioritize HIGH impact, LOW effort items first
4. **Track Improvements**: Re-run analysis after implementing changes
5. **Export Reports**: Save insights for management review

## Future Enhancements

- [ ] PDF/Excel report export
- [ ] Historical trend comparison
- [ ] Machine learning predictions
- [ ] Custom rule engine
- [ ] Industry templates
- [ ] Multi-dataset correlation

## Dependencies

- None (pure JavaScript ES6 module)
- Works with any GridDB query results
- Compatible with all modern browsers

## File Location

```
/src/features/insights/insights-analyzer.js
```

## Integration

Already integrated in `griddb_modern.html`:

```javascript
import { InsightsAnalyzer } from './src/features/insights/insights-analyzer.js';
const insightsAnalyzer = new InsightsAnalyzer();
```

## Support

For issues or feature requests, see main GridDB documentation.
