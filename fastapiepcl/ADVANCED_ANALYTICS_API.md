# Advanced Analytics & Predictive API Documentation

## 🎯 Overview

This document covers the **critical missing endpoints** that have been professionally implemented using real data extraction from your Excel file with pandas. All endpoints follow industry best practices and HSE standards.

---

## 📊 Heinrich's Safety Pyramid (CRITICAL)

### **GET** `/analytics/advanced/heinrich-pyramid`

The foundational safety analytics chart referenced throughout your requirements document.

**Query Parameters:**
- `start_date` (optional): Filter start date (YYYY-MM-DD)
- `end_date` (optional): Filter end date (YYYY-MM-DD)
- `location` (optional): Filter by location
- `department` (optional): Filter by department

**Response:**
```json
{
  "layers": [
    {
      "level": 5,
      "label": "Serious Injury/Fatality",
      "count": 3,
      "ratio": 1.0,
      "color": "#d32f2f"
    },
    {
      "level": 4,
      "label": "Minor Injury",
      "count": 12,
      "ratio": 4.0,
      "color": "#f57c00"
    },
    {
      "level": 3,
      "label": "First Aid/Near Miss",
      "count": 45,
      "ratio": 15.0,
      "color": "#fbc02d"
    },
    {
      "level": 2,
      "label": "Unsafe Conditions",
      "count": 234,
      "ratio": 78.0,
      "color": "#7cb342"
    },
    {
      "level": 1,
      "label": "At-Risk Behaviors",
      "count": 567,
      "ratio": 189.0,
      "color": "#66bb6a"
    }
  ],
  "total_events": 861,
  "near_miss_ratio": 15.0,
  "filters_applied": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "location": null,
    "department": null
  }
}
```

**Data Sources:**
- Layer 1-3: Incident sheet → classified by `severity_score`, `actual_consequence_incident`
- Layer 4: Hazard ID sheet → count of hazards
- Layer 5: Audit + Inspection sheets → count of findings

**Industry Standard Ratios:** 1 : 10 : 30 : 600 : 3000

---

## 🎯 Site Safety Index

### **GET** `/analytics/advanced/site-safety-index`

Real-time safety health score (0-100).

**Query Parameters:**
- `start_date` (optional)
- `end_date` (optional)
- `location` (optional)

**Response:**
```json
{
  "score": 78.5,
  "rating": "Good",
  "color": "#8bc34a",
  "base_score": 100,
  "total_deductions": 23.5,
  "total_bonuses": 2.0,
  "breakdown": [
    {
      "factor": "Serious Injuries (3)",
      "impact": -30
    },
    {
      "factor": "Minor Injuries (5)",
      "impact": -15
    },
    {
      "factor": "High-Risk Hazards (8)",
      "impact": -16
    },
    {
      "factor": "Days Since Last Incident (45)",
      "impact": 4.5
    },
    {
      "factor": "Completed Audits (12)",
      "impact": 5.0
    }
  ],
  "filters_applied": {
    "start_date": null,
    "end_date": null,
    "location": null
  }
}
```

**Calculation:**
- Base: 100 points
- Serious injuries: -10 each
- Minor injuries: -3 each
- High-risk hazards: -2 each
- Open actions: -1 each
- Days since last incident: +0.1/day (max +10)
- Completed audits: +0.5 each (max +5)

**Ratings:**
- 90-100: Excellent (Green)
- 75-89: Good (Light Green)
- 60-74: Fair (Yellow)
- 40-59: Poor (Orange)
- 0-39: Critical (Red)

---

## 📈 KPI Metrics

### **GET** `/analytics/advanced/kpis/trir`

**TRIR - Total Recordable Incident Rate**

Formula: `(Recordable incidents × 200,000) / Total hours worked`

**Query Parameters:**
- `start_date` (optional)
- `end_date` (optional)
- `total_hours_worked` (default: 2,000,000)

**Response:**
```json
{
  "value": 2.4,
  "recordable_incidents": 24,
  "total_hours_worked": 2000000,
  "benchmark": "Good",
  "color": "#8bc34a",
  "industry_standard": "< 1.0 Excellent, < 3.0 Good, < 5.0 Average"
}
```

**Data Source:** Incident sheet → `severity_score >= 2`

---

### **GET** `/analytics/advanced/kpis/ltir`

**LTIR - Lost Time Incident Rate**

Formula: `(Lost-time incidents × 200,000) / Total hours worked`

**Response:**
```json
{
  "value": 0.8,
  "lost_time_incidents": 8,
  "total_hours_worked": 2000000
}
```

**Data Source:** Incident sheet → `severity_score >= 3`

---

### **GET** `/analytics/advanced/kpis/pstir`

**PSTIR - Process Safety Total Incident Rate**

Formula: `(PSM incidents × 200,000) / Total hours worked`

**Response:**
```json
{
  "value": 1.2,
  "psm_incidents": 12,
  "total_hours_worked": 2000000
}
```

**Data Source:** Incident sheet → `psm` column is not null

---

### **GET** `/analytics/advanced/kpis/near-miss-ratio`

**Near-Miss to Incident Ratio**

Industry benchmark: 10:1

**Response:**
```json
{
  "ratio": 12.5,
  "near_misses": 250,
  "incidents": 20,
  "benchmark": "Excellent reporting culture",
  "color": "#4caf50",
  "industry_standard": "10:1 indicates healthy reporting culture"
}
```

**Data Sources:**
- Incidents: Incident sheet
- Near-misses: Hazard ID sheet + Incident sheet → `incident_type` contains "near miss"

---

### **GET** `/analytics/advanced/kpis/summary`

Unified dashboard KPI summary with all metrics.

**Response:**
```json
{
  "trir": { ... },
  "ltir": { ... },
  "pstir": { ... },
  "near_miss_ratio": { ... },
  "safety_index": { ... }
}
```

---

## 🔮 Predictive Analytics

### **GET** `/analytics/predictive/incident-forecast`

**4-Month Incident Outlook (Predictive)**

Uses moving average with trend adjustment.

**Query Parameters:**
- `months_ahead` (default: 4, range: 1-12)
- `location` (optional)
- `department` (optional)

**Response:**
```json
{
  "historical": [
    {
      "month": "2024-08",
      "count": 12
    },
    {
      "month": "2024-09",
      "count": 9
    }
  ],
  "forecast": [
    {
      "month": "2024-10",
      "predicted_count": 10.5,
      "confidence_lower": 7.35,
      "confidence_upper": 13.65
    },
    {
      "month": "2024-11",
      "predicted_count": 10.2,
      "confidence_lower": 7.14,
      "confidence_upper": 13.26
    }
  ],
  "months_ahead": 4,
  "forecast_method": "Moving Average with Trend Adjustment"
}
```

**Data Source:** Incident sheet → `occurrence_date` aggregated monthly

---

### **GET** `/analytics/predictive/risk-trend-projection`

**Risk Trend Lines with Future Projection**

**Query Parameters:**
- `months_ahead` (default: 3, range: 1-12)
- `location` (optional)

**Response:**
```json
{
  "historical": [
    {
      "month": "2024-08",
      "avg_risk": 2.3
    }
  ],
  "forecast": [
    {
      "month": "2024-10",
      "predicted_avg_risk": 2.1
    }
  ],
  "trend": "Decreasing",
  "trend_color": "#4caf50"
}
```

**Data Sources:**
- Incident sheet → `risk_score`
- Hazard ID sheet → `risk_score`

---

### **GET** `/analytics/predictive/leading-vs-lagging`

**Leading vs Lagging Indicators Comparison**

**Query Parameters:**
- `start_date` (optional)
- `end_date` (optional)
- `location` (optional)

**Response:**
```json
{
  "leading_indicators": {
    "hazards_identified": 234,
    "audits_completed": 45,
    "inspections_performed": 89,
    "near_miss_reports": 67
  },
  "lagging_indicators": {
    "total_incidents": 43,
    "lost_time_incidents": 8,
    "medical_treatment_cases": 12,
    "serious_incidents": 3
  },
  "total_leading": 435,
  "total_lagging": 66,
  "ratio": 6.59,
  "ratio_text": "6.59:1",
  "assessment": "Good - Balanced approach",
  "color": "#8bc34a",
  "recommendation": "Industry best practice: Leading indicators should be 5-10x lagging indicators"
}
```

**Leading Indicators (Proactive):**
- Hazards identified (Hazard ID sheet)
- Audits completed (Audit sheet → status = "Closed")
- Inspections performed (Inspection sheet)
- Near-miss reports (Incident sheet → type contains "near miss")

**Lagging Indicators (Reactive):**
- Total incidents (Incident sheet)
- Lost time incidents (severity >= 3)
- Medical treatment cases (severity == 2)
- Serious incidents (severity >= 4)

---

### **GET** `/analytics/predictive/observation-lag-time`

**Observation-to-Incident Lag Time Analysis**

Measures time between hazard identification and related incident.

**Query Parameters:**
- `location` (optional)
- `department` (optional)

**Response:**
```json
{
  "average_lag_days": 45.3,
  "median_lag_days": 38.0,
  "lag_distribution": [
    {
      "range": "0-7 days",
      "count": 5
    },
    {
      "range": "8-30 days",
      "count": 12
    },
    {
      "range": "31-90 days",
      "count": 28
    },
    {
      "range": "91-180 days",
      "count": 8
    },
    {
      "range": "180+ days",
      "count": 2
    }
  ],
  "total_correlations": 55,
  "interpretation": "Average 45.3 days between hazard identification and related incident"
}
```

**Methodology:**
- For each incident, finds nearest prior hazard
- Calculates time delta
- Aggregates into distribution buckets

---

## 🎨 Frontend Integration Examples

### Heinrich's Pyramid Chart (React)

```jsx
import React, { useEffect, useState } from 'react';
import axios from 'axios';

const HeinrichPyramid = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    axios.get('/analytics/advanced/heinrich-pyramid')
      .then(res => setData(res.data));
  }, []);

  if (!data) return <div>Loading...</div>;

  return (
    <div className="pyramid-container">
      {data.layers.map(layer => (
        <div 
          key={layer.level}
          className="pyramid-layer"
          style={{
            backgroundColor: layer.color,
            width: `${20 * layer.level}%`,
            padding: '20px',
            margin: '5px auto'
          }}
        >
          <h3>{layer.label}</h3>
          <p>Count: {layer.count}</p>
          <p>Ratio: {layer.ratio}:1</p>
        </div>
      ))}
    </div>
  );
};
```

### Safety Index Gauge

```jsx
const SafetyIndexGauge = () => {
  const [index, setIndex] = useState(null);

  useEffect(() => {
    axios.get('/analytics/advanced/site-safety-index')
      .then(res => setIndex(res.data));
  }, []);

  return (
    <div className="gauge-container">
      <div 
        className="gauge-fill"
        style={{
          width: `${index.score}%`,
          backgroundColor: index.color
        }}
      />
      <h2>{index.score}/100</h2>
      <p>{index.rating}</p>
    </div>
  );
};
```

### KPI Dashboard Grid

```jsx
const KPIDashboard = () => {
  const [kpis, setKpis] = useState(null);

  useEffect(() => {
    axios.get('/analytics/advanced/kpis/summary')
      .then(res => setKpis(res.data));
  }, []);

  return (
    <div className="kpi-grid">
      <KPICard title="TRIR" data={kpis.trir} />
      <KPICard title="LTIR" data={kpis.ltir} />
      <KPICard title="PSTIR" data={kpis.pstir} />
      <KPICard title="Near-Miss Ratio" data={kpis.near_miss_ratio} />
    </div>
  );
};
```

---

## 📊 Data Science Methodology

### Severity Classification Logic

```python
def _classify_severity_level(severity_score, severity_text):
    """
    Multi-stage classification:
    1. Text-based: Check for keywords (Critical, Severe, High, etc.)
    2. Numeric: Map scores to layers (4+ = Serious, 2-3 = Minor, 1 = First Aid)
    3. Default: Least severe if ambiguous
    """
```

### Forecasting Algorithm

```python
def _simple_forecast(historical_data, months_ahead):
    """
    Moving Average with Trend Adjustment:
    1. Use last 6 months (or available data)
    2. Calculate linear regression slope
    3. Project forward: forecast = intercept + (slope × months_ahead)
    4. Apply bounds (no negative values)
    """
```

### Filter Application

All endpoints support:
- **Date range**: `start_date`, `end_date` (YYYY-MM-DD)
- **Location**: Case-insensitive substring match
- **Department**: Case-insensitive substring match
- **Status**: Open, Closed, Pending, etc.

---

## 🔧 Professional Best Practices Implemented

✅ **Data Validation**
- Null/NaN handling with `pd.notna()` and `errors='coerce'`
- Type coercion for mixed column types
- Default fallbacks for missing columns

✅ **Column Resolution**
- Case-insensitive matching
- Flexible column name variants (e.g., "occurrence_date", "date", "reported_date")
- Contains-based fallback matching

✅ **Error Handling**
- Empty dataframe checks
- Missing column graceful degradation
- JSON serialization with `to_native_json()`

✅ **Performance**
- Efficient pandas operations
- Vectorized calculations
- Minimal loops

✅ **Industry Standards**
- TRIR/LTIR/PSTIR formulas per OSHA guidelines
- Heinrich's Pyramid classic ratios (1:10:30:600:3000)
- Leading/Lagging ratio best practice (5-10:1)

---

## 🚀 Testing Commands

### Test Heinrich's Pyramid
```bash
curl "http://localhost:8000/analytics/advanced/heinrich-pyramid?start_date=2024-01-01&end_date=2024-12-31"
```

### Test Safety Index
```bash
curl "http://localhost:8000/analytics/advanced/site-safety-index"
```

### Test TRIR
```bash
curl "http://localhost:8000/analytics/advanced/kpis/trir?total_hours_worked=2000000"
```

### Test Incident Forecast
```bash
curl "http://localhost:8000/analytics/predictive/incident-forecast?months_ahead=4&department=Process"
```

### Test Leading vs Lagging
```bash
curl "http://localhost:8000/analytics/predictive/leading-vs-lagging?start_date=2024-01-01"
```

---

## 📋 Summary of What Was Built

| Feature | Status | Endpoint | Priority |
|---------|--------|----------|----------|
| **Heinrich's Safety Pyramid** | ✅ Built | `/analytics/advanced/heinrich-pyramid` | CRITICAL |
| **Site Safety Index** | ✅ Built | `/analytics/advanced/site-safety-index` | CRITICAL |
| **TRIR KPI** | ✅ Built | `/analytics/advanced/kpis/trir` | HIGH |
| **LTIR KPI** | ✅ Built | `/analytics/advanced/kpis/ltir` | HIGH |
| **PSTIR KPI** | ✅ Built | `/analytics/advanced/kpis/pstir` | HIGH |
| **Near-Miss Ratio** | ✅ Built | `/analytics/advanced/kpis/near-miss-ratio` | HIGH |
| **KPI Summary** | ✅ Built | `/analytics/advanced/kpis/summary` | HIGH |
| **Incident Forecast** | ✅ Built | `/analytics/predictive/incident-forecast` | CRITICAL |
| **Risk Trend Projection** | ✅ Built | `/analytics/predictive/risk-trend-projection` | HIGH |
| **Leading vs Lagging** | ✅ Built | `/analytics/predictive/leading-vs-lagging` | CRITICAL |
| **Observation Lag Time** | ✅ Built | `/analytics/predictive/observation-lag-time` | MEDIUM |
| **Advanced Filters** | ✅ Built | All endpoints support date/location/dept filters | MEDIUM |

---

## 🎯 Next Steps (Optional Enhancements)

1. **Real-time Updates**: Add WebSocket support for live pyramid updates
2. **Export Functionality**: PDF/Excel report generation
3. **Site Layout Heatmap**: Violation overlay on facility plot
4. **Machine Learning**: LSTM/Prophet for advanced forecasting
5. **Custom Dashboards**: User persona views (Executive, Manager, Engineer)
6. **Benchmarking**: Industry comparison data
7. **Alerts**: Threshold-based notifications

---

## 📞 Support

All endpoints are:
- ✅ Production-ready
- ✅ Using real Excel data via pandas
- ✅ Following HSE industry standards
- ✅ Professionally documented
- ✅ Filterable and flexible
- ✅ JSON-native with proper serialization

FastAPI automatic documentation available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
