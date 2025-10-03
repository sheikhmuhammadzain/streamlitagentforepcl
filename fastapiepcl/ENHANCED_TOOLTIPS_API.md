# Enhanced Chart Tooltips API Documentation

## Overview

The Enhanced Tooltips API provides detailed breakdown information for chart data points, enabling rich, informative tooltips when hovering over trend charts.

## Endpoint

### `GET /analytics/data/incident-trend-detailed`

Returns trend data with comprehensive monthly breakdowns including top departments, incident types, severity/risk statistics, and recent items.

---

## Request Parameters

All parameters are **optional** query parameters:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `dataset` | string | Dataset to use: `"incident"` or `"hazard"` | `incident` |
| `start_date` | string | Start date filter (YYYY-MM-DD) | `2023-01-01` |
| `end_date` | string | End date filter (YYYY-MM-DD) | `2023-12-31` |
| `departments` | array[string] | Filter by specific departments | `["Operations", "Maintenance"]` |
| `locations` | array[string] | Filter by specific locations | `["Plant A", "Warehouse"]` |
| `sublocations` | array[string] | Filter by specific sublocations | `["Zone 1", "Loading Bay"]` |
| `min_severity` | float | Minimum severity score (0-5) | `2.0` |
| `max_severity` | float | Maximum severity score (0-5) | `5.0` |
| `min_risk` | float | Minimum risk score (0-5) | `3.0` |
| `max_risk` | float | Maximum risk score (0-5) | `5.0` |
| `statuses` | array[string] | Filter by status values | `["Open", "Closed"]` |
| `incident_types` | array[string] | Filter by incident types | `["Slip", "Fall"]` |
| `violation_types` | array[string] | Filter by violation types (hazards) | `["PPE", "Housekeeping"]` |

---

## Response Schema

### Root Response

```typescript
{
  labels: string[];           // Month labels (YYYY-MM format)
  series: ChartSeries[];      // Chart data series
  details: MonthDetailedData[]; // Detailed breakdown per month
}
```

### ChartSeries

```typescript
{
  name: string;    // Series name (e.g., "Count")
  data: number[];  // Data points for each month
}
```

### MonthDetailedData

```typescript
{
  month: string;              // Month label (YYYY-MM)
  total_count: number;        // Total count for the month
  departments: CountItem[];   // Top 5 departments with counts
  types: CountItem[];         // Top 5 incident/violation types
  severity: ScoreStats | null; // Severity statistics
  risk: ScoreStats | null;    // Risk statistics
  recent_items: RecentItem[]; // Up to 5 most recent items
}
```

### CountItem

```typescript
{
  name: string;   // Department or type name
  count: number;  // Number of occurrences
}
```

### ScoreStats

```typescript
{
  avg: number;  // Average score
  max: number;  // Maximum score
  min: number;  // Minimum score
}
```

### RecentItem

```typescript
{
  title: string;         // Item title/description (max 100 chars)
  department: string;    // Department name
  date: string;          // Date (YYYY-MM-DD)
  severity: number | null; // Severity score (if available)
}
```

---

## Example Request

### Basic Request (Incidents)

```bash
GET /analytics/data/incident-trend-detailed?dataset=incident
```

### Filtered Request

```bash
GET /analytics/data/incident-trend-detailed?dataset=incident&start_date=2023-01-01&end_date=2023-12-31&departments=Operations&departments=Maintenance&min_severity=3.0
```

### Using cURL

```bash
curl -X GET "http://localhost:8000/analytics/data/incident-trend-detailed?dataset=incident&start_date=2023-01-01&end_date=2023-12-31" \
  -H "accept: application/json"
```

### Using JavaScript/Fetch

```javascript
const params = new URLSearchParams({
  dataset: 'incident',
  start_date: '2023-01-01',
  end_date: '2023-12-31',
  min_severity: '3.0'
});

const response = await fetch(`/analytics/data/incident-trend-detailed?${params}`);
const data = await response.json();
```

---

## Example Response

```json
{
  "labels": ["2023-01", "2023-02", "2023-03"],
  "series": [
    {
      "name": "Count",
      "data": [15, 22, 18]
    }
  ],
  "details": [
    {
      "month": "2023-01",
      "total_count": 15,
      "departments": [
        { "name": "Operations", "count": 8 },
        { "name": "Maintenance", "count": 5 },
        { "name": "Engineering", "count": 2 }
      ],
      "types": [
        { "name": "Slip", "count": 4 },
        { "name": "Fall", "count": 3 },
        { "name": "Equipment Failure", "count": 3 },
        { "name": "Near Miss", "count": 2 }
      ],
      "severity": {
        "avg": 3.2,
        "max": 5.0,
        "min": 1.0
      },
      "risk": {
        "avg": 3.8,
        "max": 5.0,
        "min": 2.0
      },
      "recent_items": [
        {
          "title": "Worker slipped on wet floor in production area",
          "department": "Operations",
          "date": "2023-01-28",
          "severity": 3.0
        },
        {
          "title": "Equipment malfunction in Zone 2",
          "department": "Maintenance",
          "date": "2023-01-25",
          "severity": 4.0
        },
        {
          "title": "Near miss at loading bay - forklift incident",
          "department": "Operations",
          "date": "2023-01-20",
          "severity": 2.0
        }
      ]
    },
    {
      "month": "2023-02",
      "total_count": 22,
      "departments": [
        { "name": "Operations", "count": 12 },
        { "name": "Maintenance", "count": 7 },
        { "name": "Engineering", "count": 3 }
      ],
      "types": [
        { "name": "Slip", "count": 6 },
        { "name": "Equipment Failure", "count": 5 },
        { "name": "Fall", "count": 4 }
      ],
      "severity": {
        "avg": 3.5,
        "max": 5.0,
        "min": 1.5
      },
      "risk": {
        "avg": 4.0,
        "max": 5.0,
        "min": 2.5
      },
      "recent_items": [
        {
          "title": "Serious injury - fall from height",
          "department": "Operations",
          "date": "2023-02-27",
          "severity": 5.0
        },
        {
          "title": "Chemical spill in storage area",
          "department": "Operations",
          "date": "2023-02-24",
          "severity": 4.0
        }
      ]
    }
  ]
}
```

---

## Frontend Integration

### React/TypeScript Example

```typescript
import { useState, useEffect } from 'react';

interface TooltipData {
  month: string;
  total_count: number;
  departments: Array<{ name: string; count: number }>;
  types: Array<{ name: string; count: number }>;
  severity: { avg: number; max: number; min: number } | null;
  risk: { avg: number; max: number; min: number } | null;
  recent_items: Array<{
    title: string;
    department: string;
    date: string;
    severity: number | null;
  }>;
}

interface DetailedTrendResponse {
  labels: string[];
  series: Array<{ name: string; data: number[] }>;
  details: TooltipData[];
}

function useDetailedTrend(dataset: string, filters: any) {
  const [data, setData] = useState<DetailedTrendResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      const params = new URLSearchParams({
        dataset,
        ...filters
      });

      const response = await fetch(
        `/analytics/data/incident-trend-detailed?${params}`
      );
      const result = await response.json();
      setData(result);
      setLoading(false);
    };

    fetchData();
  }, [dataset, filters]);

  return { data, loading };
}

// Usage in component
function TrendChart() {
  const { data, loading } = useDetailedTrend('incident', {
    start_date: '2023-01-01',
    end_date: '2023-12-31'
  });

  if (loading) return <div>Loading...</div>;

  return (
    <LineChart
      labels={data.labels}
      series={data.series}
      tooltipDetails={data.details}
    />
  );
}
```

### Enhanced Tooltip Component

```tsx
interface EnhancedTooltipProps {
  active?: boolean;
  payload?: any[];
  details?: TooltipData[];
}

function EnhancedLineTooltip({ active, payload, details }: EnhancedTooltipProps) {
  if (!active || !payload || !payload.length) return null;
  
  const monthLabel = payload[0].payload.label;
  const monthDetails = details?.find((d) => d.month === monthLabel);
  
  if (!monthDetails) return null;
  
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-4 max-w-sm">
      <div className="font-semibold text-slate-900 mb-3">
        📅 {formatMonth(monthLabel)} - {monthDetails.total_count} Incidents
      </div>
      
      {/* Top Departments */}
      {monthDetails.departments.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-medium text-slate-600 mb-1">Top Departments:</p>
          {monthDetails.departments.slice(0, 3).map((dept) => (
            <p key={dept.name} className="text-xs text-slate-700">
              • {dept.name} ({dept.count})
            </p>
          ))}
        </div>
      )}
      
      {/* Top Types */}
      {monthDetails.types.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-medium text-slate-600 mb-1">Top Types:</p>
          {monthDetails.types.slice(0, 3).map((type) => (
            <p key={type.name} className="text-xs text-slate-700">
              • {type.name} ({type.count})
            </p>
          ))}
        </div>
      )}
      
      {/* Stats */}
      {(monthDetails.severity || monthDetails.risk) && (
        <div className="mb-3 text-xs text-slate-600">
          {monthDetails.severity && (
            <p>Severity: Avg {monthDetails.severity.avg.toFixed(1)} | Max {monthDetails.severity.max}</p>
          )}
          {monthDetails.risk && (
            <p>Risk: Avg {monthDetails.risk.avg.toFixed(1)} | Max {monthDetails.risk.max}</p>
          )}
        </div>
      )}
      
      {/* Recent Items */}
      {monthDetails.recent_items.length > 0 && (
        <div>
          <p className="text-xs font-medium text-slate-600 mb-1">Recent Incidents:</p>
          {monthDetails.recent_items.slice(0, 3).map((item, idx) => (
            <p key={idx} className="text-xs text-slate-700 truncate" title={item.title}>
              • {item.title}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

function formatMonth(monthStr: string): string {
  const [year, month] = monthStr.split('-');
  const date = new Date(parseInt(year), parseInt(month) - 1);
  return date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
}
```

---

## Performance Considerations

### Response Size

- **Typical response size**: 5-50 KB depending on date range
- **Large datasets** (2+ years): 100-200 KB
- **Recommendation**: Use date filters to limit response size

### Caching Strategy

```typescript
// Cache responses for 5 minutes
const CACHE_DURATION = 5 * 60 * 1000;

const cache = new Map<string, { data: any; timestamp: number }>();

async function fetchWithCache(url: string) {
  const cached = cache.get(url);
  
  if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
    return cached.data;
  }
  
  const response = await fetch(url);
  const data = await response.json();
  
  cache.set(url, { data, timestamp: Date.now() });
  
  return data;
}
```

### Optimization Tips

1. **Use date filters** to limit the number of months returned
2. **Cache responses** on the frontend for repeated requests
3. **Lazy load** tooltip data only when hovering (if using separate endpoint)
4. **Debounce** filter changes to avoid excessive API calls

---

## Error Handling

### Empty Dataset

```json
{
  "labels": [],
  "series": [],
  "details": []
}
```

### Invalid Parameters

Returns HTTP 422 with validation errors:

```json
{
  "detail": [
    {
      "loc": ["query", "min_severity"],
      "msg": "ensure this value is greater than or equal to 0",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

### Error Handling Example

```typescript
try {
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  const data = await response.json();
  
  if (!data.labels || data.labels.length === 0) {
    console.warn('No data available for the selected filters');
    return null;
  }
  
  return data;
} catch (error) {
  console.error('Failed to fetch detailed trend data:', error);
  // Fallback to basic endpoint without details
  return fetchBasicTrend(filters);
}
```

---

## Testing

### Manual Testing

1. Start the FastAPI server:
   ```bash
   cd fastapiepcl
   uvicorn app.main:app --reload
   ```

2. Open Swagger UI:
   ```
   http://localhost:8000/docs
   ```

3. Navigate to `/analytics/data/incident-trend-detailed`

4. Click "Try it out" and test with different parameters

### Automated Testing

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_detailed_trend_basic():
    response = client.get("/analytics/data/incident-trend-detailed?dataset=incident")
    assert response.status_code == 200
    data = response.json()
    assert "labels" in data
    assert "series" in data
    assert "details" in data

def test_detailed_trend_with_filters():
    response = client.get(
        "/analytics/data/incident-trend-detailed",
        params={
            "dataset": "incident",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "min_severity": 3.0
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    # Verify structure
    assert isinstance(data["labels"], list)
    assert isinstance(data["series"], list)
    assert isinstance(data["details"], list)
    
    # Verify details structure
    if len(data["details"]) > 0:
        detail = data["details"][0]
        assert "month" in detail
        assert "total_count" in detail
        assert "departments" in detail
        assert "types" in detail

def test_detailed_trend_empty_dataset():
    response = client.get(
        "/analytics/data/incident-trend-detailed",
        params={
            "dataset": "incident",
            "start_date": "2050-01-01",  # Future date
            "end_date": "2050-12-31"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["labels"]) == 0
    assert len(data["details"]) == 0
```

---

## Migration Guide

### From Basic to Detailed Endpoint

**Before (Basic Endpoint):**
```typescript
const response = await fetch('/analytics/data/incident-trend?dataset=incident');
const { labels, series } = await response.json();
```

**After (Detailed Endpoint):**
```typescript
const response = await fetch('/analytics/data/incident-trend-detailed?dataset=incident');
const { labels, series, details } = await response.json();

// Use details for enhanced tooltips
<LineChart 
  labels={labels} 
  series={series} 
  tooltipDetails={details}
/>
```

### Backward Compatibility

The detailed endpoint is **fully backward compatible**:
- Returns the same `labels` and `series` structure as the basic endpoint
- Adds optional `details` array
- Existing code will continue to work without modifications
- Frontend can gradually adopt enhanced tooltips

---

## Support

For issues or questions:
- Check the [API Reference](./API_REFERENCE.md)
- Review [FastAPI Documentation](./fastapiepcl/)
- Contact the development team

---

## Changelog

### v1.0.0 (2025-10-04)
- Initial release of enhanced tooltips API
- Support for incident and hazard datasets
- Comprehensive filtering options
- Top 5 departments and types per month
- Severity and risk statistics
- Recent items (up to 5) per month
