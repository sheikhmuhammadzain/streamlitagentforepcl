# Enhanced Tooltips - Quick Reference Card

## 🚀 Endpoint

```
GET /analytics/data/incident-trend-detailed
```

---

## 📥 Request

### Minimal
```bash
?dataset=incident
```

### With Filters
```bash
?dataset=incident
&start_date=2023-01-01
&end_date=2023-12-31
&departments=Operations
&min_severity=3.0
```

---

## 📤 Response Structure

```typescript
{
  labels: string[]              // ["2023-01", "2023-02", ...]
  series: [{
    name: string                // "Count"
    data: number[]              // [15, 22, 18, ...]
  }]
  details: [{
    month: string               // "2023-01"
    total_count: number         // 15
    departments: [{             // Top 5
      name: string              // "Operations"
      count: number             // 8
    }]
    types: [{                   // Top 5
      name: string              // "Slip"
      count: number             // 4
    }]
    severity: {                 // null if N/A
      avg: number               // 3.2
      max: number               // 5.0
      min: number               // 1.0
    }
    risk: {                     // null if N/A
      avg: number               // 3.8
      max: number               // 5.0
      min: number               // 2.0
    }
    recent_items: [{            // Up to 5
      title: string             // "Worker slipped..."
      department: string        // "Operations"
      date: string              // "2023-01-28"
      severity: number | null   // 3.0
    }]
  }]
}
```

---

## 🎨 Frontend - React Hook

```typescript
function useDetailedTrend(dataset: string, filters: any) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const params = new URLSearchParams({ dataset, ...filters });
    
    fetch(`/analytics/data/incident-trend-detailed?${params}`)
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      });
  }, [dataset, filters]);

  return { data, loading };
}
```

---

## 🎨 Frontend - Tooltip Component

```tsx
function EnhancedTooltip({ active, payload, details }) {
  if (!active || !payload?.length) return null;
  
  const month = payload[0].payload.label;
  const detail = details?.find(d => d.month === month);
  if (!detail) return null;
  
  return (
    <div className="tooltip">
      <h4>📅 {month} - {detail.total_count} Incidents</h4>
      
      {/* Departments */}
      <div>
        <strong>Top Departments:</strong>
        {detail.departments.slice(0, 3).map(d => (
          <div key={d.name}>• {d.name} ({d.count})</div>
        ))}
      </div>
      
      {/* Types */}
      <div>
        <strong>Top Types:</strong>
        {detail.types.slice(0, 3).map(t => (
          <div key={t.name}>• {t.name} ({t.count})</div>
        ))}
      </div>
      
      {/* Stats */}
      {detail.severity && (
        <div>
          Severity: Avg {detail.severity.avg.toFixed(1)} | 
          Max {detail.severity.max}
        </div>
      )}
      
      {detail.risk && (
        <div>
          Risk: Avg {detail.risk.avg.toFixed(1)} | 
          Max {detail.risk.max}
        </div>
      )}
      
      {/* Recent Items */}
      {detail.recent_items.length > 0 && (
        <div>
          <strong>Recent Incidents:</strong>
          {detail.recent_items.slice(0, 3).map((item, i) => (
            <div key={i} title={item.title}>
              • {item.title.substring(0, 50)}...
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

---

## 🔧 All Available Filters

| Parameter | Type | Example |
|-----------|------|---------|
| `dataset` | string | `incident` \| `hazard` |
| `start_date` | string | `2023-01-01` |
| `end_date` | string | `2023-12-31` |
| `departments` | string[] | `&departments=Ops&departments=Maint` |
| `locations` | string[] | `&locations=Plant A` |
| `sublocations` | string[] | `&sublocations=Zone 1` |
| `min_severity` | float | `3.0` (0-5) |
| `max_severity` | float | `5.0` (0-5) |
| `min_risk` | float | `2.5` (0-5) |
| `max_risk` | float | `5.0` (0-5) |
| `statuses` | string[] | `&statuses=Open&statuses=Closed` |
| `incident_types` | string[] | `&incident_types=Slip` |
| `violation_types` | string[] | `&violation_types=PPE` |

---

## 🧪 Quick Test

### cURL
```bash
curl "http://localhost:8000/analytics/data/incident-trend-detailed?dataset=incident&start_date=2023-01-01"
```

### Browser
```
http://localhost:8000/docs
→ /analytics/data/incident-trend-detailed
→ Try it out
```

### JavaScript Console
```javascript
fetch('/analytics/data/incident-trend-detailed?dataset=incident')
  .then(r => r.json())
  .then(console.log)
```

---

## ⚡ Performance Tips

1. **Use date filters** - Limit to 1-2 years max
2. **Cache responses** - 5 minute TTL recommended
3. **Debounce filter changes** - 300ms delay
4. **Show loading state** - Fetch can take 200-500ms

---

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| Empty `details` | Check date filters, verify data exists |
| Missing stats | Severity/risk columns not in dataset |
| Slow response | Reduce date range, add filters |
| Truncated titles | Intentional (100 char limit) |

---

## 📚 Full Docs

- **Complete API Docs:** `ENHANCED_TOOLTIPS_API.md`
- **Implementation Summary:** `ENHANCED_TOOLTIPS_SUMMARY.md`
- **Interactive Docs:** `http://localhost:8000/docs`

---

## ✅ Checklist

- [ ] Backend endpoint deployed
- [ ] Frontend fetches detailed data
- [ ] Tooltip component created
- [ ] Loading states added
- [ ] Error handling implemented
- [ ] Caching configured
- [ ] Tested with real data

---

**Need help?** Check the full documentation or test in Swagger UI! 🚀
