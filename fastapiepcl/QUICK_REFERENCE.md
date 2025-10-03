# Flexible Filtering - Quick Reference

## 🎯 Quick Start

### Test Filter Impact
```bash
GET /analytics/filter-summary?dataset=incident&start_date=2024-01-01&departments=Operations&min_severity=3
```

### Apply Filters to Charts
```bash
GET /analytics/data/incident-trend?start_date=2024-01-01&end_date=2024-12-31&departments=Operations&min_risk=4
```

## 📋 Available Filters

| Filter | Type | Example | Description |
|--------|------|---------|-------------|
| `dataset` | string | `incident` or `hazard` | Choose dataset |
| `start_date` | string | `2024-01-01` | Start date (YYYY-MM-DD) |
| `end_date` | string | `2024-12-31` | End date (YYYY-MM-DD) |
| `departments` | list | `?departments=Ops&departments=Maint` | Filter by departments |
| `locations` | list | `?locations=Plant A` | Filter by locations |
| `sublocations` | list | `?sublocations=Zone 1` | Filter by sublocations |
| `min_severity` | float | `3.0` | Min severity (0-5) |
| `max_severity` | float | `5.0` | Max severity (0-5) |
| `min_risk` | float | `4.0` | Min risk (0-5) |
| `max_risk` | float | `5.0` | Max risk (0-5) |
| `statuses` | list | `?statuses=Open&statuses=Closed` | Filter by status |
| `incident_types` | list | `?incident_types=Slip&incident_types=Fall` | Filter by types |
| `violation_types` | list | `?violation_types=PPE` | Filter violations (hazards) |

## 🔥 Common Use Cases

### 1. High-Risk Incidents This Quarter
```bash
GET /analytics/data/incident-trend?start_date=2024-10-01&end_date=2024-12-31&min_risk=4
```

### 2. Severe Incidents in Operations
```bash
GET /analytics/data/incident-type-distribution?departments=Operations&min_severity=4
```

### 3. Multi-Department Analysis
```bash
GET /analytics/data/department-month-heatmap?departments=Operations&departments=Maintenance&departments=Engineering
```

### 4. Date Range + Location + Risk
```bash
GET /analytics/data/incident-trend?start_date=2024-01-01&end_date=2024-06-30&locations=Plant A&min_risk=3
```

### 5. Open High-Severity Issues
```bash
GET /analytics/data/incident-type-distribution?statuses=Open&statuses=In Progress&min_severity=4
```

## 🛠️ For Developers

### Add Filtering to New Endpoint

```python
from ..services.filters import apply_analytics_filters
from typing import Optional, List
from fastapi import Query

@router.get("/your-endpoint")
async def your_endpoint(
    dataset: str = Query("incident"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    departments: Optional[List[str]] = Query(None),
    min_severity: Optional[float] = Query(None, ge=0, le=5),
    min_risk: Optional[float] = Query(None, ge=0, le=5),
):
    # Get base data
    df = get_incident_df() if dataset == "incident" else get_hazard_df()
    
    # Apply filters
    df = apply_analytics_filters(
        df,
        start_date=start_date,
        end_date=end_date,
        departments=departments,
        min_severity=min_severity,
        min_risk=min_risk
    )
    
    # Process filtered data
    # ... your logic here
```

### Use Filter Summary
```python
from ..services.filters import get_filter_summary

df_original = get_incident_df()
df_filtered = apply_analytics_filters(df_original, min_severity=3)

summary = get_filter_summary(df_original, df_filtered, {
    'min_severity': 3
})

# Returns:
# {
#   'original_count': 1250,
#   'filtered_count': 342,
#   'records_removed': 908,
#   'retention_rate': 27.36,
#   'active_filters': {'min_severity': 3},
#   'filter_count': 1
# }
```

## 📊 Updated Endpoints

### ✅ Fully Implemented
- `/analytics/data/incident-trend`
- `/analytics/data/incident-type-distribution`
- `/analytics/data/department-month-heatmap`
- `/analytics/filter-summary` (NEW)

### 🔄 To Be Updated (Same Pattern)
- `/analytics/data/root-cause-pareto`
- `/analytics/data/injury-severity-pyramid`
- `/analytics/data/consequence-gap`
- `/analytics/data/incident-cost-trend`
- `/analytics/data/hazard-cost-trend`
- `/analytics/data/ppe-violation-analysis`
- `/analytics/data/repeated-incidents`
- `/analytics/risk-calendar-heatmap`
- `/analytics/department-spider`
- `/analytics/violation-analysis`
- ... and 15+ more

## 🧪 Testing Commands

### Test Filter Summary
```bash
# No filters
curl "http://localhost:8000/analytics/filter-summary?dataset=incident"

# Date range only
curl "http://localhost:8000/analytics/filter-summary?start_date=2024-01-01&end_date=2024-03-31"

# Department filter
curl "http://localhost:8000/analytics/filter-summary?departments=Operations"

# Combined filters
curl "http://localhost:8000/analytics/filter-summary?start_date=2024-01-01&departments=Operations&min_severity=3&min_risk=4"
```

### Test Actual Endpoints
```bash
# Incident trend with filters
curl "http://localhost:8000/analytics/data/incident-trend?start_date=2024-01-01&end_date=2024-12-31&departments=Operations"

# Type distribution with severity filter
curl "http://localhost:8000/analytics/data/incident-type-distribution?min_severity=3&max_severity=5"

# Heatmap with multiple departments
curl "http://localhost:8000/analytics/data/department-month-heatmap?departments=Operations&departments=Maintenance&min_risk=3"
```

## 💡 Tips

1. **Start with Filter Summary**: Always test with `/filter-summary` first to see impact
2. **Multiple Values**: Repeat parameter for lists: `?departments=A&departments=B`
3. **Date Format**: Always use ISO format: `YYYY-MM-DD`
4. **Case Insensitive**: Department/location names are case-insensitive
5. **Combine Filters**: Mix and match any filters for complex queries
6. **Empty Results**: If filtered data is empty, charts return empty arrays

## 🚨 Common Issues

### Issue: No data returned
**Solution**: Check filter summary first to see if filters are too restrictive

### Issue: Date filter not working
**Solution**: Ensure date format is `YYYY-MM-DD`, check if date column exists in dataset

### Issue: Department filter not matching
**Solution**: Filters are case-insensitive, but check exact department names in data

### Issue: Multiple values not working
**Solution**: Use `?param=value1&param=value2` format, not comma-separated

## 📚 Full Documentation

- **User Guide**: See `FILTERING_GUIDE.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **API Docs**: Visit `/docs` (Swagger UI) when server is running

## 🎓 Examples by Scenario

### Safety Manager: "Show me all high-risk incidents in Operations last quarter"
```bash
GET /analytics/data/incident-trend?dataset=incident&start_date=2024-10-01&end_date=2024-12-31&departments=Operations&min_risk=4
```

### Compliance Officer: "What are the severe hazards across all locations?"
```bash
GET /analytics/data/incident-type-distribution?dataset=hazard&min_severity=4&max_severity=5
```

### Department Head: "Show trends for my three departments this year"
```bash
GET /analytics/data/department-month-heatmap?start_date=2024-01-01&departments=Operations&departments=Maintenance&departments=Engineering
```

### Executive: "What's the impact of filtering to high-risk only?"
```bash
GET /analytics/filter-summary?min_risk=4&min_severity=4
```

## 🔗 Related Files

- Schema: `app/models/schemas.py` → `AnalyticsFilters`
- Filter Logic: `app/services/filters.py` → `apply_analytics_filters()`
- Endpoints: `app/routers/analytics_general.py`
