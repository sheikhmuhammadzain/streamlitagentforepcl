# Analytics Filtering System - User Guide

## Overview

The analytics endpoints now support **flexible filtering** to enable dynamic data analysis. You can filter by date ranges, departments, locations, severity, risk levels, and more.

## Key Improvements

### Before (Hardcoded)
```python
# Only basic dataset selection
GET /analytics/data/incident-trend?dataset=incident
```

### After (Flexible)
```python
# Multiple filter combinations
GET /analytics/data/incident-trend?dataset=incident&start_date=2024-01-01&end_date=2024-12-31&departments=Operations&departments=Maintenance&min_severity=3&min_risk=4
```

## Available Filters

### 1. **Dataset Selection**
- `dataset`: Choose between `"incident"` or `"hazard"`
- Default: `"incident"`

### 2. **Date Range Filters**
- `start_date`: Filter records on or after this date (ISO format: `YYYY-MM-DD`)
- `end_date`: Filter records on or before this date (ISO format: `YYYY-MM-DD`)
- Works with multiple date column names: `occurrence_date`, `date_reported`, `entered_date`, `start_date`, etc.

### 3. **Department Filters**
- `departments`: List of departments to include (case-insensitive)
- Example: `?departments=Operations&departments=Maintenance&departments=Engineering`

### 4. **Location Filters**
- `locations`: List of locations to include (case-insensitive)
- `sublocations`: List of sublocations to include (case-insensitive)
- Example: `?locations=Plant A&sublocations=Zone 1`

### 5. **Severity Filters**
- `min_severity`: Minimum severity score (0-5, inclusive)
- `max_severity`: Maximum severity score (0-5, inclusive)
- Example: `?min_severity=3&max_severity=5` (high severity only)

### 6. **Risk Filters**
- `min_risk`: Minimum risk score (0-5, inclusive)
- `max_risk`: Maximum risk score (0-5, inclusive)
- Example: `?min_risk=4` (high risk only)

### 7. **Status Filters**
- `statuses`: List of status values to include
- Example: `?statuses=Open&statuses=In Progress`

### 8. **Type Filters**
- `incident_types`: Filter by incident types (supports comma-separated values in data)
- `violation_types`: Filter by violation types (for hazards)
- Example: `?incident_types=Slip&incident_types=Fall`

## Updated Endpoints

The following endpoints now support flexible filtering:

### Trend Analysis
- ✅ `/analytics/data/incident-trend`
- ✅ `/analytics/data/incident-type-distribution`
- ✅ `/analytics/data/department-month-heatmap`

### Additional Endpoints (Can be updated similarly)
- `/analytics/data/root-cause-pareto`
- `/analytics/data/injury-severity-pyramid`
- `/analytics/data/consequence-gap`
- `/analytics/data/incident-cost-trend`
- `/analytics/risk-calendar-heatmap`
- `/analytics/department-spider`
- `/analytics/violation-analysis`

## New Endpoint: Filter Summary

### `/analytics/filter-summary`

Get a preview of how filters will affect your dataset **before** applying them to charts.

**Request:**
```http
GET /analytics/filter-summary?dataset=incident&start_date=2024-01-01&end_date=2024-06-30&departments=Operations&min_severity=3
```

**Response:**
```json
{
  "original_count": 1250,
  "filtered_count": 342,
  "records_removed": 908,
  "retention_rate": 27.36,
  "active_filters": {
    "dataset": "incident",
    "start_date": "2024-01-01",
    "end_date": "2024-06-30",
    "departments": ["Operations"],
    "min_severity": 3.0
  },
  "filter_count": 5
}
```

## Usage Examples

### Example 1: Q1 2024 High-Risk Incidents in Operations
```http
GET /analytics/data/incident-trend?dataset=incident&start_date=2024-01-01&end_date=2024-03-31&departments=Operations&min_risk=4
```

### Example 2: Severe Hazards in Multiple Locations
```http
GET /analytics/data/incident-type-distribution?dataset=hazard&locations=Plant A&locations=Plant B&min_severity=4&max_severity=5
```

### Example 3: Department Heatmap for Specific Date Range
```http
GET /analytics/data/department-month-heatmap?dataset=incident&start_date=2024-01-01&end_date=2024-12-31&min_risk=3
```

### Example 4: Filter by Multiple Departments and Status
```http
GET /analytics/data/incident-trend?departments=Operations&departments=Maintenance&departments=Engineering&statuses=Open&statuses=In Progress
```

## Implementation Details

### Centralized Filtering Function

The filtering logic is centralized in `app/services/filters.py`:

```python
from app.services.filters import apply_analytics_filters

# Apply filters to any DataFrame
filtered_df = apply_analytics_filters(
    df,
    start_date="2024-01-01",
    end_date="2024-12-31",
    departments=["Operations", "Maintenance"],
    min_severity=3.0,
    min_risk=4.0
)
```

### Filter Schema

Filters are defined in `app/models/schemas.py`:

```python
class AnalyticsFilters(BaseModel):
    dataset: str = Field(default="incident")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    departments: Optional[List[str]] = None
    locations: Optional[List[str]] = None
    # ... and more
```

## Benefits

1. **Flexible Analysis**: Filter data dynamically without modifying code
2. **Consistent Logic**: Centralized filtering ensures consistency across all endpoints
3. **Performance**: Filter data before processing to improve chart rendering speed
4. **User Experience**: Frontend can provide rich filtering UI with immediate feedback
5. **Maintainability**: Single source of truth for filtering logic

## Migration Path

To add filtering to additional endpoints:

1. Import the filter utility:
   ```python
   from ..services.filters import apply_analytics_filters
   ```

2. Add filter parameters to endpoint:
   ```python
   @router.get("/your-endpoint")
   async def your_endpoint(
       dataset: str = Query("incident"),
       start_date: Optional[str] = Query(None),
       end_date: Optional[str] = Query(None),
       departments: Optional[List[str]] = Query(None),
       # ... other filters
   ):
   ```

3. Apply filters before processing:
   ```python
   df = get_incident_df() if dataset == "incident" else get_hazard_df()
   df = apply_analytics_filters(
       df, start_date=start_date, end_date=end_date,
       departments=departments, ...
   )
   ```

## Testing

Test the filtering system with various combinations:

```bash
# Test date range
curl "http://localhost:8000/analytics/filter-summary?start_date=2024-01-01&end_date=2024-03-31"

# Test department filter
curl "http://localhost:8000/analytics/filter-summary?departments=Operations&departments=Maintenance"

# Test severity and risk
curl "http://localhost:8000/analytics/filter-summary?min_severity=3&min_risk=4"

# Test combined filters
curl "http://localhost:8000/analytics/filter-summary?start_date=2024-01-01&departments=Operations&min_severity=3&min_risk=4"
```

## Future Enhancements

Potential improvements:

1. **Saved Filters**: Allow users to save and reuse filter combinations
2. **Filter Presets**: Common filter combinations (e.g., "High Risk Last Quarter")
3. **Advanced Filters**: Text search, regex patterns, custom expressions
4. **Filter Analytics**: Track which filters are most commonly used
5. **Bulk Operations**: Apply same filters across multiple chart types

## Support

For questions or issues with the filtering system:
- Check the API documentation at `/docs` (Swagger UI)
- Review the filter utility source: `app/services/filters.py`
- Test with `/analytics/filter-summary` endpoint first
