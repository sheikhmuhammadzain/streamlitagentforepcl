# Filter Options API - Documentation

## Overview

The Filter Options API provides all available filter values from your datasets, making it easy to populate frontend dropdown menus and filter UI components with actual data.

## 🎯 Purpose

Instead of hardcoding filter options in your frontend, these endpoints dynamically return:
- All available departments with record counts
- All available locations and sublocations
- All status values
- All incident/violation types
- Date ranges (min/max dates in dataset)
- Severity and risk score ranges

## 📡 Endpoints

### 1. Get Filter Options for Single Dataset

**Endpoint:** `GET /analytics/filter-options`

**Parameters:**
- `dataset` (string, optional): `"incident"` or `"hazard"` (default: `"incident"`)

**Example Request:**
```bash
GET /analytics/filter-options?dataset=incident
```

**Example Response:**
```json
{
  "dataset": "incident",
  "date_range": {
    "min_date": "2023-01-15",
    "max_date": "2024-12-20",
    "total_records": 1250
  },
  "departments": [
    {
      "value": "Operations",
      "label": "Operations",
      "count": 450
    },
    {
      "value": "Maintenance",
      "label": "Maintenance",
      "count": 320
    },
    {
      "value": "Engineering",
      "label": "Engineering",
      "count": 280
    }
  ],
  "locations": [
    {
      "value": "Plant A",
      "label": "Plant A",
      "count": 600
    },
    {
      "value": "Plant B",
      "label": "Plant B",
      "count": 450
    },
    {
      "value": "Warehouse",
      "label": "Warehouse",
      "count": 200
    }
  ],
  "sublocations": [
    {
      "value": "Zone 1",
      "label": "Zone 1",
      "count": 150
    },
    {
      "value": "Zone 2",
      "label": "Zone 2",
      "count": 130
    }
  ],
  "statuses": [
    {
      "value": "Open",
      "label": "Open",
      "count": 320
    },
    {
      "value": "In Progress",
      "label": "In Progress",
      "count": 280
    },
    {
      "value": "Closed",
      "label": "Closed",
      "count": 650
    }
  ],
  "incident_types": [
    {
      "value": "Slip",
      "label": "Slip",
      "count": 180
    },
    {
      "value": "Fall",
      "label": "Fall",
      "count": 150
    },
    {
      "value": "Equipment Failure",
      "label": "Equipment Failure",
      "count": 120
    }
  ],
  "violation_types": [],
  "severity_range": {
    "min": 1.0,
    "max": 5.0,
    "avg": 2.8,
    "median": 3.0,
    "count": 1200
  },
  "risk_range": {
    "min": 1.0,
    "max": 5.0,
    "avg": 3.2,
    "median": 3.0,
    "count": 1200
  },
  "total_records": 1250
}
```

---

### 2. Get Combined Filter Options (Both Datasets)

**Endpoint:** `GET /analytics/filter-options/combined`

**Parameters:** None

**Example Request:**
```bash
GET /analytics/filter-options/combined
```

**Example Response:**
```json
{
  "incident": {
    "dataset": "incident",
    "date_range": {
      "min_date": "2023-01-15",
      "max_date": "2024-12-20",
      "total_records": 1250
    },
    "departments": [...],
    "locations": [...],
    "statuses": [...],
    "incident_types": [...],
    "violation_types": [],
    "severity_range": {...},
    "risk_range": {...},
    "total_records": 1250
  },
  "hazard": {
    "dataset": "hazard",
    "date_range": {
      "min_date": "2023-02-01",
      "max_date": "2024-12-18",
      "total_records": 850
    },
    "departments": [...],
    "locations": [...],
    "statuses": [...],
    "incident_types": [],
    "violation_types": [
      {
        "value": "PPE Violation",
        "label": "Ppe Violation",
        "count": 120
      },
      {
        "value": "Housekeeping",
        "label": "Housekeeping",
        "count": 95
      }
    ],
    "severity_range": {...},
    "risk_range": {...},
    "total_records": 850
  },
  "last_updated": "2024-12-20T15:30:45.123456"
}
```

---

## 🎨 Frontend Integration

### React Example

```jsx
import { useState, useEffect } from 'react';

function FilterPanel() {
  const [filterOptions, setFilterOptions] = useState(null);
  const [selectedDepartments, setSelectedDepartments] = useState([]);
  const [selectedLocations, setSelectedLocations] = useState([]);
  const [dateRange, setDateRange] = useState({ start: '', end: '' });

  // Fetch filter options on component mount
  useEffect(() => {
    fetch('/analytics/filter-options?dataset=incident')
      .then(res => res.json())
      .then(data => {
        setFilterOptions(data);
        // Set default date range
        setDateRange({
          start: data.date_range.min_date,
          end: data.date_range.max_date
        });
      });
  }, []);

  if (!filterOptions) return <div>Loading filters...</div>;

  return (
    <div className="filter-panel">
      <h3>Analytics Filters</h3>
      
      {/* Date Range */}
      <div className="filter-group">
        <label>Start Date</label>
        <input 
          type="date" 
          value={dateRange.start}
          min={filterOptions.date_range.min_date}
          max={filterOptions.date_range.max_date}
          onChange={(e) => setDateRange({...dateRange, start: e.target.value})}
        />
      </div>

      <div className="filter-group">
        <label>End Date</label>
        <input 
          type="date" 
          value={dateRange.end}
          min={filterOptions.date_range.min_date}
          max={filterOptions.date_range.max_date}
          onChange={(e) => setDateRange({...dateRange, end: e.target.value})}
        />
      </div>

      {/* Departments Multi-Select */}
      <div className="filter-group">
        <label>Departments</label>
        <select 
          multiple 
          value={selectedDepartments}
          onChange={(e) => setSelectedDepartments(
            Array.from(e.target.selectedOptions, option => option.value)
          )}
        >
          {filterOptions.departments.map(dept => (
            <option key={dept.value} value={dept.value}>
              {dept.label} ({dept.count})
            </option>
          ))}
        </select>
      </div>

      {/* Locations Multi-Select */}
      <div className="filter-group">
        <label>Locations</label>
        <select 
          multiple 
          value={selectedLocations}
          onChange={(e) => setSelectedLocations(
            Array.from(e.target.selectedOptions, option => option.value)
          )}
        >
          {filterOptions.locations.map(loc => (
            <option key={loc.value} value={loc.value}>
              {loc.label} ({loc.count})
            </option>
          ))}
        </select>
      </div>

      {/* Severity Range */}
      <div className="filter-group">
        <label>
          Min Severity (Range: {filterOptions.severity_range.min} - {filterOptions.severity_range.max})
        </label>
        <input 
          type="range" 
          min={filterOptions.severity_range.min}
          max={filterOptions.severity_range.max}
          step="0.1"
        />
      </div>

      {/* Apply Filters Button */}
      <button onClick={applyFilters}>Apply Filters</button>
    </div>
  );
}
```

### Vue.js Example

```vue
<template>
  <div class="filter-panel">
    <h3>Analytics Filters</h3>
    
    <!-- Date Range -->
    <div class="filter-group">
      <label>Start Date</label>
      <input 
        type="date" 
        v-model="filters.startDate"
        :min="filterOptions?.date_range.min_date"
        :max="filterOptions?.date_range.max_date"
      />
    </div>

    <!-- Departments -->
    <div class="filter-group">
      <label>Departments</label>
      <select v-model="filters.departments" multiple>
        <option 
          v-for="dept in filterOptions?.departments" 
          :key="dept.value" 
          :value="dept.value"
        >
          {{ dept.label }} ({{ dept.count }})
        </option>
      </select>
    </div>

    <!-- Locations -->
    <div class="filter-group">
      <label>Locations</label>
      <select v-model="filters.locations" multiple>
        <option 
          v-for="loc in filterOptions?.locations" 
          :key="loc.value" 
          :value="loc.value"
        >
          {{ loc.label }} ({{ loc.count }})
        </option>
      </select>
    </div>

    <button @click="applyFilters">Apply Filters</button>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const filterOptions = ref(null);
const filters = ref({
  startDate: '',
  endDate: '',
  departments: [],
  locations: []
});

onMounted(async () => {
  const response = await fetch('/analytics/filter-options?dataset=incident');
  filterOptions.value = await response.json();
  
  // Set default date range
  filters.value.startDate = filterOptions.value.date_range.min_date;
  filters.value.endDate = filterOptions.value.date_range.max_date;
});

function applyFilters() {
  // Build query string and fetch analytics data
  const params = new URLSearchParams({
    start_date: filters.value.startDate,
    end_date: filters.value.endDate,
  });
  
  filters.value.departments.forEach(dept => {
    params.append('departments', dept);
  });
  
  filters.value.locations.forEach(loc => {
    params.append('locations', loc);
  });
  
  // Fetch analytics with filters
  fetch(`/analytics/data/incident-trend?${params}`)
    .then(res => res.json())
    .then(data => {
      // Update charts with filtered data
    });
}
</script>
```

---

## 📊 Response Schema

### FilterOption Object
```typescript
{
  value: string;      // The actual value to send in filter requests
  label: string;      // Display label for UI (title-cased)
  count: number;      // Number of records with this value
}
```

### DateRangeInfo Object
```typescript
{
  min_date: string | null;    // Earliest date (YYYY-MM-DD)
  max_date: string | null;    // Latest date (YYYY-MM-DD)
  total_records: number;      // Records with valid dates
}
```

### NumericRange Object
```typescript
{
  min: number;       // Minimum value
  max: number;       // Maximum value
  avg: number;       // Average value
  median: number;    // Median value
  count: number;     // Number of records with valid values
}
```

---

## 🔧 Features

### 1. **Automatic Value Extraction**
- Tries multiple column name variations
- Handles missing or null values gracefully
- Returns empty arrays if no data found

### 2. **Comma-Separated Value Support**
- Incident types and violation types are exploded
- Example: "Slip, Fall, Trip" → ["Slip", "Fall", "Trip"]
- Each value counted separately

### 3. **Sorted by Count**
- All options sorted by frequency (descending)
- Most common values appear first
- Helps users identify popular choices

### 4. **Record Counts**
- Each option includes count of records
- Helps users understand data distribution
- Useful for showing "(Operations - 450 records)"

### 5. **Date Range Detection**
- Automatically finds date columns
- Returns min/max dates from actual data
- Sets sensible defaults for date pickers

### 6. **Numeric Range Statistics**
- Min, max, avg, median for severity/risk
- Helps set slider ranges dynamically
- Shows data distribution

---

## 💡 Best Practices

### 1. **Cache Filter Options**
```javascript
// Cache for 5 minutes to reduce API calls
const CACHE_DURATION = 5 * 60 * 1000;
let filterOptionsCache = null;
let cacheTimestamp = null;

async function getFilterOptions(dataset) {
  const now = Date.now();
  
  if (filterOptionsCache && cacheTimestamp && (now - cacheTimestamp < CACHE_DURATION)) {
    return filterOptionsCache;
  }
  
  const response = await fetch(`/analytics/filter-options?dataset=${dataset}`);
  filterOptionsCache = await response.json();
  cacheTimestamp = now;
  
  return filterOptionsCache;
}
```

### 2. **Show Record Counts in UI**
```jsx
<option value={dept.value}>
  {dept.label} ({dept.count} records)
</option>
```

### 3. **Set Default Date Range**
```javascript
// Use full date range from data by default
setDateRange({
  start: filterOptions.date_range.min_date,
  end: filterOptions.date_range.max_date
});
```

### 4. **Disable Empty Options**
```jsx
{filterOptions.departments.length === 0 ? (
  <option disabled>No departments available</option>
) : (
  filterOptions.departments.map(dept => ...)
)}
```

### 5. **Use Combined Endpoint for Multi-Dataset UI**
```javascript
// If your UI switches between incidents and hazards
const { incident, hazard } = await fetch('/analytics/filter-options/combined')
  .then(res => res.json());

// Switch based on user selection
const options = dataset === 'incident' ? incident : hazard;
```

---

## 🚀 Performance

### Optimization Features
- **Efficient extraction**: Single pass through data
- **Limited results**: Max 100 items per filter (configurable)
- **Minimum count filter**: Excludes rare values (min_count=1)
- **Lazy evaluation**: Only computed when endpoint is called

### Caching Recommendations
```python
# Add caching in production (example with functools)
from functools import lru_cache
from datetime import datetime, timedelta

@lru_cache(maxsize=10)
def get_cached_filter_options(dataset: str, cache_key: str):
    # cache_key changes every 5 minutes to refresh cache
    return extract_filter_options(get_data(dataset), dataset)

# In endpoint
cache_key = datetime.now().strftime("%Y%m%d%H%M")[:-1]  # 5-min buckets
options = get_cached_filter_options(dataset, cache_key)
```

---

## 🧪 Testing

### Test Filter Options Endpoint
```bash
# Get incident filter options
curl "http://localhost:8000/analytics/filter-options?dataset=incident"

# Get hazard filter options
curl "http://localhost:8000/analytics/filter-options?dataset=hazard"

# Get combined options
curl "http://localhost:8000/analytics/filter-options/combined"
```

### Verify Response Structure
```bash
# Check if departments exist
curl "http://localhost:8000/analytics/filter-options?dataset=incident" | jq '.departments'

# Check date range
curl "http://localhost:8000/analytics/filter-options?dataset=incident" | jq '.date_range'

# Check severity range
curl "http://localhost:8000/analytics/filter-options?dataset=incident" | jq '.severity_range'
```

---

## 📚 Related Endpoints

- **`/analytics/filter-summary`** - Preview filter impact
- **`/analytics/data/incident-trend`** - Use filters with analytics
- **`/analytics/data/department-month-heatmap`** - Filtered heatmap data

---

## 🎓 Example Workflow

1. **Load filter options on page load**
   ```javascript
   const options = await fetch('/analytics/filter-options?dataset=incident').then(r => r.json());
   ```

2. **Populate UI dropdowns**
   ```javascript
   populateDepartmentDropdown(options.departments);
   populateLocationDropdown(options.locations);
   setDateRangeLimits(options.date_range);
   ```

3. **User selects filters**
   ```javascript
   const selectedFilters = {
     departments: ['Operations', 'Maintenance'],
     start_date: '2024-01-01',
     min_severity: 3
   };
   ```

4. **Preview filter impact (optional)**
   ```javascript
   const summary = await fetch('/analytics/filter-summary?' + buildQueryString(selectedFilters))
     .then(r => r.json());
   console.log(`Will show ${summary.filtered_count} of ${summary.original_count} records`);
   ```

5. **Apply filters to analytics**
   ```javascript
   const data = await fetch('/analytics/data/incident-trend?' + buildQueryString(selectedFilters))
     .then(r => r.json());
   updateChart(data);
   ```

---

## ✅ Summary

The Filter Options API provides a clean, efficient way to:
- ✅ Populate frontend dropdowns with actual data
- ✅ Show record counts for each option
- ✅ Set appropriate date range limits
- ✅ Configure severity/risk sliders dynamically
- ✅ Build flexible, data-driven filter UIs
- ✅ Support both single and combined dataset views

Use these endpoints to create a rich, user-friendly filtering experience without hardcoding any values!
