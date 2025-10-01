# 🔍 Filters API Documentation

## Overview

The Filters API provides dropdown options for all filter fields used across the analytics endpoints. This allows your frontend to dynamically populate filter dropdowns with actual data values from your Excel file.

---

## 📍 Endpoints

### **GET** `/filters/locations`

Get all unique locations from all datasets.

**Response:**
```json
{
  "locations": [
    "Admin Building",
    "Asset Integrity",
    "CA-1650 and HCL Loading",
    "HTDC Contstruction",
    "Karachi",
    "Manufacturing Facility",
    "PVC I Front End",
    "Projects and BD",
    "Stationary - PVC"
  ],
  "count": 9
}
```

**Data Sources:**
- Incident sheet → `location`, `sublocation`
- Hazard ID sheet → `location`, `sublocation`
- Audit sheet → `location`, `audit_location`
- Inspection sheet → `location`, `audit_location`

---

### **GET** `/filters/departments`

Get all unique departments from all datasets.

**Response:**
```json
{
  "departments": [
    "Chlor Alkali and Allied Chemicals",
    "HTDC",
    "HPO",
    "Process - EDC / VCM",
    "Process - PVC",
    "Process - UTY and PP",
    "PVC",
    "Utilities"
  ],
  "count": 8
}
```

**Data Sources:**
- Incident sheet → `department`, `sub_department`, `section`
- Hazard ID sheet → `department`, `sub_department`, `section`

---

### **GET** `/filters/statuses`

Get all unique status values from all datasets.

**Response:**
```json
{
  "statuses": [
    "Closed",
    "In Progress",
    "Open",
    "Pending",
    "Review"
  ],
  "count": 5
}
```

**Data Sources:**
- Incident sheet → `status`
- Hazard ID sheet → `status`
- Audit sheet → `audit_status`, `status`

---

### **GET** `/filters/incident-types`

Get all unique incident types.

**Response:**
```json
{
  "incident_types": [
    "Environmental",
    "Near Miss",
    "No Loss / No Injury",
    "Other",
    "Process Safety Event",
    "Property Damage"
  ],
  "count": 6
}
```

**Data Source:**
- Incident sheet → `incident_type`, `category`

---

### **GET** `/filters/violation-types`

Get all unique violation types from hazards.

**Response:**
```json
{
  "violation_types": [
    "Housekeeping",
    "PPE Violation",
    "Safety Rule Violation",
    "Unsafe Act",
    "Unsafe Condition"
  ],
  "count": 5
}
```

**Data Source:**
- Hazard ID sheet → `violation_type_hazard_id`

---

### **GET** `/filters/companies`

Get all unique company names.

**Response:**
```json
{
  "companies": [
    "Engro Polymer and Chemicals",
    "EPCL"
  ],
  "count": 2
}
```

**Data Sources:**
- Incident sheet → `company`
- Hazard ID sheet → `company`

---

### **GET** `/filters/date-range`

Get the earliest and latest dates from all datasets to set date picker ranges.

**Response:**
```json
{
  "earliest_date": "2022-04-05",
  "latest_date": "2024-12-31",
  "total_days": 1001
}
```

**Data Sources:**
- Incident sheet → `occurrence_date`, `date`, `reported_date`
- Hazard ID sheet → `occurrence_date`, `date`, `reported_date`

---

### **GET** `/filters/all` ⭐ **RECOMMENDED**

Get all filter options in a single API call for efficiency.

**Response:**
```json
{
  "locations": [
    "Admin Building",
    "Asset Integrity",
    "Karachi",
    "Manufacturing Facility",
    "..."
  ],
  "departments": [
    "Chlor Alkali and Allied Chemicals",
    "HTDC",
    "HPO",
    "Process - EDC / VCM",
    "..."
  ],
  "statuses": [
    "Closed",
    "In Progress",
    "Open",
    "..."
  ],
  "incident_types": [
    "Environmental",
    "Near Miss",
    "No Loss / No Injury",
    "..."
  ],
  "violation_types": [
    "Housekeeping",
    "PPE Violation",
    "..."
  ],
  "companies": [
    "Engro Polymer and Chemicals",
    "EPCL"
  ]
}
```

**Use Case:** Load all filter options on page load with a single API call instead of 6 separate calls.

---

## 🎨 Frontend Integration Examples

### React - Load All Filters on Mount

```jsx
import React, { useEffect, useState } from 'react';
import axios from 'axios';

const FilterDropdowns = () => {
  const [filters, setFilters] = useState(null);
  const [selectedLocation, setSelectedLocation] = useState('');
  const [selectedDepartment, setSelectedDepartment] = useState('');

  useEffect(() => {
    // Load all filters in one call
    axios.get('/filters/all')
      .then(res => setFilters(res.data));
  }, []);

  if (!filters) return <div>Loading filters...</div>;

  return (
    <div className="filter-panel">
      <select 
        value={selectedLocation} 
        onChange={(e) => setSelectedLocation(e.target.value)}
      >
        <option value="">All Locations</option>
        {filters.locations.map(loc => (
          <option key={loc} value={loc}>{loc}</option>
        ))}
      </select>

      <select 
        value={selectedDepartment} 
        onChange={(e) => setSelectedDepartment(e.target.value)}
      >
        <option value="">All Departments</option>
        {filters.departments.map(dept => (
          <option key={dept} value={dept}>{dept}</option>
        ))}
      </select>
    </div>
  );
};
```

---

### React - Date Range Picker

```jsx
const DateRangePicker = () => {
  const [dateRange, setDateRange] = useState(null);

  useEffect(() => {
    axios.get('/filters/date-range')
      .then(res => setDateRange(res.data));
  }, []);

  if (!dateRange) return null;

  return (
    <div>
      <input 
        type="date" 
        min={dateRange.earliest_date} 
        max={dateRange.latest_date}
        placeholder="Start Date"
      />
      <input 
        type="date" 
        min={dateRange.earliest_date} 
        max={dateRange.latest_date}
        placeholder="End Date"
      />
      <p>Data available from {dateRange.earliest_date} to {dateRange.latest_date}</p>
    </div>
  );
};
```

---

### Vue.js - Filter Dropdowns

```vue
<template>
  <div class="filters">
    <select v-model="selectedLocation">
      <option value="">All Locations</option>
      <option v-for="loc in filters.locations" :key="loc" :value="loc">
        {{ loc }}
      </option>
    </select>

    <select v-model="selectedDepartment">
      <option value="">All Departments</option>
      <option v-for="dept in filters.departments" :key="dept" :value="dept">
        {{ dept }}
      </option>
    </select>

    <select v-model="selectedStatus">
      <option value="">All Statuses</option>
      <option v-for="status in filters.statuses" :key="status" :value="status">
        {{ status }}
      </option>
    </select>
  </div>
</template>

<script>
export default {
  data() {
    return {
      filters: {
        locations: [],
        departments: [],
        statuses: []
      },
      selectedLocation: '',
      selectedDepartment: '',
      selectedStatus: ''
    };
  },
  async mounted() {
    const response = await fetch('/filters/all');
    this.filters = await response.json();
  }
};
</script>
```

---

### Angular - Filter Service

```typescript
// filter.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class FilterService {
  constructor(private http: HttpClient) {}

  getAllFilters(): Observable<any> {
    return this.http.get('/filters/all');
  }

  getLocations(): Observable<any> {
    return this.http.get('/filters/locations');
  }

  getDepartments(): Observable<any> {
    return this.http.get('/filters/departments');
  }

  getDateRange(): Observable<any> {
    return this.http.get('/filters/date-range');
  }
}

// filter.component.ts
export class FilterComponent implements OnInit {
  filters: any;

  constructor(private filterService: FilterService) {}

  ngOnInit() {
    this.filterService.getAllFilters().subscribe(
      data => this.filters = data
    );
  }
}
```

---

## 🔧 Data Cleaning Features

The filter endpoints automatically clean the data:

✅ **Removes null/NaN values**
✅ **Removes empty strings**
✅ **Removes "Not Specified", "Not Assigned"**
✅ **Removes duplicate values**
✅ **Sorts alphabetically**
✅ **Combines data from all relevant sheets**

---

## 🚀 Testing

### Test Individual Endpoints

```bash
# Get locations
curl http://localhost:8000/filters/locations

# Get departments
curl http://localhost:8000/filters/departments

# Get statuses
curl http://localhost:8000/filters/statuses

# Get incident types
curl http://localhost:8000/filters/incident-types

# Get violation types
curl http://localhost:8000/filters/violation-types

# Get companies
curl http://localhost:8000/filters/companies

# Get date range
curl http://localhost:8000/filters/date-range

# Get all filters (recommended)
curl http://localhost:8000/filters/all
```

### Test in Swagger UI

1. Go to: http://localhost:8000/docs
2. Find the **"filters"** tag
3. Test any endpoint with "Try it out" → "Execute"

---

## 📊 Usage in Analytics Endpoints

Use these filter values in your analytics API calls:

```javascript
// Load filter options
const filters = await fetch('/filters/all').then(r => r.json());

// Use selected filters in analytics call
const location = filters.locations[0]; // "Karachi"
const department = filters.departments[0]; // "Process - EDC / VCM"

const pyramidData = await fetch(
  `/analytics/advanced/heinrich-pyramid?location=${location}&department=${department}`
).then(r => r.json());
```

---

## 🎯 Best Practices

1. **Load once on app initialization**: Call `/filters/all` when your app loads
2. **Cache the results**: Store in state/store (Redux, Vuex, etc.)
3. **Refresh periodically**: Reload filters if data changes (e.g., every 5 minutes)
4. **Use "All" option**: Always provide an "All" or empty option in dropdowns
5. **Show counts**: Display the number of options (e.g., "Locations (9)")

---

## 📋 Summary

| Endpoint | Purpose | Response Time |
|----------|---------|---------------|
| `/filters/locations` | Location dropdown | ~50ms |
| `/filters/departments` | Department dropdown | ~50ms |
| `/filters/statuses` | Status dropdown | ~50ms |
| `/filters/incident-types` | Incident type dropdown | ~50ms |
| `/filters/violation-types` | Violation type dropdown | ~50ms |
| `/filters/companies` | Company dropdown | ~50ms |
| `/filters/date-range` | Date picker min/max | ~50ms |
| `/filters/all` ⭐ | All filters in one call | ~100ms |

**Recommendation:** Use `/filters/all` for best performance and fewer API calls.

---

## 🎉 Benefits

✅ **Dynamic dropdowns** - Always show actual data values
✅ **No hardcoding** - Filter options update automatically when data changes
✅ **Clean data** - Automatically removes invalid/empty values
✅ **Efficient** - Single API call for all filters
✅ **Sorted** - Alphabetically sorted for better UX
✅ **Consistent** - Same filter values across all analytics endpoints

Ready to use in your frontend! 🚀
