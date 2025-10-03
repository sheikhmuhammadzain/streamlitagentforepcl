# Filter Options API - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Fetch Filter Options
```bash
GET /analytics/filter-options?dataset=incident
```

### Step 2: Use in Your Frontend
```javascript
const options = await fetch('/analytics/filter-options?dataset=incident')
  .then(res => res.json());

console.log(options.departments);  // List of departments with counts
console.log(options.locations);    // List of locations with counts
console.log(options.date_range);   // Min/max dates
```

### Step 3: Populate Your UI
```jsx
<select>
  {options.departments.map(dept => (
    <option key={dept.value} value={dept.value}>
      {dept.label} ({dept.count})
    </option>
  ))}
</select>
```

---

## 📡 Available Endpoints

### 1. Single Dataset Options
```bash
GET /analytics/filter-options?dataset=incident
GET /analytics/filter-options?dataset=hazard
```

### 2. Combined Options (Both Datasets)
```bash
GET /analytics/filter-options/combined
```

---

## 📦 Response Structure

```json
{
  "dataset": "incident",
  "date_range": {
    "min_date": "2023-01-15",
    "max_date": "2024-12-20",
    "total_records": 1250
  },
  "departments": [
    {"value": "Operations", "label": "Operations", "count": 450},
    {"value": "Maintenance", "label": "Maintenance", "count": 320}
  ],
  "locations": [
    {"value": "Plant A", "label": "Plant A", "count": 600}
  ],
  "sublocations": [
    {"value": "Zone 1", "label": "Zone 1", "count": 150}
  ],
  "statuses": [
    {"value": "Open", "label": "Open", "count": 320}
  ],
  "incident_types": [
    {"value": "Slip", "label": "Slip", "count": 180}
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

## 🎨 Frontend Examples

### React Hook
```jsx
import { useState, useEffect } from 'react';

function useFilterOptions(dataset = 'incident') {
  const [options, setOptions] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`/analytics/filter-options?dataset=${dataset}`)
      .then(res => res.json())
      .then(data => {
        setOptions(data);
        setLoading(false);
      });
  }, [dataset]);

  return { options, loading };
}

// Usage
function FilterPanel() {
  const { options, loading } = useFilterOptions('incident');
  
  if (loading) return <div>Loading...</div>;
  
  return (
    <select>
      {options.departments.map(dept => (
        <option value={dept.value}>{dept.label} ({dept.count})</option>
      ))}
    </select>
  );
}
```

### Vue Composable
```javascript
import { ref, onMounted } from 'vue';

export function useFilterOptions(dataset = 'incident') {
  const options = ref(null);
  const loading = ref(true);

  onMounted(async () => {
    const response = await fetch(`/analytics/filter-options?dataset=${dataset}`);
    options.value = await response.json();
    loading.value = false;
  });

  return { options, loading };
}
```

### Vanilla JavaScript
```javascript
async function loadFilterOptions() {
  const response = await fetch('/analytics/filter-options?dataset=incident');
  const options = await response.json();
  
  // Populate departments dropdown
  const deptSelect = document.getElementById('departments');
  options.departments.forEach(dept => {
    const option = document.createElement('option');
    option.value = dept.value;
    option.textContent = `${dept.label} (${dept.count})`;
    deptSelect.appendChild(option);
  });
  
  // Set date range limits
  document.getElementById('startDate').min = options.date_range.min_date;
  document.getElementById('startDate').max = options.date_range.max_date;
  document.getElementById('endDate').min = options.date_range.min_date;
  document.getElementById('endDate').max = options.date_range.max_date;
}

loadFilterOptions();
```

---

## 🎯 Common Use Cases

### 1. Department Multi-Select
```jsx
<select multiple>
  {options.departments.map(dept => (
    <option value={dept.value}>
      {dept.label} ({dept.count} records)
    </option>
  ))}
</select>
```

### 2. Location Dropdown
```jsx
<select>
  <option value="">All Locations</option>
  {options.locations.map(loc => (
    <option value={loc.value}>{loc.label}</option>
  ))}
</select>
```

### 3. Date Range Picker
```jsx
<input 
  type="date" 
  min={options.date_range.min_date}
  max={options.date_range.max_date}
  defaultValue={options.date_range.min_date}
/>
```

### 4. Severity Slider
```jsx
<input 
  type="range" 
  min={options.severity_range.min}
  max={options.severity_range.max}
  step="0.1"
/>
<span>
  Range: {options.severity_range.min} - {options.severity_range.max}
  (Avg: {options.severity_range.avg.toFixed(1)})
</span>
```

### 5. Status Checkboxes
```jsx
{options.statuses.map(status => (
  <label key={status.value}>
    <input type="checkbox" value={status.value} />
    {status.label} ({status.count})
  </label>
))}
```

---

## 💡 Pro Tips

### 1. Cache the Response
```javascript
// Cache for 5 minutes
const cache = new Map();
const CACHE_TTL = 5 * 60 * 1000;

async function getCachedFilterOptions(dataset) {
  const cacheKey = `filter-options-${dataset}`;
  const cached = cache.get(cacheKey);
  
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.data;
  }
  
  const data = await fetch(`/analytics/filter-options?dataset=${dataset}`)
    .then(r => r.json());
  
  cache.set(cacheKey, { data, timestamp: Date.now() });
  return data;
}
```

### 2. Show "All" Option
```jsx
<option value="">All Departments ({options.total_records})</option>
{options.departments.map(dept => (
  <option value={dept.value}>{dept.label} ({dept.count})</option>
))}
```

### 3. Sort Options
```javascript
// Sort alphabetically
const sortedDepts = [...options.departments].sort((a, b) => 
  a.label.localeCompare(b.label)
);

// Sort by count (already default)
const topDepts = options.departments.slice(0, 10);
```

### 4. Search/Filter Options
```jsx
const [search, setSearch] = useState('');

const filteredDepts = options.departments.filter(dept =>
  dept.label.toLowerCase().includes(search.toLowerCase())
);
```

### 5. Group by Count
```jsx
const highVolume = options.departments.filter(d => d.count > 100);
const lowVolume = options.departments.filter(d => d.count <= 100);

<optgroup label="High Volume">
  {highVolume.map(...)}
</optgroup>
<optgroup label="Low Volume">
  {lowVolume.map(...)}
</optgroup>
```

---

## 🔗 Complete Workflow

```javascript
// 1. Load filter options on mount
const options = await fetch('/analytics/filter-options?dataset=incident')
  .then(r => r.json());

// 2. Populate UI
populateDropdowns(options);

// 3. User selects filters
const filters = {
  departments: ['Operations'],
  start_date: '2024-01-01',
  min_severity: 3
};

// 4. Check filter impact (optional)
const params = new URLSearchParams(filters);
const summary = await fetch(`/analytics/filter-summary?${params}`)
  .then(r => r.json());

console.log(`Showing ${summary.filtered_count} of ${summary.original_count} records`);

// 5. Fetch filtered analytics
const data = await fetch(`/analytics/data/incident-trend?${params}`)
  .then(r => r.json());

// 6. Update charts
updateChart(data);
```

---

## 🧪 Test Commands

```bash
# Get incident options
curl "http://localhost:8000/analytics/filter-options?dataset=incident"

# Get hazard options
curl "http://localhost:8000/analytics/filter-options?dataset=hazard"

# Get combined options
curl "http://localhost:8000/analytics/filter-options/combined"

# Check specific field
curl "http://localhost:8000/analytics/filter-options?dataset=incident" | jq '.departments'
```

---

## 📚 Full Documentation

For complete details, see:
- **`FILTER_OPTIONS_API.md`** - Complete API documentation
- **`FILTERING_GUIDE.md`** - Filter usage guide
- **`QUICK_REFERENCE.md`** - Filter parameters reference

---

## ✅ Summary

**What You Get:**
- ✅ All available departments, locations, statuses, types
- ✅ Record counts for each option
- ✅ Date range from actual data
- ✅ Severity and risk score ranges
- ✅ Ready to use in dropdowns, checkboxes, sliders

**How to Use:**
1. Fetch `/analytics/filter-options?dataset=incident`
2. Populate your UI components
3. Let users select filters
4. Apply filters to analytics endpoints

**Result:**
- 🎨 Data-driven filter UI (no hardcoding!)
- 📊 Shows actual available values
- 🔢 Displays record counts
- ⚡ Fast and efficient
