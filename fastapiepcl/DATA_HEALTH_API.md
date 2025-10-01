# 🏥 Data Health & Validation API Documentation

## Overview

The Data Health API provides endpoints to verify data quality, view raw data samples, check data sources, and validate data integrity. These endpoints allow you to cross-verify that your analytics are using the correct data from your Excel file.

---

## 📊 Endpoints

### **1. Data Health Summary** ⭐ **MOST IMPORTANT**

**GET** `/data-health/summary`

Get overall data health including record counts, date ranges, and quality metrics.

**Response:**
```json
{
  "total_records": {
    "incidents": 1845,
    "hazards": 892,
    "audits": 456,
    "inspections": 234
  },
  "grand_total": 3427,
  "date_range": {
    "earliest": "2022-04-05",
    "latest": "2024-12-31",
    "total_days": 1001
  },
  "last_sync": "2025-01-01T20:35:53",
  "data_quality": {
    "completeness_percentage": 94.5,
    "total_cells": 250000,
    "non_null_cells": 236250,
    "null_cells": 13750
  },
  "status": "healthy"
}
```

**Use Case:** Display on dashboard to show data is loaded and healthy.

---

### **2. Raw Data Samples**

View actual records from each Excel sheet to verify data accuracy.

#### **GET** `/data-health/sample/incidents`

**Query Parameters:**
- `limit` (1-100, default: 10) - Number of records
- `offset` (default: 0) - Skip N records for pagination
- `start_date` (optional) - Filter from date (YYYY-MM-DD)
- `end_date` (optional) - Filter to date (YYYY-MM-DD)
- `search` (optional) - Search text in title, description, department
- `status` (optional) - Filter by status (e.g., "Closed", "Open")
- `department` (optional) - Filter by department (e.g., "Process")
- `location` (optional) - Filter by location (e.g., "Karachi")

**Response:**
```json
{
  "records": [
    {
      "incident_id": "IN-20220405-001",
      "occurrence_date": "2022-04-05",
      "incident_type": "Other; No Loss / No Injury",
      "title": "OVR catalyst loss",
      "status": "Closed",
      "department": "Process - EDC / VCM",
      "location": "Karachi",
      "severity_score": 3,
      "risk_score": 4.5
    },
    // ... more records
  ],
  "total_count": 1845,
  "filtered_count": 245,
  "returned_count": 10,
  "offset": 0,
  "sheet_name": "Incident",
  "columns_shown": ["incident_id", "occurrence_date", "incident_type", "..."],
  "filters_applied": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "search": "catalyst",
    "status": "Closed",
    "department": null,
    "location": "Karachi"
  }
}
```

#### **GET** `/data-health/sample/hazards`

**Query Parameters:** (Same as incidents)
- `limit`, `offset`, `start_date`, `end_date`, `search`, `status`, `department`, `location`

Returns Hazard ID sheet data with same search and filter capabilities.

#### **GET** `/data-health/sample/audits`

Returns Audit sheet data with audit-specific columns.

#### **GET** `/data-health/sample/inspections`

Returns Inspection sheet data.

**Use Case:** Verify that the backend is reading the correct Excel data.

---

### **3. Data Source Info**

**GET** `/data-health/source-info`

Get information about the Excel file being used.

**Response:**
```json
{
  "excel_file": {
    "filename": "EPCL_VEHS_Data_Processed.xlsx",
    "path": "C:/Users/ibrahim/Desktop/qbit/streamlitagentforepcl/EPCL_VEHS_Data_Processed.xlsx",
    "file_size_mb": 15.2,
    "last_modified": "2024-12-31T10:00:00",
    "exists": true
  },
  "sheets": [
    {
      "name": "Incident",
      "row_count": 1845,
      "column_count": 72,
      "columns": ["incident_id", "occurrence_date", "incident_type", "..."]
    },
    {
      "name": "Hazard ID",
      "row_count": 892,
      "column_count": 35,
      "columns": ["incident_id", "occurrence_date", "..."]
    },
    {
      "name": "Audit",
      "row_count": 456,
      "column_count": 36,
      "columns": ["audit_id", "start_date", "..."]
    },
    {
      "name": "Inspection",
      "row_count": 234,
      "column_count": 41,
      "columns": ["audit_id", "start_date", "..."]
    }
  ],
  "total_sheets": 4,
  "total_rows": 3427
}
```

**Use Case:** Verify which Excel file is being used and when it was last updated.

---

### **4. Data Validation Check**

**GET** `/data-health/validation/check`

Validate data quality and identify issues.

**Response:**
```json
{
  "validation_results": {
    "incidents": {
      "dataset": "Incidents",
      "total_rows": 1845,
      "valid_rows": 1837,
      "issues_found": 3,
      "issues": [
        {
          "type": "null_values",
          "severity": "medium",
          "column": "severity_score",
          "count": 8,
          "message": "Found 8 null values in severity_score"
        },
        {
          "type": "invalid_dates",
          "severity": "high",
          "column": "occurrence_date",
          "count": 2,
          "message": "Found 2 invalid date formats in occurrence_date"
        }
      ],
      "status": "has_issues"
    },
    "hazards": {
      "dataset": "Hazards",
      "total_rows": 892,
      "valid_rows": 890,
      "issues_found": 1,
      "issues": [
        {
          "type": "duplicate_ids",
          "severity": "medium",
          "count": 2,
          "message": "Found 2 duplicate IDs in incident_id"
        }
      ],
      "status": "has_issues"
    },
    "audits": {
      "dataset": "Audits",
      "total_rows": 456,
      "valid_rows": 456,
      "issues_found": 0,
      "issues": [],
      "status": "valid"
    },
    "inspections": {
      "dataset": "Inspections",
      "total_rows": 234,
      "valid_rows": 234,
      "issues_found": 0,
      "issues": [],
      "status": "valid"
    }
  },
  "overall_health_score": 97.5,
  "total_issues": 4,
  "status": "healthy"
}
```

**Validation Checks:**
- ✅ Missing required columns
- ✅ Duplicate IDs
- ✅ Null values in critical fields
- ✅ Invalid date formats
- ✅ Data completeness

**Use Case:** Identify data quality issues before they affect analytics.

---

## 🎨 Frontend Integration Examples

### React - Data Health Dashboard

```jsx
import React, { useEffect, useState } from 'react';
import axios from 'axios';

const DataHealthDashboard = () => {
  const [health, setHealth] = useState(null);
  const [validation, setValidation] = useState(null);

  useEffect(() => {
    // Load data health
    axios.get('/data-health/summary').then(res => setHealth(res.data));
    
    // Load validation results
    axios.get('/data-health/validation/check').then(res => setValidation(res.data));
  }, []);

  if (!health) return <div>Loading...</div>;

  return (
    <div className="data-health-dashboard">
      <h2>Data Health Status: {health.status}</h2>
      
      <div className="metrics">
        <div className="metric">
          <h3>Total Records</h3>
          <p>{health.grand_total.toLocaleString()}</p>
        </div>
        
        <div className="metric">
          <h3>Data Completeness</h3>
          <p>{health.data_quality.completeness_percentage}%</p>
        </div>
        
        <div className="metric">
          <h3>Date Range</h3>
          <p>{health.date_range.earliest} to {health.date_range.latest}</p>
        </div>
        
        <div className="metric">
          <h3>Health Score</h3>
          <p>{validation?.overall_health_score}%</p>
        </div>
      </div>

      <div className="record-counts">
        <h3>Records by Dataset</h3>
        <ul>
          <li>Incidents: {health.total_records.incidents}</li>
          <li>Hazards: {health.total_records.hazards}</li>
          <li>Audits: {health.total_records.audits}</li>
          <li>Inspections: {health.total_records.inspections}</li>
        </ul>
      </div>

      {validation && validation.total_issues > 0 && (
        <div className="issues-alert">
          <h3>⚠️ Data Issues Found: {validation.total_issues}</h3>
          {Object.values(validation.validation_results).map(result => (
            result.issues.length > 0 && (
              <div key={result.dataset}>
                <h4>{result.dataset}</h4>
                <ul>
                  {result.issues.map((issue, idx) => (
                    <li key={idx} className={`severity-${issue.severity}`}>
                      {issue.message}
                    </li>
                  ))}
                </ul>
              </div>
            )
          ))}
        </div>
      )}
    </div>
  );
};
```

---

### React - Raw Data Viewer

```jsx
const RawDataViewer = () => {
  const [data, setData] = useState(null);
  const [page, setPage] = useState(0);
  const [filters, setFilters] = useState({
    startDate: '',
    endDate: '',
    search: '',
    status: '',
    department: '',
    location: ''
  });
  const limit = 10;

  const loadData = async () => {
    const params = new URLSearchParams({
      limit: limit,
      offset: page * limit,
      ...(filters.startDate && { start_date: filters.startDate }),
      ...(filters.endDate && { end_date: filters.endDate }),
      ...(filters.search && { search: filters.search }),
      ...(filters.status && { status: filters.status }),
      ...(filters.department && { department: filters.department }),
      ...(filters.location && { location: filters.location })
    });

    const response = await axios.get(
      `/data-health/sample/incidents?${params}`
    );
    setData(response.data);
  };

  useEffect(() => {
    loadData();
  }, [page, filters]);

  if (!data) return <div>Loading...</div>;

  return (
    <div className="raw-data-viewer">
      <h2>Raw Incident Data</h2>
      
      {/* Search and Filters */}
      <div className="filters">
        <input
          type="date"
          placeholder="Start Date"
          value={filters.startDate}
          onChange={(e) => setFilters({...filters, startDate: e.target.value})}
        />
        <input
          type="date"
          placeholder="End Date"
          value={filters.endDate}
          onChange={(e) => setFilters({...filters, endDate: e.target.value})}
        />
        <input
          type="text"
          placeholder="Search..."
          value={filters.search}
          onChange={(e) => setFilters({...filters, search: e.target.value})}
        />
        <input
          type="text"
          placeholder="Status"
          value={filters.status}
          onChange={(e) => setFilters({...filters, status: e.target.value})}
        />
        <input
          type="text"
          placeholder="Department"
          value={filters.department}
          onChange={(e) => setFilters({...filters, department: e.target.value})}
        />
        <input
          type="text"
          placeholder="Location"
          value={filters.location}
          onChange={(e) => setFilters({...filters, location: e.target.value})}
        />
        <button onClick={() => setFilters({
          startDate: '', endDate: '', search: '', status: '', department: '', location: ''
        })}>
          Clear Filters
        </button>
      </div>

      <p>
        Showing {data.returned_count} of {data.filtered_count} filtered 
        (Total: {data.total_count} records)
      </p>
      
      <table>
        <thead>
          <tr>
            {data.columns_shown.map(col => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.records.map((record, idx) => (
            <tr key={idx}>
              {data.columns_shown.map(col => (
                <td key={col}>{record[col]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      <div className="pagination">
        <button 
          onClick={() => setPage(p => Math.max(0, p - 1))}
          disabled={page === 0}
        >
          Previous
        </button>
        <span>
          Page {page + 1} of {Math.ceil(data.filtered_count / limit)}
        </span>
        <button 
          onClick={() => setPage(p => p + 1)}
          disabled={(page + 1) * limit >= data.filtered_count}
        >
          Next
        </button>
      </div>
    </div>
  );
};
```

---

### React - Data Source Info Display

```jsx
const DataSourceInfo = () => {
  const [sourceInfo, setSourceInfo] = useState(null);

  useEffect(() => {
    axios.get('/data-health/source-info').then(res => setSourceInfo(res.data));
  }, []);

  if (!sourceInfo) return <div>Loading...</div>;

  return (
    <div className="data-source-info">
      <h2>Data Source Information</h2>
      
      <div className="file-info">
        <h3>Excel File</h3>
        <p><strong>Filename:</strong> {sourceInfo.excel_file.filename}</p>
        <p><strong>Path:</strong> {sourceInfo.excel_file.path}</p>
        <p><strong>Size:</strong> {sourceInfo.excel_file.file_size_mb} MB</p>
        <p><strong>Last Modified:</strong> {new Date(sourceInfo.excel_file.last_modified).toLocaleString()}</p>
        <p><strong>Status:</strong> {sourceInfo.excel_file.exists ? '✅ Found' : '❌ Not Found'}</p>
      </div>

      <div className="sheets-info">
        <h3>Sheets ({sourceInfo.total_sheets})</h3>
        {sourceInfo.sheets.map(sheet => (
          <div key={sheet.name} className="sheet-card">
            <h4>{sheet.name}</h4>
            <p>Rows: {sheet.row_count.toLocaleString()}</p>
            <p>Columns: {sheet.column_count}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
```

---

## 🚀 Testing

### Test in Swagger UI

1. Go to: http://localhost:8000/docs
2. Find the **"data-health"** tag
3. Test each endpoint:

```bash
# Data health summary
GET /data-health/summary

# Sample incidents (first 10)
GET /data-health/sample/incidents?limit=10&offset=0

# Sample incidents (next 10)
GET /data-health/sample/incidents?limit=10&offset=10

# Data source info
GET /data-health/source-info

# Validation check
GET /data-health/validation/check
```

### Test via cURL

```bash
# Health summary
curl http://localhost:8000/data-health/summary

# Sample data (basic)
curl "http://localhost:8000/data-health/sample/incidents?limit=5"

# Sample data with date range filter
curl "http://localhost:8000/data-health/sample/incidents?start_date=2024-01-01&end_date=2024-12-31&limit=10"

# Sample data with search
curl "http://localhost:8000/data-health/sample/incidents?search=catalyst&limit=10"

# Sample data with multiple filters
curl "http://localhost:8000/data-health/sample/incidents?start_date=2024-01-01&status=Closed&department=Process&location=Karachi&limit=20"

# Hazards with filters
curl "http://localhost:8000/data-health/sample/hazards?search=PPE&status=Closed&limit=10"

# Source info
curl http://localhost:8000/data-health/source-info

# Validation
curl http://localhost:8000/data-health/validation/check
```

---

## 📋 Use Cases

### **1. Dashboard Health Widget**
Display data health metrics on your main dashboard:
- Total record counts
- Data completeness percentage
- Last sync time
- Health status indicator

### **2. Data Verification Page**
Create a dedicated page to verify data:
- View raw Excel records
- Check file information
- See validation issues
- Monitor data quality over time

### **3. Troubleshooting**
When analytics show unexpected results:
1. Check `/data-health/summary` - Are records loaded?
2. Check `/data-health/source-info` - Is the correct file being used?
3. Check `/data-health/sample/incidents` - Does the raw data look correct?
4. Check `/data-health/validation/check` - Are there data quality issues?

### **4. Data Quality Monitoring**
Set up alerts based on validation results:
- Alert if health score < 90%
- Alert if critical issues found
- Alert if file hasn't been updated in X days

---

## 🎯 Best Practices

1. **Display on Dashboard**: Show key metrics (total records, health score) prominently
2. **Regular Checks**: Call `/data-health/summary` on app load
3. **Validation Alerts**: Show warnings if validation issues detected
4. **Raw Data Access**: Provide a "View Raw Data" button for admins
5. **File Info**: Display Excel file name and last modified date in footer

---

## 📊 Summary

| Endpoint | Purpose | Priority |
|----------|---------|----------|
| `/data-health/summary` | Overall health metrics | ⭐ HIGH |
| `/data-health/sample/incidents` | View raw incident data | ⭐ HIGH |
| `/data-health/sample/hazards` | View raw hazard data | HIGH |
| `/data-health/sample/audits` | View raw audit data | MEDIUM |
| `/data-health/sample/inspections` | View raw inspection data | MEDIUM |
| `/data-health/source-info` | Excel file information | ⭐ HIGH |
| `/data-health/validation/check` | Data quality validation | ⭐ HIGH |

---

## ✅ Benefits

✅ **Cross-verify data** - Confirm analytics use correct Excel data
✅ **Identify issues early** - Catch data quality problems before they affect analytics
✅ **Transparency** - Show users what data is being used
✅ **Debugging** - Quickly troubleshoot unexpected results
✅ **Monitoring** - Track data health over time
✅ **Confidence** - Build trust in analytics accuracy

All **7 data health endpoints** are ready to verify your data! 🏥
