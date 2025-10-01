# 🧪 Testing Guide for Advanced Analytics API

## Quick Start

1. **Start the FastAPI server:**
   ```bash
   cd fastapiepcl
   uvicorn app.main:app --reload
   ```

2. **Access Swagger UI:**
   - Open browser: http://localhost:8000/docs
   - All endpoints have pre-filled example parameters
   - Click "Try it out" → "Execute" to test

---

## 📊 Pre-filled Example Parameters

All endpoints now include **default example values** in the Swagger UI for easy testing:

### Heinrich's Safety Pyramid
- **Endpoint:** `GET /analytics/advanced/heinrich-pyramid`
- **Pre-filled params:**
  - `start_date`: 2024-01-01
  - `end_date`: 2024-12-31
  - `location`: Karachi
  - `department`: Process

### Site Safety Index
- **Endpoint:** `GET /analytics/advanced/site-safety-index`
- **Pre-filled params:**
  - `start_date`: 2024-01-01
  - `end_date`: 2024-12-31
  - `location`: Manufacturing Facility

### TRIR / LTIR / PSTIR
- **Endpoints:** 
  - `GET /analytics/advanced/kpis/trir`
  - `GET /analytics/advanced/kpis/ltir`
  - `GET /analytics/advanced/kpis/pstir`
- **Pre-filled params:**
  - `start_date`: 2023-01-01
  - `end_date`: 2024-12-31
  - `total_hours_worked`: 2000000

### Near-Miss Ratio
- **Endpoint:** `GET /analytics/advanced/kpis/near-miss-ratio`
- **Pre-filled params:**
  - `start_date`: 2023-01-01
  - `end_date`: 2024-12-31

### KPI Summary
- **Endpoint:** `GET /analytics/advanced/kpis/summary`
- **Pre-filled params:**
  - `start_date`: 2023-01-01
  - `end_date`: 2024-12-31

### Incident Forecast
- **Endpoint:** `GET /analytics/predictive/incident-forecast`
- **Pre-filled params:**
  - `months_ahead`: 4
  - `location`: Karachi
  - `department`: Process - EDC / VCM

### Risk Trend Projection
- **Endpoint:** `GET /analytics/predictive/risk-trend-projection`
- **Pre-filled params:**
  - `months_ahead`: 3
  - `location`: Manufacturing Facility

### Leading vs Lagging
- **Endpoint:** `GET /analytics/predictive/leading-vs-lagging`
- **Pre-filled params:**
  - `start_date`: 2023-01-01
  - `end_date`: 2024-12-31
  - `location`: Karachi

### Observation Lag Time
- **Endpoint:** `GET /analytics/predictive/observation-lag-time`
- **Pre-filled params:**
  - `location`: Karachi
  - `department`: PVC

---

## 🎯 Step-by-Step Testing in Swagger UI

### Test Heinrich's Safety Pyramid

1. Go to: http://localhost:8000/docs
2. Find: `GET /analytics/advanced/heinrich-pyramid`
3. Click **"Try it out"**
4. Notice pre-filled example values:
   - `start_date`: 2024-01-01
   - `end_date`: 2024-12-31
   - `location`: Karachi
   - `department`: Process
5. Click **"Execute"**
6. View response showing 5-layer pyramid with counts and ratios

### Test Site Safety Index

1. Find: `GET /analytics/advanced/site-safety-index`
2. Click **"Try it out"**
3. Pre-filled values ready to use
4. Click **"Execute"**
5. View 0-100 score with breakdown

### Test TRIR KPI

1. Find: `GET /analytics/advanced/kpis/trir`
2. Click **"Try it out"**
3. Pre-filled: `total_hours_worked: 2000000`
4. Click **"Execute"**
5. View TRIR value with benchmark assessment

### Test Incident Forecast

1. Find: `GET /analytics/predictive/incident-forecast`
2. Click **"Try it out"**
3. Pre-filled: `months_ahead: 4`, `location: Karachi`
4. Click **"Execute"**
5. View historical data + 4-month forecast with confidence intervals

### Test Leading vs Lagging

1. Find: `GET /analytics/predictive/leading-vs-lagging`
2. Click **"Try it out"**
3. Pre-filled date range ready
4. Click **"Execute"**
5. View proactive vs reactive indicator comparison

---

## 🔧 Testing Without Filters (Full Dataset)

To test on **all data** (no filters), simply:
1. Click **"Try it out"**
2. **Clear/delete** the example values for optional parameters
3. Leave date/location/department **empty**
4. Click **"Execute"**

This will query the entire dataset without any filtering.

---

## 📋 Quick Test Checklist

Use this checklist to verify all endpoints work:

- [ ] `/analytics/advanced/heinrich-pyramid` ✅
- [ ] `/analytics/advanced/site-safety-index` ✅
- [ ] `/analytics/advanced/kpis/trir` ✅
- [ ] `/analytics/advanced/kpis/ltir` ✅
- [ ] `/analytics/advanced/kpis/pstir` ✅
- [ ] `/analytics/advanced/kpis/near-miss-ratio` ✅
- [ ] `/analytics/advanced/kpis/summary` ✅
- [ ] `/analytics/predictive/incident-forecast` ✅
- [ ] `/analytics/predictive/risk-trend-projection` ✅
- [ ] `/analytics/predictive/leading-vs-lagging` ✅
- [ ] `/analytics/predictive/observation-lag-time` ✅

---

## 🌐 cURL Testing (Alternative)

If you prefer command-line testing:

### Heinrich's Pyramid
```bash
curl "http://localhost:8000/analytics/advanced/heinrich-pyramid?start_date=2024-01-01&end_date=2024-12-31&location=Karachi&department=Process"
```

### Safety Index
```bash
curl "http://localhost:8000/analytics/advanced/site-safety-index?start_date=2024-01-01&end_date=2024-12-31"
```

### TRIR
```bash
curl "http://localhost:8000/analytics/advanced/kpis/trir?start_date=2023-01-01&end_date=2024-12-31&total_hours_worked=2000000"
```

### Incident Forecast
```bash
curl "http://localhost:8000/analytics/predictive/incident-forecast?months_ahead=4&location=Karachi"
```

### Leading vs Lagging
```bash
curl "http://localhost:8000/analytics/predictive/leading-vs-lagging?start_date=2023-01-01&end_date=2024-12-31"
```

---

## 🐛 Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process if needed (replace PID)
taskkill /PID <PID> /F

# Restart server
uvicorn app.main:app --reload
```

### Empty responses
- **Check Excel file path:** Ensure `EPCL_VEHS_Data_Processed.xlsx` is in the correct location
- **Verify data:** Check if the Excel sheets have data
- **Check filters:** Try without filters first (clear all optional params)

### Import errors
```bash
# Install missing dependencies
pip install -r requirements.txt
```

---

## 📊 Expected Response Examples

### Heinrich's Pyramid Response
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
      "count": 15,
      "ratio": 5.0,
      "color": "#f57c00"
    },
    ...
  ],
  "total_events": 450,
  "near_miss_ratio": 8.5
}
```

### Safety Index Response
```json
{
  "score": 78.5,
  "rating": "Good",
  "color": "#8bc34a",
  "breakdown": [
    {
      "factor": "Serious Injuries (3)",
      "impact": -30
    },
    {
      "factor": "Days Since Last Incident (45)",
      "impact": 4.5
    }
  ]
}
```

### TRIR Response
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

---

## ✨ Pro Tips

1. **Use the Swagger UI** - It's the easiest way to test with pre-filled params
2. **Test without filters first** - Verify data loads, then add filters
3. **Check the Response tab** - Look for HTTP 200 status and valid JSON
4. **Use the "Schemas" section** - See full response structure at bottom of docs
5. **Try different date ranges** - Your data spans 2022-2024, adjust accordingly
6. **Check department names** - Use exact names from your Excel (e.g., "Process - EDC / VCM")

---

## 📞 Next Steps

Once testing is complete:
1. ✅ All endpoints working → Ready for frontend integration
2. ❌ Errors found → Check logs and troubleshoot
3. 🎨 Need customization → Modify filters or add new params

Happy testing! 🚀
