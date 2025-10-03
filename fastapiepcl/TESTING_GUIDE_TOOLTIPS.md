# Enhanced Tooltips - Testing Guide

## 🧪 Testing Checklist

Use this guide to verify the enhanced tooltips endpoint is working correctly.

---

## Prerequisites

- [ ] FastAPI server running
- [ ] Excel data file loaded (`EPCL_VEHS_Data_Processed.xlsx`)
- [ ] Python environment activated

---

## 1. Start the Server

```bash
cd fastapiepcl
uvicorn app.main:app --reload
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

✅ Server is running

---

## 2. Test via Swagger UI

### Step 1: Open Swagger UI
Navigate to: `http://localhost:8000/docs`

### Step 2: Find the Endpoint
Scroll to: `/analytics/data/incident-trend-detailed`

### Step 3: Try Basic Request
1. Click "Try it out"
2. Set `dataset` to `incident`
3. Leave other fields empty
4. Click "Execute"

**Expected Response:**
```json
{
  "labels": ["2023-01", "2023-02", ...],
  "series": [{"name": "Count", "data": [15, 22, ...]}],
  "details": [
    {
      "month": "2023-01",
      "total_count": 15,
      "departments": [...],
      "types": [...],
      "severity": {...},
      "risk": {...},
      "recent_items": [...]
    }
  ]
}
```

✅ Basic request works

### Step 4: Try Filtered Request
1. Set `dataset` to `incident`
2. Set `start_date` to `2023-01-01`
3. Set `end_date` to `2023-12-31`
4. Set `min_severity` to `3.0`
5. Click "Execute"

**Expected:**
- Response contains only months between Jan-Dec 2023
- All incidents have severity >= 3.0

✅ Filters work correctly

---

## 3. Test via cURL

### Test 1: Basic Request

```bash
curl -X GET "http://localhost:8000/analytics/data/incident-trend-detailed?dataset=incident" \
  -H "accept: application/json"
```

**Verify:**
- [ ] HTTP 200 status
- [ ] Valid JSON response
- [ ] Contains `labels`, `series`, `details`

### Test 2: With Date Filters

```bash
curl -X GET "http://localhost:8000/analytics/data/incident-trend-detailed?dataset=incident&start_date=2023-01-01&end_date=2023-12-31" \
  -H "accept: application/json"
```

**Verify:**
- [ ] Only 2023 months returned
- [ ] Details match filtered data

### Test 3: With Multiple Filters

```bash
curl -X GET "http://localhost:8000/analytics/data/incident-trend-detailed?dataset=incident&start_date=2023-01-01&end_date=2023-12-31&min_severity=3.0&departments=Operations" \
  -H "accept: application/json"
```

**Verify:**
- [ ] Filtered by date, severity, and department
- [ ] Response size smaller than unfiltered

### Test 4: Hazard Dataset

```bash
curl -X GET "http://localhost:8000/analytics/data/incident-trend-detailed?dataset=hazard" \
  -H "accept: application/json"
```

**Verify:**
- [ ] Returns hazard data
- [ ] Types show violation types (not incident types)

---

## 4. Test via Python

Create `test_tooltips.py`:

```python
import requests
import json

BASE_URL = "http://localhost:8000"

def test_basic_request():
    """Test basic request without filters"""
    print("Test 1: Basic Request")
    response = requests.get(
        f"{BASE_URL}/analytics/data/incident-trend-detailed",
        params={"dataset": "incident"}
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    assert "labels" in data, "Missing 'labels' in response"
    assert "series" in data, "Missing 'series' in response"
    assert "details" in data, "Missing 'details' in response"
    
    print(f"✅ Passed - Found {len(data['labels'])} months")
    print(f"   First month: {data['labels'][0] if data['labels'] else 'N/A'}")
    print(f"   Details count: {len(data['details'])}")
    return data

def test_filtered_request():
    """Test request with filters"""
    print("\nTest 2: Filtered Request")
    response = requests.get(
        f"{BASE_URL}/analytics/data/incident-trend-detailed",
        params={
            "dataset": "incident",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "min_severity": 3.0
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify date filtering
    for label in data['labels']:
        year = label.split('-')[0]
        assert year == "2023", f"Expected 2023, got {year}"
    
    print(f"✅ Passed - All months in 2023")
    print(f"   Months returned: {len(data['labels'])}")
    return data

def test_detail_structure():
    """Test detail structure is correct"""
    print("\nTest 3: Detail Structure")
    response = requests.get(
        f"{BASE_URL}/analytics/data/incident-trend-detailed",
        params={"dataset": "incident"}
    )
    
    data = response.json()
    
    if len(data['details']) > 0:
        detail = data['details'][0]
        
        # Check required fields
        assert "month" in detail, "Missing 'month'"
        assert "total_count" in detail, "Missing 'total_count'"
        assert "departments" in detail, "Missing 'departments'"
        assert "types" in detail, "Missing 'types'"
        assert "recent_items" in detail, "Missing 'recent_items'"
        
        print(f"✅ Passed - Detail structure valid")
        print(f"   Month: {detail['month']}")
        print(f"   Total count: {detail['total_count']}")
        print(f"   Departments: {len(detail['departments'])}")
        print(f"   Types: {len(detail['types'])}")
        print(f"   Recent items: {len(detail['recent_items'])}")
        
        # Print sample department
        if detail['departments']:
            dept = detail['departments'][0]
            print(f"   Top department: {dept['name']} ({dept['count']})")
        
        # Print sample type
        if detail['types']:
            typ = detail['types'][0]
            print(f"   Top type: {typ['name']} ({typ['count']})")
        
        # Print severity/risk if available
        if detail.get('severity'):
            sev = detail['severity']
            print(f"   Severity: avg={sev['avg']:.1f}, max={sev['max']}, min={sev['min']}")
        
        if detail.get('risk'):
            risk = detail['risk']
            print(f"   Risk: avg={risk['avg']:.1f}, max={risk['max']}, min={risk['min']}")
    else:
        print("⚠️  No details returned (empty dataset?)")

def test_hazard_dataset():
    """Test hazard dataset"""
    print("\nTest 4: Hazard Dataset")
    response = requests.get(
        f"{BASE_URL}/analytics/data/incident-trend-detailed",
        params={"dataset": "hazard"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    print(f"✅ Passed - Hazard dataset works")
    print(f"   Months: {len(data['labels'])}")
    print(f"   Details: {len(data['details'])}")

def test_empty_result():
    """Test with filters that return no data"""
    print("\nTest 5: Empty Result")
    response = requests.get(
        f"{BASE_URL}/analytics/data/incident-trend-detailed",
        params={
            "dataset": "incident",
            "start_date": "2050-01-01",  # Future date
            "end_date": "2050-12-31"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert len(data['labels']) == 0, "Expected empty labels"
    assert len(data['details']) == 0, "Expected empty details"
    
    print(f"✅ Passed - Empty result handled correctly")

def test_performance():
    """Test response time"""
    print("\nTest 6: Performance")
    import time
    
    start = time.time()
    response = requests.get(
        f"{BASE_URL}/analytics/data/incident-trend-detailed",
        params={"dataset": "incident"}
    )
    elapsed = time.time() - start
    
    assert response.status_code == 200
    
    print(f"✅ Passed - Response time: {elapsed:.2f}s")
    
    if elapsed > 2.0:
        print(f"⚠️  Warning: Response time > 2s (consider adding filters)")
    else:
        print(f"   Performance is good!")

if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Tooltips API - Test Suite")
    print("=" * 60)
    
    try:
        test_basic_request()
        test_filtered_request()
        test_detail_structure()
        test_hazard_dataset()
        test_empty_result()
        test_performance()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
```

Run the tests:

```bash
python test_tooltips.py
```

**Expected Output:**
```
============================================================
Enhanced Tooltips API - Test Suite
============================================================
Test 1: Basic Request
✅ Passed - Found 12 months
   First month: 2023-01
   Details count: 12

Test 2: Filtered Request
✅ Passed - All months in 2023
   Months returned: 12

Test 3: Detail Structure
✅ Passed - Detail structure valid
   Month: 2023-01
   Total count: 15
   Departments: 3
   Types: 5
   Recent items: 5
   Top department: Operations (8)
   Top type: Slip (4)
   Severity: avg=3.2, max=5.0, min=1.0
   Risk: avg=3.8, max=5.0, min=2.0

Test 4: Hazard Dataset
✅ Passed - Hazard dataset works
   Months: 12
   Details: 12

Test 5: Empty Result
✅ Passed - Empty result handled correctly

Test 6: Performance
✅ Passed - Response time: 0.45s
   Performance is good!

============================================================
✅ ALL TESTS PASSED!
============================================================
```

---

## 5. Test via JavaScript/Browser Console

Open browser console on your frontend app:

```javascript
// Test 1: Basic Request
fetch('/analytics/data/incident-trend-detailed?dataset=incident')
  .then(r => r.json())
  .then(data => {
    console.log('Labels:', data.labels);
    console.log('Series:', data.series);
    console.log('Details count:', data.details.length);
    
    if (data.details.length > 0) {
      const first = data.details[0];
      console.log('First month:', first.month);
      console.log('Total count:', first.total_count);
      console.log('Departments:', first.departments);
      console.log('Types:', first.types);
      console.log('Recent items:', first.recent_items);
    }
  });

// Test 2: With Filters
const params = new URLSearchParams({
  dataset: 'incident',
  start_date: '2023-01-01',
  end_date: '2023-12-31',
  min_severity: '3.0'
});

fetch(`/analytics/data/incident-trend-detailed?${params}`)
  .then(r => r.json())
  .then(data => {
    console.log('Filtered results:', data.labels.length, 'months');
    console.table(data.details.map(d => ({
      month: d.month,
      count: d.total_count,
      top_dept: d.departments[0]?.name,
      top_type: d.types[0]?.name
    })));
  });
```

---

## 6. Validation Checklist

### Response Structure
- [ ] `labels` is an array of strings (YYYY-MM format)
- [ ] `series` is an array with at least one series object
- [ ] `series[0].name` is "Count"
- [ ] `series[0].data` is an array of numbers
- [ ] `details` is an array
- [ ] `details.length` equals `labels.length`

### Detail Object Structure
- [ ] Each detail has `month` (string)
- [ ] Each detail has `total_count` (number)
- [ ] Each detail has `departments` (array)
- [ ] Each detail has `types` (array)
- [ ] Each detail has `severity` (object or null)
- [ ] Each detail has `risk` (object or null)
- [ ] Each detail has `recent_items` (array)

### Department/Type Items
- [ ] Each has `name` (string)
- [ ] Each has `count` (number)
- [ ] Maximum 5 items per array
- [ ] Sorted by count (descending)

### Score Stats (if present)
- [ ] Has `avg` (float)
- [ ] Has `max` (float)
- [ ] Has `min` (float)
- [ ] Values are between 0-5

### Recent Items
- [ ] Each has `title` (string, max 100 chars)
- [ ] Each has `department` (string)
- [ ] Each has `date` (string, YYYY-MM-DD format)
- [ ] Each has `severity` (float or null)
- [ ] Maximum 5 items per month
- [ ] Sorted by date (most recent first)

### Filters
- [ ] Date filters work correctly
- [ ] Department filter works
- [ ] Location filter works
- [ ] Severity filter works
- [ ] Risk filter works
- [ ] Multiple filters combine with AND logic

### Edge Cases
- [ ] Empty dataset returns empty arrays
- [ ] Future dates return empty results
- [ ] Invalid severity range (e.g., min > max) handled
- [ ] Missing columns handled gracefully
- [ ] Null/NaN values filtered out

---

## 7. Performance Benchmarks

| Scenario | Expected Time | Acceptable Range |
|----------|---------------|------------------|
| 1 year data | 200-400ms | < 1s |
| 2 years data | 400-800ms | < 2s |
| 5 years data | 1-2s | < 5s |
| With filters | 100-300ms | < 1s |

**If response time > 2s:**
- Check dataset size
- Add date filters
- Consider caching
- Review server resources

---

## 8. Common Issues & Solutions

### Issue: Empty `details` array

**Symptoms:**
```json
{
  "labels": [],
  "series": [{"name": "Count", "data": []}],
  "details": []
}
```

**Possible Causes:**
1. No data in date range
2. Filters too restrictive
3. Excel file not loaded

**Solutions:**
1. Check date filters
2. Remove filters and try again
3. Verify Excel file exists in `app/` folder

---

### Issue: Missing severity/risk stats

**Symptoms:**
```json
{
  "severity": null,
  "risk": null
}
```

**Cause:** Columns not found in dataset

**Solution:** This is expected if columns don't exist. Not an error.

---

### Issue: Truncated titles

**Symptoms:**
```json
{
  "title": "This is a very long incident description that gets truncated after one hundred characters..."
}
```

**Cause:** Intentional 100-character limit

**Solution:** This is by design to keep response size manageable.

---

### Issue: Slow response

**Symptoms:** Response takes > 2 seconds

**Solutions:**
1. Add date filters to limit range
2. Use department/location filters
3. Check server resources
4. Consider caching on frontend

---

## 9. Integration Testing

### Test with Frontend

1. **Update data fetching:**
   ```typescript
   const response = await fetch('/analytics/data/incident-trend-detailed?dataset=incident');
   const data = await response.json();
   ```

2. **Pass to chart component:**
   ```tsx
   <LineChart 
     labels={data.labels}
     series={data.series}
     tooltipDetails={data.details}
   />
   ```

3. **Verify tooltip displays:**
   - Hover over data point
   - Check all sections render
   - Verify data accuracy

---

## 10. Final Checklist

Before deploying to production:

- [ ] All unit tests pass
- [ ] Manual testing completed
- [ ] Performance acceptable (< 2s)
- [ ] Edge cases handled
- [ ] Documentation reviewed
- [ ] Frontend integration tested
- [ ] Error handling verified
- [ ] Caching strategy implemented
- [ ] User acceptance testing done
- [ ] Monitoring/logging configured

---

## 🎉 Success Criteria

✅ **Backend is ready when:**
- All tests pass
- Response time < 2s
- All filters work correctly
- Edge cases handled gracefully
- Documentation complete

✅ **Frontend is ready when:**
- Data fetches successfully
- Tooltips render correctly
- Loading states work
- Error handling implemented
- User feedback positive

---

**Need help?** Check the full documentation or contact the development team! 🚀
