# Safety Co‑pilot API Reference

Base URL (local): `http://127.0.0.1:8000`

Notes
- All endpoints are GET unless explicitly stated.
- Plotly figures are returned as JSON dictionaries compatible with `react-plotly.js`.
- For the Agent endpoint, set `OPENAI_API_KEY` in your environment to enable LLM code generation; the Agent still produces sensible fallbacks without a key.

---

## Health

- GET `/health`
  - 200 OK
  - Example
    ```json
    {"status": "ok"}
    ```

---

## Workbooks

- GET `/workbooks/reload`
  - Reloads the cached Excel from `fastapiepcl/app/EPCL_VEHS_Data_Processed.xlsx`.
  - 200 OK
  - Example
    ```json
    {
      "reloaded": true,
      "sheet_count": 4,
      "sheets": ["Incidents", "Hazards", "Audits", "Inspections"]
    }
    ```

- GET `/workbooks/selection`
  - Shows which sheet is mapped to each dataset after heuristics.
  - Example
    ```json
    {
      "incident": "Incidents",
      "hazard": "Hazards",
      "audit": "Audits",
      "inspection": "Inspections"
    }
    ```

- POST `/workbooks/upload`
  - Multipart upload: form field `file` = Excel (.xlsx)
  - Response: summary of sheets (names, columns, sample rows)

- GET `/workbooks/example`
  - Loads example file `EPCL_VEHS_Data_Processed.xlsx` from project root (if present)

- POST `/workbooks/infer-schema`
  - Body
    ```json
    {
      "sheet_name": "Incidents",
      "data": { "records": [ {"col": "val"} ] }
    }
    ```

---

## Wordclouds

- GET `/wordclouds/departments?top_n=50&min_count=1&extra_stopwords=na,misc`
  - Response
    ```json
    {
      "incident": [{"text": "ops", "value": 42}],
      "hazard":   [{"text": "safety", "value": 21}],
      "html_incident": "<div>…</div>",
      "html_hazard":   "<div>…</div>"
    }
    ```

---

## Maps

- GET `/maps/combined`
  - Response
    ```json
    { "html": "<!DOCTYPE html>…" }
    ```

- GET `/maps/single?dataset=incident|hazard`
  - Response (same shape)
    ```json
    { "html": "<!DOCTYPE html>…" }
    ```

---

## Analytics (Plotly JSON)

All responses share the shape:
```json
{ "figure": { /* plotly figure dict */ } }
```

- GET `/analytics/hse-scorecard`
- GET `/analytics/hse-performance-index?dataset=incident|hazard`
- GET `/analytics/risk-calendar-heatmap?dataset=incident|hazard`
- GET `/analytics/psm-breakdown?dataset=incident|hazard`
- GET `/analytics/consequence-matrix?dataset=incident|hazard`
- GET `/analytics/data-quality-metrics?dataset=incident|hazard`
- GET `/analytics/comprehensive-timeline?dataset=incident|hazard`
- GET `/analytics/audit-inspection-tracker`
- GET `/analytics/location-risk-treemap?dataset=incident|hazard`
- GET `/analytics/department-spider?dataset=incident|hazard`
- GET `/analytics/violation-analysis?dataset=incident|hazard` (default hazard)
- GET `/analytics/cost-prediction-analysis?dataset=incident|hazard`
- GET `/analytics/facility-layout-heatmap`
- GET `/analytics/facility-3d-heatmap?dataset=incident|hazard&event_type=Incidents`

---

## Conversion Analytics (Plotly JSON)

- GET `/analytics/conversion/funnel`
- GET `/analytics/conversion/time-lag`
- GET `/analytics/conversion/sankey`
- GET `/analytics/conversion/department-matrix`
- GET `/analytics/conversion/risk-network`
- GET `/analytics/conversion/prevention-effectiveness`
- GET `/analytics/conversion/metrics-gauge`

All responses share:
```json
{ "figure": { /* plotly figure dict */ } }
```

---

## Agent (LLM Data Assistant)

- GET `/agent/run?question=...&dataset=incident|hazard|audit|inspection|all&model=gpt-4o`
  - Response shape (`AgentRunResponse`)
    ```json
    {
      "code": "# Python code produced by the agent (no imports needed)",
      "stdout": "",             
      "error": "",              
      "result_preview": [ {"col": "val"}, {"col": "val"} ],  
      "figure": { /* plotly figure dict */ },                   
      "mpl_png_base64": null,                                   
      "analysis": "Findings / Recommendations / Next steps"     
    }
    ```

- Example request
  ```bash
  curl "http://127.0.0.1:8000/agent/run?question=top%205%20departments%20where%20most%20incidents%20occurs&dataset=incident"
  ```

- Example response (abridged)
  ```json
  {
    "code": "# no imports needed...\n...",
    "stdout": "",
    "error": "",
    "result_preview": [
      {"department": "Not Assigned", "count": 3013},
      {"department": "Inspection", "count": 144},
      {"department": "Instrument", "count": 118}
    ],
    "figure": {
      "data": [ { "type": "bar", "orientation": "h", "x": [3013,144,118], "y": ["Not Assigned","Inspection","Instrument"] } ],
      "layout": {"title": {"text": "Top 5 Departments by Incident Count"}, "yaxis": {"autorange": "reversed"}}
    },
    "mpl_png_base64": null,
    "analysis": "Findings: ... Recommendations: ... Next steps: ..."
  }
  ```

### React integration snippets

- Using `fetch` and `react-plotly.js`
  ```tsx
  import React, { useEffect, useState } from 'react';
  import Plot from 'react-plotly.js';

  type AgentResponse = {
    code: string;
    stdout: string;
    error: string;
    result_preview: Array<Record<string, any>>;
    figure?: any;
    mpl_png_base64?: string | null;
    analysis: string;
  };

  export default function AgentDemo() {
    const [data, setData] = useState<AgentResponse | null>(null);

    useEffect(() => {
      const url = '/agent/run?question=' + encodeURIComponent('top 5 departments where most incidents occurs') + '&dataset=incident';
      fetch(url)
        .then(r => r.json())
        .then(setData)
        .catch(console.error);
    }, []);

    if (!data) return <div>Loading…</div>;

    return (
      <div style={{ display: 'grid', gap: 16 }}>
        <pre style={{ background: '#0b1020', color: '#d9e0ee', padding: 12 }}>
{JSON.stringify({ code: data.code, stdout: data.stdout, error: data.error }, null, 2)}
        </pre>

        {/* Result table */}
        <table>
          <thead>
            <tr>
              {data.result_preview && data.result_preview[0] && Object.keys(data.result_preview[0]).map(k => (
                <th key={k}>{k}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.result_preview?.map((row, i) => (
              <tr key={i}>
                {Object.keys(data.result_preview[0] || {}).map(k => (
                  <td key={k}>{String(row[k])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>

        {/* Plotly figure */}
        {data.figure && (
          <Plot data={data.figure.data} layout={data.figure.layout} config={{ responsive: true }} style={{ width: '100%' }} />
        )}

        {/* Matplotlib fallback */}
        {(!data.figure && data.mpl_png_base64) && (
          <img alt="mpl" src={`data:image/png;base64,${data.mpl_png_base64}`} />
        )}

        {/* Narrative */}
        <div>
          <h3>Prescriptive Analysis</h3>
          <p style={{ whiteSpace: 'pre-wrap' }}>{data.analysis}</p>
        </div>
      </div>
    );
  }
  ```

- Using Axios
  ```ts
  import axios from 'axios';

  const url = '/agent/run?question=' + encodeURIComponent('incidents per location with average severity') + '&dataset=incident';
  const res = await axios.get(url);
  const agent = res.data; // AgentResponse
  ```

### Common Agent queries to test

- `top 5 departments where most incidents occurs`
- `incidents per location with average severity`
- `weekly incident trend with average severity and total cost`
- `top 10 violation types`
- `consequence matrix for incidents`
- `audit completion rates by month`

---

## Error handling

Agent and analytics endpoints strive to return valid JSON at all times. If the Agent encounters an execution error, it still returns:
```json
{
  "code": "…",
  "stdout": "…",
  "error": "…",
  "result_preview": [ … ],
  "figure": null,
  "mpl_png_base64": null,
  "analysis": "…"
}
```

---

## CORS

CORS is fully open for local development. Adjust `allow_origins` in `fastapiepcl/app/main.py` for production.
