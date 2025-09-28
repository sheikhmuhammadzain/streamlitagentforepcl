"""
Plotly to OpenAI Insights Generator
A comprehensive solution for extracting meaningful insights from Plotly charts using OpenAI
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Safe import for OpenAI client
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


class InsightType(Enum):
    """Types of insights to generate"""

    EXECUTIVE_SUMMARY = "executive_summary"
    STATISTICAL = "statistical"
    TRENDS = "trends"
    ANOMALIES = "anomalies"
    RECOMMENDATIONS = "recommendations"
    PREDICTIVE = "predictive"


@dataclass
class PlotlyDataExtractor:
    """Extract meaningful data from Plotly figures for AI analysis"""

    @staticmethod
    def extract_chart_metadata(fig: Dict) -> Dict:
        """Extract basic metadata from the figure"""
        layout = fig.get("layout", {})

        # Handle different title formats in Plotly
        title: str
        if isinstance(layout.get("title"), dict):
            title = layout["title"].get("text", "Untitled Chart")
        elif isinstance(layout.get("title"), str):
            title = layout["title"]
        else:
            title = "Untitled Chart"

        def _axis_title(axis: Dict) -> str:
            if not isinstance(axis, dict):
                return ""
            t = axis.get("title")
            if isinstance(t, dict):
                return t.get("text", "")
            if isinstance(t, str):
                return t
            return ""

        return {
            "title": title,
            "x_axis": _axis_title(layout.get("xaxis", {})),
            "y_axis": _axis_title(layout.get("yaxis", {})),
            "chart_types": list({(trace.get("type", "unknown") or "unknown") for trace in fig.get("data", [])}),
            "num_traces": len(fig.get("data", [])),
            "has_annotations": bool(layout.get("annotations")),
            "has_legend": layout.get("showlegend", True),
        }

    @staticmethod
    def extract_data_points(trace: Dict) -> Dict:
        """Extract numerical data from a single trace"""
        trace_type = trace.get("type", "unknown")
        name = trace.get("name", trace_type)

        result: Dict[str, Any] = {
            "name": name,
            "type": trace_type,
            "statistics": {},
            "data_points": [],
            "categories": [],
            "special_values": {},
        }

        def to_numeric(arr):
            if arr is None:
                return pd.Series([], dtype=float)
            try:
                if isinstance(arr, (list, tuple)):
                    s = pd.Series(arr)
                elif isinstance(arr, np.ndarray):
                    s = pd.Series(arr.flatten())
                else:
                    s = pd.Series([arr])
                s = pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce")
                return s
            except Exception:
                return pd.Series([], dtype=float)

        if trace_type in ["bar", "scatter", "line", "scattergl"]:
            x_data = trace.get("x", [])
            y_data = trace.get("y", [])

            y_numeric = to_numeric(y_data)
            x_numeric = to_numeric(x_data)

            if not y_numeric.dropna().empty:
                result["statistics"] = {
                    "mean": float(y_numeric.mean()),
                    "median": float(y_numeric.median()),
                    "std": float(y_numeric.std()),
                    "min": float(y_numeric.min()),
                    "max": float(y_numeric.max()),
                    "sum": float(y_numeric.sum()),
                    "count": int(y_numeric.count()),
                }
                result["data_points"] = y_numeric.dropna().tolist()[:100]
                if not x_numeric.dropna().empty:
                    result["categories"] = []
                else:
                    result["categories"] = list(x_data)[:50]

            elif not x_numeric.dropna().empty:
                result["statistics"] = {
                    "mean": float(x_numeric.mean()),
                    "median": float(x_numeric.median()),
                    "std": float(x_numeric.std()),
                    "min": float(x_numeric.min()),
                    "max": float(x_numeric.max()),
                    "sum": float(x_numeric.sum()),
                    "count": int(x_numeric.count()),
                }
                result["data_points"] = x_numeric.dropna().tolist()[:100]
                result["categories"] = list(y_data)[:50] if y_data else []

        elif trace_type == "pie":
            labels = trace.get("labels", [])
            values = to_numeric(trace.get("values", []))
            if not values.dropna().empty:
                total = float(values.sum()) if float(values.sum()) != 0 else 0.0
                result["statistics"] = {
                    "total": total,
                    "count": int(values.count()),
                    "max_slice": float(values.max()),
                    "min_slice": float(values.min()),
                }
                if total > 0:
                    result["special_values"]["percentages"] = {
                        str(labels[i]): float(values.iloc[i] / total * 100)
                        for i in range(min(len(labels), len(values)))
                    }

        elif trace_type in ("heatmap", "heatmapgl", "image"):
            z_data = trace.get("z")
            parsed = False
            # Try parsing z first
            if z_data is not None:
                try:
                    # Robust numeric coercion via DataFrame to handle strings/None
                    df_z = pd.DataFrame(z_data)
                    df_num = df_z.applymap(lambda x: str(x).replace(",", "").replace("%", ""))
                    df_num = df_num.apply(pd.to_numeric, errors="coerce")
                    z_array = df_num.values.astype(float)
                    z_flat = z_array.ravel()
                    z_series = pd.Series(z_flat[np.isfinite(z_flat)])
                    if not z_series.empty:
                        result["statistics"] = {
                            "mean": float(z_series.mean()),
                            "max": float(z_series.max()),
                            "min": float(z_series.min()),
                            "std": float(z_series.std()),
                        }
                        if z_array.ndim == 2 and np.isfinite(z_array).any():
                            max_indices = np.unravel_index(np.nanargmax(z_array), z_array.shape)
                            hotspot = {
                                "row": int(max_indices[0]),
                                "col": int(max_indices[1]),
                                "value": float(z_array[max_indices]),
                            }
                            # Add labels if provided
                            x_labels = trace.get("x") or []
                            y_labels = trace.get("y") or []
                            try:
                                hotspot["x_label"] = str(x_labels[hotspot["col"]]) if x_labels else None
                                hotspot["y_label"] = str(y_labels[hotspot["row"]]) if y_labels else None
                            except Exception:
                                pass
                            result["special_values"]["hotspot"] = hotspot

                            # Department and month averages
                            try:
                                row_means = np.nanmean(z_array, axis=1)
                                col_means = np.nanmean(z_array, axis=0)
                                dept_avg = [
                                    {"department": str(y_labels[i]) if i < len(y_labels) else str(i), "avg": float(row_means[i])}
                                    for i in range(len(row_means))
                                ]
                                month_avg = [
                                    {"month": str(x_labels[j]) if j < len(x_labels) else str(j), "avg": float(col_means[j])}
                                    for j in range(len(col_means))
                                ]
                                result["special_values"]["department_averages"] = sorted(dept_avg, key=lambda d: d["avg"], reverse=True)
                                result["special_values"]["month_averages"] = month_avg

                                # Per-department start/end change and stability
                                if z_array.shape[1] >= 2:
                                    changes = []
                                    stability = []
                                    for i in range(z_array.shape[0]):
                                        series = z_array[i, :]
                                        # Use first/last finite values
                                        try:
                                            first_idx = int(np.where(np.isfinite(series))[0][0])
                                            last_idx = int(np.where(np.isfinite(series))[0][-1])
                                            start_val = float(series[first_idx])
                                            end_val = float(series[last_idx])
                                            changes.append({
                                                "department": str(y_labels[i]) if i < len(y_labels) else str(i),
                                                "start": start_val,
                                                "end": end_val,
                                                "change": end_val - start_val,
                                                "start_label": str(x_labels[first_idx]) if first_idx < len(x_labels) else str(first_idx),
                                                "end_label": str(x_labels[last_idx]) if last_idx < len(x_labels) else str(last_idx),
                                            })
                                            stability.append({
                                                "department": str(y_labels[i]) if i < len(y_labels) else str(i),
                                                "std": float(np.nanstd(series)),
                                            })
                                        except Exception:
                                            continue
                                    # Sort to find notable items
                                    inc = sorted(changes, key=lambda d: d["change"], reverse=True)
                                    dec = sorted(changes, key=lambda d: d["change"])  # most negative first
                                    stable = sorted(stability, key=lambda d: d["std"])  # lowest std
                                    result["special_values"]["top_increases"] = inc[:3]
                                    result["special_values"]["top_decreases"] = dec[:3]
                                    result["special_values"]["most_stable"] = stable[:3]
                            except Exception:
                                pass
                        parsed = True
                except Exception:
                    parsed = False

            # Fallback: parse numeric content from text matrix if available
            if not parsed:
                text_data = trace.get("text")
                if text_data is not None:
                    try:
                        # Convert 2D text to numeric by stripping commas/percent
                        df = pd.DataFrame(text_data)
                        s = pd.to_numeric(
                            df.values.astype(str).ravel()
                            .astype(str)
                            .astype(object)
                        , errors="coerce")
                        if not pd.Series(s).dropna().empty:
                            s = pd.Series(s).dropna()
                            result["statistics"] = {
                                "mean": float(s.mean()),
                                "max": float(s.max()),
                                "min": float(s.min()),
                                "std": float(s.std()),
                            }
                            # Hotspot index if shape is 2D
                            try:
                                arr = pd.to_numeric(df.applymap(lambda x: str(x).replace(",", "").replace("%", "")), errors="coerce").values.astype(float)
                                if arr.ndim == 2 and np.isfinite(arr).any():
                                    max_indices = np.unravel_index(np.nanargmax(arr), arr.shape)
                                    result["special_values"]["hotspot"] = {
                                        "row": int(max_indices[0]),
                                        "col": int(max_indices[1]),
                                        "value": float(arr[max_indices]),
                                    }
                            except Exception:
                                pass
                    except Exception:
                        pass

        elif trace_type == "indicator":
            value = trace.get("value")
            try:
                if value is not None:
                    v = float(value)
                    result["special_values"]["current_value"] = v
                    delta = trace.get("delta") or {}
                    ref = delta.get("reference")
                    if ref is not None:
                        r = float(ref)
                        result["special_values"]["reference"] = r
                        result["special_values"]["change"] = v - r
            except Exception:
                pass

        elif trace_type == "treemap":
            labels = trace.get("labels", [])
            values = to_numeric(trace.get("values", []))
            parents = trace.get("parents", [])
            if not values.dropna().empty:
                leaf_indices = [i for i, label in enumerate(labels) if label not in parents]
                if leaf_indices:
                    leaf_values = values.iloc[leaf_indices]
                    result["statistics"] = {
                        "total": float(values.sum()),
                        "leaf_count": len(leaf_indices),
                        "max_leaf": float(leaf_values.max()),
                        "mean_leaf": float(leaf_values.mean()),
                    }

        elif trace_type in ["histogram", "histogram2d"]:
            x_data = to_numeric(trace.get("x", []))
            if not x_data.dropna().empty:
                result["statistics"] = {
                    "mean": float(x_data.mean()),
                    "median": float(x_data.median()),
                    "std": float(x_data.std()),
                    "min": float(x_data.min()),
                    "max": float(x_data.max()),
                    "count": int(x_data.count()),
                }

        elif trace_type in ["scatterpolar", "barpolar"]:
            r_data = to_numeric(trace.get("r", []))
            theta = trace.get("theta", []) or trace.get("labels", [])
            if not r_data.dropna().empty:
                result["statistics"] = {
                    "mean": float(r_data.mean()),
                    "median": float(r_data.median()),
                    "std": float(r_data.std()),
                    "min": float(r_data.min()),
                    "max": float(r_data.max()),
                    "sum": float(r_data.sum()),
                    "count": int(r_data.count()),
                }
                result["data_points"] = r_data.dropna().tolist()[:100]
                result["categories"] = list(theta)[:50]

        return result

    @staticmethod
    def extract_all_data(fig: Dict) -> Dict:
        """Extract all relevant data from the figure"""
        metadata = PlotlyDataExtractor.extract_chart_metadata(fig)
        traces_data: List[Dict[str, Any]] = []
        for trace in fig.get("data", []) or []:
            trace_data = PlotlyDataExtractor.extract_data_points(trace)
            if trace_data.get("statistics") or trace_data.get("special_values"):
                traces_data.append(trace_data)
        return {
            "metadata": metadata,
            "traces": traces_data,
            "annotations": (fig.get("layout", {}) or {}).get("annotations", []),
        }


class PlotlyInsightsGenerator:
    """Generate AI insights from Plotly figures using OpenAI"""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.client = None
        try:
            if OpenAI is not None:
                self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        except Exception:
            self.client = None
        self.extractor = PlotlyDataExtractor()

    def prepare_context_for_ai(
        self,
        extracted_data: Dict,
        business_context: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
    ) -> str:
        context_parts: List[str] = []
        metadata = extracted_data.get("metadata", {})
        context_parts.append(f"Chart Title: {metadata.get('title', 'Chart')}")
        context_parts.append(f"Chart Types: {', '.join(metadata.get('chart_types', []))}")
        context_parts.append(f"Number of Data Series: {metadata.get('num_traces', 0)}")
        if business_context:
            context_parts.append(f"\nBusiness Context: {business_context}")
        if focus_areas:
            context_parts.append(f"Focus Areas: {', '.join(focus_areas)}")
        context_parts.append("\n=== DATA ANALYSIS ===")
        for trace in extracted_data.get("traces", []):
            context_parts.append(f"\nSeries: {trace['name']} (Type: {trace['type']})")
            stats = trace.get("statistics") or {}
            if stats:
                context_parts.append("Statistics:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        context_parts.append(f"  - {key}: {value:.2f}")
                    else:
                        context_parts.append(f"  - {key}: {value}")
            categories = trace.get("categories") or []
            if categories:
                top_cats = categories[:5]
                context_parts.append(f"  - Top Categories: {', '.join(map(str, top_cats))}")
            specials = trace.get("special_values") or {}
            if specials:
                context_parts.append("Special Values:")
                for key, value in specials.items():
                    if isinstance(value, dict):
                        context_parts.append(f"  - {key}: {json.dumps(value, indent=2)}")
                    else:
                        context_parts.append(f"  - {key}: {value}")
        ann = extracted_data.get("annotations") or []
        if ann:
            context_parts.append("\nChart Annotations:")
            for a in ann[:3]:
                t = a.get("text") if isinstance(a, dict) else None
                if t:
                    context_parts.append(f"  - {t}")
        return "\n".join(context_parts)

    def _get_system_prompt(self, tone: str) -> str:
        tone_prompts = {
            "professional": (
                "You are a senior data analyst specializing in HSE (Health, Safety, Environment) analytics. "
                "Provide clear, actionable insights from data visualizations. Use professional language but avoid "
                "excessive jargon. Focus on business impact and practical recommendations."
            ),
            "executive": (
                "You are an executive advisor providing high-level strategic insights. Focus on business impact, risks, "
                "and opportunities. Keep language concise and decision-focused. Highlight only the most critical findings."
            ),
            "technical": (
                "You are a data scientist providing detailed analytical insights. Include statistical significance, data "
                "quality observations, and methodological considerations. Use precise technical language where appropriate."
            ),
            "simple": (
                "You are explaining data insights to someone without technical background. Use simple, clear language. Avoid "
                "jargon and statistics. Focus on what the data means in practical terms."
            ),
        }
        return tone_prompts.get(tone, tone_prompts["professional"])

    def _build_prompt(self, insight_types: List[InsightType], tone: str) -> str:
        sections: List[str] = []
        if InsightType.EXECUTIVE_SUMMARY in insight_types:
            sections.append(
                """
## Executive Summary
Provide a 2-3 sentence high-level summary of the most important findings from this data.
Focus on what matters most for decision-making.
"""
            )
        if InsightType.STATISTICAL in insight_types:
            sections.append(
                """
## Statistical Analysis
- Identify significant statistical patterns
- Note any outliers or anomalies
- Comment on data distribution and variability
- Highlight correlations if multiple series present
"""
            )
        if InsightType.TRENDS in insight_types:
            sections.append(
                """
## Key Trends & Patterns
- Identify upward or downward trends
- Note any cyclical patterns or seasonality
- Highlight significant changes or inflection points
- Compare trends across different series if applicable
"""
            )
        if InsightType.ANOMALIES in insight_types:
            sections.append(
                """
## Anomalies & Outliers
- Identify any unusual data points or patterns
- Note any data quality issues
- Highlight unexpected findings
- Flag areas requiring investigation
"""
            )
        if InsightType.RECOMMENDATIONS in insight_types:
            sections.append(
                """
## Recommendations
Provide 3-5 actionable recommendations based on the data:
- What actions should be taken?
- What areas need attention?
- What should be monitored going forward?
- What additional analysis might be helpful?
"""
            )
        if InsightType.PREDICTIVE in insight_types:
            sections.append(
                """
## Forward-Looking Insights
- Based on current trends, what might happen next?
- What scenarios should be planned for?
- What early warning signs should be watched?
"""
            )
        prompt = f"""Analyze the provided chart data and generate insights in the following format:

{''.join(sections)}

Guidelines:
- Be specific and reference actual numbers from the data
- Focus on actionable insights
- Maintain a {tone} tone
- Use markdown formatting for clarity
- Avoid generic statements; be specific to this data
- If data is insufficient for any section, briefly note that

Generate the insights now:"""
        return prompt

    def _generate_no_data_response(self, title: str) -> str:
        return f"""## {title}

### Data Status
No meaningful numerical data was found in this chart for analysis.

### Possible Issues
- The chart may contain only categorical or text data
- Numerical values might be formatted as strings
- The chart structure might not be standard Plotly format

### Recommendations
1. Verify that numerical data is properly formatted
2. Ensure the chart contains quantitative metrics
3. Check that the Plotly figure structure is complete
4. Consider providing raw data directly for analysis"""

    def _generate_fallback_insights(self, extracted_data: Dict, error: str) -> str:
        insights: List[str] = [f"## {extracted_data['metadata'].get('title', 'Chart')}"]
        for trace in extracted_data.get("traces", []):
            stats = trace.get("statistics") or {}
            specials = trace.get("special_values") or {}
            name = trace.get("name", "series")
            if stats:
                insights.append(f"\n### {name} — Summary")
                if "mean" in stats:
                    insights.append(f"- **Average**: {stats['mean']:.2f}")
                if "min" in stats and "max" in stats:
                    insights.append(f"- **Range**: {stats['min']:.2f} to {stats['max']:.2f}")
                if "count" in stats:
                    insights.append(f"- **Data Points**: {stats['count']}")
            # Heatmap-specific bullets if available
            dept_avg = specials.get("department_averages")
            if dept_avg:
                top = dept_avg[0]
                bot = dept_avg[-1]
                insights.append(f"- **Highest average risk**: {top['department']} ({top['avg']:.2f})")
                insights.append(f"- **Lowest average risk**: {bot['department']} ({bot['avg']:.2f})")
            inc = specials.get("top_increases")
            if inc:
                a = inc[0]
                insights.append(
                    f"- **Largest increase**: {a['department']} from {a['start']:.2f} to {a['end']:.2f} between {a['start_label']} and {a['end_label']}"
                )
            dec = specials.get("top_decreases")
            if dec:
                d0 = dec[0]
                insights.append(
                    f"- **Largest decrease**: {d0['department']} from {d0['start']:.2f} to {d0['end']:.2f} between {d0['start_label']} and {d0['end_label']}"
                )
            stable = specials.get("most_stable")
            if stable:
                s0 = stable[0]
                insights.append(f"- **Most stable**: {s0['department']} (std {s0['std']:.2f})")
            hotspot = specials.get("hotspot")
            if hotspot:
                xl = hotspot.get("x_label")
                yl = hotspot.get("y_label")
                v = hotspot.get("value")
                if xl is not None and yl is not None and v is not None:
                    insights.append(f"- **Hotspot**: {yl} at {xl} ({v:.2f})")

        insights.append("\n### Note")
        insights.append(f"*Advanced AI insights unavailable: {error}*")
        insights.append("*Showing rule-based insights instead.*")
        return "\n".join(insights)

    def generate_insights(
        self,
        fig: Dict,
        insight_types: Optional[List[InsightType]] = None,
        business_context: Optional[str] = None,
        tone: str = "professional",
        max_tokens: int = 1500,
    ) -> str:
        if insight_types is None:
            insight_types = [InsightType.EXECUTIVE_SUMMARY, InsightType.TRENDS, InsightType.RECOMMENDATIONS]
        extracted_data = self.extractor.extract_all_data(fig)
        if not extracted_data.get("traces"):
            return self._generate_no_data_response(extracted_data.get("metadata", {}).get("title", "Chart"))
        context = self.prepare_context_for_ai(extracted_data, business_context)
        prompt = self._build_prompt(insight_types, tone)
        try:
            if self.client is None:
                raise RuntimeError("OpenAI client not available")
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self._get_system_prompt(tone)},
                    {"role": "user", "content": f"{prompt}\n\n===== CHART DATA =====\n{context}"},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content  # type: ignore[attr-defined]
        except Exception as e:
            return self._generate_fallback_insights(extracted_data, str(e))
