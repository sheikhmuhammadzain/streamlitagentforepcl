"""Tool-Based AI Agent with Function Calling
Uses Grok to decide which data analysis tools to use

PERFORMANCE OPTIMIZATIONS:
- Connection pooling for OpenAI client (reuse)
- Data caching with TTL (5min workbook, 1min queries)
- Parallel tool execution when independent
- Reduced streaming overhead
- Fast-path for simple queries
"""

from typing import AsyncGenerator, Dict, Any, List, Optional
import pandas as pd
import json
import os
import hashlib
import asyncio
import re
from openai import AsyncOpenAI

from .agent import load_default_sheets
from .data_cache import (
    get_cached_workbook,
    cache_workbook,
    get_cached_query,
    cache_query
)


# ==================== Data Analysis Tools ====================

def _safe_json_serialize(obj):
    """Convert pandas/numpy objects to JSON-serializable values (for dict values)."""
    # Handle Timestamp
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    # Handle Period (monthly, quarterly, etc.)
    if isinstance(obj, pd.Period):
        return str(obj)
    # Handle Timedelta
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    # Handle pandas Series
    if isinstance(obj, pd.Series):
        return obj.tolist()
    # Handle lists
    if isinstance(obj, list):
        return [_safe_json_serialize(item) for item in obj]
    # Handle dict values (not keys)
    if isinstance(obj, dict):
        return {k: _safe_json_serialize(v) for k, v in obj.items()}
    # Handle NaNs
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    # Handle numpy scalar types via item()
    try:
        # numpy scalars have .item()
        return obj.item()  # type: ignore[attr-defined]
    except Exception:
        return obj


def _stringify_key(k):
    """Ensure dict keys are strings (timestamps to ISO)."""
    if isinstance(k, pd.Timestamp):
        return k.isoformat()
    return str(k)


def _to_jsonable(obj):
    """Recursively convert an object into a JSON-serializable structure.
    - Dict keys become strings (Timestamp -> ISO)
    - Values are converted via _safe_json_serialize
    """
    if isinstance(obj, dict):
        return { _stringify_key(k): _to_jsonable(v) for k, v in obj.items() }
    if isinstance(obj, list):
        return [ _to_jsonable(v) for v in obj ]
    return _safe_json_serialize(obj)


def _normalize_name(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _resolve_sheet_name(name: str, dfs_keys) -> str | None:
    """Resolve a requested sheet name to the closest available key in dfs_keys (all lowercase)."""
    if not name:
        return None
    want = name.lower()
    if want in dfs_keys:
        return want
    norm_want = _normalize_name(want)
    # Exact normalized match
    for k in dfs_keys:
        if _normalize_name(k) == norm_want:
            return k
    # Substring match (e.g., 'hazard' -> 'hazard id')
    for k in dfs_keys:
        if norm_want and norm_want in _normalize_name(k):
            return k
    # Token overlap heuristic
    tokens = set(want.split())
    for k in dfs_keys:
        if any(t in k for t in tokens):
            return k
    return None


def get_data_summary(sheet_name: str = "incident") -> str:
    """
    Get summary statistics for a dataset
    
    Args:
        sheet_name: Name of the sheet (incident, hazard, audit, inspection)
    
    Returns:
        JSON string with summary statistics
    """
    try:
        # OPTIMIZATION: Use cached workbook
        workbook = get_cached_workbook()
        if workbook is None:
            workbook = load_default_sheets()
            if workbook:
                cache_workbook(workbook)
        
        dfs = {str(k).lower(): v for k, v in (workbook or {}).items()}
        
        resolved = _resolve_sheet_name(sheet_name, dfs.keys())
        if not resolved:
            return json.dumps({"error": f"Sheet '{sheet_name}' not found. Available: {list(dfs.keys())}"})
        
        df = dfs[resolved]
        
        # Convert sample data safely
        sample_data = df.head(3).to_dict('records')
        sample_data = _safe_json_serialize(sample_data)
        
        summary = {
            "sheet": resolved,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": list(df.columns)[:20],  # Limit to first 20 columns
            "dtypes": {col: str(dtype) for col, dtype in list(df.dtypes.items())[:20]},
            "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items() if v > 0},
            "sample_data": sample_data
        }
        
        return json.dumps(_to_jsonable(summary), indent=2)
    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


def query_data(sheet_name: str, query_description: str) -> str:
    """
    Execute a data query based on natural language description
    
    Args:
        sheet_name: Name of the sheet to query
        query_description: Natural language description of what to find
    
    Returns:
        JSON string with query results
    """
    try:
        # OPTIMIZATION: Check query cache first
        cache_key = f"query_{sheet_name}_{hashlib.md5(query_description.encode()).hexdigest()}"
        cached_result = get_cached_query(cache_key)
        if cached_result:
            return cached_result
        
        # OPTIMIZATION: Use cached workbook
        workbook = get_cached_workbook()
        if workbook is None:
            workbook = load_default_sheets()
            if workbook:
                cache_workbook(workbook)
        
        dfs = {str(k).lower(): v for k, v in (workbook or {}).items()}
        
        resolved = _resolve_sheet_name(sheet_name, dfs.keys())
        if not resolved:
            return json.dumps({"error": f"Sheet '{sheet_name}' not found. Available: {list(dfs.keys())}"})
        
        df = dfs[resolved]
        
        # Simple query patterns
        query_lower = query_description.lower()
        
        if "top" in query_lower and "department" in query_lower:
            # Extract number
            import re
            match = re.search(r'top\s+(\d+)', query_lower)
            n = int(match.group(1)) if match else 10
            
            # Find department column
            dept_col = None
            for col in df.columns:
                if 'dept' in col.lower() or 'department' in col.lower():
                    dept_col = col
                    break
            
            if dept_col:
                result = df[dept_col].value_counts().head(n).to_dict()
                payload = {
                    "query": query_description,
                    "results": result,
                    "total_rows": len(df)
                }
                return json.dumps(_to_jsonable(payload), indent=2)
        
        elif "count" in query_lower or "total" in query_lower:
            payload = {
                "query": query_description,
                "total_count": len(df),
                "by_columns": {col: df[col].nunique() for col in df.columns[:5]}
            }
            return json.dumps(_to_jsonable(payload), indent=2)
        
        # Default: return sample
        payload = {
            "query": query_description,
            "sample_results": df.head(10).to_dict('records'),
            "total_rows": len(df)
        }
        result = json.dumps(_to_jsonable(payload), indent=2)
        
        # Cache the result
        cache_query(cache_key, result)
        return result
        
    except Exception as e:
        return json.dumps({"error": str(e)})


def aggregate_data(sheet_name: str, group_by: str, aggregate_column: str = "", operation: str = "count") -> str:
    """
    Aggregate data by grouping
    
    Args:
        sheet_name: Name of the sheet
        group_by: Column to group by
        aggregate_column: Column to aggregate (optional for count)
        operation: count, sum, mean, max, min
    
    Returns:
        JSON string with aggregated results
    """
    try:
        # OPTIMIZATION: Check cache
        cache_key = f"agg_{sheet_name}_{group_by}_{aggregate_column}_{operation}"
        cached_result = get_cached_query(cache_key)
        if cached_result:
            return cached_result
        
        # OPTIMIZATION: Use cached workbook
        workbook = get_cached_workbook()
        if workbook is None:
            workbook = load_default_sheets()
            if workbook:
                cache_workbook(workbook)
        
        dfs = {str(k).lower(): v for k, v in (workbook or {}).items()}
        
        if sheet_name.lower() not in dfs:
            return json.dumps({"error": f"Sheet '{sheet_name}' not found"})
        
        df = dfs[sheet_name.lower()]
        
        if group_by not in df.columns:
            return json.dumps({"error": f"Column '{group_by}' not found. Available: {list(df.columns)}"})
        
        if operation == "count":
            result = df.groupby(group_by).size().sort_values(ascending=False).head(20).to_dict()
        elif operation == "sum" and aggregate_column in df.columns:
            result = df.groupby(group_by)[aggregate_column].sum().sort_values(ascending=False).head(20).to_dict()
        elif operation == "mean" and aggregate_column in df.columns:
            result = df.groupby(group_by)[aggregate_column].mean().sort_values(ascending=False).head(20).to_dict()
        else:
            result = df.groupby(group_by).size().sort_values(ascending=False).head(20).to_dict()
        
        # Ensure keys are strings (e.g., Timestamp -> ISO)
        result_str_keys = { _stringify_key(k): _to_jsonable(v) for k, v in result.items() }
        
        # Convert to tabular format for better display
        table_data = [
            {group_by: k, aggregate_column or "count": v}
            for k, v in result_str_keys.items()
        ]
        
        payload = {
            "operation": operation,
            "group_by": group_by,
            "aggregate_column": aggregate_column,
            "results": result_str_keys,
            "table": table_data  # Add tabular format
        }
        result = json.dumps(_to_jsonable(payload), indent=2)
        
        # Cache the result
        cache_query(cache_key, result)
        return result
        
    except Exception as e:
        return json.dumps({"error": str(e)})


def compare_sheets(sheet1: str, sheet2: str, comparison_type: str = "count") -> str:
    """
    Compare data between two sheets
    
    Args:
        sheet1: First sheet name
        sheet2: Second sheet name
        comparison_type: Type of comparison (count, columns, overlap)
    
    Returns:
        JSON string with comparison results
    """
    try:
        # OPTIMIZATION: Use cached workbook
        workbook = get_cached_workbook()
        if workbook is None:
            workbook = load_default_sheets()
            if workbook:
                cache_workbook(workbook)
        
        dfs = {str(k).lower(): v for k, v in (workbook or {}).items()}
        
        r1 = _resolve_sheet_name(sheet1, dfs.keys())
        r2 = _resolve_sheet_name(sheet2, dfs.keys())
        if not r1 or not r2:
            return json.dumps({"error": f"One or both sheets not found. Available: {list(dfs.keys())}"})
        
        df1 = dfs[r1]
        df2 = dfs[r2]
        
        comparison = {
            "sheet1": r1,
            "sheet2": r2,
            "sheet1_rows": len(df1),
            "sheet2_rows": len(df2),
            "sheet1_columns": list(df1.columns),
            "sheet2_columns": list(df2.columns),
            "common_columns": list(set(df1.columns) & set(df2.columns))
        }
        
        return json.dumps(comparison, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


def create_chart(
    sheet_name: str, 
    chart_type: str, 
    x_column: str, 
    y_column: str = "", 
    title: str = "",
    filter_column: str = "",
    filter_value: str = ""
) -> str:
    """
    Create a chart/visualization from data with optional filtering
    
    Args:
        sheet_name: Name of the sheet
        chart_type: Type of chart (bar, line, pie, scatter)
        x_column: Column for x-axis (or labels for pie)
        y_column: Column for y-axis (optional for pie/bar with value_counts)
        title: Chart title
        filter_column: Column to filter by (optional, e.g., 'occurrence_date')
        filter_value: Value or condition to filter (e.g., '2023-06' for June 2023, supports partial matching)
    
    Returns:
        JSON string with Plotly chart specification
    """
    try:
        # OPTIMIZATION: Use cached workbook
        workbook = get_cached_workbook()
        if workbook is None:
            workbook = load_default_sheets()
            if workbook:
                cache_workbook(workbook)
        
        dfs = {str(k).lower(): v for k, v in (workbook or {}).items()}
        
        resolved = _resolve_sheet_name(sheet_name, dfs.keys())
        if not resolved:
            return json.dumps({"error": f"Sheet '{sheet_name}' not found. Available: {list(dfs.keys())}"})
        
        df = dfs[resolved]
        original_count = len(df)
        
        # ENHANCEMENT: Apply filter if provided
        if filter_column and filter_value:
            if filter_column not in df.columns:
                return json.dumps({"error": f"Filter column '{filter_column}' not found. Available: {list(df.columns)}"})
            
            # Convert filter column to string for partial matching
            df_filtered = df[df[filter_column].astype(str).str.contains(filter_value, case=False, na=False)]
            
            if len(df_filtered) == 0:
                return json.dumps({
                    "error": f"No data found matching filter: {filter_column} contains '{filter_value}'",
                    "suggestion": f"Try a different filter value or check the data in column '{filter_column}'"
                })
            
            df = df_filtered
            filtered_count = len(df)
        else:
            filtered_count = original_count
        
        # Validate columns
        if x_column not in df.columns:
            return json.dumps({"error": f"Column '{x_column}' not found. Available: {list(df.columns)}"})
        
        if y_column and y_column not in df.columns:
            return json.dumps({"error": f"Column '{y_column}' not found. Available: {list(df.columns)}"})
        
        chart_data = {
            "chart_type": chart_type,
            "title": title or f"{chart_type.title()} Chart",
            "x_label": x_column,
            "y_label": y_column or "Count",
            "total_records": filtered_count,
            "filtered": bool(filter_column and filter_value),
            "filter_info": f"{filter_column} contains '{filter_value}'" if filter_column and filter_value else None
        }
        
        # Generate chart data based on type
        if chart_type == "pie":
            # For pie charts, count values in x_column
            counts = df[x_column].value_counts().head(10)
            chart_data["labels"] = [_stringify_key(k) for k in counts.index.tolist()]
            chart_data["values"] = [int(v) if pd.notna(v) else 0 for v in counts.values.tolist()]
            
        elif chart_type == "bar":
            if y_column:
                # Group by x_column and sum/mean y_column
                grouped = df.groupby(x_column)[y_column].sum().sort_values(ascending=False).head(20)
                chart_data["x_data"] = [_stringify_key(k) for k in grouped.index.tolist()]
                chart_data["y_data"] = [float(v) if pd.notna(v) else 0 for v in grouped.values.tolist()]
            else:
                # Count values in x_column
                counts = df[x_column].value_counts().head(20)
                chart_data["x_data"] = [_stringify_key(k) for k in counts.index.tolist()]
                chart_data["y_data"] = [int(v) for v in counts.values.tolist()]
                
        elif chart_type == "line":
            if y_column:
                # Group by x_column (usually time-based)
                grouped = df.groupby(x_column)[y_column].sum().sort_index()
                chart_data["x_data"] = [_stringify_key(k) for k in grouped.index.tolist()]
                chart_data["y_data"] = [float(v) if pd.notna(v) else 0 for v in grouped.values.tolist()]
            else:
                # Count over time
                counts = df[x_column].value_counts().sort_index()
                chart_data["x_data"] = [_stringify_key(k) for k in counts.index.tolist()]
                chart_data["y_data"] = [int(v) for v in counts.values.tolist()]
                
        elif chart_type == "scatter":
            if not y_column:
                return json.dumps({"error": "Scatter plot requires both x_column and y_column"})
            # Sample data for scatter (limit to 1000 points)
            sample_df = df[[x_column, y_column]].dropna().head(1000)
            chart_data["x_data"] = [_stringify_key(v) for v in sample_df[x_column].tolist()]
            chart_data["y_data"] = [float(v) if pd.notna(v) else 0 for v in sample_df[y_column].tolist()]
        
        else:
            return json.dumps({"error": f"Unsupported chart type: {chart_type}. Use: bar, line, pie, scatter"})
        
        return json.dumps(_to_jsonable(chart_data), indent=2)
        
    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


# ==================== Tool Definitions for Grok ====================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_data_summary",
            "description": "Get summary statistics and schema information for a dataset. Use this to understand the data structure before querying.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet_name": {
                        "type": "string",
                        "description": "Name of the dataset to summarize"
                    }
                },
                "required": ["sheet_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": "Query data based on natural language description. Use this to find specific records or patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet_name": {
                        "type": "string",
                        "description": "Name of the dataset to query"
                    },
                    "query_description": {
                        "type": "string",
                        "description": "Natural language description of what to find (e.g., 'top 10 departments', 'total count')"
                    }
                },
                "required": ["sheet_name", "query_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_data",
            "description": "Aggregate data by grouping and applying operations. Use 'count' to count records per group, or sum/mean/max/min with aggregate_column for numeric operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "group_by": {
                        "type": "string",
                        "description": "Column name to group by (e.g., 'department', 'location', 'severity'). Common columns: department, location, audit_location, hazard_location"
                    },
                    "aggregate_column": {
                        "type": "string",
                        "description": "Column to aggregate (optional, not needed for count operation)"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["count", "sum", "mean", "max", "min"],
                        "description": "Aggregation operation. Use 'count' to count records, others need aggregate_column",
                        "default": "count"
                    }
                },
                "required": ["sheet_name", "group_by"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_sheets",
            "description": "Compare data between two datasets. Use this for cross-dataset analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet1": {
                        "type": "string",
                        "description": "First dataset name"
                    },
                    "sheet2": {
                        "type": "string",
                        "description": "Second dataset name"
                    },
                    "comparison_type": {
                        "type": "string",
                        "enum": ["count", "columns", "overlap"],
                        "description": "Type of comparison"
                    }
                },
                "required": ["sheet1", "sheet2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": "Create a visualization/chart from data with optional filtering. Supports bar, line, pie, and scatter plots. Can filter data by date ranges or other criteria. Perfect for creating pie charts of specific time periods (e.g., June 2023 hazards).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "pie", "scatter"],
                        "description": "Type of chart to create. Use 'pie' for distribution charts."
                    },
                    "x_column": {
                        "type": "string",
                        "description": "Column for x-axis or labels. For pie charts, this is the category column (e.g., 'hazard_title', 'department'). Common columns: department, location, occurrence_date, severity, hazard_title"
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Column for y-axis (optional for pie/bar charts when just counting). Use for numeric values to plot."
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title (optional)"
                    },
                    "filter_column": {
                        "type": "string",
                        "description": "Column to filter by (optional). Use 'occurrence_date' for date filtering, or any other column name. Example: 'occurrence_date', 'department', 'severity'"
                    },
                    "filter_value": {
                        "type": "string",
                        "description": "Value or pattern to filter (optional). Supports partial matching. Examples: '2023-06' for June 2023, '2023' for all of 2023, 'Operations' for Operations department. Case-insensitive."
                    }
                },
                "required": ["sheet_name", "chart_type", "x_column"]
            }
        }
    }
]


# Tool execution mapping
TOOL_FUNCTIONS = {
    "get_data_summary": get_data_summary,
    "query_data": query_data,
    "aggregate_data": aggregate_data,
    "compare_sheets": compare_sheets,
    "create_chart": create_chart
}

# ==================== Response Formatting ====================

def enhance_response_formatting(response: str) -> str:
    """
    Enhance response formatting to ensure it's well-structured
    Adds missing sections and improves readability
    """
    
    # Check if response already has proper formatting
    has_sections = any(marker in response for marker in ["## 📊", "## 💡", "## 📈", "## 📋"])
    
    if has_sections:
        # Already formatted, just ensure consistency
        return response
    
    # If not formatted, add basic structure
    lines = response.split('\n')
    
    # Try to identify different parts
    formatted = "## 📊 Analysis Results\n\n"
    formatted += response
    
    # Add a summary section if not present
    if "summary" not in response.lower() and "## 📋" not in response:
        formatted += "\n\n## 📋 Summary\n"
        formatted += "Analysis completed successfully. See findings above for detailed insights."
    
    return formatted


# ==================== Connection Pool ====================

# OPTIMIZATION: Reuse OpenAI client (connection pooling)
_client_cache: Optional[AsyncOpenAI] = None

def get_openai_client() -> AsyncOpenAI:
    """Get or create cached OpenAI client for connection reuse"""
    global _client_cache
    
    if _client_cache is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        
        _client_cache = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=30.0,  # 30s timeout
            max_retries=2   # Retry failed requests
        )
    
    return _client_cache


# ==================== Tool-Based Agent ====================

async def run_tool_based_agent(
    query: str,
    model: str = "z-ai/glm-4.6",  # Free model with function calling
    max_iterations: int = 100
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run tool-based agent where AI decides which tools to use
    
    OPTIMIZATIONS:
    - Connection pooling (reuse HTTP connections)
    - Data caching (5min TTL for workbook)
    - Reduced streaming overhead
    - Fast model selection
    
    Flow:
    1. User asks question
    2. AI analyzes and decides which tools to call
    3. Tools execute and return results (streamed)
    4. AI synthesizes final answer (streamed)
    
    Free models that support function calling (in order of speed):
    - google/gemini-flash-1.5:free (FASTEST - recommended)
    - qwen/qwen-2.5-7b-instruct:free
    - mistralai/mistral-7b-instruct:free
    - meta-llama/llama-3.1-8b-instruct:free
    
    Note: Rate limits apply to all free models during high demand
    """
    
    # Send start signal
    yield {
        "type": "start",
        "message": "🚀 Starting..."
    }
    
    # OPTIMIZATION: Reuse client connection
    try:
        client = get_openai_client()
    except ValueError as e:
        yield {"type": "error", "message": str(e)}
        return
    
    # OPTIMIZATION: Use cached workbook for available sheets
    try:
        workbook_ctx = get_cached_workbook()
        if workbook_ctx is None:
            workbook_ctx = load_default_sheets()
            if workbook_ctx:
                cache_workbook(workbook_ctx)
        available_sheets = list({str(k).lower() for k in (workbook_ctx or {}).keys()})
    except Exception:
        available_sheets = ["incident", "hazard", "audit", "inspection"]
    available_keys = set(available_sheets)

    # System prompt with runtime sheet list
    system_prompt = f"""You are an expert data analyst with access to safety management data.

Available datasets (use EXACT names from this list in tool calls):
{available_sheets}

Instructions:
- Always pick sheet names from the list above. If the user mentions an alias (e.g., 'hazard'), map it to the closest available name (e.g., 'hazard id') and use that exact name in tool calls.
- Before deep analysis, use get_data_summary to confirm structure if unsure.
- Use tools to query/aggregate data; then synthesize concise insights with exact numbers.

RESPONSE FORMATTING REQUIREMENTS:
When providing your final answer, format it using Markdown with the following structure:

## 📊 Key Findings
- List the most important data points with **exact numbers**
- Highlight trends and patterns discovered
- Use bullet points for clarity

## 💡 Insights
- Explain what the data means
- Identify root causes or contributing factors
- Connect findings to business impact

## 📈 Recommendations
- Provide actionable next steps (prioritized)
- Suggest areas requiring attention
- Include specific, data-driven suggestions

## 📋 Summary
- Brief overview of the analysis
- Key metrics in a concise format
- Use tables or lists for structured data

FORMATTING RULES:
✅ Use **bold** for important numbers and metrics
✅ Use bullet points (•) for lists
✅ Use emojis for visual clarity (📊 📈 💡 ⚠️ ✅)
✅ Include exact numbers from tool results
✅ Format tables using Markdown table syntax when showing multiple data points
✅ Keep paragraphs short and scannable
✅ Use headings (##) to organize sections
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # OPTIMIZATION: Only show thinking on first iteration
        if iteration == 1:
            yield {
                "type": "thinking",
                "message": "🤔 Analyzing..."
            }
        
        try:
            # OPTIMIZATION: Call AI with tools (streaming enabled)
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,  # Low temp for consistency
                max_tokens=12000,
                stream=True,  # Enable streaming
                extra_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Safety Copilot"
                }
            )
            
            # Collect streaming response
            assistant_message = None
            content_buffer = ""
            tool_calls_buffer = {}
            token_count = 0
            
            async for chunk in stream:
                delta = chunk.choices[0].delta
                
                # OPTIMIZATION: Batch content tokens (every 10 tokens)
                if delta.content:
                    content_buffer += delta.content
                    token_count += 1
                    
                    # Stream in batches of 10 tokens for efficiency
                    if token_count >= 10:
                        yield {
                            "type": "thinking_token",
                            "token": content_buffer[-len(delta.content) * 10:]
                        }
                        token_count = 0
                
                # Collect tool calls (don't stream partial args - reduces overhead)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }
                        
                        if tc.function.name:
                            tool_calls_buffer[idx]["function"]["name"] = tc.function.name
                        
                        if tc.function.arguments:
                            tool_calls_buffer[idx]["function"]["arguments"] += tc.function.arguments
                
                # Get finish reason
                if chunk.choices[0].finish_reason:
                    # Build proper message dict for API
                    assistant_message_dict = {
                        "role": "assistant",
                        "content": content_buffer or None
                    }
                    
                    # Add tool calls if present
                    if tool_calls_buffer:
                        assistant_message_dict["tool_calls"] = [
                            {
                                "id": tc["id"],
                                "type": tc["type"],
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"]
                                }
                            } for tc in tool_calls_buffer.values()
                        ]
                        assistant_message = type('obj', (), assistant_message_dict)()
                        assistant_message.tool_calls = [
                            type('obj', (), {
                                'id': tc["id"],
                                'function': type('obj', (), {
                                    'name': tc["function"]["name"],
                                    'arguments': tc["function"]["arguments"]
                                })()
                            })() for tc in tool_calls_buffer.values()
                        ]
                    else:
                        assistant_message = type('obj', (), {'content': content_buffer, 'tool_calls': None})()
            
            # Check if AI wants to use tools
            if assistant_message and hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                # AI decided to use tools - append dict to messages
                messages.append(assistant_message_dict)
                
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # Pre-resolve sheet names to closest available dataset(s)
                    corrected_pre_args = dict(function_args)
                    pre_changed = False

                    def _pre_fix(name: str | None) -> str | None:
                        if not name:
                            return None
                        resolved = _resolve_sheet_name(str(name), available_keys)
                        return resolved or name

                    if "sheet_name" in corrected_pre_args:
                        new_name = _pre_fix(corrected_pre_args.get("sheet_name"))
                        if new_name and new_name != corrected_pre_args["sheet_name"]:
                            corrected_pre_args["sheet_name"] = new_name
                            pre_changed = True

                    if "sheet1" in corrected_pre_args or "sheet2" in corrected_pre_args:
                        n1 = _pre_fix(corrected_pre_args.get("sheet1")) if corrected_pre_args.get("sheet1") else None
                        n2 = _pre_fix(corrected_pre_args.get("sheet2")) if corrected_pre_args.get("sheet2") else None
                        if n1 and corrected_pre_args.get("sheet1") != n1:
                            corrected_pre_args["sheet1"] = n1
                            pre_changed = True
                        if n2 and corrected_pre_args.get("sheet2") != n2:
                            corrected_pre_args["sheet2"] = n2
                            pre_changed = True

                    if pre_changed:
                        function_args = corrected_pre_args
                        yield {
                            "type": "tool_call",
                            "tool": function_name,
                            "arguments": function_args,
                            "message": "🧭 Resolved sheet name(s) to available dataset(s)"
                        }
                    
                    yield {
                        "type": "tool_call",
                        "tool": function_name,
                        "arguments": function_args,
                        "message": f"🔧 Using tool: {function_name}"
                    }
                    
                    # Execute tool
                    if function_name in TOOL_FUNCTIONS:
                        result = TOOL_FUNCTIONS[function_name](**function_args)
                        
                        yield {
                            "type": "tool_result",
                            "tool": function_name,
                            "result": result
                        }
                        
                        # Auto-correct sheet names on 'not found' errors and retry once
                        try:
                            parsed = json.loads(result)
                        except Exception:
                            parsed = None
                        if isinstance(parsed, dict) and isinstance(parsed.get("error"), str) and "not found" in parsed["error"].lower():
                            # Determine available sheets at runtime
                            try:
                                workbook_ctx = load_default_sheets()
                                dfs_keys = {str(k).lower() for k in (workbook_ctx or {}).keys()}
                            except Exception:
                                dfs_keys = set()

                            corrected_args = dict(function_args)
                            changed = False

                            def _fix(name: str | None) -> str | None:
                                if not name:
                                    return None
                                resolved = _resolve_sheet_name(name, dfs_keys)
                                return resolved or name

                            # Fix for single-sheet tools
                            if "sheet_name" in corrected_args:
                                new_name = _fix(str(corrected_args["sheet_name"]))
                                if new_name and new_name != corrected_args["sheet_name"]:
                                    corrected_args["sheet_name"] = new_name
                                    changed = True

                            # Fix for compare_sheets style args
                            if "sheet1" in corrected_args or "sheet2" in corrected_args:
                                n1 = _fix(str(corrected_args.get("sheet1"))) if corrected_args.get("sheet1") else None
                                n2 = _fix(str(corrected_args.get("sheet2"))) if corrected_args.get("sheet2") else None
                                if n1 and corrected_args.get("sheet1") != n1:
                                    corrected_args["sheet1"] = n1
                                    changed = True
                                if n2 and corrected_args.get("sheet2") != n2:
                                    corrected_args["sheet2"] = n2
                                    changed = True

                            if changed:
                                yield {
                                    "type": "tool_call",
                                    "tool": function_name,
                                    "arguments": corrected_args,
                                    "message": "🔁 Auto-corrected sheet name(s) and retried"
                                }
                                result2 = TOOL_FUNCTIONS[function_name](**corrected_args)
                                yield {
                                    "type": "tool_result",
                                    "tool": function_name,
                                    "result": result2
                                }
                                # Feed corrected result to the model
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": function_name,
                                    "content": result2
                                })
                                continue

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": result
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps({"error": "Tool not found"})
                        })
                
                # Continue loop to get final answer
                continue
            
            else:
                # AI provided final answer (already streamed via thinking_token)
                final_answer = content_buffer
                
                # ENHANCEMENT: Ensure response is well-formatted
                if final_answer:
                    formatted_answer = enhance_response_formatting(final_answer)
                    
                    yield {
                        "type": "answer_complete",
                        "content": formatted_answer
                    }
                
                yield {
                    "type": "complete",
                    "data": {
                        "answer": formatted_answer if final_answer else "",
                        "iterations": iteration,
                        "tools_used": len([msg for msg in messages if msg.get("role") == "tool"])
                    }
                }
                
                break
        
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Error: {str(e)}",
                "iteration": iteration
            }
            break
    
    if iteration >= max_iterations:
        yield {
            "type": "error",
            "message": "Max iterations reached without final answer"
        }
