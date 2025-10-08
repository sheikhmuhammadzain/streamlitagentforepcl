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
import time
import re
import httpx
import sqlite3
from openai import AsyncOpenAI

from .agent import load_default_sheets
from .data_cache import (
    get_cached_workbook,
    cache_workbook,
    get_cached_query,
    cache_query
)


# ==================== Web Search Tool ====================

SERPER_API_KEY = "1a7343c2485b3e95dde021b5bb0b24296f6ce659"

async def search_web(query: str, num_results: int = 5) -> str:
    """
    Search the web for safety standards, best practices, and solutions
    
    Args:
        query: Search query (e.g., 'OSHA fall protection standards', 'workplace hazard prevention')
        num_results: Number of results to return (default 5, max 10)
    
    Returns:
        JSON string with search results including titles, links, and snippets
    """
    try:
        url = "https://google.serper.dev/search"
        
        payload = json.dumps({
            "q": query,
            "num": min(num_results, 10)
        })
        
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, content=payload)
            response.raise_for_status()
            data = response.json()
        
        # Extract organic results with images
        results = []
        for item in data.get('organic', [])[:num_results]:
            results.append({
                "title": item.get('title', ''),
                "link": item.get('link', ''),
                "snippet": item.get('snippet', ''),
                "position": item.get('position', 0),
                "thumbnail": item.get('thumbnail', None)  # Image thumbnail if available
            })
        
        # Extract knowledge graph if available (with image)
        knowledge_graph = None
        if 'knowledgeGraph' in data:
            kg = data['knowledgeGraph']
            knowledge_graph = {
                "title": kg.get('title', ''),
                "type": kg.get('type', ''),
                "description": kg.get('description', ''),
                "imageUrl": kg.get('imageUrl', None)  # Knowledge graph image
            }
        
        payload = {
            "query": query,
            "results_count": len(results),
            "results": results,
            "knowledge_graph": knowledge_graph,
            "search_metadata": {
                "total_results": data.get('searchInformation', {}).get('totalResults', 0)
            }
        }
        
        return json.dumps(payload, indent=2)
        
    except httpx.HTTPError as e:
        return json.dumps({"error": f"HTTP error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})


async def search_images(query: str, num_results: int = 10) -> str:
    """
    Search for images related to safety topics
    
    Args:
        query: Image search query (e.g., 'workplace hazard signs', 'PPE equipment', 'safety infographics')
        num_results: Number of images to return (default 10, max 10)
    
    Returns:
        JSON string with image URLs, thumbnails, dimensions, and sources
    """
    try:
        url = "https://google.serper.dev/images"
        
        payload = json.dumps({
            "q": query,
            "num": min(num_results, 10)
        })
        
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, content=payload)
            response.raise_for_status()
            data = response.json()
        
        # Extract image results
        images = []
        for item in data.get('images', [])[:num_results]:
            images.append({
                "title": item.get('title', ''),
                "imageUrl": item.get('imageUrl', ''),
                "thumbnailUrl": item.get('thumbnailUrl', ''),
                "source": item.get('source', ''),
                "link": item.get('link', ''),
                "width": item.get('imageWidth', 0),
                "height": item.get('imageHeight', 0)
            })
        
        payload = {
            "query": query,
            "images_count": len(images),
            "images": images,
            "search_metadata": {
                "total_results": data.get('searchInformation', {}).get('totalResults', 0)
            }
        }
        
        return json.dumps(payload, indent=2)
        
    except httpx.HTTPError as e:
        return json.dumps({"error": f"HTTP error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Image search failed: {str(e)}"})


# ==================== SQL Database Tools ====================

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "epcl_vehs.db")

def get_database_schema(database_path: str = None) -> str:
    """
    Get database schema (tables and columns)
    
    Args:
        database_path: Path to SQLite database (default: epcl_vehs.db)
    
    Returns:
        JSON string with schema information
    """
    try:
        db_path = database_path or DB_PATH
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            # Get columns for each table
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [
                {
                    "name": row[1],
                    "type": row[2],
                    "nullable": not row[3],
                    "primary_key": bool(row[5])
                }
                for row in cursor.fetchall()
            ]
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            # Get sample data (3 rows)
            cursor.execute(f"SELECT * FROM {table} LIMIT 3")
            sample_rows = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            sample_data = [
                {column_names[i]: row[i] for i in range(len(column_names))}
                for row in sample_rows
            ]
            
            schema[table] = {
                "columns": columns,
                "row_count": row_count,
                "sample_data": sample_data
            }
        
        conn.close()
        
        return json.dumps({
            "database": db_path,
            "table_count": len(tables),
            "tables": schema
        }, indent=2)
        
    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


def execute_sql_query(
    query: str,
    database_path: str = None,
    limit: int = 100
) -> str:
    """
    Execute SQL query on SQLite database
    
    Args:
        query: SQL query to execute (SELECT only for safety)
        database_path: Path to SQLite database (default: epcl_vehs.db)
        limit: Maximum rows to return (default 100)
    
    Returns:
        JSON string with query results
    """
    try:
        # Security: Only allow SELECT queries
        query_upper = query.strip().upper()
        if not query_upper.startswith('SELECT'):
            return json.dumps({
                "error": "Only SELECT queries are allowed for safety. Query must start with SELECT."
            })
        
        # Prevent dangerous operations
        forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'PRAGMA']
        for word in forbidden:
            if word in query_upper:
                return json.dumps({
                    "error": f"Forbidden operation '{word}' detected. Only SELECT queries are allowed."
                })
        
        # Add LIMIT if not present
        if 'LIMIT' not in query_upper:
            query = f"{query.rstrip(';')} LIMIT {limit}"
        
        # Execute query
        db_path = database_path or DB_PATH
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dicts
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Fetch results
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        
        # Get column names
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        
        conn.close()
        
        return json.dumps({
            "query": query,
            "rows_returned": len(results),
            "columns": columns,
            "results": results
        }, indent=2)
        
    except sqlite3.Error as e:
        return json.dumps({"error": f"SQL error: {str(e)}"})
    except Exception as e:
        import traceback
        return json.dumps({"error": f"Execution error: {str(e)}", "traceback": traceback.format_exc()})


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


def get_top_values(sheet_name: str, column_name: str, limit: int = 10) -> str:
    """
    Get the top N most common values in a column
    
    Args:
        sheet_name: Name of the sheet
        column_name: Column to analyze
        limit: Number of top values to return (default 10)
    
    Returns:
        JSON string with top values and their counts
    """
    try:
        # OPTIMIZATION: Check cache
        cache_key = f"top_{sheet_name}_{column_name}_{limit}"
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
        
        if column_name not in df.columns:
            return json.dumps({"error": f"Column '{column_name}' not found. Available: {list(df.columns)}"})
        
        # Get value counts
        value_counts = df[column_name].value_counts().head(limit)
        
        # Convert to list of dicts for better readability
        results = [
            {"value": _stringify_key(val), "count": int(count)}
            for val, count in value_counts.items()
        ]
        
        payload = {
            "column": column_name,
            "total_unique_values": int(df[column_name].nunique()),
            "total_rows": len(df),
            "top_values": results
        }
        
        result = json.dumps(_to_jsonable(payload), indent=2)
        
        # Cache the result
        cache_query(cache_key, result)
        return result
        
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_trend(sheet_name: str, date_column: str, value_column: str = "", period: str = "month") -> str:
    """
    Analyze trends over time
    
    Args:
        sheet_name: Name of the sheet
        date_column: Column containing dates
        value_column: Column to aggregate (optional, will count if not provided)
        period: Time period for grouping (day, week, month, quarter, year)
    
    Returns:
        JSON string with trend data
    """
    try:
        # OPTIMIZATION: Check cache
        cache_key = f"trend_{sheet_name}_{date_column}_{value_column}_{period}"
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
        
        if date_column not in df.columns:
            return json.dumps({"error": f"Column '{date_column}' not found. Available: {list(df.columns)}"})
        
        # Convert to datetime
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
        df_copy = df_copy.dropna(subset=[date_column])
        
        # Group by period
        period_map = {
            "day": "D",
            "week": "W",
            "month": "M",
            "quarter": "Q",
            "year": "Y"
        }
        
        freq = period_map.get(period.lower(), "M")
        df_copy['period'] = df_copy[date_column].dt.to_period(freq)
        
        # Aggregate
        if value_column and value_column in df_copy.columns:
            trend_data = df_copy.groupby('period')[value_column].sum().sort_index()
            metric = "sum"
        else:
            trend_data = df_copy.groupby('period').size().sort_index()
            metric = "count"
        
        # Convert to list format
        results = [
            {"period": str(period), "value": float(value) if pd.notna(value) else 0}
            for period, value in trend_data.items()
        ]
        
        payload = {
            "date_column": date_column,
            "value_column": value_column or "count",
            "period": period,
            "metric": metric,
            "trend_data": results,
            "total_periods": len(results)
        }
        
        result = json.dumps(_to_jsonable(payload), indent=2)
        
        # Cache the result
        cache_query(cache_key, result)
        return result
        
    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


def filter_data(sheet_name: str, filter_column: str, filter_value: str, return_columns: str = "") -> str:
    """
    Filter data based on column value
    
    Args:
        sheet_name: Name of the sheet
        filter_column: Column to filter by
        filter_value: Value or pattern to match (supports partial matching)
        return_columns: Comma-separated list of columns to return (optional, returns all if empty)
    
    Returns:
        JSON string with filtered data
    """
    try:
        # OPTIMIZATION: Check cache
        cache_key = f"filter_{sheet_name}_{filter_column}_{filter_value}_{return_columns}"
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
        
        if filter_column not in df.columns:
            return json.dumps({"error": f"Column '{filter_column}' not found. Available: {list(df.columns)}"})
        
        # Apply filter (case-insensitive partial match)
        mask = df[filter_column].astype(str).str.contains(str(filter_value), case=False, na=False)
        filtered_df = df[mask]
        
        # Select columns if specified
        if return_columns:
            cols = [c.strip() for c in return_columns.split(',')]
            available_cols = [c for c in cols if c in filtered_df.columns]
            if available_cols:
                filtered_df = filtered_df[available_cols]
        
        # Limit to 100 rows for performance
        filtered_df = filtered_df.head(100)
        
        payload = {
            "filter_column": filter_column,
            "filter_value": filter_value,
            "matched_rows": len(filtered_df),
            "total_rows": len(df),
            "data": filtered_df.to_dict('records')
        }
        
        result = json.dumps(_to_jsonable(payload), indent=2)
        
        # Cache the result
        cache_query(cache_key, result)
        return result
        
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_values",
            "description": "Get the top N most common values in a column. Perfect for finding most frequent categories, departments, or types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "column_name": {
                        "type": "string",
                        "description": "Column to analyze (e.g., 'department', 'violation_type_hazard_id', 'location')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of top values to return (default 10)",
                        "default": 10
                    }
                },
                "required": ["sheet_name", "column_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_trend",
            "description": "Analyze trends over time. Shows how values change across days, weeks, months, quarters, or years.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "date_column": {
                        "type": "string",
                        "description": "Column containing dates (e.g., 'occurrence_date', 'start_date')"
                    },
                    "value_column": {
                        "type": "string",
                        "description": "Column to aggregate (optional, will count records if not provided)"
                    },
                    "period": {
                        "type": "string",
                        "enum": ["day", "week", "month", "quarter", "year"],
                        "description": "Time period for grouping (default: month)",
                        "default": "month"
                    }
                },
                "required": ["sheet_name", "date_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_data",
            "description": "Filter data based on column value with partial matching. Returns matching records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet_name": {
                        "type": "string",
                        "description": "Name of the dataset"
                    },
                    "filter_column": {
                        "type": "string",
                        "description": "Column to filter by (e.g., 'department', 'status', 'location')"
                    },
                    "filter_value": {
                        "type": "string",
                        "description": "Value or pattern to match (case-insensitive, supports partial matching)"
                    },
                    "return_columns": {
                        "type": "string",
                        "description": "Comma-separated list of columns to return (optional, returns all if empty)"
                    }
                },
                "required": ["sheet_name", "filter_column", "filter_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for safety standards, best practices, regulations, and solutions. Use this to find OSHA standards, industry guidelines, safety protocols, or expert recommendations. Perfect for providing authoritative sources and compliance information. Results may include thumbnail images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query focused on safety standards, regulations, or best practices. Examples: 'OSHA fall protection standards', 'workplace chemical hazard prevention', 'confined space safety requirements'"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_images",
            "description": "Search for images related to safety topics. Use this to find visual examples of hazards, safety equipment, signage, infographics, or diagrams. Perfect for visual references in safety training or documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Image search query. Examples: 'workplace hazard signs', 'PPE safety equipment', 'confined space entry diagram', 'safety infographic', 'OSHA poster'"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of images to return (default 10, max 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_database_schema",
            "description": "Get the database schema showing all tables, columns, data types, and sample data. ALWAYS use this tool first before executing SQL queries to understand the database structure. Shows table names, column names, types, and 3 sample rows per table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "database_path": {
                        "type": "string",
                        "description": "Path to SQLite database (optional, defaults to epcl_vehs.db)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql_query",
            "description": "Execute SQL SELECT query on the database for complex analysis, joins, aggregations, and custom queries. Use get_database_schema first to see available tables and columns. Only SELECT queries are allowed for security. Perfect for complex multi-table analysis, custom aggregations, and advanced filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query to execute. Examples: 'SELECT department, COUNT(*) as count FROM incidents GROUP BY department ORDER BY count DESC', 'SELECT i.*, h.hazard_type FROM incidents i JOIN hazards h ON i.hazard_id = h.id WHERE i.severity = \"High\"'"
                    },
                    "database_path": {
                        "type": "string",
                        "description": "Path to SQLite database (optional, defaults to epcl_vehs.db)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum rows to return (default 100, max 1000)",
                        "default": 100
                    }
                },
                "required": ["query"]
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
    "get_top_values": get_top_values,
    "get_trend": get_trend,
    "filter_data": filter_data,
    "create_chart": create_chart,
    "search_web": search_web,
    "search_images": search_images,
    "get_database_schema": get_database_schema,
    "execute_sql_query": execute_sql_query
}

# ==================== Response Formatting ====================

def enhance_response_formatting(response: str) -> str:
    """
    Clean and normalize response formatting without forcing structure.
    Removes excessive formatting that causes rendering issues.
    """
    
    # Just return the response as-is - let frontend handle markdown rendering
    # This prevents double-escaping and literal markdown display issues
    return response.strip()


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
    system_prompt = f"""You are Safety Copilot, an AI workplace safety advisor and data analyst built by Qbit Dynamics.

You have access to:
- Excel workbook data: {available_sheets}
- SQLite database: epcl_vehs.db
- Web search: OSHA/NIOSH standards
- Image search: Safety signs, PPE, diagrams

═══════════════════════════════════════════════════════════════
QUERY ANALYSIS & PLANNING (Do this FIRST before calling tools)
═══════════════════════════════════════════════════════════════

Step 1: UNDERSTAND THE QUESTION
- What is the user really asking?
- What type of analysis is needed? (trend, comparison, root cause, compliance)
- What data sources are required?
- What level of detail is expected?

Step 2: CREATE A PLAN
List the tools you'll use in order:
Example: "I will: 1) get_database_schema to see structure, 2) execute_sql_query to get data, 3) search_web for OSHA standards, 4) create_chart for visualization"

Step 3: EXECUTE PLAN
Call tools in the planned sequence

═══════════════════════════════════════════════════════════════
TOOL SELECTION STRATEGY (Critical for efficiency)
═══════════════════════════════════════════════════════════════

ALWAYS use get_database_schema BEFORE execute_sql_query

For SIMPLE queries (top N, counts, single table):
✓ Use: get_top_values, aggregate_data, query_data
✗ Avoid: SQL (overkill)
Example: "top 10 departments" → get_top_values

For COMPLEX queries (joins, multi-table, advanced aggregations):
✓ Use: get_database_schema + execute_sql_query
✗ Avoid: Multiple Pandas tools (inefficient)
Example: "incidents with hazard types by severity" → SQL JOIN

For TRENDS over time:
✓ Use: get_trend + create_chart
Example: "monthly incident trends" → get_trend(period="month") + create_chart

For COMPARISONS:
✓ Use: compare_sheets OR execute_sql_query with UNION/JOIN
Example: "compare Q1 vs Q2" → SQL with date filters

For STANDARDS/COMPLIANCE:
✓ Use: search_web (OSHA, NIOSH, industry standards)
Example: "fall protection requirements" → search_web

For VISUAL REFERENCES:
✓ Use: search_images
Example: "show me PPE examples" → search_images

For VISUALIZATIONS:
✓ Always use create_chart after getting data
✓ Specify chart_type based on data: bar (comparisons), line (trends), pie (distributions)

═══════════════════════════════════════════════════════════════
ERROR HANDLING & SELF-CORRECTION
═══════════════════════════════════════════════════════════════

If a tool returns an error:
1. Analyze what went wrong
2. Try alternative approach:
   - Sheet not found? → Use get_data_summary to see available sheets
   - SQL error? → Try Pandas tools instead
   - No results? → Broaden search criteria
   - Timeout? → Add LIMIT or simplify query
3. Maximum 2 retry attempts per tool
4. If still failing, explain limitation to user honestly

═══════════════════════════════════════════════════════════════
DATA QUALITY & VALIDATION
═══════════════════════════════════════════════════════════════

Before presenting insights:
- Check if results make sense (e.g., negative counts = error)
- Mention if data is incomplete or has quality issues
- Note the time period of data
- Indicate confidence level: High (complete data), Medium (partial), Low (limited)

═══════════════════════════════════════════════════════════════
RESPONSE STRUCTURE (Tell a compelling story)
═══════════════════════════════════════════════════════════════

## What's Happening (Key Findings)
- Lead with the most important insight
- Use specific numbers with context: "245 incidents (28% of total)"
- Make comparisons relatable: "That's 3x higher than last quarter"
- Confidence: [High/Medium/Low]

## Why It Matters (Root Cause Analysis)
- Explain WHY the pattern exists (not just WHAT)
- Connect to safety outcomes: "This increases injury risk by..."
- Identify contributing factors
- Show business impact: costs, downtime, compliance risks

## What To Do About It (Actionable Recommendations)
Priority 1 (Immediate - Do this week):
- Specific action with clear owner
- Expected impact

Priority 2 (Short-term - Do this month):
- Preventive measures
- Process improvements

Priority 3 (Long-term - Do this quarter):
- Systemic changes
- Culture shifts

**Standards & Compliance:**
- Cite OSHA/NIOSH standards when applicable
- Include links to official guidelines
- Note any compliance gaps

## The Bottom Line
- One-sentence summary
- Key metric to track: "Monitor [metric] weekly"

---
**Data Sources:**
- List all tools used with their data sources
- Example: "Excel workbook (via get_top_values), OSHA.gov (via search_web)"

═══════════════════════════════════════════════════════════════
COMMUNICATION STYLE
═══════════════════════════════════════════════════════════════

✓ DO:
- Speak like advising a colleague over coffee
- Use analogies: "Think of it like..."
- Make it personal: "Your team is experiencing..."
- Create urgency for safety issues: "This needs immediate attention because..."
- Celebrate wins: "Great news - incidents dropped 30%!"

✗ DON'T:
- Write like a technical report
- Use jargon without explanation
- Be vague: "Some departments" → "Operations (245) and Maintenance (189)"
- Overwhelm with data dumps
- Ignore the human impact

═══════════════════════════════════════════════════════════════
FORMATTING RULES
═══════════════════════════════════════════════════════════════

- Use ## for main sections
- Use - for bullet points (not numbers unless prioritizing)
- Bold only for critical emphasis: **URGENT**
- Keep paragraphs to 2-3 sentences max
- Format in Table if it result contains any Top values 
- Use tables for comparing 3+ items also for any tabular data display
- Include emojis sparingly for visual breaks: 🚨 ⚠️ ✅ 📊

═══════════════════════════════════════════════════════════════
AVAILABLE TOOLS (12 total)
═══════════════════════════════════════════════════════════════

Excel/Pandas Tools:
- get_data_summary: Schema and stats for a dataset
- query_data: Natural language queries
- aggregate_data: Group by operations (count, sum, mean, max, min)
- compare_sheets: Compare two datasets
- get_top_values: Most common values in a column
- get_trend: Time-series analysis (day/week/month/quarter/year)
- filter_data: Filter rows by column value
- create_chart: Visualizations (bar, line, pie, scatter)

SQL Tools:
- get_database_schema: See database structure (ALWAYS use before SQL)
- execute_sql_query: Complex queries, joins, aggregations

Web Tools:
- search_web: OSHA/NIOSH standards, best practices
- search_images: Safety signs, PPE, diagrams

═══════════════════════════════════════════════════════════════
CRITICAL REMINDERS
═══════════════════════════════════════════════════════════════

1. PLAN before executing tools
2. Use get_database_schema BEFORE execute_sql_query
3. Always cite specific numbers with context
4. Include confidence levels
5. Provide prioritized, actionable recommendations
6. Cite OSHA/NIOSH standards when relevant
7. List data sources at the end
8. Make it conversational, not technical
9. Focus on human impact and safety outcomes
10. Self-correct if tools fail

Available datasets: {available_sheets}
Database: epcl_vehs.db
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    # Deduplication: avoid repeating identical tool calls
    recent_tool_calls: Dict[str, str] = {}
    repeat_counts: Dict[str, int] = {}

    def _tool_key(name: str, args: Dict[str, Any]) -> str:
        """Stable key for a tool call to detect repeats."""
        try:
            return f"{name}:{json.dumps(args, sort_keys=True)}"
        except Exception:
            return f"{name}:{str(args)}"

    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # NOTE: Suppress separate 'thinking' event; keep only reasoning tokens
        
        try:
            # OPTIMIZATION: Call AI with tools (streaming enabled)
            # Note: reasoning parameter requires OpenAI SDK >= 1.58.0 or use extra_body
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,  # Low temp for consistency
                max_tokens=30000,
                stream=True,  # Enable streaming
                extra_body={  # Use extra_body for OpenRouter-specific params
                    "reasoning": {
                        "effort": "high",
                        "exclude": False
                    }
                },
                extra_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Safety Copilot"
                }
            )
            
            # Collect streaming response
            assistant_message = None
            content_buffer = ""
            sent_answer_tokens = False
            token_buffer = ""
            last_flush_time = time.perf_counter()
            min_flush_chars = 64
            max_wait_seconds = 0.2
            tool_calls_buffer = {}
            reasoning_buffer = ""  # For reasoning tokens
            reasoning_details_buffer = []  # For reasoning_details array
            has_tool_calls = False
            
            async for chunk in stream:
                delta = chunk.choices[0].delta
                
                # Collect reasoning tokens (OpenRouter reasoning models)
                if hasattr(delta, 'reasoning') and delta.reasoning:
                    reasoning_buffer += delta.reasoning
                    yield {
                        "type": "reasoning_token",
                        "token": delta.reasoning
                    }
                
                # Collect reasoning_details (for preserving reasoning blocks)
                if hasattr(delta, 'reasoning_details') and delta.reasoning_details:
                    reasoning_details_buffer.extend(delta.reasoning_details)
                
                # Collect tool calls (don't stream partial args - reduces overhead)
                if delta.tool_calls:
                    has_tool_calls = True
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
                
                # Accumulate assistant content silently; do not emit 'thinking_token'
                if delta.content:
                    content_buffer += delta.content
                    token_buffer += delta.content
                    now = time.perf_counter()
                    should_flush = False
                    if "\n" in token_buffer:
                        should_flush = True
                    elif len(token_buffer) >= min_flush_chars:
                        should_flush = True
                    elif token_buffer[-1:] in ('.', '!', '?') and len(token_buffer) >= 20:
                        should_flush = True
                    elif (now - last_flush_time) >= max_wait_seconds and len(token_buffer) >= 1:
                        should_flush = True
                    if should_flush:
                        yield {
                            "type": "answer",
                            "content": token_buffer
                        }
                        sent_answer_tokens = True
                        token_buffer = ""
                        last_flush_time = now
                
                # Get finish reason
                if chunk.choices[0].finish_reason:
                    # Flush any remaining buffered answer tokens
                    if token_buffer:
                        yield {
                            "type": "answer",
                            "content": token_buffer
                        }
                        sent_answer_tokens = True
                        token_buffer = ""
                    # Build proper message dict for API
                    assistant_message_dict = {
                        "role": "assistant",
                        "content": content_buffer or None
                    }
                    
                    # Add reasoning_details if present (for preserving reasoning blocks)
                    if reasoning_details_buffer:
                        assistant_message_dict["reasoning_details"] = reasoning_details_buffer
                    
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
                        # Add reasoning to object if present
                        if reasoning_buffer:
                            assistant_message.reasoning = reasoning_buffer
                        if reasoning_details_buffer:
                            assistant_message.reasoning_details = reasoning_details_buffer
                    else:
                        assistant_message = type('obj', (), {
                            'content': content_buffer,
                            'tool_calls': None,
                            'reasoning': reasoning_buffer if reasoning_buffer else None,
                            'reasoning_details': reasoning_details_buffer if reasoning_details_buffer else None
                        })()
            
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
                    
                    # Deduplicate identical tool calls (short-circuit repeats)
                    key = _tool_key(function_name, function_args)
                    if key in recent_tool_calls:
                        repeat_counts[key] = repeat_counts.get(key, 1) + 1
                        cached = recent_tool_calls[key]
                        # Return cached result again so the model can consume it
                        yield {
                            "type": "tool_result",
                            "tool": function_name,
                            "result": cached,
                            "repeat": True
                        }
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": cached
                        })
                        # On second repeat, nudge synthesis explicitly
                        if repeat_counts[key] >= 2:
                            messages.append({
                                "role": "assistant",
                                "content": "I already have the data from previous tool calls. I will synthesize the final answer now without re-calling the tool."
                            })
                        continue

                    # Execute tool
                    if function_name in TOOL_FUNCTIONS:
                        # Handle async tools (search_web)
                        if asyncio.iscoroutinefunction(TOOL_FUNCTIONS[function_name]):
                            result = await TOOL_FUNCTIONS[function_name](**function_args)
                        else:
                            result = TOOL_FUNCTIONS[function_name](**function_args)
                        # Cache successful result for deduplication
                        recent_tool_calls[key] = result
                        
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
                                # Handle async tools
                                if asyncio.iscoroutinefunction(TOOL_FUNCTIONS[function_name]):
                                    result2 = await TOOL_FUNCTIONS[function_name](**corrected_args)
                                else:
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
                # AI provided final answer (content was accumulated silently)
                final_answer = content_buffer
                
                # If we already streamed tokens, avoid duplicating the answer
                if sent_answer_tokens:
                    yield {
                        "type": "complete",
                        "data": {
                            "iterations": iteration,
                            "tools_used": len([msg for msg in messages if msg.get("role") == "tool"])
                        }
                    }
                else:
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
