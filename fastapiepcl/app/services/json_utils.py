from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd


def to_native_json(obj: Any) -> Any:
    """Recursively convert common scientific Python objects into JSON-safe native types.
    - numpy scalars -> Python scalars
    - numpy arrays -> lists
    - pandas Timestamp -> ISO string
    - pandas Series -> list
    - pandas Index -> list of strings
    - set/tuple -> list
    - dict keys and values converted recursively
    Fallback: string representation for unknown objects.
    """
    try:
        # Basic scalars
        if obj is None:
            return None
        # Handle native floats explicitly to catch NaN/Inf
        if isinstance(obj, float):
            return None if not math.isfinite(obj) else obj
        if isinstance(obj, (bool, int, str)):
            return obj
        # Numpy scalar -> convert then re-run through sanitizer
        if isinstance(obj, np.generic):
            return to_native_json(obj.item())
        # Pandas Timestamp/Timedelta
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Timedelta):
            return obj.total_seconds()
        # Numpy array
        if isinstance(obj, np.ndarray):
            return to_native_json(obj.tolist())
        # Pandas structures
        if isinstance(obj, pd.Series):
            return to_native_json(obj.tolist())
        if isinstance(obj, (pd.Index, pd.PeriodIndex, pd.DatetimeIndex)):
            try:
                return to_native_json(list(obj.astype(str)))
            except Exception:
                return to_native_json(list(map(str, list(obj))))
        # Containers
        if isinstance(obj, (list, tuple, set)):
            return [to_native_json(v) for v in obj]
        if isinstance(obj, dict):
            return {str(to_native_json(k)): to_native_json(v) for k, v in obj.items()}
        # Handle NaN/NaT for remaining scalar-like objects
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return obj
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None
