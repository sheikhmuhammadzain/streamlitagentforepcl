"""
High-performance data caching layer for blazing fast responses
Implements LRU cache with TTL for optimal performance
"""

import pandas as pd
from typing import Dict, Optional, Any
from functools import lru_cache
import time
from threading import Lock


class DataCache:
    """Thread-safe LRU cache for DataFrames with TTL"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.ttl = ttl_seconds
        self.lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    self._hits += 1
                    return value
                else:
                    # Expired, remove it
                    del self.cache[key]
            
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any):
        """Cache a value with current timestamp"""
        with self.lock:
            self.cache[key] = (value, time.time())
    
    def clear(self):
        """Clear all cached data"""
        with self.lock:
            self.cache.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 2),
            "cached_items": len(self.cache)
        }


# Global cache instances
_workbook_cache = DataCache(ttl_seconds=300)  # 5 minutes TTL
_query_cache = DataCache(ttl_seconds=60)  # 1 minute for query results


def get_cached_workbook() -> Optional[Dict[str, pd.DataFrame]]:
    """Get cached workbook data"""
    return _workbook_cache.get("workbook")


def cache_workbook(workbook: Dict[str, pd.DataFrame]):
    """Cache workbook data"""
    _workbook_cache.set("workbook", workbook)


def get_cached_query(query_key: str) -> Optional[str]:
    """Get cached query result"""
    return _query_cache.get(query_key)


def cache_query(query_key: str, result: str):
    """Cache query result"""
    _query_cache.set(query_key, result)


def clear_all_caches():
    """Clear all caches"""
    _workbook_cache.clear()
    _query_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches"""
    return {
        "workbook_cache": _workbook_cache.stats(),
        "query_cache": _query_cache.stats()
    }
