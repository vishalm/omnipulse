"""
Caching module for OmniPulse dashboard.
"""

import streamlit as st
import os
import json
import pickle
import hashlib
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union, TypeVar, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import functools

from src.config.settings import get_setting

# Type variable for generic function return types
T = TypeVar('T')


class Cache:
    """In-memory and disk cache for OmniPulse dashboard."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_memory_items: int = 1000):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory for persistent cache files (None for in-memory only)
            max_memory_items: Maximum number of items to keep in memory
        """
        self.memory_cache: Dict[str, Tuple[Any, float, Optional[float]]] = {}  # key -> (value, timestamp, expiry)
        self.max_memory_items = max_memory_items
        self.cache_lock = threading.RLock()
        
        # Set up disk cache if a directory is provided
        self.disk_cache_enabled = cache_dir is not None
        if self.disk_cache_enabled:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        # Check memory cache first
        with self.cache_lock:
            if key in self.memory_cache:
                value, timestamp, expiry = self.memory_cache[key]
                
                # Check if expired
                if expiry is not None and time.time() > expiry:
                    del self.memory_cache[key]
                else:
                    return value
        
        # If not in memory cache and disk cache is enabled, check disk
        if self.disk_cache_enabled:
            file_path = self._get_file_path(key)
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Check if the disk cache item is expired
                    if 'expiry' in data and data['expiry'] is not None and time.time() > data['expiry']:
                        # Remove expired item
                        file_path.unlink(missing_ok=True)
                    else:
                        # Load into memory cache
                        with self.cache_lock:
                            self.memory_cache[key] = (data['value'], data['timestamp'], data.get('expiry'))
                        return data['value']
                except Exception:
                    # If any error occurs, consider the cache miss
                    pass
        
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for no expiry)
        """
        timestamp = time.time()
        expiry = timestamp + ttl if ttl is not None else None
        
        # Set in memory cache
        with self.cache_lock:
            self.memory_cache[key] = (value, timestamp, expiry)
            
            # If exceeding max items, remove oldest
            if len(self.memory_cache) > self.max_memory_items:
                oldest_key = min(self.memory_cache.items(), key=lambda x: x[1][1])[0]
                del self.memory_cache[oldest_key]
        
        # If disk cache is enabled, also write to disk
        if self.disk_cache_enabled:
            file_path = self._get_file_path(key)
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump({
                        'value': value,
                        'timestamp': timestamp,
                        'expiry': expiry
                    }, f)
            except Exception:
                # Ignore disk write errors
                pass
    
    def delete(self, key: str):
        """
        Delete a value from cache.
        
        Args:
            key: Cache key to delete
        """
        # Remove from memory cache
        with self.cache_lock:
            if key in self.memory_cache:
                del self.memory_cache[key]
        
        # Remove from disk cache if enabled
        if self.disk_cache_enabled:
            file_path = self._get_file_path(key)
            file_path.unlink(missing_ok=True)
    
    def clear(self):
        """Clear all cached values."""
        # Clear memory cache
        with self.cache_lock:
            self.memory_cache.clear()
        
        # Clear disk cache if enabled
        if self.disk_cache_enabled:
            for file_path in self.cache_dir.glob('*.cache'):
                file_path.unlink(missing_ok=True)
    
    def _get_file_path(self, key: str) -> Path:
        """
        Get file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Path object for the cache file
        """
        # Hash the key to create a filename
        filename = hashlib.md5(key.encode()).hexdigest() + '.cache'
        return self.cache_dir / filename
    
    def memoize(self, ttl: Optional[int] = None):
        """
        Decorator to memoize function results.
        
        Args:
            ttl: Time to live in seconds (None for no expiry)
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key based on function name and arguments
                key_parts = [func.__module__, func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
                
                # Try to get from cache
                cached_value = self.get(key)
                if cached_value is not None:
                    return cached_value
                
                # Call the function and cache the result
                result = func(*args, **kwargs)
                self.set(key, result, ttl)
                return result
            return wrapper
        return decorator


# Initialize global cache
cache_dir = get_setting("cache_dir", "cache")
memory_cache_size = get_setting("memory_cache_size", 1000)
cache = Cache(cache_dir=cache_dir, max_memory_items=memory_cache_size)


def cached(ttl: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds (None for no expiry)
        
    Returns:
        Decorated function
    """
    return cache.memoize(ttl)


@st.cache_data
def streamlit_cache(func):
    """
    Decorator for Streamlit's built-in caching.
    
    Args:
        func: Function to cache
        
    Returns:
        Cached function
    """
    return func


def clear_all_caches():
    """Clear all caches (both custom and Streamlit's)."""
    # Clear custom cache
    cache.clear()
    
    # Clear Streamlit caches
    st.cache_data.clear()
    st.cache_resource.clear()