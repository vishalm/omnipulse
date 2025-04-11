"""
Helper functions for the OmniPulse dashboard.
"""

import time
import datetime
from typing import Dict, List, Any, Optional, Union
import math


def format_number(num: Union[int, float]) -> str:
    """
    Format a number with appropriate suffixes (K, M, B, T).
    
    Args:
        num: Number to format
        
    Returns:
        Formatted number string
    """
    if num is None:
        return "N/A"
    
    abs_num = abs(num)
    
    if abs_num < 1000:
        return str(num)
    elif abs_num < 1_000_000:
        return f"{num/1000:.1f}K"
    elif abs_num < 1_000_000_000:
        return f"{num/1_000_000:.1f}M"
    elif abs_num < 1_000_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    else:
        return f"{num/1_000_000_000_000:.1f}T"


def format_bytes(bytes_value: Union[int, float]) -> str:
    """
    Format bytes value to appropriate unit (B, KB, MB, GB, TB).
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted bytes string
    """
    if bytes_value is None:
        return "N/A"
    
    if bytes_value < 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB", "PB"]
    
    if bytes_value == 0:
        return "0 B"
    
    i = int(math.floor(math.log(bytes_value, 1024)))
    p = math.pow(1024, i)
    s = round(bytes_value / p, 2)
    
    return f"{s} {size_names[i]}"


def format_time(seconds: Union[int, float]) -> str:
    """
    Format seconds into a readable time string.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted time string
    """
    if seconds is None:
        return "N/A"
    
    if seconds < 0:
        return "0s"
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def format_date(dt: Union[datetime.datetime, str], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a datetime object or string to a readable date string.
    
    Args:
        dt: Datetime object or string
        format_str: Format string for output
        
    Returns:
        Formatted date string
    """
    if dt is None:
        return "N/A"
    
    if isinstance(dt, str):
        try:
            dt = datetime.datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except ValueError:
            try:
                dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                return dt
    
    return dt.strftime(format_str)


def format_percent(value: Union[int, float], decimal_places: int = 1) -> str:
    """
    Format a number as a percentage string.
    
    Args:
        value: Number to format (0-100 or 0-1)
        decimal_places: Number of decimal places to include
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"
    
    # If value is in range 0-1, convert to 0-100
    if 0 <= value <= 1:
        value = value * 100
    
    return f"{value:.{decimal_places}f}%"


def get_color_for_value(value: float, thresholds: List[float], 
                        colors: List[str] = ["green", "yellow", "red"]) -> str:
    """
    Get a color based on thresholds.
    
    Args:
        value: Value to check
        thresholds: List of threshold values
        colors: List of colors to use
        
    Returns:
        Color string
    """
    if len(thresholds) != len(colors) - 1:
        raise ValueError("Number of thresholds must be one less than number of colors")
    
    for i, threshold in enumerate(thresholds):
        if value < threshold:
            return colors[i]
    
    return colors[-1]


def time_ago(dt: Union[datetime.datetime, str]) -> str:
    """
    Convert datetime to 'time ago' string (e.g., "5 minutes ago").
    
    Args:
        dt: Datetime object or string
        
    Returns:
        Time ago string
    """
    if dt is None:
        return "N/A"
    
    if isinstance(dt, str):
        try:
            dt = datetime.datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except ValueError:
            try:
                dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                return "Unknown"
    
    now = datetime.datetime.now(dt.tzinfo if dt.tzinfo else None)
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{int(minutes)} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{int(hours)} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = seconds // 86400
        return f"{int(days)} day{'s' if days != 1 else ''} ago"
    elif seconds < 2419200:
        weeks = seconds // 604800
        return f"{int(weeks)} week{'s' if weeks != 1 else ''} ago"
    elif seconds < 29030400:
        months = seconds // 2419200
        return f"{int(months)} month{'s' if months != 1 else ''} ago"
    else:
        years = seconds // 29030400
        return f"{int(years)} year{'s' if years != 1 else ''} ago"


def get_trend_icon(current: float, previous: float, inverted: bool = False) -> str:
    """
    Get a trend icon based on the comparison of current and previous values.
    
    Args:
        current: Current value
        previous: Previous value
        inverted: Whether a decrease is positive (e.g., for error rates)
        
    Returns:
        Trend icon (emoji)
    """
    if current is None or previous is None:
        return "➖"
    
    if current > previous:
        return "🔻" if inverted else "🔺"
    elif current < previous:
        return "🔺" if inverted else "🔻"
    else:
        return "➖"


def truncate_string(s: str, max_length: int = 50) -> str:
    """
    Truncate a string to a maximum length, adding ellipsis if needed.
    
    Args:
        s: String to truncate
        max_length: Maximum length
        
    Returns:
        Truncated string
    """
    if s is None:
        return ""
    
    if len(s) <= max_length:
        return s
    
    return s[:max_length-3] + "..."


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between a minimum and maximum.
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def moving_average(data: List[float], window_size: int = 5) -> List[float]:
    """
    Calculate moving average of a list of values.
    
    Args:
        data: List of values
        window_size: Window size for moving average
        
    Returns:
        List of moving averages
    """
    if len(data) < window_size:
        return data
    
    result = []
    for i in range(len(data)):
        if i < window_size - 1:
            # Not enough data for full window yet
            window = data[:i+1]
        else:
            window = data[i-window_size+1:i+1]
        
        result.append(sum(window) / len(window))
    
    return result


def calculate_change(current: float, previous: float, as_percent: bool = True) -> float:
    """
    Calculate change between current and previous values.
    
    Args:
        current: Current value
        previous: Previous value
        as_percent: Whether to return as percentage
        
    Returns:
        Change value
    """
    if current is None or previous is None or previous == 0:
        return 0
    
    change = current - previous
    
    if as_percent:
        return (change / abs(previous)) * 100
    
    return change


def safe_divide(a: float, b: float, default: float = 0) -> float:
    """
    Safely divide two numbers, returning a default if denominator is zero.
    
    Args:
        a: Numerator
        b: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Result of division or default
    """
    if b == 0:
        return default
    
    return a / b