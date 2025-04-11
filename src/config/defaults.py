"""
Default configuration values for OmniPulse dashboard.
"""

from typing import Dict, Any, List

# Default dashboard settings
DEFAULT_DASHBOARD_SETTINGS = {
    "title": "OmniPulse Dashboard",
    "refresh_interval": 5,  # seconds
    "theme": "light",
    "default_time_range": "Last hour",
    "sidebar_collapsed": False,
    "show_footer": True
}

# Default Ollama monitor settings
DEFAULT_OLLAMA_SETTINGS = {
    "api_url": "http://localhost:11434",
    "test_prompt": "Explain quantum computing in simple terms",
    "monitor_interval": 5,  # seconds
    "history_limit": 1000,  # data points
    "default_model": None  # Use first available model
}

# Default system monitor settings
DEFAULT_SYSTEM_SETTINGS = {
    "monitor_interval": 5,  # seconds
    "history_limit": 1000,  # data points
    "disk_paths": [],  # Auto-detect
    "network_interfaces": [],  # Auto-detect
    "gpu_monitoring": True
}

# Default Python monitor settings
DEFAULT_PYTHON_SETTINGS = {
    "monitor_interval": 10,  # seconds
    "history_limit": 1000,  # data points
    "process_filter": ["streamlit", "jupyter", "ollama", "python"],
    "include_streamlit": True,
    "include_jupyter": True
}

# Default custom monitor settings
DEFAULT_CUSTOM_SETTINGS = {
    "config_path": "config/custom_monitors.json",
    "monitor_interval": 60,  # seconds
    "history_limit": 1000  # data points
}

# Default metric card colors
DEFAULT_METRIC_COLORS = {
    "normal": "blue",
    "success": "green",
    "warning": "yellow",
    "danger": "red",
    "info": "purple"
}

# Default authentication settings
DEFAULT_AUTH_SETTINGS = {
    "enabled": False,
    "username": "admin",
    "password_hash": None,  # Generated from admin/admin if not specified
    "cookie_expiry_days": 30,
    "preauthorized_emails": []
}

# Default alert settings
DEFAULT_ALERT_SETTINGS = {
    "enabled": False,
    "email_alerts": False,
    "email_address": "",
    "smtp_server": "",
    "smtp_port": 587,
    "smtp_username": "",
    "smtp_password": "",
    "log_alerts": True,
    "alert_log_path": "logs/alerts.log",
    "notification_sound": False,
    "browser_notifications": False,
    "thresholds": {
        "cpu_percent": 90,
        "memory_percent": 90,
        "disk_percent": 90,
        "gpu_percent": 90,
        "ollama_error_rate": 10
    }
}

# Default chart colors
DEFAULT_CHART_COLORS = [
    "#4e8df5",  # Blue
    "#4CAF50",  # Green
    "#F44336",  # Red
    "#FFC107",  # Yellow
    "#9C27B0",  # Purple
    "#00BCD4",  # Cyan
    "#FF9800",  # Orange
    "#607D8B",  # Blue Grey
    "#8BC34A",  # Light Green
    "#E91E63"   # Pink
]

# Default time ranges in seconds
DEFAULT_TIME_RANGES = {
    "Last 15 minutes": 60 * 15,
    "Last hour": 60 * 60,
    "Last 3 hours": 60 * 60 * 3,
    "Last day": 60 * 60 * 24,
    "Last week": 60 * 60 * 24 * 7
}

# Default example custom metrics
DEFAULT_EXAMPLE_METRICS = [
    {
        "name": "system_load",
        "type": "command",
        "command": "cat /proc/loadavg | awk '{print \"{\\\"load_1m\\\":\\\"\" $1 \"\\\",\\\"load_5m\\\":\\\"\" $2 \"\\\",\\\"load_15m\\\":\\\"\" $3 \"\\\"}\"}'",
        "parser": "json",
        "interval": 60,
        "enabled": False
    },
    {
        "name": "memory_usage",
        "type": "command",
        "command": "free -m | grep Mem | awk '{print \"{\\\"total\\\":\" $2 \",\\\"used\\\":\" $3 \",\\\"free\\\":\" $4 \",\\\"percent_used\\\":\" ($3/$2*100) \"}\"}'",
        "parser": "json", 
        "interval": 60,
        "enabled": False
    },
    {
        "name": "example_http",
        "type": "http",
        "url": "https://httpstat.us/200",
        "method": "GET",
        "headers": {},
        "interval": 60,
        "enabled": False
    }
]

# Default dashboard layouts
DEFAULT_DASHBOARD_LAYOUTS = {
    "system_overview": [
        {"type": "metric_card", "title": "CPU Usage", "metric": "cpu_percent", "width": "third"},
        {"type": "metric_card", "title": "Memory Usage", "metric": "memory_percent", "width": "third"},
        {"type": "metric_card", "title": "Disk Usage", "metric": "disk_percent", "width": "third"},
        {"type": "chart", "title": "CPU Over Time", "metric": "cpu_percent", "chart_type": "line", "width": "half"},
        {"type": "chart", "title": "Memory Over Time", "metric": "memory_percent", "chart_type": "line", "width": "half"}
    ],
    "ollama_overview": [
        {"type": "metric_card", "title": "Requests", "metric": "total_requests", "width": "quarter"},
        {"type": "metric_card", "title": "Success Rate", "metric": "success_rate", "width": "quarter"},
        {"type": "metric_card", "title": "Avg. Latency", "metric": "average_latency", "width": "quarter"},
        {"type": "metric_card", "title": "Tokens/sec", "metric": "average_throughput", "width": "quarter"},
        {"type": "chart", "title": "Latency Over Time", "metric": "latency", "chart_type": "line", "width": "full"}
    ]
}

# Default widget types
DEFAULT_WIDGET_TYPES = [
    "metric_card",
    "line_chart",
    "bar_chart",
    "area_chart", 
    "pie_chart",
    "gauge",
    "table",
    "stat_comparison"
]

# Default system metrics to track
DEFAULT_SYSTEM_METRICS = [
    "cpu_percent",
    "memory_percent",
    "disk_percent",
    "network_send_rate",
    "network_recv_rate",
    "load_average"
]

# Default Ollama metrics to track
DEFAULT_OLLAMA_METRICS = [
    "latency",
    "throughput",
    "success_rate", 
    "token_count",
    "requests_per_second"
]

# Default Python app metrics to track
DEFAULT_PYTHON_METRICS = [
    "process_count",
    "cpu_usage",
    "memory_usage",
    "thread_count"
]

# Function to get default configuration
def get_default_config() -> Dict[str, Any]:
    """
    Get the complete default configuration.
    
    Returns:
        Dictionary with all default configuration values
    """
    return {
        "dashboard": DEFAULT_DASHBOARD_SETTINGS,
        "ollama": DEFAULT_OLLAMA_SETTINGS,
        "system": DEFAULT_SYSTEM_SETTINGS,
        "python": DEFAULT_PYTHON_SETTINGS,
        "custom": DEFAULT_CUSTOM_SETTINGS,
        "auth": DEFAULT_AUTH_SETTINGS,
        "alerts": DEFAULT_ALERT_SETTINGS,
        "metrics": {
            "colors": DEFAULT_METRIC_COLORS,
            "system": DEFAULT_SYSTEM_METRICS,
            "ollama": DEFAULT_OLLAMA_METRICS,
            "python": DEFAULT_PYTHON_METRICS
        },
        "charts": {
            "colors": DEFAULT_CHART_COLORS
        },
        "time_ranges": DEFAULT_TIME_RANGES,
        "example_metrics": DEFAULT_EXAMPLE_METRICS,
        "layouts": DEFAULT_DASHBOARD_LAYOUTS,
        "widgets": DEFAULT_WIDGET_TYPES
    }