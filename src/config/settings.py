"""
Settings management for OmniPulse dashboard.
"""

import os
import json
import streamlit as st
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import dotenv

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Default settings
DEFAULT_SETTINGS = {
    # Ollama settings
    "ollama_api_url": "http://localhost:11434",
    
    # System monitor settings
    "system_monitor_interval": 5,  # seconds
    "system_disk_paths": [],  # Auto-detect if empty
    "system_network_interfaces": [],  # Auto-detect if empty
    
    # Python monitor settings
    "python_monitor_interval": 10,  # seconds
    "python_process_filter": ["streamlit", "jupyter", "ollama"],
    
    # Dashboard settings
    "dashboard_theme": "light",
    "dashboard_refresh_interval": 5,  # seconds
    "dashboard_default_time_range": "Last hour",
    
    # Custom monitor settings
    "custom_monitors_config_path": "config/custom_monitors.json",
    
    # Authentication settings
    "enable_authentication": False,
    "auth_username": "",
    "auth_password_hash": "",
    
    # Alert settings
    "enable_alerts": False,
    "alert_email": "",
    "alert_thresholds": {
        "cpu_percent": 90,
        "memory_percent": 90,
        "disk_percent": 90
    }
}


def get_config_path() -> Path:
    """
    Get path to config file.
    
    Returns:
        Path to config file
    """
    # First, check for config file in the current directory
    local_config = Path("config/settings.json")
    if local_config.exists():
        return local_config
    
    # Then check for config file in user's home directory
    home_config = Path.home() / ".omnipulse" / "settings.json"
    if home_config.exists():
        return home_config
    
    # Return local config path as default
    return local_config


def load_settings() -> Dict[str, Any]:
    """
    Load settings from file or environment variables.
    
    Returns:
        Dictionary of settings
    """
    settings = DEFAULT_SETTINGS.copy()
    
    # Try to load from file
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                file_settings = json.load(f)
                settings.update(file_settings)
        except Exception as e:
            print(f"Error loading settings from {config_path}: {e}")
    
    # Override with environment variables
    env_prefix = "OMNIPULSE_"
    for key in settings.keys():
        env_key = f"{env_prefix}{key.upper()}"
        if env_key in os.environ:
            value = os.environ[env_key]
            
            # Convert some types
            if isinstance(settings[key], bool):
                settings[key] = value.lower() in ("true", "yes", "1")
            elif isinstance(settings[key], int):
                try:
                    settings[key] = int(value)
                except ValueError:
                    pass
            elif isinstance(settings[key], float):
                try:
                    settings[key] = float(value)
                except ValueError:
                    pass
            elif isinstance(settings[key], list):
                try:
                    settings[key] = json.loads(value)
                except json.JSONDecodeError:
                    # Try comma-separated list
                    settings[key] = [item.strip() for item in value.split(",")]
            elif isinstance(settings[key], dict):
                try:
                    settings[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
            else:
                settings[key] = value
    
    # Override with Streamlit session state
    for key in settings.keys():
        if key in st.session_state:
            settings[key] = st.session_state[key]
    
    return settings


def save_settings(settings: Dict[str, Any]) -> bool:
    """
    Save settings to file.
    
    Args:
        settings: Dictionary of settings
        
    Returns:
        True if successful, False otherwise
    """
    config_path = get_config_path()
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, "w") as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings to {config_path}: {e}")
        return False


def update_setting(key: str, value: Any) -> bool:
    """
    Update a specific setting.
    
    Args:
        key: Setting key
        value: Setting value
        
    Returns:
        True if successful, False otherwise
    """
    # Update in session state
    st.session_state[key] = value
    
    # Load current settings
    settings = load_settings()
    
    # Update setting
    settings[key] = value
    
    # Save settings
    return save_settings(settings)


def get_setting(key: str, default: Any = None) -> Any:
    """
    Get a specific setting.
    
    Args:
        key: Setting key
        default: Default value if setting doesn't exist
        
    Returns:
        Setting value
    """
    # Check session state first
    if key in st.session_state:
        return st.session_state[key]
    
    # Then load from settings
    settings = load_settings()
    return settings.get(key, default)


# Convenience functions for commonly used settings

def get_ollama_api_url() -> str:
    """
    Get Ollama API URL.
    
    Returns:
        Ollama API URL
    """
    return get_setting("ollama_api_url", DEFAULT_SETTINGS["ollama_api_url"])


def get_refresh_interval() -> int:
    """
    Get dashboard refresh interval.
    
    Returns:
        Refresh interval in seconds
    """
    return get_setting("dashboard_refresh_interval", DEFAULT_SETTINGS["dashboard_refresh_interval"])


def get_system_monitor_interval() -> int:
    """
    Get system monitor interval.
    
    Returns:
        System monitor interval in seconds
    """
    return get_setting("system_monitor_interval", DEFAULT_SETTINGS["system_monitor_interval"])


def get_python_monitor_interval() -> int:
    """
    Get Python monitor interval.
    
    Returns:
        Python monitor interval in seconds
    """
    return get_setting("python_monitor_interval", DEFAULT_SETTINGS["python_monitor_interval"])


def get_custom_monitors_config_path() -> str:
    """
    Get path to custom monitors configuration.
    
    Returns:
        Path to custom monitors configuration
    """
    return get_setting("custom_monitors_config_path", DEFAULT_SETTINGS["custom_monitors_config_path"])


def get_theme() -> str:
    """
    Get dashboard theme.
    
    Returns:
        Dashboard theme
    """
    return get_setting("dashboard_theme", DEFAULT_SETTINGS["dashboard_theme"])


def is_authentication_enabled() -> bool:
    """
    Check if authentication is enabled.
    
    Returns:
        True if authentication is enabled, False otherwise
    """
    return get_setting("enable_authentication", DEFAULT_SETTINGS["enable_authentication"])


def is_alerts_enabled() -> bool:
    """
    Check if alerts are enabled.
    
    Returns:
        True if alerts are enabled, False otherwise
    """
    return get_setting("enable_alerts", DEFAULT_SETTINGS["enable_alerts"])


def get_alert_thresholds() -> Dict[str, float]:
    """
    Get alert thresholds.
    
    Returns:
        Dictionary of alert thresholds
    """
    return get_setting("alert_thresholds", DEFAULT_SETTINGS["alert_thresholds"])