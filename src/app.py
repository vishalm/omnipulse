"""
OmniPulse: Enterprise LLM & System Monitoring Hub
Main application entry point
"""

import streamlit as st
import asyncio
import time
from datetime import datetime
import os
import sys
from streamlit_autorefresh import st_autorefresh

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.config import settings
from src.dashboards import ollama_dashboard, system_dashboard, python_dashboard, custom_dashboard
from src.monitors.ollama_monitor import OllamaMonitor
from src.monitors.system_monitor import SystemMonitor
from src.monitors.python_monitor import PythonMonitor
from src.monitors.custom_monitor import CustomMonitor


# Initialize monitors
@st.cache_resource
def initialize_monitors():
    """Initialize monitoring components."""
    ollama_mon = OllamaMonitor(api_url=settings.get_ollama_api_url())
    system_mon = SystemMonitor()
    python_mon = PythonMonitor(
        process_name_filter=settings.get_setting("python_process_filter")
    )
    custom_mon = CustomMonitor(
        config_path=settings.get_custom_monitors_config_path()
    )
    
    return {
        "ollama": ollama_mon,
        "system": system_mon,
        "python": python_mon,
        "custom": custom_mon
    }


def main():
    """Main application entry point."""
    
    # Set page configuration
    st.set_page_config(
        page_title="OmniPulse Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Apply custom styling
    apply_custom_styles()
    
    # Initialize monitors
    monitors = initialize_monitors()
    
    # Auto-refresh dashboard
    refresh_interval = settings.get_refresh_interval()
    if refresh_interval > 0:
        st_autorefresh(interval=refresh_interval * 1000, key="data_refresh")
    
    # Dashboard header
    render_header()
    
    # Sidebar navigation
    selected_dashboard = render_sidebar()
    
    # Render selected dashboard
    if selected_dashboard == "Ollama LLM":
        asyncio.run(ollama_dashboard.render_dashboard(monitors["ollama"], get_time_range()))
    elif selected_dashboard == "System":
        system_dashboard.render_dashboard(monitors["system"], get_time_range())
    elif selected_dashboard == "Python":
        python_dashboard.render_dashboard(monitors["python"], get_time_range())
    elif selected_dashboard == "Custom":
        custom_dashboard.render_dashboard(monitors["custom"], get_time_range())
    elif selected_dashboard == "Settings":
        render_settings_page(monitors)
    
    # Render footer
    render_footer()


def apply_custom_styles():
    """Apply custom CSS styling to the dashboard."""
    
    # Get theme
    theme = settings.get_theme()
    
    # Define CSS for light and dark themes
    if theme == "dark":
        background_color = "#121212"
        card_background = "#1E1E1E"
        text_color = "#E0E0E0"
        secondary_text_color = "#B0B0B0"
        border_color = "#333333"
    else:
        background_color = "#f5f7fa"
        card_background = "#FFFFFF"
        text_color = "#333333"
        secondary_text_color = "#666666"
        border_color = "#E0E0E0"
    
    # Apply the CSS
    st.markdown(f"""
    <style>
    .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    
    .stApp {{
        background-color: {background_color};
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color};
    }}
    
    .metric-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
    }}
    
    .metric-card {{
        background-color: {card_background};
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        flex: 1;
        min-width: 200px;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {card_background};
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {text_color};
    }}
    
    div.stButton > button:first-child {{
        background-color: #4e8df5;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }}
    
    div.stButton > button:hover {{
        background-color: #3a7bd5;
        color: white;
        border: none;
    }}
    
    [data-testid="StyledLinkIconContainer"] {{
        color: {secondary_text_color};
    }}
    
    h1 {{
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }}
    
    h2 {{
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }}
    
    h3 {{
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }}
    
    .dashboard-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        border-bottom: 1px solid {border_color};
        padding-bottom: 1rem;
    }}
    
    .dashboard-footer {{
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid {border_color};
        color: {secondary_text_color};
        font-size: 0.8rem;
        text-align: center;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the dashboard header."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("📊 OmniPulse Dashboard")
        st.caption("Enterprise LLM & System Monitoring Hub")
    
    with col2:
        st.text(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        if st.button("Refresh Now"):
            st.rerun()


def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.title("OmniPulse")
        
        # Navigation
        st.subheader("Dashboards")
        
        selected = st.radio(
            "Select Dashboard",
            options=["Ollama LLM", "System", "Python", "Custom", "Settings"],
            key="selected_dashboard",
            label_visibility="collapsed"
        )
        
        # Time range filter
        st.subheader("Filter")
        time_range = st.selectbox(
            "Time Range",
            options=["Last 15 minutes", "Last hour", "Last 3 hours", "Last day", "Last week"],
            index=1,
            key="time_range"
        )
        
        # About section
        st.sidebar.markdown("---")
        st.sidebar.subheader("About")
        st.sidebar.info(
            "OmniPulse v0.1.0\n\n"
            "Enterprise monitoring for Ollama, system resources, and Python applications."
        )
    
    return selected


def render_footer():
    """Render the dashboard footer."""
    st.markdown(
        '<div class="dashboard-footer">OmniPulse - Enterprise Monitoring Dashboard © 2025</div>',
        unsafe_allow_html=True
    )


def render_settings_page(monitors):
    """Render the settings page."""
    st.header("Dashboard Settings")
    
    # Create tabs for different settings categories
    tabs = st.tabs(["General", "Ollama", "System", "Python", "Custom"])
    
    # General Settings
    with tabs[0]:
        st.subheader("General Settings")
        
        # Theme
        current_theme = settings.get_theme()
        new_theme = st.selectbox(
            "Dashboard Theme",
            options=["light", "dark"],
            index=0 if current_theme == "light" else 1
        )
        
        # Refresh interval
        current_refresh = settings.get_refresh_interval()
        new_refresh = st.slider(
            "Dashboard Refresh Interval (seconds)",
            min_value=0,
            max_value=60,
            value=current_refresh,
            step=1
        )
        
        # Default time range
        current_time_range = settings.get_setting("dashboard_default_time_range")
        new_time_range = st.selectbox(
            "Default Time Range",
            options=["Last 15 minutes", "Last hour", "Last 3 hours", "Last day", "Last week"],
            index=1 if current_time_range == "Last hour" else 0
        )
        
        # Authentication
        enable_auth = st.checkbox(
            "Enable Authentication",
            value=settings.is_authentication_enabled()
        )
        
        auth_username = ""
        auth_password = ""
        
        if enable_auth:
            auth_username = st.text_input(
                "Username",
                value=settings.get_setting("auth_username", "")
            )
            
            auth_password = st.text_input(
                "Password",
                type="password"
            )
        
        # Save general settings
        if st.button("Save General Settings"):
            # Update settings
            settings.update_setting("dashboard_theme", new_theme)
            settings.update_setting("dashboard_refresh_interval", new_refresh)
            settings.update_setting("dashboard_default_time_range", new_time_range)
            settings.update_setting("enable_authentication", enable_auth)
            
            if enable_auth and auth_username:
                settings.update_setting("auth_username", auth_username)
                
                if auth_password:
                    import hashlib
                    password_hash = hashlib.sha256(auth_password.encode()).hexdigest()
                    settings.update_setting("auth_password_hash", password_hash)
            
            st.success("Settings saved successfully! Refresh the page to apply changes.")
    
    # Ollama Settings
    with tabs[1]:
        st.subheader("Ollama Settings")
        
        # API URL
        current_api_url = settings.get_ollama_api_url()
        new_api_url = st.text_input("Ollama API URL", value=current_api_url)
        
        # Test connection
        if st.button("Test Ollama Connection"):
            with st.spinner("Testing connection..."):
                try:
                    result = asyncio.run(monitors["ollama"].perform_health_check())
                    
                    if result.get("healthy", False):
                        st.success("✅ Connection successful! Ollama service is running.")
                    else:
                        st.error(f"❌ Connection failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")
        
        # Save Ollama settings
        if st.button("Save Ollama Settings"):
            settings.update_setting("ollama_api_url", new_api_url)
            monitors["ollama"].api_url = new_api_url
            st.success("Ollama settings saved successfully!")
    
    # System Monitor Settings
    with tabs[2]:
        st.subheader("System Monitor Settings")
        
        # Monitoring interval
        current_interval = settings.get_system_monitor_interval()
        new_interval = st.slider(
            "System Monitoring Interval (seconds)",
            min_value=1,
            max_value=60,
            value=current_interval,
            step=1
        )
        
        # Disk paths
        st.write("Disk Paths to Monitor")
        disk_paths = monitors["system"].disk_paths
        st.write(", ".join(disk_paths))
        
        # Network interfaces
        st.write("Network Interfaces to Monitor")
        network_interfaces = monitors["system"].network_interfaces
        st.write(", ".join(network_interfaces))
        
        # Save System settings
        if st.button("Save System Settings"):
            settings.update_setting("system_monitor_interval", new_interval)
            st.success("System monitor settings saved successfully!")
    
    # Python Monitor Settings
    with tabs[3]:
        st.subheader("Python Monitor Settings")
        
        # Monitoring interval
        current_interval = settings.get_python_monitor_interval()
        new_interval = st.slider(
            "Python Monitoring Interval (seconds)",
            min_value=1,
            max_value=60,
            value=current_interval,
            step=1
        )
        
        # Process filter
        current_filter = settings.get_setting("python_process_filter")
        new_filter = st.text_input(
            "Python Process Filter (comma-separated)",
            value=",".join(current_filter) if isinstance(current_filter, list) else current_filter
        )
        
        # Save Python settings
        if st.button("Save Python Settings"):
            settings.update_setting("python_monitor_interval", new_interval)
            
            # Parse filter
            if new_filter:
                filter_list = [item.strip() for item in new_filter.split(",")]
                settings.update_setting("python_process_filter", filter_list)
            
            st.success("Python monitor settings saved successfully!")
    
    # Custom Monitor Settings
    with tabs[4]:
        st.subheader("Custom Monitor Settings")
        
        # Config path
        current_path = settings.get_custom_monitors_config_path()
        new_path = st.text_input("Custom Monitors Config Path", value=current_path)
        
        # Show current custom monitors
        st.write("Current Custom Monitors")
        
        # Get monitor status
        monitor_status = monitors["custom"].get_all_metrics_status()
        
        if monitor_status:
            # Convert to DataFrame for display
            import pandas as pd
            
            status_data = []
            for name, status in monitor_status.items():
                status_data.append({
                    "Name": name,
                    "Type": status.get("type", "Unknown"),
                    "Enabled": "Yes" if status.get("enabled", False) else "No",
                    "Interval (sec)": status.get("interval", 0),
                    "Last Collection": status.get("last_collection", "Never"),
                    "Status": status.get("latest_status", "Unknown")
                })
            
            if status_data:
                st.dataframe(pd.DataFrame(status_data), use_container_width=True)
            else:
                st.info("No custom monitors configured yet.")
        else:
            st.info("No custom monitors configured yet.")
        
        # Add new monitor
        with st.expander("Add New Custom Monitor"):
            monitor_name = st.text_input("Monitor Name", key="new_monitor_name")
            monitor_type = st.selectbox(
                "Monitor Type",
                options=["http", "command", "script", "file", "function"],
                key="new_monitor_type"
            )
            
            # Different fields based on type
            if monitor_type == "http":
                url = st.text_input("URL", key="new_monitor_url")
                method = st.selectbox("Method", options=["GET", "POST", "PUT", "DELETE"], key="new_monitor_method")
                headers = st.text_area("Headers (JSON)", key="new_monitor_headers")
                
                config = {
                    "name": monitor_name,
                    "type": monitor_type,
                    "url": url,
                    "method": method,
                    "interval": 60,
                    "enabled": True
                }
                
                # Parse headers if provided
                if headers:
                    try:
                        import json
                        headers_dict = json.loads(headers)
                        config["headers"] = headers_dict
                    except:
                        st.error("Invalid JSON for headers")
            
            elif monitor_type == "command":
                command = st.text_input("Command", key="new_monitor_command")
                parser = st.selectbox("Parser", options=["text", "json"], key="new_monitor_parser")
                
                config = {
                    "name": monitor_name,
                    "type": monitor_type,
                    "command": command,
                    "parser": parser,
                    "interval": 60,
                    "enabled": True
                }
            
            elif monitor_type == "script":
                script_path = st.text_input("Script Path", key="new_monitor_script_path")
                args = st.text_input("Arguments (comma-separated)", key="new_monitor_args")
                
                config = {
                    "name": monitor_name,
                    "type": monitor_type,
                    "script_path": script_path,
                    "interval": 60,
                    "enabled": True
                }
                
                # Parse arguments if provided
                if args:
                    config["arguments"] = [arg.strip() for arg in args.split(",")]
            
            elif monitor_type == "file":
                file_path = st.text_input("File Path", key="new_monitor_file_path")
                parser = st.selectbox("Parser", options=["text", "json", "lines"], key="new_monitor_parser")
                
                config = {
                    "name": monitor_name,
                    "type": monitor_type,
                    "file_path": file_path,
                    "parser": parser,
                    "interval": 60,
                    "enabled": True
                }
            
            elif monitor_type == "function":
                module_path = st.text_input("Module Path", key="new_monitor_module_path")
                function_name = st.text_input("Function Name", key="new_monitor_function_name")
                
                config = {
                    "name": monitor_name,
                    "type": monitor_type,
                    "module_path": module_path,
                    "function_name": function_name,
                    "interval": 60,
                    "enabled": True
                }
            
            # Add monitor button
            if st.button("Add Monitor"):
                if not monitor_name:
                    st.error("Monitor name is required")
                else:
                    # Add the monitor
                    result = monitors["custom"].add_metric(config)
                    
                    if result:
                        st.success(f"Added custom monitor: {monitor_name}")
                    else:
                        st.error("Failed to add custom monitor")
        
        # Save Custom settings
        if st.button("Save Custom Settings"):
            settings.update_setting("custom_monitors_config_path", new_path)
            st.success("Custom monitor settings saved successfully!")


def get_time_range():
    """Get the current time range selection."""
    if "time_range" in st.session_state:
        return st.session_state.time_range
    else:
        return settings.get_setting("dashboard_default_time_range", "Last hour")


if __name__ == "__main__":
    main()