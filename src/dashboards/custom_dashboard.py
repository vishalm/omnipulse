"""
Custom dashboard module for user-defined metrics and visualizations.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.monitors.custom_monitor import CustomMonitor
from src.widgets.metric_cards import render_metric_card, render_metric_row
from src.widgets.charts import render_time_series, render_bar_chart, render_gauge_chart
from src.utils.helpers import format_number, format_bytes, format_percent, time_ago


def render_dashboard(monitor: Optional[CustomMonitor] = None, time_range: str = "Last hour"):
    """
    Render the custom monitoring dashboard.
    
    Args:
        monitor: Custom monitor instance
        time_range: Time range filter for metrics
    """
    st.subheader("Custom Monitoring Dashboard", anchor=False)
    
    # Initialize monitor if not provided
    if monitor is None:
        monitor = CustomMonitor()
    
    # Get metrics status
    try:
        metrics_status = monitor.get_all_metrics_status()
    except AttributeError:
        metrics_status = []
        st.warning("Custom metrics configuration is not available. Please check your monitor implementation.")
    
    # Check if any metrics are configured
    if not metrics_status:
        render_empty_dashboard(monitor)
        return
    
    # Collect latest metrics
    with st.spinner("Collecting metrics..."):
        metrics_data = st.session_state.get("custom_metrics_data", {})
        
        # Only fetch new data if not already in session state
        if not metrics_data:
            import asyncio
            metrics_data = asyncio.run(monitor.collect_all_metrics())
            st.session_state.custom_metrics_data = metrics_data
    
    # Tabs for different sections
    tabs = st.tabs(["Dashboard", "Metrics Explorer", "Configure"])
    
    # DASHBOARD TAB
    with tabs[0]:
        render_dashboard_tab(monitor, metrics_status, metrics_data, time_range)
    
    # METRICS EXPLORER TAB
    with tabs[1]:
        render_explorer_tab(monitor, metrics_status, time_range)
    
    # CONFIGURE TAB
    with tabs[2]:
        render_configure_tab(monitor, metrics_status)


def render_dashboard_tab(monitor: CustomMonitor, metrics_status: Dict[str, Dict[str, Any]], metrics_data: Dict[str, Dict[str, Any]], time_range: str):
    """Render the main dashboard tab with configurable widgets."""
    # Dashboard controls
    st.markdown("### Dashboard Configuration")
    
    # Check if dashboard config exists in session state
    if "custom_dashboard_config" not in st.session_state:
        # Default configuration
        st.session_state.custom_dashboard_config = {
            "title": "Custom Monitoring Dashboard",
            "refresh_interval": 60,
            "layout": [],
            "metrics": []
        }
    
    dashboard_config = st.session_state.custom_dashboard_config
    
    # Dashboard configuration expander
    with st.expander("Dashboard Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            dashboard_config["title"] = st.text_input(
                "Dashboard Title",
                value=dashboard_config.get("title", "Custom Monitoring Dashboard")
            )
        
        with col2:
            dashboard_config["refresh_interval"] = st.number_input(
                "Refresh Interval (seconds)",
                min_value=10,
                max_value=3600,
                value=dashboard_config.get("refresh_interval", 60),
                step=10
            )
    
    # Metric selection
    with st.expander("Add Metrics to Dashboard"):
        available_metrics = list(metrics_status.keys())
        
        if available_metrics:
            # Add a metric row
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                selected_metric = st.selectbox(
                    "Select Metric",
                    options=available_metrics
                )
            
            with col2:
                widget_type = st.selectbox(
                    "Display As",
                    options=["Metric Card", "Line Chart", "Bar Chart", "Gauge", "Table", "JSON"]
                )
            
            with col3:
                if st.button("Add to Dashboard"):
                    # Add to dashboard config
                    dashboard_config["metrics"].append({
                        "metric_name": selected_metric,
                        "widget_type": widget_type,
                        "title": metrics_status[selected_metric].get("name", selected_metric),
                        "position": len(dashboard_config["metrics"]) + 1,
                        "width": "half"  # half or full
                    })
                    
                    st.success(f"Added {selected_metric} to dashboard")
                    st.experimental_rerun()
        else:
            st.info("No metrics available. Configure custom metrics in the 'Configure' tab.")
    
    # Render the dashboard
    st.markdown(f"### {dashboard_config['title']}")
    
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if dashboard_config["metrics"]:
        # Create a list of metric widgets to display
        metrics_to_display = dashboard_config["metrics"]
        
        # Group metrics by rows (based on width)
        row_widgets = []
        current_row = []
        current_width = 0
        
        for metric in metrics_to_display:
            widget_width = 1 if metric["width"] == "half" else 2
            
            if current_width + widget_width > 2:
                # Start a new row
                row_widgets.append(current_row)
                current_row = [metric]
                current_width = widget_width
            else:
                # Add to current row
                current_row.append(metric)
                current_width += widget_width
        
        # Add the last row if not empty
        if current_row:
            row_widgets.append(current_row)
        
        # Render rows
        for row in row_widgets:
            cols = st.columns([1 if w["width"] == "half" else 2 for w in row])
            
            for i, widget in enumerate(row):
                with cols[i]:
                    render_metric_widget(
                        monitor,
                        widget["metric_name"],
                        widget["widget_type"],
                        widget.get("title", widget["metric_name"]),
                        metrics_data.get(widget["metric_name"], {}),
                        time_range
                    )
        
        # Dashboard controls
        with st.expander("Edit Dashboard"):
            st.write("Drag and drop to reorder widgets (coming soon)")
            
            # List current widgets with edit/remove buttons
            for i, metric in enumerate(dashboard_config["metrics"]):
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                
                with col1:
                    st.write(f"{i+1}. {metric['title']}")
                
                with col2:
                    new_width = st.selectbox(
                        "Width",
                        options=["half", "full"],
                        index=0 if metric["width"] == "half" else 1,
                        key=f"width_{i}"
                    )
                    dashboard_config["metrics"][i]["width"] = new_width
                
                with col3:
                    if st.button("↑", key=f"up_{i}") and i > 0:
                        # Move up
                        dashboard_config["metrics"][i], dashboard_config["metrics"][i-1] = dashboard_config["metrics"][i-1], dashboard_config["metrics"][i]
                        st.experimental_rerun()
                
                with col4:
                    if st.button("✖", key=f"remove_{i}"):
                        # Remove
                        dashboard_config["metrics"].pop(i)
                        st.experimental_rerun()
    else:
        st.info("No widgets added to the dashboard yet. Add metrics using the controls above.")
    
    # Save dashboard
    if st.button("Save Dashboard Configuration"):
        # Save to session state
        st.session_state.custom_dashboard_config = dashboard_config
        
        # TODO: Save to file
        
        st.success("Dashboard configuration saved!")


def render_explorer_tab(monitor: CustomMonitor, metrics_status: Dict[str, Dict[str, Any]], time_range: str):
    """Render the metrics explorer tab for detailed metric analysis."""
    st.markdown("### Metrics Explorer")
    
    # Metric selection
    available_metrics = list(metrics_status.keys())
    
    if not available_metrics:
        st.info("No metrics available. Configure custom metrics in the 'Configure' tab.")
        return
    
    selected_metric = st.selectbox(
        "Select Metric to Explore",
        options=available_metrics
    )
    
    # Get metric history
    metric_history = monitor.get_metric_history(selected_metric)
    
    if not metric_history:
        st.info(f"No data available for metric: {selected_metric}")
        return
    
    # Metric details
    st.markdown(f"#### {selected_metric}")
    
    # Metric type and status
    metric_type = metrics_status[selected_metric].get("type", "Unknown")
    metric_enabled = metrics_status[selected_metric].get("enabled", False)
    metric_interval = metrics_status[selected_metric].get("interval", 0)
    
    # Display metric metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Type", metric_type.capitalize())
    
    with col2:
        st.metric("Status", "Enabled" if metric_enabled else "Disabled")
    
    with col3:
        st.metric("Interval", f"{metric_interval}s")
    
    # Convert history to DataFrame
    if metric_history:
        try:
            # Extract basic data
            basic_data = []
            
            for entry in metric_history:
                item = {
                    "timestamp": entry.get("timestamp"),
                    "success": entry.get("success", False),
                    "duration_seconds": entry.get("duration_seconds", 0)
                }
                
                # Extract useful data if available
                if "data" in entry and isinstance(entry["data"], dict):
                    for key, value in entry["data"].items():
                        # Only include scalar values
                        if isinstance(value, (int, float, str, bool)) or value is None:
                            item[f"data_{key}"] = value
                
                if "error" in entry:
                    item["error"] = entry["error"]
                
                basic_data.append(item)
            
            # Create DataFrame
            df = pd.DataFrame(basic_data)
            
            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                # Apply time range filter
                df = filter_dataframe_by_time(df, time_range)
            
            if df.empty:
                st.info(f"No data available for metric: {selected_metric} in the selected time range")
                return
            
            # Plot main metric data
            st.markdown("#### Metric Data")
            
            # Determine which columns to plot
            numeric_cols = [col for col in df.columns if col.startswith("data_") and pd.api.types.is_numeric_dtype(df[col])]
            
            if numeric_cols:
                fig = px.line(
                    df,
                    x="timestamp",
                    y=numeric_cols,
                    title=f"{selected_metric} Metric Values",
                    labels={col: col.replace("data_", "") for col in numeric_cols}
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(t=30, b=50, l=80, r=30),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Plot performance data
            st.markdown("#### Performance Metrics")
            
            if "duration_seconds" in df.columns:
                fig = px.line(
                    df,
                    x="timestamp",
                    y="duration_seconds",
                    title=f"{selected_metric} Response Time",
                    labels={"duration_seconds": "Duration (seconds)", "timestamp": "Time"}
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=30, l=80, r=30)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Success rate
            if "success" in df.columns:
                # Calculate success rate over time
                df["success_int"] = df["success"].astype(int)
                
                # Group by hour
                df["hour"] = df["timestamp"].dt.floor("H")
                success_by_hour = df.groupby("hour").agg(
                    success_rate=("success_int", "mean"),
                    count=("success_int", "count")
                ).reset_index()
                
                success_by_hour["success_rate"] = success_by_hour["success_rate"] * 100
                
                fig = px.line(
                    success_by_hour,
                    x="hour",
                    y="success_rate",
                    title=f"{selected_metric} Success Rate",
                    labels={"success_rate": "Success Rate (%)", "hour": "Time"}
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=30, l=80, r=30),
                    yaxis_range=[0, 100]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Raw data table
            with st.expander("Raw Data"):
                st.dataframe(df, use_container_width=True)
            
            # Latest data
            st.markdown("#### Latest Data")
            
            if not df.empty:
                latest = df.iloc[-1].to_dict()
                
                # Format latest data
                formatted_data = {}
                
                for key, value in latest.items():
                    if key.startswith("data_"):
                        formatted_key = key.replace("data_", "")
                        formatted_data[formatted_key] = value
                
                if formatted_data:
                    st.json(formatted_data)
                else:
                    st.info("No structured data available in latest metric")
            
            # Run metric manually
            if st.button("Collect Metric Now"):
                with st.spinner(f"Collecting metric: {selected_metric}..."):
                    import asyncio
                    result = asyncio.run(monitor.collect_metric(selected_metric))
                    
                    if result.get("success", False):
                        st.success("Metric collected successfully!")
                        st.json(result.get("data", {}))
                    else:
                        st.error(f"Failed to collect metric: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Error processing metric data: {str(e)}")
    else:
        st.info(f"No data available for metric: {selected_metric}")


def render_configure_tab(monitor: CustomMonitor, metrics_status: Dict[str, Dict[str, Any]]):
    """Render the configuration tab for managing custom metrics."""
    st.markdown("### Configure Custom Metrics")
    
    # List current metrics
    st.subheader("Current Metrics", anchor=False)
    
    if metrics_status:
        # Convert to DataFrame for display
        metrics_data = []
        
        for name, status in metrics_status.items():
            # Format last collection time
            last_collection = status.get("last_collection", "Never")
            if last_collection != "Never":
                try:
                    last_collection = time_ago(last_collection)
                except:
                    pass
            
            metrics_data.append({
                "Name": name,
                "Type": status.get("type", "Unknown"),
                "Enabled": "✅" if status.get("enabled", False) else "❌",
                "Interval": f"{status.get('interval', 0)}s",
                "Last Collection": last_collection,
                "Status": status.get("latest_status", "Unknown")
            })
        
        # Create DataFrame
        df_metrics = pd.DataFrame(metrics_data)
        
        # Display table
        st.dataframe(df_metrics, use_container_width=True)
        
        # Metric details and editing
        st.subheader("Edit Metric", anchor=False)
        
        selected_metric = st.selectbox(
            "Select Metric to Edit",
            options=list(metrics_status.keys())
        )
        
        if selected_metric:
            metric_config = metrics_status[selected_metric]
            
            # Edit metric configuration
            with st.form(key="edit_metric_form"):
                st.write(f"Editing: {selected_metric}")
                
                # Enable/disable
                enabled = st.checkbox(
                    "Enabled",
                    value=metric_config.get("enabled", False)
                )
                
                # Interval
                interval = st.number_input(
                    "Collection Interval (seconds)",
                    min_value=10,
                    max_value=3600,
                    value=metric_config.get("interval", 60),
                    step=10
                )
                
                # Type-specific configuration
                metric_type = metric_config.get("type", "unknown")
                
                if metric_type == "http":
                    # HTTP settings
                    url = st.text_input(
                        "URL",
                        value=metric_config.get("url", "")
                    )
                    
                    method = st.selectbox(
                        "Method",
                        options=["GET", "POST", "PUT", "DELETE"],
                        index=0 if metric_config.get("method", "GET") == "GET" else 1
                    )
                    
                    headers_str = st.text_area(
                        "Headers (JSON)",
                        value=json.dumps(metric_config.get("headers", {}), indent=2) if metric_config.get("headers") else "{}"
                    )
                    
                    update_config = {
                        "enabled": enabled,
                        "interval": interval,
                        "url": url,
                        "method": method
                    }
                    
                    # Parse headers
                    try:
                        headers = json.loads(headers_str)
                        update_config["headers"] = headers
                    except:
                        st.error("Invalid JSON for headers")
                
                elif metric_type == "command":
                    # Command settings
                    command = st.text_input(
                        "Command",
                        value=metric_config.get("command", "")
                    )
                    
                    parser = st.selectbox(
                        "Output Parser",
                        options=["text", "json"],
                        index=0 if metric_config.get("parser", "text") == "text" else 1
                    )
                    
                    update_config = {
                        "enabled": enabled,
                        "interval": interval,
                        "command": command,
                        "parser": parser
                    }
                
                elif metric_type == "script":
                    # Script settings
                    script_path = st.text_input(
                        "Script Path",
                        value=metric_config.get("script_path", "")
                    )
                    
                    arguments = st.text_input(
                        "Arguments (comma-separated)",
                        value=",".join(metric_config.get("arguments", []))
                    )
                    
                    update_config = {
                        "enabled": enabled,
                        "interval": interval,
                        "script_path": script_path
                    }
                    
                    if arguments:
                        update_config["arguments"] = [arg.strip() for arg in arguments.split(",")]
                
                elif metric_type == "file":
                    # File settings
                    file_path = st.text_input(
                        "File Path",
                        value=metric_config.get("file_path", "")
                    )
                    
                    parser = st.selectbox(
                        "File Parser",
                        options=["text", "json", "lines"],
                        index=0 if metric_config.get("parser", "text") == "text" else 1
                    )
                    
                    update_config = {
                        "enabled": enabled,
                        "interval": interval,
                        "file_path": file_path,
                        "parser": parser
                    }
                
                elif metric_type == "function":
                    # Function settings
                    module_path = st.text_input(
                        "Module Path",
                        value=metric_config.get("module_path", "")
                    )
                    
                    function_name = st.text_input(
                        "Function Name",
                        value=metric_config.get("function_name", "")
                    )
                    
                    arguments_str = st.text_area(
                        "Arguments (JSON)",
                        value=json.dumps(metric_config.get("arguments", {}), indent=2) if metric_config.get("arguments") else "{}"
                    )
                    
                    update_config = {
                        "enabled": enabled,
                        "interval": interval,
                        "module_path": module_path,
                        "function_name": function_name
                    }
                    
                    # Parse arguments
                    try:
                        arguments = json.loads(arguments_str)
                        update_config["arguments"] = arguments
                    except:
                        st.error("Invalid JSON for arguments")
                
                else:
                    # Generic settings
                    update_config = {
                        "enabled": enabled,
                        "interval": interval
                    }
                
                # Submit button
                submit = st.form_submit_button("Update Metric")
                
                if submit:
                    # Update metric configuration
                    result = monitor.update_metric(selected_metric, update_config)
                    
                    if result:
                        st.success(f"Updated metric: {selected_metric}")
                    else:
                        st.error(f"Failed to update metric: {selected_metric}")
            
            # Delete metric button
            if st.button(f"Delete Metric: {selected_metric}"):
                # Confirm deletion
                if st.checkbox("Confirm deletion", key="confirm_delete"):
                    result = monitor.remove_metric(selected_metric)
                    
                    if result:
                        st.success(f"Deleted metric: {selected_metric}")
                        # Clear session state
                        if "custom_metrics_data" in st.session_state:
                            del st.session_state.custom_metrics_data
                        
                        # Rerun to refresh the page
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to delete metric: {selected_metric}")
    else:
        st.info("No custom metrics configured yet.")
    
    # Add new metric
    st.subheader("Add New Metric", anchor=False)
    
    with st.form(key="add_metric_form"):
        # Basic info
        metric_name = st.text_input("Metric Name")
        metric_type = st.selectbox(
            "Metric Type",
            options=["http", "command", "script", "file", "function"]
        )
        
        # Common settings
        enabled = st.checkbox("Enabled", value=True)
        interval = st.number_input(
            "Collection Interval (seconds)",
            min_value=10,
            max_value=3600,
            value=60,
            step=10
        )
        
        # Type-specific configuration
        if metric_type == "http":
            # HTTP settings
            url = st.text_input("URL")
            method = st.selectbox("Method", options=["GET", "POST", "PUT", "DELETE"])
            headers = st.text_area("Headers (JSON)", value="{}")
            
            config = {
                "name": metric_name,
                "type": metric_type,
                "enabled": enabled,
                "interval": interval,
                "url": url,
                "method": method
            }
            
            # Parse headers
            try:
                headers_dict = json.loads(headers)
                config["headers"] = headers_dict
            except:
                st.error("Invalid JSON for headers")
        
        elif metric_type == "command":
            # Command settings
            command = st.text_input("Command")
            parser = st.selectbox("Output Parser", options=["text", "json"])
            
            config = {
                "name": metric_name,
                "type": metric_type,
                "enabled": enabled,
                "interval": interval,
                "command": command,
                "parser": parser
            }
        
        elif metric_type == "script":
            # Script settings
            script_path = st.text_input("Script Path")
            arguments = st.text_input("Arguments (comma-separated)")
            
            config = {
                "name": metric_name,
                "type": metric_type,
                "enabled": enabled,
                "interval": interval,
                "script_path": script_path
            }
            
            if arguments:
                config["arguments"] = [arg.strip() for arg in arguments.split(",")]
        
        elif metric_type == "file":
            # File settings
            file_path = st.text_input("File Path")
            parser = st.selectbox("File Parser", options=["text", "json", "lines"])
            
            config = {
                "name": metric_name,
                "type": metric_type,
                "enabled": enabled,
                "interval": interval,
                "file_path": file_path,
                "parser": parser
            }
        
        elif metric_type == "function":
            # Function settings
            module_path = st.text_input("Module Path")
            function_name = st.text_input("Function Name")
            arguments = st.text_area("Arguments (JSON)", value="{}")
            
            config = {
                "name": metric_name,
                "type": metric_type,
                "enabled": enabled,
                "interval": interval,
                "module_path": module_path,
                "function_name": function_name
            }
            
            # Parse arguments
            try:
                arguments_dict = json.loads(arguments)
                config["arguments"] = arguments_dict
            except:
                st.error("Invalid JSON for arguments")
        
        # Submit button
        submit = st.form_submit_button("Add Metric")
        
        if submit:
            if not metric_name:
                st.error("Metric name is required")
            else:
                # Add the metric
                result = monitor.add_metric(config)
                
                if result:
                    st.success(f"Added metric: {metric_name}")
                    # Clear session state
                    if "custom_metrics_data" in st.session_state:
                        del st.session_state.custom_metrics_data
                    
                    # Rerun to refresh the page
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to add metric: {metric_name}")


def render_empty_dashboard(monitor: CustomMonitor):
    """Render an empty dashboard with instructions for adding metrics."""
    st.info("No custom metrics configured yet. Add metrics in the 'Configure' tab.")
    
    # Quick setup section
    st.markdown("### Quick Setup")
    
    st.markdown("""
    This dashboard allows you to monitor custom metrics from various sources:
    
    - **HTTP APIs**: Monitor web services, REST APIs, or any HTTP endpoint
    - **Commands**: Run shell commands and monitor their output
    - **Scripts**: Execute Python scripts and collect their results
    - **Files**: Monitor file content for changes
    - **Functions**: Call custom Python functions for advanced monitoring
    
    To get started, switch to the 'Configure' tab and add your first metric.
    """)
    
    # Example metrics
    st.markdown("### Example Metrics")
    
    st.markdown("Here are some example metrics you can add:")
    
    with st.expander("System Load Average"):
        st.code("""
        Name: system_load
        Type: command
        Command: cat /proc/loadavg | awk '{print "{\"load_1m\":" $1 ",\"load_5m\":" $2 ",\"load_15m\":" $3 "}"}'
        Parser: json
        Interval: 60
        """)
    
    with st.expander("Disk Space Usage"):
        st.code("""
        Name: disk_usage
        Type: command
        Command: df -h / | tail -n 1 | awk '{print "{\"total\":\"" $2 "\",\"used\":\"" $3 "\",\"available\":\"" $4 "\",\"percent_used\":\"" $5 "\"}"}'
        Parser: json
        Interval: 300
        """)
    
    with st.expander("Memory Usage"):
        st.code("""
        Name: memory_usage
        Type: command
        Command: free -m | grep Mem | awk '{print "{\"total\":" $2 ",\"used\":" $3 ",\"free\":" $4 ",\"percent_used\":" ($3/$2*100) "}"}'
        Parser: json
        Interval: 60
        """)
    
    with st.expander("External API Status"):
        st.code("""
        Name: api_status
        Type: http
        URL: https://httpstat.us/200
        Method: GET
        Interval: 60
        """)


def render_metric_widget(monitor: CustomMonitor, metric_name: str, widget_type: str, title: str, metric_data: Dict[str, Any], time_range: str):
    """Render a metric widget based on the specified type."""
    # Placeholder for empty widget
    if not metric_data:
        st.markdown(f"### {title}")
        st.info(f"No data available for metric: {metric_name}")
        return
    
    # Get metric history
    metric_history = monitor.get_metric_history(metric_name)
    
    if not metric_history:
        st.markdown(f"### {title}")
        st.info(f"No history available for metric: {metric_name}")
        return
    
    # Render the appropriate widget based on type
    if widget_type == "Metric Card":
        st.markdown(f"### {title}")
        
        # Extract useful data from latest metric
        if "data" in metric_data and isinstance(metric_data["data"], dict):
            # Find the first numeric value to display
            value_to_display = None
            label_to_display = None
            
            for key, value in metric_data["data"].items():
                if isinstance(value, (int, float)):
                    value_to_display = value
                    label_to_display = key
                    break
            
            if value_to_display is not None:
                # Format value based on magnitude
                if abs(value_to_display) >= 1000000:
                    display_value = f"{value_to_display/1000000:.2f}M"
                elif abs(value_to_display) >= 1000:
                    display_value = f"{value_to_display/1000:.2f}K"
                else:
                    display_value = f"{value_to_display:.2f}" if isinstance(value_to_display, float) else str(value_to_display)
                
                # Get trend data
                if len(metric_history) > 1:
                    # Find previous value
                    prev_data = metric_history[-2].get("data", {})
                    prev_value = prev_data.get(label_to_display, None)
                    
                    if prev_value is not None and isinstance(prev_value, (int, float)):
                        # Calculate change
                        change = value_to_display - prev_value
                        render_metric_card(
                            title=title,
                            value=display_value,
                            description=f"{label_to_display}",
                            delta=change,
                            icon="📊"
                        )
                    else:
                        render_metric_card(
                            title=title,
                            value=display_value,
                            description=f"{label_to_display}",
                            icon="📊"
                        )
                else:
                    render_metric_card(
                        title=title,
                        value=display_value,
                        description=f"{label_to_display}",
                        icon="📊"
                    )
            else:
                # Just show generic status
                status = "Success" if metric_data.get("success", False) else "Failed"
                render_metric_card(
                    title=title,
                    value=status,
                    description=f"Last updated: {time_ago(metric_data.get('timestamp', ''))}",
                    icon="📊"
                )
        else:
            # Just show generic status
            status = "Success" if metric_data.get("success", False) else "Failed"
            render_metric_card(
                title=title,
                value=status,
                description=f"Last updated: {time_ago(metric_data.get('timestamp', ''))}",
                icon="📊"
            )
    
    elif widget_type == "Line Chart":
        st.markdown(f"### {title}")
        
        try:
            # Convert history to DataFrame
            chart_data = []
            
            for entry in metric_history:
                if "data" in entry and isinstance(entry["data"], dict):
                    item = {
                        "timestamp": entry.get("timestamp")
                    }
                    
                    # Extract numeric values
                    for key, value in entry["data"].items():
                        if isinstance(value, (int, float)):
                            item[key] = value
                    
                    if len(item) > 1:  # Ensure we have at least one metric value
                        chart_data.append(item)
            
            if chart_data:
                df = pd.DataFrame(chart_data)
                
                # Convert timestamp to datetime
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    
                    # Apply time range filter
                    df = filter_dataframe_by_time(df, time_range)
                
                if df.empty:
                    st.info(f"No data available for the selected time range")
                    return
                
                # Determine which columns to plot
                numeric_cols = [col for col in df.columns if col != "timestamp" and pd.api.types.is_numeric_dtype(df[col])]
                
                if numeric_cols:
                    fig = px.line(
                        df,
                        x="timestamp",
                        y=numeric_cols,
                        title=f"{title} - Time Series",
                        labels={"timestamp": "Time"}
                    )
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(t=30, b=50, l=80, r=30),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No numeric data available for chart")
            else:
                st.info("No numeric data available for chart")
        
        except Exception as e:
            st.error(f"Error rendering chart: {str(e)}")
    
    elif widget_type == "Bar Chart":
        st.markdown(f"### {title}")
        
        try:
            # Extract the latest data
            if "data" in metric_data and isinstance(metric_data["data"], dict):
                data = metric_data["data"]
                
                # Convert to DataFrame
                bar_data = []
                
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        bar_data.append({
                            "key": key,
                            "value": value
                        })
                
                if bar_data:
                    df = pd.DataFrame(bar_data)
                    
                    # Create bar chart
                    fig = px.bar(
                        df,
                        x="key",
                        y="value",
                        title=f"{title} - Current Values",
                        labels={"key": "Metric", "value": "Value"}
                    )
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(t=30, b=50, l=80, r=30)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No numeric data available for chart")
            else:
                st.info("No structured data available for chart")
        
        except Exception as e:
            st.error(f"Error rendering chart: {str(e)}")
    
    elif widget_type == "Gauge":
        st.markdown(f"### {title}")
        
        try:
            # Extract the latest data
            if "data" in metric_data and isinstance(metric_data["data"], dict):
                data = metric_data["data"]
                
                # Find the first numeric value to display
                value_to_display = None
                label_to_display = None
                
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        value_to_display = value
                        label_to_display = key
                        break
                
                if value_to_display is not None:
                    # Determine reasonable min/max based on value
                    if value_to_display <= 1 and value_to_display >= 0:
                        # Probably a percentage in decimal form
                        min_val = 0
                        max_val = 1
                        display_val = value_to_display
                        suffix = ""
                    elif value_to_display <= 100 and value_to_display >= 0:
                        # Probably a percentage
                        min_val = 0
                        max_val = 100
                        display_val = value_to_display
                        suffix = "%"
                    else:
                        # Some other value
                        order = 10 ** (len(str(int(abs(value_to_display)))) - 1)
                        min_val = 0
                        max_val = max(100, order * 10)
                        display_val = value_to_display
                        suffix = ""
                    
                    render_gauge_chart(
                        value=display_val,
                        title=f"{title} - {label_to_display}",
                        min_value=min_val,
                        max_value=max_val,
                        suffix=suffix
                    )
                else:
                    st.info("No numeric data available for gauge")
            else:
                st.info("No structured data available for gauge")
        
        except Exception as e:
            st.error(f"Error rendering gauge: {str(e)}")
    
    elif widget_type == "Table":
        st.markdown(f"### {title}")
        
        try:
            # Extract the latest data
            if "data" in metric_data and isinstance(metric_data["data"], dict):
                data = metric_data["data"]
                
                # Convert to DataFrame
                df = pd.DataFrame([data])
                
                # Transpose for better display
                df_display = df.T.reset_index()
                df_display.columns = ["Metric", "Value"]
                
                # Display table
                st.dataframe(df_display, use_container_width=True)
            else:
                st.info("No structured data available for table")
        
        except Exception as e:
            st.error(f"Error rendering table: {str(e)}")
    
    elif widget_type == "JSON":
        st.markdown(f"### {title}")
        
        try:
            # Extract the latest data
            if "data" in metric_data:
                data = metric_data["data"]
                
                # Display as JSON
                st.json(data)
            else:
                st.info("No structured data available")
        
        except Exception as e:
            st.error(f"Error rendering JSON: {str(e)}")
    
    else:
        st.markdown(f"### {title}")
        st.info(f"Unsupported widget type: {widget_type}")


def filter_dataframe_by_time(df: pd.DataFrame, time_range: str) -> pd.DataFrame:
    """Filter DataFrame based on time range."""
    if "timestamp" not in df.columns:
        return df
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    now = pd.Timestamp.now()
    
    if time_range == "Last 15 minutes":
        cutoff = now - pd.Timedelta(minutes=15)
    elif time_range == "Last hour":
        cutoff = now - pd.Timedelta(hours=1)
    elif time_range == "Last 3 hours":
        cutoff = now - pd.Timedelta(hours=3)
    elif time_range == "Last day":
        cutoff = now - pd.Timedelta(days=1)
    elif time_range == "Last week":
        cutoff = now - pd.Timedelta(days=7)
    else:
        return df
    
    return df[df["timestamp"] >= cutoff]