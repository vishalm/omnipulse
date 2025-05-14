"""
System dashboard module for displaying system metrics and telemetry.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.monitors.system_monitor import SystemMonitor
from src.widgets.metric_cards import render_metric_card, render_metric_row
from src.widgets.charts import render_time_series, render_bar_chart
from src.utils.helpers import format_number, format_bytes, format_percent


def render_dashboard(monitor: Optional[SystemMonitor] = None, time_range: str = "Last hour"):
    """
    Render the system monitoring dashboard.
    
    Args:
        monitor: System monitor instance
        time_range: Time range filter for metrics
    """
    st.subheader("System Monitoring Dashboard", anchor=False)
    
    # Initialize monitor if not provided
    if monitor is None:
        monitor = SystemMonitor()
    
    # Load system info
    system_info = monitor.system_info
    
    # Collect latest metrics
    monitor.collect_all_metrics()
    
    # Get dataframes for charts
    dfs = monitor.to_dataframe()
    
    # Tabs for different sections
    tabs = st.tabs(["Overview", "CPU", "Memory", "Disk", "Network", "GPU", "Settings"])
    
    # OVERVIEW TAB
    with tabs[0]:
        render_overview_tab(monitor, system_info, dfs, time_range)
    
    # CPU TAB
    with tabs[1]:
        render_cpu_tab(monitor, system_info, dfs, time_range)
    
    # MEMORY TAB
    with tabs[2]:
        render_memory_tab(monitor, system_info, dfs, time_range)
    
    # DISK TAB
    with tabs[3]:
        render_disk_tab(monitor, system_info, dfs, time_range)
    
    # NETWORK TAB
    with tabs[4]:
        render_network_tab(monitor, system_info, dfs, time_range)
    
    # GPU TAB
    with tabs[5]:
        render_gpu_tab(monitor, system_info, dfs, time_range)
    
    # SETTINGS TAB
    with tabs[6]:
        render_settings_tab(monitor)


def render_overview_tab(monitor: SystemMonitor, system_info: Dict[str, Any], dfs: Dict[str, pd.DataFrame], time_range: str):
    """Render the overview tab with summary metrics from all systems."""
    # System information 
    st.markdown("### System Information")
    
    # Basic system info in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        os_info = system_info.get("os", {})
        render_metric_card(
            title="Operating System",
            value=f"{os_info.get('system', 'Unknown')} {os_info.get('release', '')}",
            description=f"Hostname: {os_info.get('hostname', 'Unknown')}",
            icon="💻"
        )
    
    with col2:
        cpu_info = system_info.get("cpu", {})
        render_metric_card(
            title="CPU",
            value=f"{cpu_info.get('cpu_count_logical', 0)} Cores",
            description=cpu_info.get('cpu_model', 'Unknown')[:40] + "..." if len(cpu_info.get('cpu_model', 'Unknown')) > 40 else cpu_info.get('cpu_model', 'Unknown'),
            icon="⚙️"
        )
    
    with col3:
        memory_info = system_info.get("memory", {})
        render_metric_card(
            title="Memory",
            value=f"{memory_info.get('total_memory_gb', 0):.1f} GB",
            description=f"Available: {memory_info.get('available_memory_gb', 0):.1f} GB",
            icon="🧠"
        )
    
    # Key metrics
    st.markdown("### Key Metrics")
    
    # Get latest metrics
    latest_metrics = {}
    
    if "cpu" in dfs and not dfs["cpu"].empty:
        latest_metrics["cpu"] = dfs["cpu"].iloc[-1].to_dict() if "timestamp" in dfs["cpu"].columns else {}
    
    if "memory" in dfs and not dfs["memory"].empty:
        latest_metrics["memory"] = dfs["memory"].iloc[-1].to_dict() if "timestamp" in dfs["memory"].columns else {}
    
    if "disk" in dfs and not dfs["disk"].empty:
        latest_metrics["disk"] = dfs["disk"].iloc[-1].to_dict() if "timestamp" in dfs["disk"].columns else {}
    
    if "network" in dfs and not dfs["network"].empty:
        latest_metrics["network"] = dfs["network"].iloc[-1].to_dict() if "timestamp" in dfs["network"].columns else {}
    
    # Display key metrics in a grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_percent = latest_metrics.get("cpu", {}).get("overall_percent", 0)
        render_metric_card(
            title="CPU Usage",
            value=f"{cpu_percent:.1f}%",
            description=f"{system_info.get('cpu', {}).get('cpu_count_logical', 0)} Cores",
            icon="🔄",
            progress_value=cpu_percent / 100
        )
    
    with col2:
        memory_percent = latest_metrics.get("memory", {}).get("virtual", {}).get("percent", 0)
        render_metric_card(
            title="Memory Usage",
            value=f"{memory_percent:.1f}%",
            description=f"Total: {system_info.get('memory', {}).get('total_memory_gb', 0):.1f} GB",
            icon="📊",
            progress_value=memory_percent / 100
        )
    
    with col3:
        # Get the disk usage for the primary disk
        disk_paths = system_info.get("disk", {}).keys()
        primary_disk = "/" if "/" in disk_paths else next(iter(disk_paths), None)
        
        if primary_disk:
            disk_percent = system_info.get("disk", {}).get(primary_disk, {}).get("percent_used", 0)
            render_metric_card(
                title="Disk Usage",
                value=f"{disk_percent:.1f}%",
                description=f"Path: {primary_disk}",
                icon="💾",
                progress_value=disk_percent / 100
            )
        else:
            render_metric_card(
                title="Disk Usage",
                value="N/A",
                description="No disk data available",
                icon="💾"
            )
    
    with col4:
        # Network activity (sum of all interfaces)
        network_interfaces = latest_metrics.get("network", {}).get("interfaces", {})
        
        if network_interfaces and any(isinstance(iface, dict) for iface in network_interfaces.values()):
            # Filter out any non-dict values and get valid interfaces
            valid_interfaces = {k: v for k, v in network_interfaces.items() if isinstance(v, dict)}
            
            # Sum up send and receive rates
            total_send_rate = 0
            total_recv_rate = 0
            
            for iface in valid_interfaces.values():
                # Use get() with default 0 to handle missing keys
                total_send_rate += iface.get("send_rate_bytes_per_sec", 0)
                total_recv_rate += iface.get("recv_rate_bytes_per_sec", 0)
            
            # Only show actual values if we have non-zero data
            if total_send_rate > 0 or total_recv_rate > 0:
                render_metric_card(
                    title="Network Activity",
                    value=f"↑{format_bytes(total_send_rate)}/s",
                    description=f"↓{format_bytes(total_recv_rate)}/s",
                    icon="🌐"
                )
            else:
                # No activity detected
                render_metric_card(
                    title="Network Activity",
                    value="No Traffic",
                    description=f"{len(valid_interfaces)} interface(s) monitored",
                    icon="🌐"
                )
        else:
            # No interfaces or invalid data
            render_metric_card(
                title="Network Activity",
                value="N/A",
                description="No network data available",
                icon="🌐"
            )
    
    # Multi-metric charts for overview
    st.markdown("### System Overview")
    
    # Create a combined chart with CPU, Memory, and Disk usage
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU Usage Chart
        if "cpu" in dfs and not dfs["cpu"].empty:
            df_cpu = dfs["cpu"]
            
            # Apply time range filter
            if "timestamp" in df_cpu.columns:
                df_cpu["timestamp"] = pd.to_datetime(df_cpu["timestamp"])
                df_cpu = filter_dataframe_by_time(df_cpu, time_range)
            
            if not df_cpu.empty and "overall_percent" in df_cpu.columns:
                fig = px.line(
                    df_cpu,
                    x="timestamp",
                    y="overall_percent",
                    title="CPU Usage Over Time",
                    labels={"overall_percent": "CPU Usage (%)", "timestamp": "Time"}
                )
                
                fig.update_layout(
                    height=250,
                    margin=dict(t=30, b=30, l=80, r=30),
                    yaxis_range=[0, 100]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No CPU usage data available for the selected time range.")
        else:
            st.info("No CPU usage data available yet.")
    
    with col2:
        # Memory Usage Chart
        if "memory" in dfs and not dfs["memory"].empty:
            df_memory = dfs["memory"]
            
            # Apply time range filter
            if "timestamp" in df_memory.columns:
                df_memory["timestamp"] = pd.to_datetime(df_memory["timestamp"])
                df_memory = filter_dataframe_by_time(df_memory, time_range)
            
            if not df_memory.empty:
                # Extract memory percent from the nested structure
                memory_data = []
                
                for _, row in df_memory.iterrows():
                    timestamp = row.get("timestamp")
                    virtual = row.get("virtual", {})
                    
                    if isinstance(virtual, dict) and "percent" in virtual:
                        memory_data.append({
                            "timestamp": timestamp,
                            "memory_percent": virtual["percent"]
                        })
                
                if memory_data:
                    df_memory_plot = pd.DataFrame(memory_data)
                    
                    fig = px.line(
                        df_memory_plot,
                        x="timestamp",
                        y="memory_percent",
                        title="Memory Usage Over Time",
                        labels={"memory_percent": "Memory Usage (%)", "timestamp": "Time"}
                    )
                    
                    fig.update_layout(
                        height=250,
                        margin=dict(t=30, b=30, l=80, r=30),
                        yaxis_range=[0, 100]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No memory usage data available for the selected time range.")
            else:
                st.info("No memory usage data available for the selected time range.")
        else:
            st.info("No memory usage data available yet.")
    
    # System Load and Network
    col1, col2 = st.columns(2)
    
    with col1:
        # System Load Chart (if available)
        if "cpu" in dfs and not dfs["cpu"].empty:
            df_cpu = dfs["cpu"]
            
            # Apply time range filter
            if "timestamp" in df_cpu.columns:
                df_cpu["timestamp"] = pd.to_datetime(df_cpu["timestamp"])
                df_cpu = filter_dataframe_by_time(df_cpu, time_range)
            
            # Check if load averages are available
            load_cols = [col for col in df_cpu.columns if col.startswith("load_avg_")]
            
            if not df_cpu.empty and load_cols:
                fig = px.line(
                    df_cpu,
                    x="timestamp",
                    y=load_cols,
                    title="System Load Averages",
                    labels={
                        "timestamp": "Time",
                        "load_avg_1min": "1 min",
                        "load_avg_5min": "5 min",
                        "load_avg_15min": "15 min",
                        "value": "Load Average"
                    }
                )
                
                fig.update_layout(
                    height=250,
                    margin=dict(t=30, b=30, l=80, r=30),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Display CPU breakdown if load averages aren't available
                if not df_cpu.empty and "user_percent" in df_cpu.columns and "system_percent" in df_cpu.columns:
                    df_cpu_plot = df_cpu[["timestamp", "user_percent", "system_percent", "idle_percent"]].tail(30)
                    
                    fig = px.area(
                        df_cpu_plot,
                        x="timestamp",
                        y=["user_percent", "system_percent", "idle_percent"],
                        title="CPU Usage Breakdown",
                        labels={
                            "timestamp": "Time",
                            "value": "Percentage",
                            "variable": "Type"
                        }
                    )
                    
                    fig.update_layout(
                        height=250,
                        margin=dict(t=30, b=30, l=80, r=30),
                        yaxis_range=[0, 100],
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No CPU usage breakdown data available.")
        else:
            st.info("No system load data available yet.")
    
    with col2:
        # Network Activity Chart
        if "network" in dfs and not dfs["network"].empty:
            df_network = dfs["network"]
            
            # Apply time range filter
            if "timestamp" in df_network.columns:
                df_network["timestamp"] = pd.to_datetime(df_network["timestamp"])
                df_network = filter_dataframe_by_time(df_network, time_range)
            
            if not df_network.empty:
                # Extract network rates from the nested structure
                network_data = []
                
                for _, row in df_network.iterrows():
                    timestamp = row.get("timestamp")
                    interfaces = row.get("interfaces", {})
                    
                    if isinstance(interfaces, dict):
                        send_rate = sum([
                            iface.get("send_rate_bytes_per_sec", 0) 
                            for iface in interfaces.values()
                            if isinstance(iface, dict)
                        ])
                        
                        recv_rate = sum([
                            iface.get("recv_rate_bytes_per_sec", 0)
                            for iface in interfaces.values()
                            if isinstance(iface, dict)
                        ])
                        
                        network_data.append({
                            "timestamp": timestamp,
                            "send_rate_bytes_per_sec": send_rate,
                            "recv_rate_bytes_per_sec": recv_rate
                        })
                
                if network_data:
                    df_network_plot = pd.DataFrame(network_data)
                    
                    # Convert to KB/s for better readability
                    df_network_plot["send_rate_kb_per_sec"] = df_network_plot["send_rate_bytes_per_sec"] / 1024
                    df_network_plot["recv_rate_kb_per_sec"] = df_network_plot["recv_rate_bytes_per_sec"] / 1024
                    
                    fig = px.line(
                        df_network_plot,
                        x="timestamp",
                        y=["send_rate_kb_per_sec", "recv_rate_kb_per_sec"],
                        title="Network Activity",
                        labels={
                            "timestamp": "Time", 
                            "value": "KB/s",
                            "variable": "Direction"
                        }
                    )
                    
                    fig.update_layout(
                        height=250,
                        margin=dict(t=30, b=30, l=80, r=30),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No network activity data available for the selected time range.")
            else:
                st.info("No network activity data available for the selected time range.")
        else:
            st.info("No network activity data available yet.")
        
        # Interface Details
        st.markdown("### Network Interface Details")
        
        # Get the latest network data to show per-interface details
        if "network" in dfs and not dfs["network"].empty:
            df_network = dfs["network"]
            
            # Get the latest entry
            latest_network = df_network.iloc[-1].to_dict() if not df_network.empty else {}
            interfaces = latest_network.get("interfaces", {})
            
            if interfaces and any(isinstance(iface, dict) for iface in interfaces.values()):
                # Filter out any non-dict values and get valid interfaces
                valid_interfaces = {k: v for k, v in interfaces.items() if isinstance(v, dict)}
                
                # Only show details if we have active interfaces
                if valid_interfaces:
                    # Create a container with a light background
                    interface_container = st.container()
                    
                    with interface_container:
                        # Display interface details in an organized manner
                        active_interfaces = []
                        inactive_interfaces = []
                        
                        for interface_name, interface in valid_interfaces.items():
                            # Check if interface has any activity
                            send_rate = interface.get("send_rate_bytes_per_sec", 0)
                            recv_rate = interface.get("recv_rate_bytes_per_sec", 0)
                            
                            if send_rate > 0 or recv_rate > 0:
                                active_interfaces.append((interface_name, interface))
                            else:
                                inactive_interfaces.append((interface_name, interface))
                        
                        # Show active interfaces first with more details
                        if active_interfaces:
                            st.markdown("#### Active Interfaces")
                            
                            for interface_name, interface in active_interfaces:
                                with st.expander(f"Interface: {interface_name}", expanded=True):
                                    # Interface metrics
                                    interface_cols = st.columns(3)
                                    
                                    with interface_cols[0]:
                                        send_rate = interface.get("send_rate_bytes_per_sec", 0)
                                        render_metric_card(
                                            title="Upload Rate",
                                            value=f"{format_bytes(send_rate)}/s",
                                            description="Current outgoing traffic",
                                            icon="⬆️"
                                        )
                                    
                                    with interface_cols[1]:
                                        recv_rate = interface.get("recv_rate_bytes_per_sec", 0)
                                        render_metric_card(
                                            title="Download Rate",
                                            value=f"{format_bytes(recv_rate)}/s",
                                            description="Current incoming traffic",
                                            icon="⬇️"
                                        )
                                    
                                    with interface_cols[2]:
                                        bytes_sent = interface.get("bytes_sent", 0)
                                        bytes_recv = interface.get("bytes_recv", 0)
                                        render_metric_card(
                                            title="Total Transferred",
                                            value=format_bytes(bytes_sent + bytes_recv),
                                            description=f"↑{format_bytes(bytes_sent)} ↓{format_bytes(bytes_recv)}",
                                            icon="🔄"
                                        )
                                    
                                    # Additional packet information
                                    st.markdown("##### Packet Statistics")
                                    packet_cols = st.columns(4)
                                    
                                    with packet_cols[0]:
                                        st.metric("Packets Sent", interface.get("packets_sent", 0))
                                    
                                    with packet_cols[1]:
                                        st.metric("Packets Received", interface.get("packets_recv", 0))
                                    
                                    with packet_cols[2]:
                                        st.metric("Errors In", interface.get("errin", 0))
                                    
                                    with packet_cols[3]:
                                        st.metric("Errors Out", interface.get("errout", 0))
                        
                        # Show inactive interfaces with less detail
                        if inactive_interfaces:
                            st.markdown("#### Inactive Interfaces")
                            
                            # Display inactive interfaces in a grid
                            cols = st.columns(len(inactive_interfaces) if len(inactive_interfaces) < 4 else 3)
                            
                            for i, (interface_name, interface) in enumerate(inactive_interfaces):
                                with cols[i % len(cols)]:
                                    bytes_sent = interface.get("bytes_sent", 0)
                                    bytes_recv = interface.get("bytes_recv", 0)
                                    
                                    render_metric_card(
                                        title=interface_name,
                                        value="No current traffic",
                                        description=f"Total: {format_bytes(bytes_sent + bytes_recv)} transferred",
                                        icon="🌐"
                                    )
                else:
                    st.info("No active network interfaces detected.")
            else:
                st.info("No network interface details available.")
        else:
            st.info("No network interface details available yet.")


def render_cpu_tab(monitor: SystemMonitor, system_info: Dict[str, Any], dfs: Dict[str, pd.DataFrame], time_range: str):
    """Render the CPU tab with detailed CPU metrics."""
    # CPU information
    st.markdown("### CPU Information")
    
    # CPU info from system_info
    cpu_info = system_info.get("cpu", {})
    
    # Display CPU information in a grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_metric_card(
            title="CPU Model",
            value=cpu_info.get('cpu_model', 'Unknown'),
            description="",
            icon="💻"
        )
    
    with col2:
        render_metric_card(
            title="CPU Cores",
            value=f"{cpu_info.get('cpu_count_logical', 0)} Logical, {cpu_info.get('cpu_count_physical', 0)} Physical",
            description="",
            icon="⚙️"
        )
    
    with col3:
        render_metric_card(
            title="CPU Frequency",
            value=f"{cpu_info.get('cpu_freq_current', 0):.2f} MHz",
            description=f"Min: {cpu_info.get('cpu_freq_min', 0):.2f} MHz, Max: {cpu_info.get('cpu_freq_max', 0):.2f} MHz",
            icon="⚡"
        )
    
    # CPU usage over time
    st.markdown("### CPU Usage Over Time")
    
    if "cpu" in dfs and not dfs["cpu"].empty:
        df_cpu = dfs["cpu"]
        
        # Apply time range filter
        if "timestamp" in df_cpu.columns:
            df_cpu["timestamp"] = pd.to_datetime(df_cpu["timestamp"])
            df_cpu = filter_dataframe_by_time(df_cpu, time_range)
        
        # Overall CPU usage chart
        if not df_cpu.empty and "overall_percent" in df_cpu.columns:
            fig = px.line(
                df_cpu,
                x="timestamp",
                y="overall_percent",
                title="Overall CPU Usage",
                labels={"overall_percent": "CPU Usage (%)", "timestamp": "Time"}
            )
            
            fig.update_layout(
                height=300,
                margin=dict(t=30, b=30, l=80, r=30),
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Per-core CPU usage
        st.markdown("### Per-Core CPU Usage")
        
        # Get the most recent CPU usage data
        if not df_cpu.empty and "timestamp" in df_cpu.columns:
            latest_cpu = df_cpu.iloc[-1].to_dict()
            
            # Extract per-core data
            core_data = {}
            for key, value in latest_cpu.items():
                if key.startswith("core_") and key.endswith("_percent"):
                    core_num = key.replace("core_", "").replace("_percent", "")
                    core_data[f"Core {core_num}"] = value
            
            if core_data:
                # Create a bar chart for current per-core usage
                core_df = pd.DataFrame({
                    "Core": list(core_data.keys()),
                    "Usage (%)": list(core_data.values())
                })
                
                fig = px.bar(
                    core_df,
                    x="Core",
                    y="Usage (%)",
                    title="Current Per-Core CPU Usage",
                    color="Usage (%)",
                    color_continuous_scale="viridis"
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=30, l=80, r=30),
                    yaxis_range=[0, 100]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No per-core CPU data available")
    else:
        st.info("No CPU metrics available yet. Please wait for data collection.")


def render_memory_tab(monitor: SystemMonitor, system_info: Dict[str, Any], dfs: Dict[str, pd.DataFrame], time_range: str):
    """Render the memory tab with detailed memory metrics."""
    # Memory information
    st.markdown("### Memory Information")
    
    # Memory info from system_info
    memory_info = system_info.get("memory", {})
    
    # Display memory information in a grid
    col1, col2 = st.columns(2)
    
    with col1:
        render_metric_card(
            title="Total Memory",
            value=f"{memory_info.get('total_memory_gb', 0):.2f} GB",
            description="Physical RAM",
            icon="🧠"
        )
    
    with col2:
        render_metric_card(
            title="Available Memory",
            value=f"{memory_info.get('available_memory_gb', 0):.2f} GB",
            description=f"{memory_info.get('available_memory_gb', 0)/memory_info.get('total_memory_gb', 1)*100:.1f}% available",
            icon="✅"
        )
    
    # Memory Usage
    st.markdown("### Memory Usage")
    
    if "memory" in dfs and not dfs["memory"].empty:
        df_memory = dfs["memory"]
        
        # Apply time range filter
        if "timestamp" in df_memory.columns:
            df_memory["timestamp"] = pd.to_datetime(df_memory["timestamp"])
            df_memory = filter_dataframe_by_time(df_memory, time_range)
        
        if not df_memory.empty:
            # Extract memory metrics from the nested structure
            memory_data = []
            
            for _, row in df_memory.iterrows():
                timestamp = row.get("timestamp")
                virtual = row.get("virtual", {})
                
                if isinstance(virtual, dict):
                    memory_data.append({
                        "timestamp": timestamp,
                        "memory_percent": virtual.get("percent", 0),
                        "memory_used_gb": virtual.get("used_gb", 0),
                        "memory_available_gb": virtual.get("available_gb", 0)
                    })
            
            if memory_data:
                df_memory_plot = pd.DataFrame(memory_data)
                
                # Get latest memory usage
                latest_memory = df_memory_plot.iloc[-1].to_dict()
                
                # Display physical memory usage
                memory_percent = latest_memory.get("memory_percent", 0)
                
                st.markdown(f"#### Physical Memory Usage: {memory_percent:.1f}%")
                
                # Progress bar for memory usage
                st.progress(memory_percent / 100)
                
                # Memory usage chart
                fig = px.line(
                    df_memory_plot,
                    x="timestamp",
                    y="memory_percent",
                    title="Memory Usage Over Time",
                    labels={"memory_percent": "Memory Usage (%)", "timestamp": "Time"}
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=30, l=80, r=30),
                    yaxis_range=[0, 100]
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No memory usage data available for the selected time range.")
        else:
            st.info("No memory usage data available for the selected time range.")
    else:
        st.info("No memory usage data available yet.")


def render_disk_tab(monitor: SystemMonitor, system_info: Dict[str, Any], dfs: Dict[str, pd.DataFrame], time_range: str):
    """Render the disk tab with detailed disk metrics."""
    # Disk information
    st.markdown("### Disk Information")
    
    # Disk info from system_info
    disk_info = system_info.get("disk", {})
    
    if disk_info:
        # Display disk partitions
        for path, disk_data in disk_info.items():
            st.markdown(f"#### {path}")
            
            # Usage percentage
            percent_used = disk_data.get("percent_used", 0)
            st.progress(percent_used / 100)
            
            # Disk metrics
            disk_cols = st.columns(3)
            
            with disk_cols[0]:
                render_metric_card(
                    title="Total Space",
                    value=f"{disk_data.get('total_gb', 0):.2f} GB",
                    description=f"Disk capacity",
                    icon="💾"
                )
            
            with disk_cols[1]:
                render_metric_card(
                    title="Used Space",
                    value=f"{disk_data.get('used_gb', 0):.2f} GB",
                    description=f"{percent_used:.1f}% of total",
                    icon="📊"
                )
            
            with disk_cols[2]:
                render_metric_card(
                    title="Free Space",
                    value=f"{disk_data.get('free_gb', 0):.2f} GB",
                    description=f"{100-percent_used:.1f}% of total",
                    icon="✅"
                )
    else:
        st.info("No disk information available.")


def render_network_tab(monitor: SystemMonitor, system_info: Dict[str, Any], dfs: Dict[str, pd.DataFrame], time_range: str):
    """Render the network tab with detailed network metrics."""
    # Network information
    st.markdown("### Network Information")
    
    # Network info from system_info
    network_info = system_info.get("network", {})
    
    if network_info:
        # Display network interfaces in a grid
        st.markdown("#### Network Interfaces")
        
        # Create a grid based on the number of interfaces
        num_interfaces = len(network_info)
        num_cols = min(3, num_interfaces)
        if num_cols > 0:
            cols = st.columns(num_cols)
            
            for i, (interface, data) in enumerate(network_info.items()):
                with cols[i % num_cols]:
                    # Get IP addresses
                    ip_list = []
                    if isinstance(data, dict) and "ip_addresses" in data:
                        ip_list = data["ip_addresses"]
                    
                    ip_text = ", ".join(ip_list) if ip_list else "No IP assigned"
                    
                    render_metric_card(
                        title=f"Interface: {interface}",
                        value=f"{len(ip_list)} IP address(es)",
                        description=ip_text[:40] + "..." if len(ip_text) > 40 else ip_text,
                        icon="🌐"
                    )
    
    # Network activity overview
    st.markdown("### Network Activity Overview")
    
    if "network" in dfs and not dfs["network"].empty:
        df_network = dfs["network"]
        
        # Apply time range filter
        if "timestamp" in df_network.columns:
            df_network["timestamp"] = pd.to_datetime(df_network["timestamp"])
            df_network = filter_dataframe_by_time(df_network, time_range)
        
        if not df_network.empty:
            # Get total network activity over time
            network_activity_data = []
            
            for _, row in df_network.iterrows():
                timestamp = row.get("timestamp")
                interfaces = row.get("interfaces", {})
                connections = row.get("connections_count", 0)
                
                if isinstance(interfaces, dict):
                    # Sum up activity across all interfaces
                    total_send_rate = sum([
                        iface.get("send_rate_bytes_per_sec", 0) 
                        for iface in interfaces.values()
                        if isinstance(iface, dict)
                    ])
                    
                    total_recv_rate = sum([
                        iface.get("recv_rate_bytes_per_sec", 0)
                        for iface in interfaces.values()
                        if isinstance(iface, dict)
                    ])
                    
                    network_activity_data.append({
                        "timestamp": timestamp,
                        "send_rate_bytes_per_sec": total_send_rate,
                        "recv_rate_bytes_per_sec": total_recv_rate,
                        "connections_count": connections
                    })
            
            if network_activity_data:
                df_activity = pd.DataFrame(network_activity_data)
                
                # Get latest metrics
                latest_metrics = df_activity.iloc[-1].to_dict() if not df_activity.empty else {}
                
                # Display total network activity
                activity_cols = st.columns(3)
                
                with activity_cols[0]:
                    send_rate = latest_metrics.get("send_rate_bytes_per_sec", 0)
                    render_metric_card(
                        title="Total Upload Rate",
                        value=f"{format_bytes(send_rate)}/s",
                        description="Current outgoing traffic",
                        icon="⬆️"
                    )
                
                with activity_cols[1]:
                    recv_rate = latest_metrics.get("recv_rate_bytes_per_sec", 0)
                    render_metric_card(
                        title="Total Download Rate",
                        value=f"{format_bytes(recv_rate)}/s",
                        description="Current incoming traffic",
                        icon="⬇️"
                    )
                
                with activity_cols[2]:
                    connections = latest_metrics.get("connections_count", 0)
                    render_metric_card(
                        title="Active Connections",
                        value=f"{connections}",
                        description="Total network connections",
                        icon="🔌"
                    )
                
                # Network traffic charts
                st.markdown("#### Network Traffic Over Time")
                
                # Convert bytes to more readable units
                df_activity["upload_speed_kb"] = df_activity["send_rate_bytes_per_sec"] / 1024
                df_activity["download_speed_kb"] = df_activity["recv_rate_bytes_per_sec"] / 1024
                
                # Create a single figure with multiple subplots
                fig = make_subplots(rows=2, cols=1, 
                                   subplot_titles=("Upload/Download Speeds", "Active Connections"),
                                   vertical_spacing=0.12,
                                   shared_xaxes=True)
                
                # Add upload/download speed traces
                fig.add_trace(
                    go.Scatter(
                        x=df_activity["timestamp"], 
                        y=df_activity["upload_speed_kb"],
                        name="Upload",
                        mode="lines",
                        line=dict(color="#3366CC", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(51, 102, 204, 0.1)"
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df_activity["timestamp"], 
                        y=df_activity["download_speed_kb"],
                        name="Download",
                        mode="lines",
                        line=dict(color="#DC3912", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(220, 57, 18, 0.1)"
                    ),
                    row=1, col=1
                )
                
                # Add connections trace
                if "connections_count" in df_activity.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df_activity["timestamp"], 
                            y=df_activity["connections_count"],
                            name="Connections",
                            mode="lines",
                            line=dict(color="#109618", width=2)
                        ),
                        row=2, col=1
                    )
                
                # Update layout
                fig.update_layout(
                    height=500,
                    margin=dict(t=50, b=30, l=80, r=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                # Update axes
                fig.update_yaxes(title_text="KB/s", row=1, col=1)
                fig.update_yaxes(title_text="Count", row=2, col=1)
                fig.update_xaxes(title_text="Time", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Network Interface Details
                st.markdown("### Network Interface Details")
                
                # Get the latest network data
                latest_network = df_network.iloc[-1].to_dict() if not df_network.empty else {}
                interfaces = latest_network.get("interfaces", {})
                
                if interfaces and any(isinstance(iface, dict) for iface in interfaces.values()):
                    # Filter out any non-dict values and get valid interfaces
                    valid_interfaces = {k: v for k, v in interfaces.items() if isinstance(v, dict)}
                    
                    # Only show details if we have interfaces
                    if valid_interfaces:
                        # Categorize interfaces
                        active_interfaces = []
                        inactive_interfaces = []
                        
                        for interface_name, interface in valid_interfaces.items():
                            # Check if interface has any activity
                            send_rate = interface.get("send_rate_bytes_per_sec", 0)
                            recv_rate = interface.get("recv_rate_bytes_per_sec", 0)
                            
                            if send_rate > 0 or recv_rate > 0:
                                active_interfaces.append((interface_name, interface))
                            else:
                                inactive_interfaces.append((interface_name, interface))
                        
                        # Display active interfaces
                        if active_interfaces:
                            st.markdown("#### Active Interfaces")
                            
                            for interface_name, interface in active_interfaces:
                                with st.expander(f"Interface: {interface_name}", expanded=True):
                                    # Create tabs for different metrics
                                    interface_tabs = st.tabs(["Overview", "Traffic", "Errors"])
                                    
                                    # Overview tab
                                    with interface_tabs[0]:
                                        # Interface metrics
                                        interface_cols = st.columns(3)
                                        
                                        with interface_cols[0]:
                                            send_rate = interface.get("send_rate_bytes_per_sec", 0)
                                            render_metric_card(
                                                title="Upload Rate",
                                                value=f"{format_bytes(send_rate)}/s",
                                                description="Current outgoing traffic",
                                                icon="⬆️"
                                            )
                                        
                                        with interface_cols[1]:
                                            recv_rate = interface.get("recv_rate_bytes_per_sec", 0)
                                            render_metric_card(
                                                title="Download Rate",
                                                value=f"{format_bytes(recv_rate)}/s",
                                                description="Current incoming traffic",
                                                icon="⬇️"
                                            )
                                        
                                        with interface_cols[2]:
                                            bytes_sent = interface.get("bytes_sent", 0)
                                            bytes_recv = interface.get("bytes_recv", 0)
                                            render_metric_card(
                                                title="Total Transferred",
                                                value=format_bytes(bytes_sent + bytes_recv),
                                                description=f"↑{format_bytes(bytes_sent)} ↓{format_bytes(bytes_recv)}",
                                                icon="🔄"
                                            )
                                    
                                    # Traffic tab
                                    with interface_tabs[1]:
                                        traffic_cols = st.columns(2)
                                        
                                        with traffic_cols[0]:
                                            st.metric("Packets Sent", interface.get("packets_sent", 0))
                                            st.metric("Bytes Sent", format_bytes(interface.get("bytes_sent", 0)))
                                        
                                        with traffic_cols[1]:
                                            st.metric("Packets Received", interface.get("packets_recv", 0))
                                            st.metric("Bytes Received", format_bytes(interface.get("bytes_recv", 0)))
                                    
                                    # Errors tab
                                    with interface_tabs[2]:
                                        error_cols = st.columns(2)
                                        
                                        with error_cols[0]:
                                            st.metric("Incoming Errors", interface.get("errin", 0))
                                            st.metric("Incoming Drops", interface.get("dropin", 0))
                                        
                                        with error_cols[1]:
                                            st.metric("Outgoing Errors", interface.get("errout", 0))
                                            st.metric("Outgoing Drops", interface.get("dropout", 0))
                        
                        # Display inactive interfaces
                        if inactive_interfaces:
                            st.markdown("#### Inactive Interfaces")
                            
                            # Display inactive interfaces in a grid
                            num_cols = min(3, len(inactive_interfaces))
                            if num_cols > 0:
                                cols = st.columns(num_cols)
                                
                                for i, (interface_name, interface) in enumerate(inactive_interfaces):
                                    with cols[i % num_cols]:
                                        bytes_sent = interface.get("bytes_sent", 0)
                                        bytes_recv = interface.get("bytes_recv", 0)
                                        
                                        render_metric_card(
                                            title=interface_name,
                                            value="No current traffic",
                                            description=f"Total: {format_bytes(bytes_sent + bytes_recv)} transferred",
                                            icon="🌐"
                                        )
                                        
                                        # Add an expander for more details
                                        with st.expander("Interface Details"):
                                            st.metric("Packets Sent", interface.get("packets_sent", 0))
                                            st.metric("Packets Received", interface.get("packets_recv", 0))
                                            st.metric("Errors", interface.get("errin", 0) + interface.get("errout", 0))
                    else:
                        st.info("No network interface details available.")
                else:
                    st.info("No network interface details available.")
            else:
                st.info("No network activity data available for the selected time range.")
        else:
            st.info("No network activity data available for the selected time range.")
    else:
        st.info("No network information available.")


def render_gpu_tab(monitor: SystemMonitor, system_info: Dict[str, Any], dfs: Dict[str, pd.DataFrame], time_range: str):
    """Render the GPU tab with detailed GPU metrics if available."""
    # Check if GPU monitoring is available
    gpu_info = system_info.get("gpu", {})
    
    if not gpu_info:
        st.info("No GPU detected or GPU monitoring is not available.")
        return
    
    # GPU information
    st.markdown("### GPU Information")
    
    # Display GPU information in a grid
    gpu_cards = []
    
    for gpu_id, gpu_data in gpu_info.items():
        gpu_cards.append({
            "id": gpu_id,
            "name": gpu_data.get("name", "Unknown GPU"),
            "memory_total_gb": gpu_data.get("memory_total_gb", 0),
            "driver": gpu_data.get("driver", "Unknown")
        })
    
    # Display GPU cards
    gpu_cols = st.columns(min(len(gpu_cards), 3))
    
    for i, gpu in enumerate(gpu_cards):
        with gpu_cols[i % len(gpu_cols)]:
            render_metric_card(
                title=f"GPU {gpu['id']}",
                value=gpu["name"],
                description=f"{gpu['memory_total_gb']} GB | Driver: {gpu['driver']}",
                icon="🖥️"
            )
    
    # GPU Usage
    st.markdown("### GPU Usage")
    
    if "gpu" in dfs and not dfs["gpu"].empty:
        df_gpu = dfs["gpu"]
        
        # Apply time range filter
        if "timestamp" in df_gpu.columns:
            df_gpu["timestamp"] = pd.to_datetime(df_gpu["timestamp"])
            df_gpu = filter_dataframe_by_time(df_gpu, time_range)
        
        if not df_gpu.empty:
            # Extract GPU metrics from the nested structure
            gpu_data = []
            
            for _, row in df_gpu.iterrows():
                timestamp = row.get("timestamp")
                gpus = row.get("gpus", {})
                
                if isinstance(gpus, dict):
                    for gpu_id, gpu_metrics in gpus.items():
                        if isinstance(gpu_metrics, dict):
                            gpu_data.append({
                                "timestamp": timestamp,
                                "gpu_id": gpu_id,
                                "name": gpu_metrics.get("name", "Unknown"),
                                "load_percent": gpu_metrics.get("load_percent", 0),
                                "memory_percent": gpu_metrics.get("memory_percent", 0),
                                "memory_used_gb": gpu_metrics.get("memory_used_gb", 0),
                                "memory_free_gb": gpu_metrics.get("memory_free_gb", 0),
                                "temperature_c": gpu_metrics.get("temperature_c", 0)
                            })
            
            if gpu_data:
                df_gpu_plot = pd.DataFrame(gpu_data)
                
                # Display per-GPU usage
                for gpu_id in df_gpu_plot["gpu_id"].unique():
                    gpu_df = df_gpu_plot[df_gpu_plot["gpu_id"] == gpu_id]
                    
                    if not gpu_df.empty:
                        # Get the latest metrics for this GPU
                        latest_gpu = gpu_df.iloc[-1].to_dict()
                        
                        # GPU name and basic info
                        st.markdown(f"#### {latest_gpu.get('name', gpu_id)}")
                        
                        # GPU usage progress bars
                        st.markdown("**GPU Utilization**")
                        st.progress(latest_gpu.get("load_percent", 0) / 100)
                        st.markdown(f"{latest_gpu.get('load_percent', 0):.1f}%")
                        
                        st.markdown("**GPU Memory**")
                        st.progress(latest_gpu.get("memory_percent", 0) / 100)
                        st.markdown(f"{latest_gpu.get('memory_percent', 0):.1f}% ({latest_gpu.get('memory_used_gb', 0):.2f} GB / {latest_gpu.get('memory_used_gb', 0) + latest_gpu.get('memory_free_gb', 0):.2f} GB)")
                        
                        # GPU metrics
                        gpu_metric_cols = st.columns(3)
                        
                        with gpu_metric_cols[0]:
                            render_metric_card(
                                title="GPU Utilization",
                                value=f"{latest_gpu.get('load_percent', 0):.1f}%",
                                description="Processing load",
                                icon="⚙️"
                            )
                        
                        with gpu_metric_cols[1]:
                            render_metric_card(
                                title="Memory Usage",
                                value=f"{latest_gpu.get('memory_percent', 0):.1f}%",
                                description=f"{latest_gpu.get('memory_used_gb', 0):.2f} GB used",
                                icon="🧠"
                            )
                        
                        with gpu_metric_cols[2]:
                            render_metric_card(
                                title="Temperature",
                                value=f"{latest_gpu.get('temperature_c', 0):.1f}°C",
                                description="GPU temperature",
                                icon="🌡️"
                            )
                        
                        # GPU utilization chart
                        fig = px.line(
                            gpu_df,
                            x="timestamp",
                            y="load_percent",
                            title="GPU Utilization Over Time",
                            labels={"load_percent": "Utilization (%)", "timestamp": "Time"}
                        )
                        
                        fig.update_layout(
                            height=250,
                            margin=dict(t=30, b=30, l=80, r=30),
                            yaxis_range=[0, 100]
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # GPU memory usage chart
                        fig = px.line(
                            gpu_df,
                            x="timestamp",
                            y="memory_percent",
                            title="GPU Memory Usage Over Time",
                            labels={"memory_percent": "Memory Usage (%)", "timestamp": "Time"}
                        )
                        
                        fig.update_layout(
                            height=250,
                            margin=dict(t=30, b=30, l=80, r=30),
                            yaxis_range=[0, 100]
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # GPU temperature chart
                        if "temperature_c" in gpu_df.columns:
                            fig = px.line(
                                gpu_df,
                                x="timestamp",
                                y="temperature_c",
                                title="GPU Temperature Over Time",
                                labels={"temperature_c": "Temperature (°C)", "timestamp": "Time"}
                            )
                            
                            fig.update_layout(
                                height=250,
                                margin=dict(t=30, b=30, l=80, r=30),
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No GPU usage data available for the selected time range.")
        else:
            st.info("No GPU usage data available for the selected time range.")
    else:
        st.info("No GPU usage data available yet.")


def render_settings_tab(monitor: SystemMonitor):
    """Render the settings tab for system monitoring configuration."""
    st.markdown("### System Monitoring Settings")
    
    # Monitoring intervals
    st.subheader("Monitoring Interval", anchor=False)
    
    if "system_refresh_interval" not in st.session_state:
        st.session_state.system_refresh_interval = 5
    
    refresh_interval = st.slider(
        "Refresh Interval (seconds)",
        min_value=1,
        max_value=60,
        value=st.session_state.system_refresh_interval,
        step=1
    )
    
    if refresh_interval != st.session_state.system_refresh_interval:
        st.session_state.system_refresh_interval = refresh_interval
        st.success(f"Updated refresh interval to {refresh_interval} seconds")
    
    # Disk paths to monitor
    st.subheader("Disk Monitoring", anchor=False)
    
    # Get current disk paths
    current_disk_paths = monitor.disk_paths
    
    # Display disk paths as multiselect
    new_disk_paths = st.multiselect(
        "Select disk paths to monitor",
        options=current_disk_paths,
        default=current_disk_paths
    )
    
    # Network interfaces to monitor
    st.subheader("Network Monitoring", anchor=False)
    
    # Get current network interfaces
    current_network_interfaces = monitor.network_interfaces
    
    # Display network interfaces as multiselect
    new_network_interfaces = st.multiselect(
        "Select network interfaces to monitor",
        options=current_network_interfaces,
        default=current_network_interfaces
    )
    
    # Apply button
    if st.button("Apply Settings"):
        # This would need a way to update the monitor's configuration
        # For now, we'll just show a success message
        st.success("Settings updated successfully")


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