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
        
        if network_interfaces:
            total_send_rate = sum([
                iface.get("send_rate_bytes_per_sec", 0) 
                for iface in network_interfaces.values()
            ])
            
            total_recv_rate = sum([
                iface.get("recv_rate_bytes_per_sec", 0)
                for iface in network_interfaces.values()
            ])
            
            render_metric_card(
                title="Network Activity",
                value=f"↑{format_bytes(total_send_rate)}/s",
                description=f"↓{format_bytes(total_recv_rate)}/s",
                icon="🌐"
            )
        else:
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
            value=f"{cpu_info.get('cpu_count_physical', 0)} Physical / {cpu_info.get('cpu_count_logical', 0)} Logical",
            description="Physical and logical cores",
            icon="⚙️"
        )
    
    with col3:
        freq = cpu_info.get('cpu_frequency', {})
        if freq and freq.get('current_mhz'):
            render_metric_card(
                title="CPU Frequency",
                value=f"{freq.get('current_mhz', 0)/1000:.2f} GHz",
                description=f"Min: {freq.get('min_mhz', 0)/1000:.2f} GHz, Max: {freq.get('max_mhz', 0)/1000:.2f} GHz" if freq.get('min_mhz') and freq.get('max_mhz') else "",
                icon="⚡"
            )
        else:
            render_metric_card(
                title="CPU Frequency",
                value="Unknown",
                description="",
                icon="⚡"
            )
    
    # CPU Usage
    st.markdown("### CPU Usage")
    
    if "cpu" in dfs and not dfs["cpu"].empty:
        df_cpu = dfs["cpu"]
        
        # Apply time range filter
        if "timestamp" in df_cpu.columns:
            df_cpu["timestamp"] = pd.to_datetime(df_cpu["timestamp"])
            df_cpu = filter_dataframe_by_time(df_cpu, time_range)
        
        if not df_cpu.empty:
            # Get latest CPU usage
            latest_cpu = df_cpu.iloc[-1].to_dict()
            
            # Display overall CPU usage
            overall_percent = latest_cpu.get("overall_percent", 0)
            
            st.markdown(f"#### Overall CPU Usage: {overall_percent:.1f}%")
            
            # Progress bar for overall CPU usage
            st.progress(overall_percent / 100)
            
            # CPU usage chart over time
            fig = px.line(
                df_cpu,
                x="timestamp",
                y="overall_percent",
                title="CPU Usage Over Time",
                labels={"overall_percent": "CPU Usage (%)", "timestamp": "Time"}
            )
            
            fig.update_layout(
                height=300,
                margin=dict(t=30, b=30, l=80, r=30),
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Per-CPU usage if available
            if "per_cpu_percent" in latest_cpu and isinstance(latest_cpu["per_cpu_percent"], list):
                st.markdown("#### Per-CPU Usage")
                
                per_cpu = latest_cpu["per_cpu_percent"]
                num_cpus = len(per_cpu)
                
                # Calculate number of rows based on number of CPUs
                cpus_per_row = 4
                num_rows = (num_cpus + cpus_per_row - 1) // cpus_per_row
                
                for row in range(num_rows):
                    cols = st.columns(cpus_per_row)
                    
                    for col_idx in range(cpus_per_row):
                        cpu_idx = row * cpus_per_row + col_idx
                        
                        if cpu_idx < num_cpus:
                            with cols[col_idx]:
                                cpu_percent = per_cpu[cpu_idx]
                                st.markdown(f"**CPU {cpu_idx}:** {cpu_percent:.1f}%")
                                st.progress(cpu_percent / 100)
            
            # CPU usage breakdown
            st.markdown("#### CPU Usage Breakdown")
            
            if "user_percent" in latest_cpu and "system_percent" in latest_cpu and "idle_percent" in latest_cpu:
                breakdown_cols = st.columns(3)
                
                with breakdown_cols[0]:
                    render_metric_card(
                        title="User",
                        value=f"{latest_cpu.get('user_percent', 0):.1f}%",
                        description="User processes",
                        icon="👤"
                    )
                
                with breakdown_cols[1]:
                    render_metric_card(
                        title="System",
                        value=f"{latest_cpu.get('system_percent', 0):.1f}%",
                        description="Kernel & system processes",
                        icon="🔧"
                    )
                
                with breakdown_cols[2]:
                    render_metric_card(
                        title="Idle",
                        value=f"{latest_cpu.get('idle_percent', 0):.1f}%",
                        description="Idle time",
                        icon="💤"
                    )
                
                # CPU usage breakdown chart
                fig = px.area(
                    df_cpu[["timestamp", "user_percent", "system_percent", "idle_percent"]],
                    x="timestamp",
                    y=["user_percent", "system_percent", "idle_percent"],
                    title="CPU Usage Breakdown Over Time",
                    labels={
                        "timestamp": "Time",
                        "value": "Percentage",
                        "variable": "Type"
                    }
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=30, l=80, r=30),
                    yaxis_range=[0, 100],
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # System load averages
            load_cols = [col for col in df_cpu.columns if col.startswith("load_avg_")]
            
            if load_cols:
                st.markdown("#### System Load Averages")
                
                # Get latest load averages
                latest_load = {
                    "1 minute": latest_cpu.get("load_avg_1min", 0),
                    "5 minutes": latest_cpu.get("load_avg_5min", 0),
                    "15 minutes": latest_cpu.get("load_avg_15min", 0)
                }
                
                # Display load averages
                load_cols = st.columns(3)
                
                for i, (period, value) in enumerate(latest_load.items()):
                    with load_cols[i]:
                        render_metric_card(
                            title=f"Load ({period})",
                            value=f"{value:.2f}",
                            description=f"Relative to {cpu_info.get('cpu_count_logical', 1)} cores",
                            icon="🔄"
                        )
                
                # Load averages chart
                fig = px.line(
                    df_cpu,
                    x="timestamp",
                    y=load_cols,
                    title="System Load Averages Over Time",
                    labels={
                        "timestamp": "Time",
                        "load_avg_1min": "1 min",
                        "load_avg_5min": "5 min",
                        "load_avg_15min": "15 min",
                        "value": "Load Average"
                    }
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=30, l=80, r=30),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CPU data available for the selected time range.")
    else:
        st.info("No CPU data available yet.")


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
                swap = row.get("swap", {})
                
                if isinstance(virtual, dict) and isinstance(swap, dict):
                    memory_data.append({
                        "timestamp": timestamp,
                        "memory_percent": virtual.get("percent", 0),
                        "memory_used_gb": virtual.get("used_gb", 0),
                        "memory_available_gb": virtual.get("available_gb", 0),
                        "memory_free_gb": virtual.get("free_gb", 0),
                        "swap_percent": swap.get("percent", 0),
                        "swap_used_gb": swap.get("used_gb", 0),
                        "swap_free_gb": swap.get("free_gb", 0)
                    })
            
            if memory_data:
                df_memory_plot = pd.DataFrame(memory_data)
                
                # Get latest memory usage
                latest_memory = df_memory_plot.iloc[-1].to_dict()
                
                # Display physical memory usage
                memory_percent = latest_memory.get("memory_percent", 0)
                memory_used = latest_memory.get("memory_used_gb", 0)
                memory_available = latest_memory.get("memory_available_gb", 0)
                memory_total = memory_used + memory_available
                
                st.markdown(f"#### Physical Memory Usage: {memory_percent:.1f}%")
                
                # Progress bar for memory usage
                st.progress(memory_percent / 100)
                
                # Memory usage details
                mem_cols = st.columns(3)
                
                with mem_cols[0]:
                    render_metric_card(
                        title="Used Memory",
                        value=f"{memory_used:.2f} GB",
                        description=f"{memory_percent:.1f}% of total",
                        icon="📊"
                    )
                
                with mem_cols[1]:
                    render_metric_card(
                        title="Available Memory",
                        value=f"{memory_available:.2f} GB",
                        description=f"{100-memory_percent:.1f}% of total",
                        icon="✅"
                    )
                
                with mem_cols[2]:
                    render_metric_card(
                        title="Total Memory",
                        value=f"{memory_total:.2f} GB",
                        description="Physical RAM",
                        icon="🧠"
                    )
                
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
                
                # Memory usage breakdown chart
                fig = px.area(
                    df_memory_plot,
                    x="timestamp",
                    y=["memory_used_gb", "memory_free_gb"],
                    title="Memory Allocation Over Time",
                    labels={
                        "timestamp": "Time",
                        "value": "GB",
                        "variable": "Type"
                    }
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=30, l=80, r=30),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )                

                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=30, l=80, r=30),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Swap usage
                swap_percent = latest_memory.get("swap_percent", 0)
                swap_used = latest_memory.get("swap_used_gb", 0)
                swap_free = latest_memory.get("swap_free_gb", 0)
                swap_total = swap_used + swap_free
                
                if swap_total > 0:
                    st.markdown(f"#### Swap Memory Usage: {swap_percent:.1f}%")
                    
                    # Progress bar for swap usage
                    st.progress(swap_percent / 100)
                    
                    # Swap usage details
                    swap_cols = st.columns(3)
                    
                    with swap_cols[0]:
                        render_metric_card(
                            title="Used Swap",
                            value=f"{swap_used:.2f} GB",
                            description=f"{swap_percent:.1f}% of total",
                            icon="📊"
                        )
                    
                    with swap_cols[1]:
                        render_metric_card(
                            title="Free Swap",
                            value=f"{swap_free:.2f} GB",
                            description=f"{100-swap_percent:.1f}% of total",
                            icon="✅"
                        )
                    
                    with swap_cols[2]:
                        render_metric_card(
                            title="Total Swap",
                            value=f"{swap_total:.2f} GB",
                            description="Virtual memory",
                            icon="💾"
                        )
                    
                    # Swap usage chart
                    fig = px.line(
                        df_memory_plot,
                        x="timestamp",
                        y="swap_percent",
                        title="Swap Usage Over Time",
                        labels={"swap_percent": "Swap Usage (%)", "timestamp": "Time"}
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
        # Display disk information in a grid
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate total disk space across all monitored paths
            total_disk_gb = sum([
                disk.get("total_gb", 0)
                for disk in disk_info.values()
            ])
            
            render_metric_card(
                title="Total Disk Space",
                value=f"{total_disk_gb:.2f} GB",
                description=f"Across {len(disk_info)} monitored paths",
                icon="💾"
            )
        
        with col2:
            # Calculate total used disk space across all monitored paths
            used_disk_gb = sum([
                disk.get("used_gb", 0)
                for disk in disk_info.values()
            ])
            
            render_metric_card(
                title="Used Disk Space",
                value=f"{used_disk_gb:.2f} GB",
                description=f"{used_disk_gb/total_disk_gb*100 if total_disk_gb > 0 else 0:.1f}% of total",
                icon="📊"
            )
        
        # Disk Usage by Partition
        st.markdown("### Disk Usage by Partition")
        
        # Create a list of partitions sorted by total size
        partitions = []
        
        for path, data in disk_info.items():
            partitions.append({
                "path": path,
                "total_gb": data.get("total_gb", 0),
                "used_gb": data.get("used_gb", 0),
                "free_gb": data.get("free_gb", 0),
                "percent": data.get("percent", 0)
            })
        
        partitions.sort(key=lambda x: x["total_gb"], reverse=True)
        
        # Display partitions
        for partition in partitions:
            st.markdown(f"#### {partition['path']}")
            
            # Progress bar for disk usage
            st.progress(partition["percent"] / 100)
            
            # Disk usage details
            disk_cols = st.columns(3)
            
            with disk_cols[0]:
                render_metric_card(
                    title="Used Space",
                    value=f"{partition['used_gb']:.2f} GB",
                    description=f"{partition['percent']:.1f}% of total",
                    icon="📊"
                )
            
            with disk_cols[1]:
                render_metric_card(
                    title="Free Space",
                    value=f"{partition['free_gb']:.2f} GB",
                    description=f"{100-partition['percent']:.1f}% of total",
                    icon="✅"
                )
            
            with disk_cols[2]:
                render_metric_card(
                    title="Total Space",
                    value=f"{partition['total_gb']:.2f} GB",
                    description=f"Mount point: {partition['path']}",
                    icon="💾"
                )
        
        # Disk I/O
        st.markdown("### Disk I/O")
        
        if "disk" in dfs and not dfs["disk"].empty:
            df_disk = dfs["disk"]
            
            # Apply time range filter
            if "timestamp" in df_disk.columns:
                df_disk["timestamp"] = pd.to_datetime(df_disk["timestamp"])
                df_disk = filter_dataframe_by_time(df_disk, time_range)
            
            if not df_disk.empty:
                # Extract disk I/O metrics from the nested structure
                disk_io_data = []
                
                for _, row in df_disk.iterrows():
                    timestamp = row.get("timestamp")
                    io = row.get("io", {})
                    
                    if isinstance(io, dict):
                        # Sum I/O across all disks
                        read_bytes = sum([
                            disk_data.get("read_bytes", 0)
                            for disk_data in io.values()
                            if isinstance(disk_data, dict)
                        ])
                        
                        write_bytes = sum([
                            disk_data.get("write_bytes", 0)
                            for disk_data in io.values()
                            if isinstance(disk_data, dict)
                        ])
                        
                        disk_io_data.append({
                            "timestamp": timestamp,
                            "read_bytes": read_bytes,
                            "write_bytes": write_bytes
                        })
                
                if disk_io_data:
                    df_io = pd.DataFrame(disk_io_data)
                    
                    # Calculate I/O rates by differencing consecutive rows
                    if len(df_io) > 1:
                        df_io["read_rate"] = df_io["read_bytes"].diff() / ((df_io["timestamp"].diff().dt.total_seconds()))
                        df_io["write_rate"] = df_io["write_bytes"].diff() / ((df_io["timestamp"].diff().dt.total_seconds()))
                        
                        # Replace invalid values with 0
                        df_io.replace([float('inf'), float('-inf'), float('nan')], 0, inplace=True)
                        
                        # Convert to KB/s for better readability
                        df_io["read_rate_kb"] = df_io["read_rate"] / 1024
                        df_io["write_rate_kb"] = df_io["write_rate"] / 1024
                        
                        # Disk I/O rate chart
                        fig = px.line(
                            df_io.dropna(),
                            x="timestamp",
                            y=["read_rate_kb", "write_rate_kb"],
                            title="Disk I/O Rate",
                            labels={
                                "timestamp": "Time",
                                "value": "KB/s",
                                "variable": "Operation"
                            }
                        )
                        
                        fig.update_layout(
                            height=300,
                            margin=dict(t=30, b=30, l=80, r=30),
                            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough data points to calculate I/O rates.")
                else:
                    st.info("No disk I/O data available for the selected time range.")
            else:
                st.info("No disk I/O data available for the selected time range.")
        else:
            st.info("No disk I/O data available yet.")
    else:
        st.info("No disk information available.")


def render_network_tab(monitor: SystemMonitor, system_info: Dict[str, Any], dfs: Dict[str, pd.DataFrame], time_range: str):
    """Render the network tab with detailed network metrics."""
    # Network information
    st.markdown("### Network Information")
    
    # Network info from system_info
    network_info = system_info.get("network", {})
    
    if network_info:
        # Display network information in a grid
        col1, col2 = st.columns(2)
        
        with col1:
            interfaces_text = ", ".join(network_info.keys())
            render_metric_card(
                title="Network Interfaces",
                value=f"{len(network_info)} Active",
                description=interfaces_text[:50] + "..." if len(interfaces_text) > 50 else interfaces_text,
                icon="🌐"
            )
        
        with col2:
            # Get IP addresses
            ip_addresses = []
            
            for interface, data in network_info.items():
                if isinstance(data, dict) and "ip_addresses" in data:
                    ip_addresses.extend(data["ip_addresses"])
            
            ip_text = ", ".join(ip_addresses)
            render_metric_card(
                title="IP Addresses",
                value=f"{len(ip_addresses)} Assigned",
                description=ip_text[:50] + "..." if len(ip_text) > 50 else ip_text,
                icon="🔢"
            )
        
        # Network Activity
        st.markdown("### Network Activity")
        
        if "network" in dfs and not dfs["network"].empty:
            df_network = dfs["network"]
            
            # Apply time range filter
            if "timestamp" in df_network.columns:
                df_network["timestamp"] = pd.to_datetime(df_network["timestamp"])
                df_network = filter_dataframe_by_time(df_network, time_range)
            
            if not df_network.empty:
                # Extract network metrics from the nested structure
                network_data = []
                
                for _, row in df_network.iterrows():
                    timestamp = row.get("timestamp")
                    interfaces = row.get("interfaces", {})
                    connections_count = row.get("connections_count", 0)
                    
                    if isinstance(interfaces, dict):
                        # Sum rates across all interfaces
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
                        
                        # Sum total bytes
                        bytes_sent = sum([
                            iface.get("bytes_sent", 0) 
                            for iface in interfaces.values()
                            if isinstance(iface, dict)
                        ])
                        
                        bytes_recv = sum([
                            iface.get("bytes_recv", 0)
                            for iface in interfaces.values()
                            if isinstance(iface, dict)
                        ])
                        
                        network_data.append({
                            "timestamp": timestamp,
                            "send_rate_bytes_per_sec": send_rate,
                            "recv_rate_bytes_per_sec": recv_rate,
                            "bytes_sent": bytes_sent,
                            "bytes_recv": bytes_recv,
                            "connections_count": connections_count
                        })
                
                if network_data:
                    df_network_plot = pd.DataFrame(network_data)
                    
                    # Get latest network activity
                    latest_network = df_network_plot.iloc[-1].to_dict()
                    
                    # Display current network activity
                    net_cols = st.columns(3)
                    
                    with net_cols[0]:
                        send_rate = latest_network.get("send_rate_bytes_per_sec", 0)
                        render_metric_card(
                            title="Upload Rate",
                            value=f"{format_bytes(send_rate)}/s",
                            description="Current outgoing traffic",
                            icon="⬆️"
                        )
                    
                    with net_cols[1]:
                        recv_rate = latest_network.get("recv_rate_bytes_per_sec", 0)
                        render_metric_card(
                            title="Download Rate",
                            value=f"{format_bytes(recv_rate)}/s",
                            description="Current incoming traffic",
                            icon="⬇️"
                        )
                    
                    with net_cols[2]:
                        connections = latest_network.get("connections_count", 0)
                        render_metric_card(
                            title="Active Connections",
                            value=f"{connections}",
                            description="Network connections",
                            icon="🔌"
                        )
                    
                    # Network rate chart
                    # Convert to KB/s for better readability
                    df_network_plot["send_rate_kb_per_sec"] = df_network_plot["send_rate_bytes_per_sec"] / 1024
                    df_network_plot["recv_rate_kb_per_sec"] = df_network_plot["recv_rate_bytes_per_sec"] / 1024
                    
                    fig = px.line(
                        df_network_plot,
                        x="timestamp",
                        y=["send_rate_kb_per_sec", "recv_rate_kb_per_sec"],
                        title="Network Traffic Rate",
                        labels={
                            "timestamp": "Time", 
                            "value": "KB/s",
                            "variable": "Direction"
                        }
                    )
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(t=30, b=30, l=80, r=30),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Total data transferred
                    st.markdown("### Data Transferred")
                    
                    data_cols = st.columns(3)
                    
                    with data_cols[0]:
                        sent_total = latest_network.get("bytes_sent", 0)
                        render_metric_card(
                            title="Total Uploaded",
                            value=format_bytes(sent_total),
                            description="Since system boot",
                            icon="⬆️"
                        )
                    
                    with data_cols[1]:
                        recv_total = latest_network.get("bytes_recv", 0)
                        render_metric_card(
                            title="Total Downloaded",
                            value=format_bytes(recv_total),
                            description="Since system boot",
                            icon="⬇️"
                        )
                    
                    with data_cols[2]:
                        total = sent_total + recv_total
                        render_metric_card(
                            title="Total Transferred",
                            value=format_bytes(total),
                            description="Since system boot",
                            icon="🔄"
                        )
                    
                    # Network connections chart
                    if "connections_count" in df_network_plot.columns:
                        fig = px.line(
                            df_network_plot,
                            x="timestamp",
                            y="connections_count",
                            title="Active Network Connections",
                            labels={
                                "timestamp": "Time", 
                                "connections_count": "Connections"
                            }
                        )
                        
                        fig.update_layout(
                            height=300,
                            margin=dict(t=30, b=30, l=80, r=30),
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
            
            if interfaces:
                for interface_name, interface in interfaces.items():
                    if isinstance(interface, dict):
                        st.markdown(f"#### Interface: {interface_name}")
                        
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
            else:
                st.info("No network interface details available.")
        else:
            st.info("No network interface details available yet.")
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
        
        if network_interfaces:
            total_send_rate = sum([
                iface.get("send_rate_bytes_per_sec", 0) 
                for iface in network_interfaces.values()
            ])
            
            total_recv_rate = sum([
                iface.get("recv_rate_bytes_per_sec", 0)
                for iface in network_interfaces.values()
            ])
            
            render_metric_card(
                title="Network Activity",
                value=f"↑{format_bytes(total_send_rate)}/s",
                description=f"↓{format_bytes(total_recv_rate)}/s",
                icon="🌐"
            )
        else:
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
            value=f"{cpu_info.get('cpu_count_physical', 0)} Physical / {cpu_info.get('cpu_count_logical', 0)} Logical",
            description="Physical and logical cores",
            icon="⚙️"
        )
    
    with col3:
        freq = cpu_info.get('cpu_frequency', {})
        if freq and freq.get('current_mhz'):
            render_metric_card(
                title="CPU Frequency",
                value=f"{freq.get('current_mhz', 0)/1000:.2f} GHz",
                description=f"Min: {freq.get('min_mhz', 0)/1000:.2f} GHz, Max: {freq.get('max_mhz', 0)/1000:.2f} GHz" if freq.get('min_mhz') and freq.get('max_mhz') else "",
                icon="⚡"
            )
        else:
            render_metric_card(
                title="CPU Frequency",
                value="Unknown",
                description="",
                icon="⚡"
            )
    
    # CPU Usage
    st.markdown("### CPU Usage")
    
    if "cpu" in dfs and not dfs["cpu"].empty:
        df_cpu = dfs["cpu"]
        
        # Apply time range filter
        if "timestamp" in df_cpu.columns:
            df_cpu["timestamp"] = pd.to_datetime(df_cpu["timestamp"])
            df_cpu = filter_dataframe_by_time(df_cpu, time_range)
        
        if not df_cpu.empty:
            # Get latest CPU usage
            latest_cpu = df_cpu.iloc[-1].to_dict()
            
            # Display overall CPU usage
            overall_percent = latest_cpu.get("overall_percent", 0)
            
            st.markdown(f"#### Overall CPU Usage: {overall_percent:.1f}%")
            
            # Progress bar for overall CPU usage
            st.progress(overall_percent / 100)
            
            # CPU usage chart over time
            fig = px.line(
                df_cpu,
                x="timestamp",
                y="overall_percent",
                title="CPU Usage Over Time",
                labels={"overall_percent": "CPU Usage (%)", "timestamp": "Time"}
            )
            
            fig.update_layout(
                height=300,
                margin=dict(t=30, b=30, l=80, r=30),
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Per-CPU usage if available
            if "per_cpu_percent" in latest_cpu and isinstance(latest_cpu["per_cpu_percent"], list):
                st.markdown("#### Per-CPU Usage")
                
                per_cpu = latest_cpu["per_cpu_percent"]
                num_cpus = len(per_cpu)
                
                # Calculate number of rows based on number of CPUs
                cpus_per_row = 4
                num_rows = (num_cpus + cpus_per_row - 1) // cpus_per_row
                
                for row in range(num_rows):
                    cols = st.columns(cpus_per_row)
                    
                    for col_idx in range(cpus_per_row):
                        cpu_idx = row * cpus_per_row + col_idx
                        
                        if cpu_idx < num_cpus:
                            with cols[col_idx]:
                                cpu_percent = per_cpu[cpu_idx]
                                st.markdown(f"**CPU {cpu_idx}:** {cpu_percent:.1f}%")
                                st.progress(cpu_percent / 100)
            
            # CPU usage breakdown
            st.markdown("#### CPU Usage Breakdown")
            
            if "user_percent" in latest_cpu and "system_percent" in latest_cpu and "idle_percent" in latest_cpu:
                breakdown_cols = st.columns(3)
                
                with breakdown_cols[0]:
                    render_metric_card(
                        title="User",
                        value=f"{latest_cpu.get('user_percent', 0):.1f}%",
                        description="User processes",
                        icon="👤"
                    )
                
                with breakdown_cols[1]:
                    render_metric_card(
                        title="System",
                        value=f"{latest_cpu.get('system_percent', 0):.1f}%",
                        description="Kernel & system processes",
                        icon="🔧"
                    )
                
                with breakdown_cols[2]:
                    render_metric_card(
                        title="Idle",
                        value=f"{latest_cpu.get('idle_percent', 0):.1f}%",
                        description="Idle time",
                        icon="💤"
                    )
                
                # CPU usage breakdown chart
                fig = px.area(
                    df_cpu[["timestamp", "user_percent", "system_percent", "idle_percent"]],
                    x="timestamp",
                    y=["user_percent", "system_percent", "idle_percent"],
                    title="CPU Usage Breakdown Over Time",
                    labels={
                        "timestamp": "Time",
                        "value": "Percentage",
                        "variable": "Type"
                    }
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=30, l=80, r=30),
                    yaxis_range=[0, 100],
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # System load averages
            load_cols = [col for col in df_cpu.columns if col.startswith("load_avg_")]
            
            if load_cols:
                st.markdown("#### System Load Averages")
                
                # Get latest load averages
                latest_load = {
                    "1 minute": latest_cpu.get("load_avg_1min", 0),
                    "5 minutes": latest_cpu.get("load_avg_5min", 0),
                    "15 minutes": latest_cpu.get("load_avg_15min", 0)
                }
                
                # Display load averages
                load_cols = st.columns(3)
                
                for i, (period, value) in enumerate(latest_load.items()):
                    with load_cols[i]:
                        render_metric_card(
                            title=f"Load ({period})",
                            value=f"{value:.2f}",
                            description=f"Relative to {cpu_info.get('cpu_count_logical', 1)} cores",
                            icon="🔄"
                        )
                
                fig = go.Figure()
                for col in load_cols:
                    if col in df_cpu.columns:
                        fig.add_trace(go.Scatter(x=df_cpu["timestamp"], y=df_cpu[col], mode='lines', name=col))

                fig.update_layout(
                    title="System Load Averages Over Time",
                    xaxis_title="Time",
                    yaxis_title="Load Average",
                    height=300,
                    margin=dict(t=30, b=30, l=80, r=30),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CPU data available for the selected time range.")
    else:
        st.info("No CPU data available yet.")


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
                swap = row.get("swap", {})
                
                if isinstance(virtual, dict) and isinstance(swap, dict):
                    memory_data.append({
                        "timestamp": timestamp,
                        "memory_percent": virtual.get("percent", 0),
                        "memory_used_gb": virtual.get("used_gb", 0),
                        "memory_available_gb": virtual.get("available_gb", 0),
                        "memory_free_gb": virtual.get("free_gb", 0),
                        "swap_percent": swap.get("percent", 0),
                        "swap_used_gb": swap.get("used_gb", 0),
                        "swap_free_gb": swap.get("free_gb", 0)
                    })
            
            if memory_data:
                df_memory_plot = pd.DataFrame(memory_data)
                
                # Get latest memory usage
                latest_memory = df_memory_plot.iloc[-1].to_dict()
                
                # Display physical memory usage
                memory_percent = latest_memory.get("memory_percent", 0)
                memory_used = latest_memory.get("memory_used_gb", 0)
                memory_available = latest_memory.get("memory_available_gb", 0)
                memory_total = memory_used + memory_available
                
                st.markdown(f"#### Physical Memory Usage: {memory_percent:.1f}%")
                
                # Progress bar for memory usage
                st.progress(memory_percent / 100)
                
                # Memory usage details
                mem_cols = st.columns(3)
                
                with mem_cols[0]:
                    render_metric_card(
                        title="Used Memory",
                        value=f"{memory_used:.2f} GB",
                        description=f"{memory_percent:.1f}% of total",
                        icon="📊"
                    )
                
                with mem_cols[1]:
                    render_metric_card(
                        title="Available Memory",
                        value=f"{memory_available:.2f} GB",
                        description=f"{100-memory_percent:.1f}% of total",
                        icon="✅"
                    )
                
                with mem_cols[2]:
                    render_metric_card(
                        title="Total Memory",
                        value=f"{memory_total:.2f} GB",
                        description="Physical RAM",
                        icon="🧠"
                    )
                
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
                
                # Memory usage breakdown chart
                fig = px.area(
                    df_memory_plot,
                    x="timestamp",
                    y=["memory_used_gb", "memory_free_gb"],
                    title="Memory Allocation Over Time",
                    labels={
                        "timestamp": "Time",
                        "value": "GB",
                        "variable": "Type"
                    }
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=30, l=80, r=30),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )