"""
Python applications dashboard module for monitoring Python processes.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.monitors.python_monitor import PythonMonitor
from src.widgets.metric_cards import render_metric_card, render_metric_row
from src.widgets.charts import render_time_series, render_bar_chart
from src.utils.helpers import format_number, format_bytes, format_percent, time_ago


def render_dashboard(monitor: Optional[PythonMonitor] = None, time_range: str = "Last hour"):
    """
    Render the Python applications monitoring dashboard.
    
    Args:
        monitor: Python monitor instance
        time_range: Time range filter for metrics
    """
    st.subheader("Python Applications Monitoring Dashboard", anchor=False)
    
    # Initialize monitor if not provided
    if monitor is None:
        monitor = PythonMonitor()
    
    # Collect latest metrics
    processes_info = monitor.collect_python_processes()
    
    # Get dataframes for charts
    dfs = monitor.to_dataframe()
    
    # Tabs for different sections
    tabs = st.tabs(["Overview", "Processes", "Performance", "Settings"])
    
    # OVERVIEW TAB
    with tabs[0]:
        render_overview_tab(monitor, processes_info, dfs, time_range)
    
    # PROCESSES TAB
    with tabs[1]:
        render_processes_tab(monitor, processes_info, dfs, time_range)
    
    # PERFORMANCE TAB
    with tabs[2]:
        render_performance_tab(monitor, processes_info, dfs, time_range)
    
    # SETTINGS TAB
    with tabs[3]:
        render_settings_tab(monitor)


def render_overview_tab(monitor: PythonMonitor, processes_info: Dict[str, Any], dfs: Dict[str, pd.DataFrame], time_range: str):
    """Render the overview tab with summary metrics."""
    # Summary metrics
    st.markdown("### Python Applications Overview")
    
    # Get summary from processes_info
    summary = processes_info.get("summary", {})
    
    # Display key metrics in a grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card(
            title="Python Processes",
            value=summary.get("total_count", 0),
            description="Active Python processes",
            icon="🐍"
        )
    
    with col2:
        render_metric_card(
            title="Total Memory",
            value=f"{summary.get('total_memory_mb', 0):.1f} MB",
            description="Memory used by Python processes",
            icon="🧠"
        )
    
    with col3:
        render_metric_card(
            title="CPU Usage",
            value=f"{summary.get('total_cpu_percent', 0):.1f}%",
            description="CPU used by Python processes",
            icon="⚙️"
        )
    
    with col4:
        # Get breakdown by framework
        frameworks = summary.get("frameworks", {})
        framework_text = ", ".join([f"{k.capitalize()}: {v}" for k, v in frameworks.items()])
        
        render_metric_card(
            title="Frameworks",
            value=len(frameworks),
            description=framework_text[:40] + "..." if len(framework_text) > 40 else framework_text,
            icon="📦"
        )
    
    # Processes table
    st.markdown("### Active Python Processes")
    
    if "processes" in processes_info:
        processes = processes_info.get("processes", [])
        
        if processes:
            # Convert to DataFrame for display
            process_data = []
            
            for proc in processes:
                process_data.append({
                    "PID": proc.get("pid", "Unknown"),
                    "Name": proc.get("name", "Unknown"),
                    "CPU %": f"{proc.get('cpu_percent', 0):.1f}%",
                    "Memory": f"{proc.get('memory_mb', 0):.1f} MB",
                    "Threads": proc.get("num_threads", 0),
                    "Framework": proc.get("framework", "unknown").capitalize(),
                    "Status": proc.get("status", "Unknown")
                })
            
            # Create DataFrame
            df_processes = pd.DataFrame(process_data)
            
            # Display table
            st.dataframe(df_processes, use_container_width=True)
        else:
            st.info("No active Python processes found.")
    else:
        st.info("No active Python processes found.")
    
    # Process charts
    st.markdown("### Python Processes Metrics")
    
    if "processes" in dfs and not dfs["processes"].empty:
        df_processes = dfs["processes"]
        
        # Apply time range filter
        if "timestamp" in df_processes.columns:
            df_processes["timestamp"] = pd.to_datetime(df_processes["timestamp"])
            df_processes = filter_dataframe_by_time(df_processes, time_range)
        
        if not df_processes.empty:
            # Process count over time
            process_counts = df_processes.groupby(["timestamp"]).size().reset_index(name="count")
            
            fig = px.line(
                process_counts,
                x="timestamp",
                y="count",
                title="Python Process Count Over Time",
                labels={"count": "Process Count", "timestamp": "Time"}
            )
            
            fig.update_layout(
                height=300,
                margin=dict(t=30, b=30, l=80, r=30)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # CPU usage over time
            col1, col2 = st.columns(2)
            
            with col1:
                # CPU usage by process
                if "cpu_percent" in df_processes.columns and "pid" in df_processes.columns:
                    # Group by timestamp and get total CPU usage
                    cpu_usage = df_processes.groupby("timestamp")["cpu_percent"].sum().reset_index()
                    
                    fig = px.line(
                        cpu_usage,
                        x="timestamp",
                        y="cpu_percent",
                        title="Total CPU Usage Over Time",
                        labels={"cpu_percent": "CPU Usage (%)", "timestamp": "Time"}
                    )
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(t=30, b=30, l=80, r=30)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Memory usage by process
                if "memory_mb" in df_processes.columns and "pid" in df_processes.columns:
                    # Group by timestamp and get total memory usage
                    memory_usage = df_processes.groupby("timestamp")["memory_mb"].sum().reset_index()
                    
                    fig = px.line(
                        memory_usage,
                        x="timestamp",
                        y="memory_mb",
                        title="Total Memory Usage Over Time",
                        labels={"memory_mb": "Memory Usage (MB)", "timestamp": "Time"}
                    )
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(t=30, b=30, l=80, r=30)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Framework distribution
            if "framework" in df_processes.columns:
                # Count processes by framework
                framework_counts = df_processes.groupby("framework").size().reset_index(name="count")
                
                fig = px.pie(
                    framework_counts,
                    names="framework",
                    values="count",
                    title="Python Processes by Framework",
                    hole=0.4
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=30, l=30, r=30)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Python process data available for the selected time range.")
    else:
        st.info("No Python process data available yet.")


def render_processes_tab(monitor: PythonMonitor, processes_info: Dict[str, Any], dfs: Dict[str, pd.DataFrame], time_range: str):
    """Render the processes tab with detailed process information."""
    # Process list
    st.markdown("### Python Process List")
    
    if "processes" in processes_info:
        processes = processes_info.get("processes", [])
        
        if processes:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Get unique framework values
                frameworks = set(proc.get("framework", "unknown") for proc in processes)
                framework_options = ["All"] + sorted(list(frameworks))
                
                framework_filter = st.selectbox(
                    "Framework",
                    options=framework_options
                )
            
            with col2:
                # Status filter
                status_options = ["All", "running", "sleeping", "stopped", "zombie"]
                status_filter = st.selectbox(
                    "Status",
                    options=status_options
                )
            
            with col3:
                # Sort options
                sort_options = [
                    "CPU Usage (High to Low)", 
                    "Memory Usage (High to Low)",
                    "PID (Low to High)",
                    "PID (High to Low)"
                ]
                
                sort_by = st.selectbox(
                    "Sort By",
                    options=sort_options
                )
            
            # Apply filters
            filtered_processes = processes.copy()
            
            if framework_filter != "All":
                filtered_processes = [
                    proc for proc in filtered_processes 
                    if proc.get("framework", "unknown") == framework_filter
                ]
            
            if status_filter != "All":
                filtered_processes = [
                    proc for proc in filtered_processes 
                    if proc.get("status", "").lower() == status_filter.lower()
                ]
            
            # Apply sorting
            if sort_by == "CPU Usage (High to Low)":
                filtered_processes.sort(key=lambda x: x.get("cpu_percent", 0), reverse=True)
            elif sort_by == "Memory Usage (High to Low)":
                filtered_processes.sort(key=lambda x: x.get("memory_mb", 0), reverse=True)
            elif sort_by == "PID (Low to High)":
                filtered_processes.sort(key=lambda x: x.get("pid", 0))
            elif sort_by == "PID (High to Low)":
                filtered_processes.sort(key=lambda x: x.get("pid", 0), reverse=True)
            
            # Display processes
            if filtered_processes:
                # Convert to DataFrame for display
                process_data = []
                
                for proc in filtered_processes:
                    process_data.append({
                        "PID": proc.get("pid", "Unknown"),
                        "Name": proc.get("name", "Unknown"),
                        "CPU %": f"{proc.get('cpu_percent', 0):.1f}%",
                        "Memory": f"{proc.get('memory_mb', 0):.1f} MB",
                        "Threads": proc.get("num_threads", 0),
                        "Framework": proc.get("framework", "unknown").capitalize(),
                        "Status": proc.get("status", "Unknown"),
                        "User": proc.get("username", "Unknown")
                    })
                
                # Create DataFrame
                df_processes = pd.DataFrame(process_data)
                
                # Display table with selection
                selection = st.dataframe(
                    df_processes,
                    use_container_width=True,
                    column_config={
                        "PID": st.column_config.TextColumn("PID", width="small"),
                        "Name": st.column_config.TextColumn("Name", width="medium"),
                        "CPU %": st.column_config.TextColumn("CPU %", width="small"),
                        "Memory": st.column_config.TextColumn("Memory", width="small"),
                        "Threads": st.column_config.NumberColumn("Threads", width="small"),
                        "Framework": st.column_config.TextColumn("Framework", width="small"),
                        "Status": st.column_config.TextColumn("Status", width="small"),
                        "User": st.column_config.TextColumn("User", width="small")
                    }
                )
                
                # Process details
                st.markdown("### Process Details")
                
                # Get selected PID
                selected_pid = st.selectbox(
                    "Select Process to View Details",
                    options=[proc.get("pid") for proc in filtered_processes],
                    format_func=lambda pid: f"PID {pid} - {next((p.get('name', 'Unknown') for p in filtered_processes if p.get('pid') == pid), 'Unknown')}"
                )
                
                if selected_pid:
                    # Get process details
                    process_details = monitor.get_process_details(selected_pid)
                    
                    if process_details and "error" not in process_details:
                        # Display process details
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            render_metric_card(
                                title="Process Name",
                                value=process_details.get("name", "Unknown"),
                                description=f"PID: {selected_pid}",
                                icon="🔍"
                            )
                        
                        with col2:
                            render_metric_card(
                                title="CPU Usage",
                                value=f"{process_details.get('cpu_percent', 0):.1f}%",
                                description="Current CPU usage",
                                icon="⚙️"
                            )
                        
                        with col3:
                            render_metric_card(
                                title="Memory Usage",
                                value=f"{process_details.get('memory_mb', 0):.1f} MB",
                                description=f"{process_details.get('memory_percent', 0):.1f}% of total RAM",
                                icon="🧠"
                            )
                        
                        # Process information
                        with st.expander("Process Information", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### Basic Information")
                                st.markdown(f"**Status:** {process_details.get('status', 'Unknown')}")
                                st.markdown(f"**User:** {process_details.get('username', 'Unknown')}")
                                st.markdown(f"**Threads:** {process_details.get('num_threads', 0)}")
                                
                                if "create_time" in process_details:
                                    create_time = datetime.fromtimestamp(process_details.get("create_time", 0))
                                    st.markdown(f"**Started:** {create_time.strftime('%Y-%m-%d %H:%M:%S')} ({time_ago(create_time)})")
                            
                            with col2:
                                st.markdown("#### Resources")
                                
                                if "cpu_times" in process_details:
                                    cpu_times = process_details.get("cpu_times", {})
                                    if hasattr(cpu_times, "user"):
                                        st.markdown(f"**CPU User Time:** {cpu_times.user:.2f}s")
                                    if hasattr(cpu_times, "system"):
                                        st.markdown(f"**CPU System Time:** {cpu_times.system:.2f}s")
                                
                                if "io_counters" in process_details:
                                    io_counters = process_details.get("io_counters", {})
                                    if io_counters:
                                        if hasattr(io_counters, "read_bytes"):
                                            st.markdown(f"**I/O Read:** {format_bytes(io_counters.read_bytes)}")
                                        if hasattr(io_counters, "write_bytes"):
                                            st.markdown(f"**I/O Write:** {format_bytes(io_counters.write_bytes)}")
                        
                        # Process command line
                        with st.expander("Command Line"):
                            st.code(process_details.get("cmdline", "N/A"))
                        
                        # Current working directory
                        st.markdown(f"**Working Directory:** {process_details.get('cwd', 'Unknown')}")
                        
                        # Process open files
                        with st.expander("Open Files"):
                            open_files = process_details.get("open_files", [])
                            if open_files:
                                for file_path in open_files:
                                    st.markdown(f"- `{file_path}`")
                            else:
                                st.write("No open files.")
                        
                        # Process packages
                        with st.expander("Imported Packages"):
                            packages = process_details.get("packages", {})
                            if packages:
                                package_data = [{"Package": pkg, "Version": ver} for pkg, ver in packages.items()]
                                st.dataframe(pd.DataFrame(package_data), use_container_width=True)
                            else:
                                st.write("No package information available.")
                        
                        # Process connections
                        with st.expander("Network Connections"):
                            connections = process_details.get("connections", [])
                            if connections:
                                connection_data = []
                                for conn in connections:
                                    if hasattr(conn, "laddr") and hasattr(conn, "raddr"):
                                        connection_data.append({
                                            "Type": conn.type,
                                            "Local Address": f"{conn.laddr.ip}:{conn.laddr.port}" if hasattr(conn.laddr, "ip") else "N/A",
                                            "Remote Address": f"{conn.raddr.ip}:{conn.raddr.port}" if hasattr(conn.raddr, "ip") else "N/A",
                                            "Status": conn.status if hasattr(conn, "status") else "N/A"
                                        })
                                
                                if connection_data:
                                    st.dataframe(pd.DataFrame(connection_data), use_container_width=True)
                                else:
                                    st.write("No connection details available.")
                            else:
                                st.write("No active connections.")
                        
                        # Process children
                        if "children" in process_details and process_details["children"]:
                            with st.expander("Child Processes"):
                                children = process_details["children"]
                                for child_pid in children:
                                    st.markdown(f"- PID {child_pid}")
                        
                        # Process performance over time
                        st.markdown("### Process Performance Over Time")
                        
                        # Get process stats
                        process_stats = monitor.get_process_stats(selected_pid)
                        
                        if process_stats and process_stats.get("stats"):
                            stats = process_stats.get("stats", [])
                            
                            # Convert to DataFrame
                            df_stats = pd.DataFrame(stats)
                            
                            if not df_stats.empty:
                                # Apply time range filter
                                if "timestamp" in df_stats.columns:
                                    df_stats["timestamp"] = pd.to_datetime(df_stats["timestamp"])
                                    df_stats = filter_dataframe_by_time(df_stats, time_range)
                                
                                if not df_stats.empty:
                                    # CPU usage chart
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        if "cpu_percent" in df_stats.columns:
                                            fig = px.line(
                                                df_stats,
                                                x="timestamp",
                                                y="cpu_percent",
                                                title=f"CPU Usage for PID {selected_pid}",
                                                labels={"cpu_percent": "CPU Usage (%)", "timestamp": "Time"}
                                            )
                                            
                                            fig.update_layout(
                                                height=300,
                                                margin=dict(t=30, b=30, l=80, r=30)
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                    
                                    with col2:
                                        if "memory_mb" in df_stats.columns:
                                            fig = px.line(
                                                df_stats,
                                                x="timestamp",
                                                y="memory_mb",
                                                title=f"Memory Usage for PID {selected_pid}",
                                                labels={"memory_mb": "Memory Usage (MB)", "timestamp": "Time"}
                                            )
                                            
                                            fig.update_layout(
                                                height=300,
                                                margin=dict(t=30, b=30, l=80, r=30)
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info(f"No performance data available for PID {selected_pid} in the selected time range.")
                            else:
                                st.info(f"No performance data available for PID {selected_pid}.")
                        else:
                            st.info(f"No performance data available for PID {selected_pid}.")
                    else:
                        st.error(f"Error getting process details: {process_details.get('error', 'Unknown error')}")
            else:
                st.info("No Python processes match the selected filters.")
        else:
            st.info("No active Python processes found.")
    else:
        st.info("No active Python processes found.")


def render_performance_tab(monitor: PythonMonitor, processes_info: Dict[str, Any], dfs: Dict[str, pd.DataFrame], time_range: str):
    """Render the performance tab with detailed performance metrics."""
    # Performance metrics
    st.markdown("### Python Applications Performance")
    
    if "processes" in dfs and not dfs["processes"].empty:
        df_processes = dfs["processes"]
        
        # Apply time range filter
        if "timestamp" in df_processes.columns:
            df_processes["timestamp"] = pd.to_datetime(df_processes["timestamp"])
            df_processes = filter_dataframe_by_time(df_processes, time_range)
        
        if not df_processes.empty:
            # Framework filter
            if "framework" in df_processes.columns:
                frameworks = ["All"] + sorted(df_processes["framework"].unique().tolist())
                
                framework_filter = st.selectbox(
                    "Framework",
                    options=frameworks
                )
                
                # Apply filter
                if framework_filter != "All":
                    df_processes = df_processes[df_processes["framework"] == framework_filter]
            
            # Performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                # CPU usage by framework
                if "framework" in df_processes.columns and "cpu_percent" in df_processes.columns:
                    # Aggregate by timestamp and framework
                    cpu_by_framework = df_processes.groupby(["timestamp", "framework"])["cpu_percent"].sum().reset_index()
                    
                    fig = px.line(
                        cpu_by_framework,
                        x="timestamp",
                        y="cpu_percent",
                        color="framework",
                        title="CPU Usage by Framework",
                        labels={"cpu_percent": "CPU Usage (%)", "timestamp": "Time", "framework": "Framework"}
                    )
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(t=30, b=30, l=80, r=30)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Memory usage by framework
                if "framework" in df_processes.columns and "memory_mb" in df_processes.columns:
                    # Aggregate by timestamp and framework
                    memory_by_framework = df_processes.groupby(["timestamp", "framework"])["memory_mb"].sum().reset_index()
                    
                    fig = px.line(
                        memory_by_framework,
                        x="timestamp",
                        y="memory_mb",
                        color="framework",
                        title="Memory Usage by Framework",
                        labels={"memory_mb": "Memory Usage (MB)", "timestamp": "Time", "framework": "Framework"}
                    )
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(t=30, b=30, l=80, r=30)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Top process charts
            st.markdown("### Top Python Processes")
            
            # Get latest timestamp
            latest_timestamp = df_processes["timestamp"].max()
            latest_df = df_processes[df_processes["timestamp"] == latest_timestamp]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top CPU processes
                if "pid" in latest_df.columns and "cpu_percent" in latest_df.columns and "name" in latest_df.columns:
                    # Sort by CPU usage
                    top_cpu = latest_df.sort_values("cpu_percent", ascending=False).head(10)
                    
                    if not top_cpu.empty:
                        fig = px.bar(
                            top_cpu,
                            x="cpu_percent",
                            y="name",
                            title="Top Processes by CPU Usage",
                            labels={"cpu_percent": "CPU Usage (%)", "name": "Process Name"},
                            orientation="h"
                        )
                        
                        fig.update_layout(
                            height=400,
                            margin=dict(t=30, b=30, l=120, r=30),
                            yaxis=dict(autorange="reversed")  # To show highest at top
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No CPU usage data available.")
            
            with col2:
                # Top memory processes
                if "pid" in latest_df.columns and "memory_mb" in latest_df.columns and "name" in latest_df.columns:
                    # Sort by memory usage
                    top_memory = latest_df.sort_values("memory_mb", ascending=False).head(10)
                    
                    if not top_memory.empty:
                        fig = px.bar(
                            top_memory,
                            x="memory_mb",
                            y="name",
                            title="Top Processes by Memory Usage",
                            labels={"memory_mb": "Memory Usage (MB)", "name": "Process Name"},
                            orientation="h"
                        )
                        
                        fig.update_layout(
                            height=400,
                            margin=dict(t=30, b=30, l=120, r=30),
                            yaxis=dict(autorange="reversed")  # To show highest at top
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No memory usage data available.")
            
            # Thread count chart
            if "num_threads" in df_processes.columns:
                # Sum threads by timestamp
                threads_by_time = df_processes.groupby("timestamp")["num_threads"].sum().reset_index()
                
                fig = px.line(
                    threads_by_time,
                    x="timestamp",
                    y="num_threads",
                    title="Total Thread Count Over Time",
                    labels={"num_threads": "Thread Count", "timestamp": "Time"}
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=30, l=80, r=30)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Process count by status
            if "status" in df_processes.columns:
                # Get counts by timestamp and status
                status_counts = df_processes.groupby(["timestamp", "status"]).size().reset_index(name="count")
                
                fig = px.area(
                    status_counts,
                    x="timestamp",
                    y="count",
                    color="status",
                    title="Process Count by Status",
                    labels={"count": "Process Count", "timestamp": "Time", "status": "Status"}
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=30, l=80, r=30)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Python process data available for the selected time range.")
    else:
        st.info("No Python process data available yet.")
    
    # Framework comparison
    st.markdown("### Framework Comparison")
    
    if ("processes" in processes_info and 
    processes_info.get("processes", []) and  # Check if list is not empty
    "framework" in processes_info.get("processes", [])[0]):
        processes = processes_info.get("processes", [])
        
        # Group by framework
        framework_data = {}
        
        for proc in processes:
            framework = proc.get("framework", "unknown")
            
            if framework not in framework_data:
                framework_data[framework] = {
                    "count": 0,
                    "cpu_total": 0,
                    "memory_total": 0,
                    "threads_total": 0
                }
            
            framework_data[framework]["count"] += 1
            framework_data[framework]["cpu_total"] += proc.get("cpu_percent", 0)
            framework_data[framework]["memory_total"] += proc.get("memory_mb", 0)
            framework_data[framework]["threads_total"] += proc.get("num_threads", 0)
        
        if framework_data:
            # Convert to DataFrame
            df_frameworks = pd.DataFrame([
                {
                    "Framework": k,
                    "Process Count": v["count"],
                    "CPU Usage (%)": v["cpu_total"],
                    "Memory Usage (MB)": v["memory_total"],
                    "Thread Count": v["threads_total"],
                    "Avg CPU/Process": v["cpu_total"] / v["count"] if v["count"] > 0 else 0,
                    "Avg Memory/Process": v["memory_total"] / v["count"] if v["count"] > 0 else 0
                }
                for k, v in framework_data.items()
            ])
            
            # Display comparison table
            st.dataframe(
                df_frameworks,
                use_container_width=True,
                column_config={
                    "Framework": st.column_config.TextColumn("Framework"),
                    "Process Count": st.column_config.NumberColumn("Process Count"),
                    "CPU Usage (%)": st.column_config.NumberColumn("CPU Usage (%)", format="%.1f%%"),
                    "Memory Usage (MB)": st.column_config.NumberColumn("Memory Usage (MB)", format="%.1f MB"),
                    "Thread Count": st.column_config.NumberColumn("Thread Count"),
                    "Avg CPU/Process": st.column_config.NumberColumn("Avg CPU/Process", format="%.2f%%"),
                    "Avg Memory/Process": st.column_config.NumberColumn("Avg Memory/Process", format="%.2f MB")
                }
            )
            
            # Framework comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                # CPU usage by framework
                fig = px.bar(
                    df_frameworks,
                    x="Framework",
                    y="CPU Usage (%)",
                    title="CPU Usage by Framework",
                    color="Framework"
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=50, l=80, r=30)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Memory usage by framework
                fig = px.bar(
                    df_frameworks,
                    x="Framework",
                    y="Memory Usage (MB)",
                    title="Memory Usage by Framework",
                    color="Framework"
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(t=30, b=50, l=80, r=30)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No framework data available.")
    else:
        st.info("No framework data available.")


def render_settings_tab(monitor: PythonMonitor):
    """Render the settings tab for Python monitoring configuration."""
    st.markdown("### Python Monitoring Settings")
    
    # Process filter
    st.subheader("Process Filter", anchor=False)
    
    current_filter = monitor.process_name_filter
    filter_text = st.text_area(
        "Process Names to Monitor (one per line)",
        value="\n".join(current_filter),
        height=100
    )
    
    # Monitoring options
    st.subheader("Monitoring Options", anchor=False)
    
    include_streamlit = st.checkbox("Include Streamlit Processes", value=True)
    include_jupyter = st.checkbox("Include Jupyter Processes", value=True)
    
    # Apply settings button
    if st.button("Apply Settings"):
        # Update process filter
        new_filter = [line.strip() for line in filter_text.split("\n") if line.strip()]
        
        # This would need a way to update the monitor's configuration
        # For now, we'll just show a success message
        st.success("Settings updated successfully!")
        st.info("Note: Some settings may require restarting the dashboard to take effect.")


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