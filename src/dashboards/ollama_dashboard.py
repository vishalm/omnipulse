# """
# Ollama dashboard module for displaying Ollama LLM metrics and performance.
# """

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import pandas as pd
# import time
# from datetime import datetime, timedelta
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import pandas as pd
# import time
# from datetime import datetime, timedelta
# import json
# from typing import Dict, List, Any, Optional, Tuple

# from src.monitors.ollama_monitor import OllamaMonitor
# from src.widgets.metric_cards import render_metric_card, render_metric_row
# from src.widgets.charts import render_time_series, render_bar_chart
# from src.utils.helpers import format_number, format_time
# from src.config.settings import get_ollama_api_url
# from src.core.database import db
# from src.core.cache import cached

# def time_range_to_filter(time_range: str) -> dict:
#     """
#     Convert a time range string to a filter dictionary for database queries.
    
#     Args:
#         time_range: Time range string (e.g., "Last hour", "Last 24 hours", etc.)
        
#     Returns:
#         Dictionary with filter parameters
#     """
#     now = datetime.now()
    
#     if time_range == "Last 15 minutes":
#         start_time = now - timedelta(minutes=15)
#     elif time_range == "Last hour":
#         start_time = now - timedelta(hours=1)
#     elif time_range == "Last 3 hours":
#         start_time = now - timedelta(hours=3)
#     elif time_range == "Last 6 hours":
#         start_time = now - timedelta(hours=6)
#     elif time_range == "Last 12 hours":
#         start_time = now - timedelta(hours=12)
#     elif time_range == "Last 24 hours":
#         start_time = now - timedelta(hours=24)
#     elif time_range == "Last 7 days":
#         start_time = now - timedelta(days=7)
#     elif time_range == "Last 30 days":
#         start_time = now - timedelta(days=30)
#     else:
#         # Default to Last hour if not recognized
#         start_time = now - timedelta(hours=1)
    
#     return {
#         "start_time": start_time,
#         "end_time": now
#     }

# def filter_dataframe_by_time(df: pd.DataFrame, time_range: str) -> pd.DataFrame:
#     """
#     Filter a DataFrame based on a time range string.
    
#     Args:
#         df: DataFrame to filter
#         time_range: Time range string
        
#     Returns:
#         Filtered DataFrame
#     """
#     if "timestamp" not in df.columns:
#         return df
    
#     filter_params = time_range_to_filter(time_range)
    
#     # Filter by timestamp
#     return df[(df["timestamp"] >= filter_params["start_time"]) & 
#               (df["timestamp"] <= filter_params["end_time"])]



# async def render_dashboard(monitor: Optional[OllamaMonitor] = None, time_range: str = "Last hour"):
#     """
#     Render the Ollama LLM dashboard.
    
#     Args:
#         monitor: Ollama monitor instance
#         time_range: Time range filter for metrics
#     """
#     st.subheader("Ollama LLM Performance Dashboard", anchor=False)
    
#     # Initialize monitor if not provided
#     if monitor is None:
#         monitor = OllamaMonitor(api_url=get_ollama_api_url())
    
#     # Perform health check
#     with st.spinner("Checking Ollama service..."):
#         health = await monitor.perform_health_check()
    
#     # Health status
#     health_status = health.get("healthy", False)
#     if health_status:
#         st.success("✅ Ollama service is running", icon="✅")
#     else:
#         st.error(f"❌ Ollama service is not available: {health.get('error', 'Unknown error')}", icon="❌")
        
#         # Show connection settings
#         with st.expander("Connection Settings"):
#             current_url = get_ollama_api_url()
#             new_url = st.text_input("Ollama API URL", value=current_url)
#             if st.button("Update Connection") and new_url != current_url:
#                 # Update connection settings
#                 st.session_state.ollama_api_url = new_url
#                 st.success(f"Updated Ollama API URL to {new_url}")
#                 st.experimental_rerun()
        
#         return
    
#     # Tabs for different sections
#     tabs = st.tabs(["Overview", "Models", "Performance", "Requests", "Settings"])
    
#     # OVERVIEW TAB
#     with tabs[0]:
#         await render_overview_tab(monitor, time_range)
    
#     # MODELS TAB
#     with tabs[1]:
#         await render_models_tab(monitor)
    
#     # PERFORMANCE TAB
#     with tabs[2]:
#         await render_performance_tab(monitor, time_range)
    
#     # REQUESTS TAB
#     with tabs[3]:
#         await render_model_tab(monitor, time_range)
    
#     # SETTINGS TAB
#     with tabs[4]:
#         render_settings_tab(monitor)


# async def render_overview_tab(monitor: OllamaMonitor, time_range: str):
#     """Render the overview tab with summary metrics."""
#     # Collect metrics if needed
#     with st.spinner("Collecting Ollama metrics..."):
#         metrics = await monitor.collect_all_metrics()
#         performance_stats = monitor.get_performance_stats(time_range_to_filter(time_range))
#         token_stats = monitor.get_token_usage_stats(time_range_to_filter(time_range))
    
#     # Top metrics row
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         render_metric_card(
#             title="Available Models",
#             value=len(metrics.get("models", [])),
#             description="Total models loaded in Ollama",
#             icon="📚",
#             delta=None
#         )
    
#     with col2:
#         render_metric_card(
#             title="Total Requests",
#             value=performance_stats.get("total_requests", 0),
#             description=f"Requests in {time_range.lower()}",
#             icon="🔄",
#             delta=None
#         )
    
#     with col3:
#         render_metric_card(
#             title="Avg. Latency",
#             value=f"{performance_stats.get('average_latency', 0):.2f}s",
#             description="Average response time",
#             icon="⏱️",
#             delta=None
#         )
    
#     with col4:
#         render_metric_card(
#             title="Success Rate",
#             value=f"{performance_stats.get('success_rate', 0):.1f}%",
#             description="Request success rate",
#             icon="✅",
#             delta=None
#         )
    
#     # Second metrics row
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         render_metric_card(
#             title="Total Tokens",
#             value=format_number(token_stats.get("total_tokens", 0)),
#             description=f"Tokens in {time_range.lower()}",
#             icon="🔤",
#             delta=None
#         )
    
#     with col2:
#         render_metric_card(
#             title="Avg. Throughput",
#             value=f"{performance_stats.get('average_throughput', 0):.1f}",
#             description="Tokens per second",
#             icon="⚡",
#             delta=None
#         )
    
#     with col3:
#         render_metric_card(
#             title="Prompt Tokens",
#             value=format_number(token_stats.get("prompt_tokens", 0)),
#             description="Input tokens used",
#             icon="📥",
#             delta=None
#         )
    
#     with col4:
#         render_metric_card(
#             title="Completion Tokens",
#             value=format_number(token_stats.get("completion_tokens", 0)),
#             description="Output tokens generated",
#             icon="📤",
#             delta=None
#         )
    
#     # Get dataframes for charts
#     dfs = monitor.to_dataframe()
    
#     # Store metrics in database if enabled
#     if hasattr(db, 'store_metric'):
#         # Store overall stats
#         db.store_metric('ollama', 'total_requests', performance_stats.get("total_requests", 0))
#         db.store_metric('ollama', 'average_latency', performance_stats.get("average_latency", 0))
#         db.store_metric('ollama', 'success_rate', performance_stats.get("success_rate", 0))
#         db.store_metric('ollama', 'total_tokens', token_stats.get("total_tokens", 0))
#         db.store_metric('ollama', 'average_throughput', performance_stats.get("average_throughput", 0))
    
#     # Token usage chart
#     st.subheader("Token Usage Over Time", anchor=False)
    
#     if "token_throughput" in dfs and not dfs["token_throughput"].empty:
#         df_tokens = dfs["token_throughput"]
        
#         # Apply time range filter
#         if "timestamp" in df_tokens.columns:
#             df_tokens["timestamp"] = pd.to_datetime(df_tokens["timestamp"])
#             df_tokens = filter_dataframe_by_time(df_tokens, time_range)
        
#         if not df_tokens.empty:
#             # Prepare data for plotting
#             if "model" in df_tokens.columns and "tokens_per_second" in df_tokens.columns:
#                 fig = px.line(
#                     df_tokens, 
#                     x="timestamp", 
#                     y="tokens_per_second",
#                     color="model",
#                     title="Token Throughput Over Time",
#                     labels={"tokens_per_second": "Tokens/second", "timestamp": "Time"},
#                 )
                
#                 fig.update_layout(
#                     height=400,
#                     legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
#                     margin=dict(t=30, b=50, l=80, r=30),
#                 )
                
#                 st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("No token usage data available for the selected time range.")
#     else:
#         st.info("No token usage data available yet. Make some requests to Ollama to see metrics.")
    
#     # Latency chart
#     st.subheader("Response Latency", anchor=False)
    
#     if "latency" in dfs and not dfs["latency"].empty:
#         df_latency = dfs["latency"]
        
#         # Apply time range filter
#         if "timestamp" in df_latency.columns:
#             df_latency["timestamp"] = pd.to_datetime(df_latency["timestamp"])
#             df_latency = filter_dataframe_by_time(df_latency, time_range)
        
#         if not df_latency.empty:
#             # Prepare data for plotting
#             if "model" in df_latency.columns and "latency_seconds" in df_latency.columns:
#                 fig = px.line(
#                     df_latency, 
#                     x="timestamp", 
#                     y="latency_seconds",
#                     color="model",
#                     title="Response Latency Over Time",
#                     labels={"latency_seconds": "Latency (seconds)", "timestamp": "Time"},
#                 )
                
#                 fig.update_layout(
#                     height=400,
#                     legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
#                     margin=dict(t=30, b=50, l=80, r=30),
#                 )
                
#                 st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("No latency data available for the selected time range.")
#     else:
#         st.info("No latency data available yet. Make some requests to Ollama to see metrics.")
    
#     # Quick test section
#     st.subheader("Quick Test", anchor=False)
    
#     # Get available models
#     models = [model["name"] for model in metrics.get("models", [])]
    
#     if models:
#         col1, col2 = st.columns([3, 1])
        
#         with col1:
#             test_prompt = st.text_area("Test Prompt", value="Explain quantum computing in simple terms", height=100)
        
#         with col2:
#             selected_model = st.selectbox("Model", options=models)
#             test_button = st.button("Run Test", type="primary", use_container_width=True)
        
#         if test_button and test_prompt:
#             with st.spinner(f"Testing {selected_model} with prompt..."):
#                 start_time = time.time()
#                 result = await monitor.simulate_request(selected_model, test_prompt)
#                 duration = time.time() - start_time
                
#                 if result.get("success", False):
#                     st.success(f"Test completed in {duration:.2f} seconds")
                    
#                     # Display metrics
#                     metrics_cols = st.columns(3)
#                     with metrics_cols[0]:
#                         st.metric("Prompt Tokens", result.get("prompt_tokens", "N/A"))
#                     with metrics_cols[1]:
#                         st.metric("Completion Tokens", result.get("completion_tokens", "N/A"))
#                     with metrics_cols[2]:
#                         st.metric("Tokens/second", round(result.get("total_tokens", 0) / max(duration, 0.001), 1))
                    
#                     # Display response in expandable panel
#                     with st.expander("View Response", expanded=True):
#                         st.text_area("Response", value=result.get("response", "No response"), height=200, disabled=True)
#                 else:
#                     st.error(f"Test failed: {result.get('error', 'Unknown error')}")
#     else:
#         st.warning("No models available. Please load models in Ollama first.")


# async def render_models_tab(monitor: OllamaMonitor):
#     """Render the models tab with detailed model information."""
#     # Get available models
#     with st.spinner("Loading model information..."):
#         models = await monitor.get_models()
    
#     if not models:
#         st.warning("No models available in Ollama. Please load models first.")
        
#         # Show instructions for loading models
#         with st.expander("How to load models"):
#             st.markdown("""
#             To load models into Ollama, use the following command:
#             ```bash
#             ollama pull <model>
#             ```
            
#             Available models can be found on [Ollama Library](https://ollama.ai/library).
#             """)
        
#         return
    
#     # Model selector
#     selected_model = st.selectbox(
#         "Select Model",
#         options=[model["name"] for model in models],
#         index=0,
#         key="models_tab_model_selector"  # Add this unique key
#     )
#     # Get detailed information for the selected model
#     with st.spinner(f"Loading details for {selected_model}..."):
#         model_info = await monitor.get_model_info(selected_model)
    
#     # Display model information
#     if model_info:
#         # Basic info columns
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             size_gb = model_info.get("size", 0) / (1024 * 1024 * 1024)
#             render_metric_card(
#                 title="Model Size",
#                 value=f"{size_gb:.2f} GB",
#                 description="Disk space used",
#                 icon="💾"
#             )
        
#         with col2:
#             render_metric_card(
#                 title="Parameters",
#                 value=model_info.get("parameter_size", "Unknown"),
#                 description="Model parameters",
#                 icon="🧮"
#             )
        
#         with col3:
#             render_metric_card(
#                 title="Format",
#                 value=model_info.get("format", "Unknown"),
#                 description="Model format",
#                 icon="📄"
#             )
        
#         # Model family and details
#         st.subheader("Model Information", anchor=False)
        
#         # Display model metadata
#         metadata = model_info.get("metadata", {})
        
#         if metadata:
#             info_cols = st.columns(2)
            
#             with info_cols[0]:
#                 st.markdown("#### Model Details")
#                 details = {
#                     "Family": metadata.get("family", "Unknown"),
#                     "Template": metadata.get("template", "N/A"),
#                     "Quantization": metadata.get("quantization", "N/A"),
#                     "Quantization Level": metadata.get("quantization_level", "N/A"),
#                 }
                
#                 for key, value in details.items():
#                     st.markdown(f"**{key}:** {value}")
            
#             with info_cols[1]:
#                 st.markdown("#### System Requirements")
#                 requirements = {
#                     "RAM Required": f"{metadata.get('ram_required', 'Unknown')} GB" if metadata.get('ram_required') else "Unknown",
#                     "VRAM Required": f"{metadata.get('vram_required', 'Unknown')} GB" if metadata.get('vram_required') else "Unknown",
#                     "Parameters": metadata.get("parameter_size", "Unknown"),
#                     "Context Length": metadata.get("context_length", "Unknown"),
#                 }
                
#                 for key, value in details.items():
#                     st.markdown(f"**{key}:** {value}")
        
#         # Model parameters
#         # Replace the problematic code segment in render_models_tab function
#         # around line 457 with this:

#         # Model parameters
#         if "parameters" in model_info:
#             st.markdown("#### Model Parameters")
            
#             params = model_info.get("parameters", {})
            
#             # Check if params is a dictionary
#             if isinstance(params, dict) and params:
#                 param_df = pd.DataFrame({
#                     "Parameter": list(params.keys()),
#                     "Value": list(params.values())
#                 })
                
#                 st.dataframe(param_df, use_container_width=True, hide_index=True)
#             # Check if params is a string
#             elif isinstance(params, str) and params.strip():
#                 # Try to parse it as JSON in case it's a stringified JSON
#                 try:
#                     parsed_params = json.loads(params)
#                     if isinstance(parsed_params, dict):
#                         param_df = pd.DataFrame({
#                             "Parameter": list(parsed_params.keys()),
#                             "Value": list(parsed_params.values())
#                         })
#                         st.dataframe(param_df, use_container_width=True, hide_index=True)
#                     else:
#                         st.text(params)
#                 except json.JSONDecodeError:
#                     # If it's not valid JSON, just display as text
#                     st.text(params)
#             else:
#                 st.info("No parameter information available for this model.")
        
#         # Model files
#         if "modelfile" in model_info:
#             with st.expander("View Modelfile"):
#                 st.code(model_info.get("modelfile", ""), language="dockerfile")
        
#         # Model license
#         if "license" in model_info:
#             with st.expander("View License"):
#                 st.text(model_info.get("license", ""))
#     else:
#         st.error(f"Failed to load detailed information for {selected_model}")
    
#     # Model performance metrics
#     st.subheader("Performance Metrics", anchor=False)
    
#     # Get request history for this model
#     dfs = monitor.to_dataframe()
    
#     if "requests" in dfs and not dfs["requests"].empty:
#         df_requests = dfs["requests"]
        
#         # Filter for selected model
#         if "model" in df_requests.columns:
#             df_model = df_requests[df_requests["model"] == selected_model]
            
#             if not df_model.empty:
#                 # Calculate average metrics
#                 avg_latency = df_model["duration_seconds"].mean() if "duration_seconds" in df_model.columns else 0
                
#                 if "total_tokens" in df_model.columns and "duration_seconds" in df_model.columns:
#                     df_model["tokens_per_second"] = df_model["total_tokens"] / df_model["duration_seconds"].clip(lower=0.001)
#                     avg_throughput = df_model["tokens_per_second"].mean()
#                 else:
#                     avg_throughput = 0
                
#                 # Display metrics
#                 metric_cols = st.columns(3)
                
#                 with metric_cols[0]:
#                     st.metric("Average Latency", f"{avg_latency:.2f}s")
                
#                 with metric_cols[1]:
#                     st.metric("Average Throughput", f"{avg_throughput:.1f} tokens/sec")
                
#                 with metric_cols[2]:
#                     st.metric("Total Requests", len(df_model))
                
#                 # Latency chart
#                 if "timestamp" in df_model.columns and "duration_seconds" in df_model.columns:
#                     df_model["timestamp"] = pd.to_datetime(df_model["timestamp"])
#                     df_model = df_model.sort_values("timestamp")
                    
#                     fig = px.line(
#                         df_model, 
#                         x="timestamp", 
#                         y="duration_seconds",
#                         title="Response Latency Over Time",
#                         labels={"duration_seconds": "Latency (seconds)", "timestamp": "Time"},
#                     )
                    
#                     fig.update_layout(
#                         height=300,
#                         margin=dict(t=30, b=30, l=80, r=30),
#                     )
                    
#                     st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.info(f"No performance data available for {selected_model} yet.")
#         else:
#             st.info("No model-specific performance data available.")
#     else:
#         st.info("No performance data available yet. Make some requests to see metrics.")
    
#     # List of all models
#     st.subheader("All Available Models", anchor=False)
    
#     # Convert models list to DataFrame for display
#     if models:
#         models_data = []
#         for model in models:
#             model_data = {
#                 "Name": model.get("name", "Unknown"),
#                 "Size": f"{model.get('size', 0) / (1024 * 1024 * 1024):.2f} GB" if model.get("size") else "Unknown",
#                 "Modified": model.get("modified_at", "Unknown"),
#                 "Format": model.get("format", "Unknown"),
#             }
#             models_data.append(model_data)
        
#         models_df = pd.DataFrame(models_data)
#         st.dataframe(models_df, use_container_width=True, hide_index=True)
#     else:
#         st.info("No models available.")


# async def render_performance_tab(monitor: OllamaMonitor, time_range: str):
#     """Render the performance tab with detailed performance metrics."""
#     # Get performance statistics
#     with st.spinner("Loading performance metrics..."):
#         performance_stats = monitor.get_performance_stats(time_range_to_filter(time_range))
#         token_stats = monitor.get_token_usage_stats(time_range_to_filter(time_range))
#         dfs = monitor.to_dataframe()
    
#     # Overview metrics
#     st.subheader("Performance Overview", anchor=False)
    
#     # Display metrics in cards
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         render_metric_card(
#             title="Avg. Latency",
#             value=f"{performance_stats.get('average_latency', 0):.2f}s",
#             description="Average response time",
#             icon="⏱️"
#         )
    
#     with col2:
#         render_metric_card(
#             title="Avg. Throughput",
#             value=f"{performance_stats.get('average_throughput', 0):.1f}",
#             description="Tokens per second",
#             icon="⚡"
#         )
    
#     with col3:
#         render_metric_card(
#             title="Success Rate",
#             value=f"{performance_stats.get('success_rate', 0):.1f}%",
#             description="Request success rate",
#             icon="✅"
#         )
    
#     with col4:
#         render_metric_card(
#             title="Error Rate",
#             value=f"{performance_stats.get('error_rate', 0):.1f}%",
#             description="Request error rate",
#             icon="❌"
#         )
    
#     # Performance charts
#     st.subheader("Performance Metrics Over Time", anchor=False)
    
#     chart_tabs = st.tabs(["Latency", "Throughput", "Token Usage", "Success Rate"])
    
#     # Latency chart
#     with chart_tabs[0]:
#         if "latency" in dfs and not dfs["latency"].empty:
#             df_latency = dfs["latency"]
            
#             # Apply time range filter
#             if "timestamp" in df_latency.columns:
#                 df_latency["timestamp"] = pd.to_datetime(df_latency["timestamp"])
#                 df_latency = filter_dataframe_by_time(df_latency, time_range)
            
#             if not df_latency.empty:
#                 # Prepare data for plotting
#                 if "model" in df_latency.columns and "latency_seconds" in df_latency.columns:
#                     # Overall latency trend
#                     fig = px.line(
#                         df_latency, 
#                         x="timestamp", 
#                         y="latency_seconds",
#                         color="model",
#                         title="Response Latency Over Time",
#                         labels={"latency_seconds": "Latency (seconds)", "timestamp": "Time"},
#                     )
                    
#                     fig.update_layout(
#                         height=400,
#                         margin=dict(t=30, b=50, l=80, r=30),
#                     )
                    
#                     st.plotly_chart(fig, use_container_width=True)
                    
#                     # Latency histogram
#                     fig = px.histogram(
#                         df_latency,
#                         x="latency_seconds",
#                         color="model",
#                         nbins=20,
#                         title="Latency Distribution",
#                         labels={"latency_seconds": "Latency (seconds)", "count": "Frequency"},
#                     )
                    
#                     fig.update_layout(
#                         height=300,
#                         margin=dict(t=30, b=50, l=80, r=30),
#                     )
                    
#                     st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.info("No latency data available for the selected time range.")
#         else:
#             st.info("No latency data available yet.")
    
#     # Throughput chart
#     with chart_tabs[1]:
#         if "token_throughput" in dfs and not dfs["token_throughput"].empty:
#             df_throughput = dfs["token_throughput"]
            
#             # Apply time range filter
#             if "timestamp" in df_throughput.columns:
#                 df_throughput["timestamp"] = pd.to_datetime(df_throughput["timestamp"])
#                 df_throughput = filter_dataframe_by_time(df_throughput, time_range)
            
#             if not df_throughput.empty:
#                 # Prepare data for plotting
#                 if "model" in df_throughput.columns and "tokens_per_second" in df_throughput.columns:
#                     # Overall throughput trend
#                     fig = px.line(
#                         df_throughput, 
#                         x="timestamp", 
#                         y="tokens_per_second",
#                         color="model",
#                         title="Token Throughput Over Time",
#                         labels={"tokens_per_second": "Tokens/second", "timestamp": "Time"},
#                     )
                    
#                     fig.update_layout(
#                         height=400,
#                         margin=dict(t=30, b=50, l=80, r=30),
#                     )
                    
#                     st.plotly_chart(fig, use_container_width=True)
                    
#                     # Throughput by model
#                     fig = px.box(
#                         df_throughput,
#                         x="model",
#                         y="tokens_per_second",
#                         title="Token Throughput Distribution by Model",
#                         labels={"tokens_per_second": "Tokens/second", "model": "Model"},
#                     )
                    
#                     fig.update_layout(
#                         height=300,
#                         margin=dict(t=30, b=50, l=80, r=30),
#                     )
                    
#                     st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.info("No throughput data available for the selected time range.")
#         else:
#             st.info("No throughput data available yet.")
    
#     # Token usage chart
#     with chart_tabs[2]:
#         if "requests" in dfs and not dfs["requests"].empty:
#             df_requests = dfs["requests"]
            
#             # Apply time range filter
#             if "timestamp" in df_requests.columns:
#                 df_requests["timestamp"] = pd.to_datetime(df_requests["timestamp"])
#                 df_requests = filter_dataframe_by_time(df_requests, time_range)
            
#             if not df_requests.empty:
#                 # Token usage by model
#                 if "model" in df_requests.columns and "total_tokens" in df_requests.columns:
#                     token_by_model = df_requests.groupby("model")[["total_tokens", "prompt_tokens", "completion_tokens"]].sum().reset_index()
                    
#                     fig = px.bar(
#                         token_by_model,
#                         x="model",
#                         y=["prompt_tokens", "completion_tokens"],
#                         title="Token Usage by Model",
#                         labels={"value": "Tokens", "model": "Model", "variable": "Token Type"},
#                     )
                    
#                     fig.update_layout(
#                         height=400,
#                         margin=dict(t=30, b=50, l=80, r=30),
#                     )
                    
#                     st.plotly_chart(fig, use_container_width=True)
                    
#                     # Cumulative token usage over time
#                     if "timestamp" in df_requests.columns:
#                         df_requests = df_requests.sort_values("timestamp")
#                         df_requests["cumulative_tokens"] = df_requests["total_tokens"].cumsum()
                        
#                         fig = px.line(
#                             df_requests,
#                             x="timestamp",
#                             y="cumulative_tokens",
#                             color="model",
#                             title="Cumulative Token Usage Over Time",
#                             labels={"cumulative_tokens": "Total Tokens", "timestamp": "Time"},
#                         )
                        
#                         fig.update_layout(
#                             height=300,
#                             margin=dict(t=30, b=50, l=80, r=30),
#                         )
                        
#                         st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.info("No token usage data available for the selected time range.")
#         else:
#             st.info("No token usage data available yet.")
    
#     # Success rate chart
#     with chart_tabs[3]:
#         if "requests" in dfs and not dfs["requests"].empty:
#             df_requests = dfs["requests"]
            
#             # Apply time range filter
#             if "timestamp" in df_requests.columns:
#                 df_requests["timestamp"] = pd.to_datetime(df_requests["timestamp"])
#                 df_requests = filter_dataframe_by_time(df_requests, time_range)
            
#             if not df_requests.empty and "success" in df_requests.columns:
#                 # Success rate over time
#                 df_requests["success_int"] = df_requests["success"].astype(int)
#                 df_requests["hour"] = df_requests["timestamp"].dt.floor("H")
                
#                 success_by_hour = df_requests.groupby(["hour", "model"]).agg(
#                     success_rate=("success_int", "mean"),
#                     count=("success_int", "count")
#                 ).reset_index()
                
#                 success_by_hour["success_rate"] = success_by_hour["success_rate"] * 100
                
#                 fig = px.line(
#                     success_by_hour,
#                     x="hour",
#                     y="success_rate",
#                     color="model",
#                     title="Success Rate Over Time",
#                     labels={"success_rate": "Success Rate (%)", "hour": "Time"},
#                 )
                
#                 fig.update_layout(
#                     height=400,
#                     margin=dict(t=30, b=50, l=80, r=30),
#                 )
                
#                 st.plotly_chart(fig, use_container_width=True)
                
#                 # Success rate by model
#                 success_by_model = df_requests.groupby("model").agg(
#                     success_rate=("success_int", "mean"),
#                     count=("success_int", "count")
#                 ).reset_index()
                
#                 success_by_model["success_rate"] = success_by_model["success_rate"] * 100
                
#                 fig = px.bar(
#                     success_by_model,
#                     x="model",
#                     y="success_rate",
#                     text="count",
#                     title="Success Rate by Model",
#                     labels={"success_rate": "Success Rate (%)", "model": "Model", "count": "Requests"},
#                 )
                
#                 fig.update_layout(
#                     height=300,
#                     yaxis_range=[0, 100],
#                     margin=dict(t=30, b=50, l=80, r=30),
#                 )
                
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.info("No success rate data available for the selected time range.")
#         else:
#             st.info("No success rate data available yet.")
    
    
#     # Performance comparison
#     st.subheader("Model Performance Comparison", anchor=False)

#     if "requests" in dfs and not dfs["requests"].empty:
#         df_requests = dfs["requests"]
        
#         # Apply time range filter
#         if "timestamp" in df_requests.columns:
#             df_requests["timestamp"] = pd.to_datetime(df_requests["timestamp"])
#             df_requests = filter_dataframe_by_time(df_requests, time_range)
        
#         if not df_requests.empty and "model" in df_requests.columns:
#             # Create a performance comparison table
#             performance_by_model = df_requests.groupby("model").agg(
#                 avg_latency=("duration_seconds", "mean"),
#                 min_latency=("duration_seconds", "min"),
#                 max_latency=("duration_seconds", "max"),
#                 success_rate=("success", "mean"),
#                 request_count=("request_id", "count")
#             )
            
#             if "total_tokens" in df_requests.columns and "duration_seconds" in df_requests.columns:
#                 # Calculate tokens per second
#                 df_requests["tokens_per_second"] = df_requests["total_tokens"] / df_requests["duration_seconds"].clip(lower=0.001)
                
#                 tokens_by_model = df_requests.groupby("model").agg(
#                     avg_throughput=("tokens_per_second", "mean"),
#                     total_tokens=("total_tokens", "sum")
#                 )
                
#                 performance_by_model = performance_by_model.join(tokens_by_model)
            
#             # Format the table
#             performance_by_model["success_rate"] = performance_by_model["success_rate"] * 100
#             performance_by_model = performance_by_model.round(2)
            
#             # Display the table
#             st.dataframe(performance_by_model, use_container_width=True)
#         else:
#             st.info("No performance comparison data available for the selected time range.")
#     else:
#         st.info("No performance comparison data available yet.")
    
#     async def render_requests_tab(monitor: OllamaMonitor, time_range: str):
#         """Render the requests tab with request history and analytics."""
#         # Get request history
#         with st.spinner("Loading request history..."):
#             dfs = monitor.to_dataframe()
        
#         # Display request metrics
#         st.subheader("Request Analytics", anchor=False)
        
#         if "requests" in dfs and not dfs["requests"].empty:
#             df_requests = dfs["requests"].copy()
            
#             # Apply time range filter
#             if "timestamp" in df_requests.columns:
#                 df_requests["timestamp"] = pd.to_datetime(df_requests["timestamp"])
#                 df_requests = filter_dataframe_by_time(df_requests, time_range)
            
#             if not df_requests.empty:
#                 # Calculate summary metrics
#                 total_requests = len(df_requests)
#                 if "success" in df_requests.columns:
#                     success_count = df_requests["success"].sum()
#                     success_rate = (success_count / total_requests) * 100 if total_requests > 0 else 0
#                     error_count = total_requests - success_count
#                 else:
#                     success_count = 0
#                     success_rate = 0
#                     error_count = 0
                
#                 # Show metrics
#                 col1, col2, col3, col4 = st.columns(4)
                
#                 with col1:
#                     render_metric_card(
#                         title="Total Requests",
#                         value=total_requests,
#                         description=f"In {time_range.lower()}",
#                         icon="🔄"
#                     )
                
#                 with col2:
#                     render_metric_card(
#                         title="Successful",
#                         value=success_count,
#                         description=f"{success_rate:.1f}% success rate",
#                         icon="✅"
#                     )
                
#                 with col3:
#                     render_metric_card(
#                         title="Errors",
#                         value=error_count,
#                         description=f"{100 - success_rate:.1f}% error rate",
#                         icon="❌"
#                     )
                
#                 with col4:
#                     if "duration_seconds" in df_requests.columns:
#                         avg_latency = df_requests["duration_seconds"].mean()
#                         render_metric_card(
#                             title="Avg. Latency",
#                             value=f"{avg_latency:.2f}s",
#                             description="Average response time",
#                             icon="⏱️"
#                         )
#                     else:
#                         render_metric_card(
#                             title="Avg. Latency",
#                             value="N/A",
#                             description="No latency data",
#                             icon="⏱️"
#                         )
                
#                 # Request volume over time
#                 st.subheader("Request Volume", anchor=False)
                
#                 if "timestamp" in df_requests.columns:
#                     # Group by hour and model
#                     df_requests["hour"] = df_requests["timestamp"].dt.floor("H")
#                     requests_by_hour = df_requests.groupby(["hour", "model"]).size().reset_index(name="count")
                    
#                     fig = px.line(
#                         requests_by_hour,
#                         x="hour",
#                         y="count",
#                         color="model",
#                         title="Requests Volume Over Time",
#                         labels={"count": "Number of Requests", "hour": "Time"},
#                     )
                    
#                     fig.update_layout(
#                         height=400,
#                         margin=dict(t=30, b=50, l=80, r=30),
#                     )
                    
#                     st.plotly_chart(fig, use_container_width=True)
                
#                 # Request history table
#                 st.subheader("Request History", anchor=False)
                
#                 # Add filters
#                 col1, col2, col3 = st.columns(3)
                
#                 with col1:
#                     models = ["All"] + sorted(df_requests["model"].unique().tolist())
#                     selected_model = st.selectbox("Filter by Model", models)
                
#                 with col2:
#                     if "success" in df_requests.columns:
#                         status_options = ["All", "Success", "Error"]
#                         selected_status = st.selectbox("Filter by Status", status_options)
#                     else:
#                         selected_status = "All"
                
#                 with col3:
#                     sort_options = ["Newest First", "Oldest First", "Longest Duration", "Shortest Duration"]
#                     sort_option = st.selectbox("Sort by", sort_options)
                
#                 # Apply filters
#                 filtered_df = df_requests.copy()
                
#                 if selected_model != "All":
#                     filtered_df = filtered_df[filtered_df["model"] == selected_model]
                
#                 if selected_status != "All" and "success" in filtered_df.columns:
#                     if selected_status == "Success":
#                         filtered_df = filtered_df[filtered_df["success"] == True]
#                     else:
#                         filtered_df = filtered_df[filtered_df["success"] == False]
                
#                 # Apply sorting
#                 if sort_option == "Newest First":
#                     filtered_df = filtered_df.sort_values("timestamp", ascending=False)
#                 elif sort_option == "Oldest First":
#                     filtered_df = filtered_df.sort_values("timestamp", ascending=True)
#                 elif sort_option == "Longest Duration" and "duration_seconds" in filtered_df.columns:
#                     filtered_df = filtered_df.sort_values("duration_seconds", ascending=False)
#                 elif sort_option == "Shortest Duration" and "duration_seconds" in filtered_df.columns:
#                     filtered_df = filtered_df.sort_values("duration_seconds", ascending=True)
                
#                 # Display the filtered table
#                 if not filtered_df.empty:
#                     # Prepare display columns
#                     display_columns = []
                    
#                     if "timestamp" in filtered_df.columns:
#                         filtered_df["formatted_time"] = filtered_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
#                         display_columns.append("formatted_time")
                    
#                     if "model" in filtered_df.columns:
#                         display_columns.append("model")
                    
#                     if "request_type" in filtered_df.columns:
#                         display_columns.append("request_type")
                    
#                     if "duration_seconds" in filtered_df.columns:
#                         filtered_df["duration_formatted"] = filtered_df["duration_seconds"].apply(lambda x: f"{x:.2f}s")
#                         display_columns.append("duration_formatted")
                    
#                     if "prompt_tokens" in filtered_df.columns:
#                         display_columns.append("prompt_tokens")
                    
#                     if "completion_tokens" in filtered_df.columns:
#                         display_columns.append("completion_tokens")
                    
#                     if "total_tokens" in filtered_df.columns:
#                         display_columns.append("total_tokens")
                    
#                     if "success" in filtered_df.columns:
#                         display_columns.append("success")
                    
#                     # Create display dataframe
#                     if display_columns:
#                         display_df = filtered_df[display_columns].copy()
                        
#                         # Rename columns for better display
#                         column_rename = {
#                             "formatted_time": "Time",
#                             "model": "Model",
#                             "request_type": "Type",
#                             "duration_formatted": "Duration",
#                             "prompt_tokens": "Prompt Tokens",
#                             "completion_tokens": "Completion Tokens",
#                             "total_tokens": "Total Tokens",
#                             "success": "Success"
#                         }
                        
#                         display_df = display_df.rename(columns=column_rename)
                        
#                         # Display the table
#                         st.dataframe(display_df, use_container_width=True, height=400)
                        
#                         # Add export option
#                         if st.button("Export to CSV"):
#                             csv = display_df.to_csv(index=False)
                            
#                             st.download_button(
#                                 label="Download CSV",
#                                 data=csv,
#                                 file_name=f"ollama_requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                                 mime="text/csv"
#                             )
#                     else:
#                         st.info("No columns available to display")
#                 else:
#                     st.info("No requests match the selected filters")
                
#                 # Request details
#                 with st.expander("View Request Details"):
#                     request_id = st.selectbox(
#                         "Select a request to view details",
#                         options=filtered_df["request_id"].tolist() if "request_id" in filtered_df.columns else []
#                     )
                    
#                     if request_id:
#                         selected_request = filtered_df[filtered_df["request_id"] == request_id].iloc[0]
                        
#                         # Display request details
#                         col1, col2 = st.columns(2)
                        
#                         with col1:
#                             st.markdown("#### Request Information")
                            
#                             details = {
#                                 "Request ID": selected_request.get("request_id", "N/A"),
#                                 "Model": selected_request.get("model", "N/A"),
#                                 "Type": selected_request.get("request_type", "N/A"),
#                                 "Time": selected_request.get("formatted_time", "N/A"),
#                                 "Success": "✅ Yes" if selected_request.get("success", False) else "❌ No"
#                             }
                            
#                             for key, value in details.items():
#                                 st.markdown(f"**{key}:** {value}")
                        
#                         with col2:
#                             st.markdown("#### Performance Metrics")
                            
#                             metrics = {
#                                 "Duration": f"{selected_request.get('duration_seconds', 0):.2f} seconds",
#                                 "Prompt Tokens": selected_request.get("prompt_tokens", "N/A"),
#                                 "Completion Tokens": selected_request.get("completion_tokens", "N/A"),
#                                 "Total Tokens": selected_request.get("total_tokens", "N/A"),
#                                 "Tokens/Second": f"{selected_request.get('tokens_per_second', 0):.1f}"
#                             }
                            
#                             for key, value in metrics.items():
#                                 st.markdown(f"**{key}:** {value}")
                        
#                         # Display prompt and response if available
#                         if "prompt" in selected_request:
#                             st.markdown("#### Prompt")
#                             st.text_area("", value=selected_request["prompt"], height=150, disabled=True)
                        
#                         if "response" in selected_request:
#                             st.markdown("#### Response")
#                             st.text_area("", value=selected_request["response"], height=200, disabled=True)
                        
#                         # Display error information if available
#                         if "error" in selected_request and selected_request["error"]:
#                             st.markdown("#### Error Information")
#                             st.error(selected_request["error"])
#             else:
#                 st.info("No request data available for the selected time range.")
#         else:
#             st.info("No request history available yet. Make some requests to Ollama to see data.")
        
#         # Request simulator
#         st.subheader("Request Simulator", anchor=False)
        
#         # Get available models
#         with st.spinner("Loading models..."):
#             models = await monitor.get_models()
        
#         if models:
#             model_names = [model["name"] for model in models]
            
#             col1, col2 = st.columns([3, 1])
            
#             with col1:
#                 sim_prompt = st.text_area("Prompt", height=100)
            
#             with col2:
#                 sim_model = st.selectbox("Model", options=model_names)
#                 sim_button = st.button("Send Request", type="primary", use_container_width=True)
            
#             if sim_button and sim_prompt:
#                 with st.spinner(f"Sending request to {sim_model}..."):
#                     start_time = time.time()
#                     result = await monitor.simulate_request(sim_model, sim_prompt)
#                     duration = time.time() - start_time
                    
#                     if result.get("success", False):
#                         st.success(f"Request completed in {duration:.2f} seconds")
                        
#                         # Display metrics
#                         metrics_cols = st.columns(3)
#                         with metrics_cols[0]:
#                             st.metric("Prompt Tokens", result.get("prompt_tokens", "N/A"))
#                         with metrics_cols[1]:
#                             st.metric("Completion Tokens", result.get("completion_tokens", "N/A"))
#                         with metrics_cols[2]:
#                             st.metric("Tokens/second", round(result.get("total_tokens", 0) / max(duration, 0.001), 1))
                        
#                         # Display response
#                         st.markdown("#### Response")
#                         st.text_area("", value=result.get("response", "No response"), height=200, disabled=True)
#                     else:
#                         st.error(f"Request failed: {result.get('error', 'Unknown error')}")
#         else:
#             st.warning("No models available for testing. Please load models in Ollama first.")

# def render_settings_tab(monitor: OllamaMonitor):
#     """Render the settings tab for Ollama configuration."""
#     st.subheader("Ollama Connection Settings", anchor=False)
    
#     # Current connection info
#     current_url = get_ollama_api_url()
    
#     with st.form("ollama_settings_form"):
#         new_url = st.text_input("Ollama API URL", value=current_url)
        
#         col1, col2 = st.columns(2)
#         with col1:
#             test_button = st.form_submit_button("Test Connection")
#         with col2:
#             save_button = st.form_submit_button("Save Settings")
    
#         if test_button:
#             with st.spinner("Testing connection..."):
#                 # Create a temporary monitor with the new URL
#                 temp_monitor = OllamaMonitor(api_url=new_url)
#                 health = asyncio.run(temp_monitor.perform_health_check())
                
#                 if health.get("healthy", False):
#                     st.success(f"✅ Successfully connected to Ollama at {new_url}")
#                 else:
#                     st.error(f"❌ Failed to connect to Ollama at {new_url}: {health.get('error', 'Unknown error')}")
        
#         if save_button and new_url != current_url:
#             # Update connection settings
#             st.session_state.ollama_api_url = new_url
#             st.success(f"Updated Ollama API URL to {new_url}")
#             st.info("Reload the page to apply changes")
    
#     # Data retention settings
#     st.subheader("Data Retention Settings", anchor=False)
    
#     with st.form("data_retention_form"):
#         current_retention = st.session_state.get("ollama_data_retention_days", 7)
#         retention_days = st.slider(
#             "Data Retention Period (days)",
#             min_value=1,
#             max_value=90,
#             value=current_retention
#         )
        
#         save_retention = st.form_submit_button("Save Retention Settings")
        
#         if save_retention and retention_days != current_retention:
#             st.session_state.ollama_data_retention_days = retention_days
#             st.success(f"Updated data retention period to {retention_days} days")
    
#     # Model management
#     st.subheader("Model Management", anchor=False)
    
#     # Get available models
#     with st.spinner("Loading model information..."):
#         models = asyncio.run(monitor.get_models())
    
#     if models:
#         # Convert models list to DataFrame for display
#         models_data = []
#         for model in models:
#             model_data = {
#                 "Name": model.get("name", "Unknown"),
#                 "Size": f"{model.get('size', 0) / (1024 * 1024 * 1024):.2f} GB" if model.get("size") else "Unknown",
#                 "Modified": model.get("modified_at", "Unknown"),
#             }
#             models_data.append(model_data)
        
#         models_df = pd.DataFrame(models_data)
        
#         # Display models with actions
#         edited_df = st.data_editor(
#             models_df,
#             column_config={
#                 "Name": st.column_config.TextColumn("Model Name"),
#                 "Size": st.column_config.TextColumn("Size"),
#                 "Modified": st.column_config.TextColumn("Last Modified"),
#                 "Action": st.column_config.SelectboxColumn(
#                     "Action",
#                     options=["None", "Delete"],
#                     required=True,
#                     default="None"
#                 )
#             },
#             hide_index=True,
#             use_container_width=True
#         )
        
#         # Handle model actions
#         if st.button("Apply Actions"):
#             for i, row in edited_df.iterrows():
#                 if row.get("Action") == "Delete":
#                     model_name = row.get("Name")
#                     st.warning(f"Would delete model: {model_name}")
#                     # In a real implementation, you would call an API to delete the model
#                     # monitor.delete_model(model_name)
#     else:
#         st.info("No models available in Ollama")
    
#     # Advanced settings
#     with st.expander("Advanced Settings"):
#         st.caption("These settings are for advanced users and may affect system performance.")
        
#         monitoring_interval = st.session_state.get("ollama_monitoring_interval", 5)
#         new_interval = st.slider(
#             "Monitoring Interval (seconds)",
#             min_value=1,
#             max_value=60,
#             value=monitoring_interval
#         )
        
#         if st.button("Save Advanced Settings"):
#             st.session_state.ollama_monitoring_interval = new_interval
#             st.success(f"Updated monitoring interval to {new_interval} seconds")
#             st.info("Changes will take effect after restarting the application")
    
#     # Debug information
#     with st.expander("Debug Information"):
#         st.code(f"""
#         Ollama API URL: {current_url}
#         Dashboard Version: 0.1.0
#         Python Version: {sys.version}
#         Operating System: {platform.platform()}
#         """)

# async def render_model_tab(monitor: OllamaMonitor, time_range: str):
#     """
#     Render the detailed model tab with specific model analytics and performance.
    
#     Args:
#         monitor: Ollama monitor instance
#         time_range: Time range filter for metrics
#     """
#     st.subheader("Model Analytics", anchor=False)
    
#     # Get available models
#     with st.spinner("Loading model information..."):
#         models = await monitor.get_models()
#         dfs = monitor.to_dataframe()
    
#     if not models:
#         st.warning("No models available in Ollama. Please load models first.")
#         return
    
#     # Model selector
#     model_names = [model["name"] for model in models]
#     selected_model = st.selectbox(
#         "Select Model",
#         options=model_names,
#         index=0,
#         key="model_tab_model_selector"  # Add this unique key
#     )
    
#     # Get detailed information for the selected model
#     with st.spinner(f"Loading details for {selected_model}..."):
#         model_info = await monitor.get_model_info(selected_model)
    
#     # Display model basic info
#     if model_info:
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             size_gb = model_info.get("size", 0) / (1024 * 1024 * 1024)
#             render_metric_card(
#                 title="Model Size",
#                 value=f"{size_gb:.2f} GB",
#                 description="Disk space used",
#                 icon="💾"
#             )
        
#         with col2:
#             param_size = model_info.get("parameter_size", "Unknown")
#             if isinstance(param_size, (int, float)):
#                 param_display = f"{param_size / 1e9:.1f}B" if param_size >= 1e9 else param_size
#             else:
#                 param_display = param_size
                
#             render_metric_card(
#                 title="Parameters",
#                 value=param_display,
#                 description="Model parameters",
#                 icon="🧮"
#             )
        
#         with col3:
#             render_metric_card(
#                 title="Format",
#                 value=model_info.get("format", "Unknown"),
#                 description="Model format",
#                 icon="📄"
#             )
    
#     # Filter data for selected model
#     if "requests" in dfs and not dfs["requests"].empty:
#         df_requests = dfs["requests"].copy()
        
#         # Apply model filter
#         if "model" in df_requests.columns:
#             df_model = df_requests[df_requests["model"] == selected_model]
            
#             # Apply time range filter
#             if "timestamp" in df_model.columns:
#                 df_model["timestamp"] = pd.to_datetime(df_model["timestamp"])
#                 df_model = filter_dataframe_by_time(df_model, time_range)
            
#             if not df_model.empty:
#                 # Performance metrics
#                 st.subheader("Performance Metrics", anchor=False)
                
#                 # Calculate metrics
#                 request_count = len(df_model)
                
#                 if "success" in df_model.columns:
#                     success_count = df_model["success"].sum()
#                     success_rate = (success_count / request_count) * 100 if request_count > 0 else 0
#                 else:
#                     success_count = 0
#                     success_rate = 0
                
#                 if "duration_seconds" in df_model.columns:
#                     avg_latency = df_model["duration_seconds"].mean()
#                     min_latency = df_model["duration_seconds"].min()
#                     max_latency = df_model["duration_seconds"].max()
#                     p95_latency = df_model["duration_seconds"].quantile(0.95)
#                 else:
#                     avg_latency = min_latency = max_latency = p95_latency = 0
                
#                 if "total_tokens" in df_model.columns and "duration_seconds" in df_model.columns:
#                     df_model["tokens_per_second"] = df_model["total_tokens"] / df_model["duration_seconds"].clip(lower=0.001)
#                     avg_throughput = df_model["tokens_per_second"].mean()
#                     total_tokens = df_model["total_tokens"].sum()
#                     prompt_tokens = df_model["prompt_tokens"].sum() if "prompt_tokens" in df_model.columns else 0
#                     completion_tokens = df_model["completion_tokens"].sum() if "completion_tokens" in df_model.columns else 0
#                 else:
#                     avg_throughput = total_tokens = prompt_tokens = completion_tokens = 0
                
#                 # Display metrics in cards
#                 col1, col2, col3, col4 = st.columns(4)
                
#                 with col1:
#                     render_metric_card(
#                         title="Requests",
#                         value=request_count,
#                         description=f"In {time_range.lower()}",
#                         icon="🔄"
#                     )
                
#                 with col2:
#                     render_metric_card(
#                         title="Success Rate",
#                         value=f"{success_rate:.1f}%",
#                         description=f"{success_count} successful requests",
#                         icon="✅"
#                     )
                
#                 with col3:
#                     render_metric_card(
#                         title="Avg. Latency",
#                         value=f"{avg_latency:.2f}s",
#                         description=f"P95: {p95_latency:.2f}s",
#                         icon="⏱️"
#                     )
                
#                 with col4:
#                     render_metric_card(
#                         title="Avg. Throughput",
#                         value=f"{avg_throughput:.1f}",
#                         description="Tokens per second",
#                         icon="⚡"
#                     )
                
#                 # Second row of metrics
#                 col1, col2, col3, col4 = st.columns(4)
                
#                 with col1:
#                     render_metric_card(
#                         title="Total Tokens",
#                         value=format_number(total_tokens),
#                         description=f"In {time_range.lower()}",
#                         icon="🔤"
#                     )
                
#                 with col2:
#                     render_metric_card(
#                         title="Prompt Tokens",
#                         value=format_number(prompt_tokens),
#                         description=f"{(prompt_tokens / total_tokens * 100):.1f}% of total" if total_tokens > 0 else "0% of total",
#                         icon="📥"
#                     )
                
#                 with col3:
#                     render_metric_card(
#                         title="Completion Tokens",
#                         value=format_number(completion_tokens),
#                         description=f"{(completion_tokens / total_tokens * 100):.1f}% of total" if total_tokens > 0 else "0% of total",
#                         icon="📤"
#                     )
                
#                 with col4:
#                     if "duration_seconds" in df_model.columns:
#                         total_duration = df_model["duration_seconds"].sum()
#                         render_metric_card(
#                             title="Total Duration",
#                             value=f"{total_duration:.1f}s",
#                             description=f"Avg: {avg_latency:.2f}s per request",
#                             icon="⏰"
#                         )
#                     else:
#                         render_metric_card(
#                             title="Total Duration",
#                             value="N/A",
#                             description="No duration data available",
#                             icon="⏰"
#                         )
                
#                 # Performance charts
#                 st.subheader("Performance Charts", anchor=False)
                
#                 chart_tabs = st.tabs(["Latency", "Throughput", "Usage Over Time"])
                
#                 # Latency chart
#                 with chart_tabs[0]:
#                     if "timestamp" in df_model.columns and "duration_seconds" in df_model.columns:
#                         # Latency over time
#                         fig = px.line(
#                             df_model.sort_values("timestamp"),
#                             x="timestamp",
#                             y="duration_seconds",
#                             title="Latency Over Time",
#                             labels={"duration_seconds": "Latency (seconds)", "timestamp": "Time"},
#                         )
                        
#                         fig.update_layout(
#                             height=350,
#                             margin=dict(t=30, b=30, l=80, r=30),
#                         )
                        
#                         st.plotly_chart(fig, use_container_width=True)
                        
#                         # Latency distribution
#                         fig = px.histogram(
#                             df_model,
#                             x="duration_seconds",
#                             nbins=20,
#                             title="Latency Distribution",
#                             labels={"duration_seconds": "Latency (seconds)", "count": "Frequency"},
#                         )
                        
#                         fig.update_layout(
#                             height=300,
#                             margin=dict(t=30, b=30, l=80, r=30),
#                         )
                        
#                         st.plotly_chart(fig, use_container_width=True)
#                     else:
#                         st.info("No latency data available for this model.")
                
#                 # Throughput chart
#                 with chart_tabs[1]:
#                     if "timestamp" in df_model.columns and "tokens_per_second" in df_model.columns:
#                         # Throughput over time
#                         fig = px.line(
#                             df_model.sort_values("timestamp"),
#                             x="timestamp",
#                             y="tokens_per_second",
#                             title="Token Throughput Over Time",
#                             labels={"tokens_per_second": "Tokens/second", "timestamp": "Time"},
#                         )
                        
#                         fig.update_layout(
#                             height=350,
#                             margin=dict(t=30, b=30, l=80, r=30),
#                         )
                        
#                         st.plotly_chart(fig, use_container_width=True)
                        
#                         # Throughput distribution
#                         fig = px.histogram(
#                             df_model,
#                             x="tokens_per_second",
#                             nbins=20,
#                             title="Throughput Distribution",
#                             labels={"tokens_per_second": "Tokens/second", "count": "Frequency"},
#                         )
                        
#                         fig.update_layout(
#                             height=300,
#                             margin=dict(t=30, b=30, l=80, r=30),
#                         )
                        
#                         st.plotly_chart(fig, use_container_width=True)
#                     else:
#                         st.info("No throughput data available for this model.")
                
#                 # Usage over time
#                 with chart_tabs[2]:
#                     if "timestamp" in df_model.columns:
#                         # Group by hour
#                         df_model["hour"] = df_model["timestamp"].dt.floor("H")
#                         usage_by_hour = df_model.groupby("hour").agg(
#                             request_count=("model", "count")
#                         ).reset_index()
                        
#                         if "total_tokens" in df_model.columns:
#                             tokens_by_hour = df_model.groupby("hour").agg(
#                                 total_tokens=("total_tokens", "sum"),
#                                 prompt_tokens=("prompt_tokens", "sum") if "prompt_tokens" in df_model.columns else None,
#                                 completion_tokens=("completion_tokens", "sum") if "completion_tokens" in df_model.columns else None,
#                             ).reset_index()
                            
#                             # Create a figure with two y-axes
#                             fig = make_subplots(specs=[[{"secondary_y": True}]])
                            
#                             # Add request count trace on primary y-axis
#                             fig.add_trace(
#                                 go.Scatter(
#                                     x=usage_by_hour["hour"],
#                                     y=usage_by_hour["request_count"],
#                                     name="Requests",
#                                     line=dict(color=COLORS['primary'], width=2),
#                                 ),
#                                 secondary_y=False,
#                             )
                            
#                             # Add token count trace on secondary y-axis
#                             fig.add_trace(
#                                 go.Scatter(
#                                     x=tokens_by_hour["hour"],
#                                     y=tokens_by_hour["total_tokens"],
#                                     name="Tokens",
#                                     line=dict(color=COLORS['secondary'], width=2),
#                                 ),
#                                 secondary_y=True,
#                             )
                            
#                             # Set titles
#                             fig.update_layout(
#                                 title_text="Model Usage Over Time",
#                                 height=400,
#                                 margin=dict(t=30, b=30, l=80, r=80),
#                             )
                            
#                             # Set y-axes titles
#                             fig.update_yaxes(title_text="Request Count", secondary_y=False)
#                             fig.update_yaxes(title_text="Token Count", secondary_y=True)
                            
#                             st.plotly_chart(fig, use_container_width=True)
                            
#                             # Token breakdown over time
#                             if "prompt_tokens" in tokens_by_hour.columns and "completion_tokens" in tokens_by_hour.columns:
#                                 fig = px.area(
#                                     tokens_by_hour,
#                                     x="hour",
#                                     y=["prompt_tokens", "completion_tokens"],
#                                     title="Token Usage Breakdown Over Time",
#                                     labels={"value": "Token Count", "hour": "Time", "variable": "Token Type"},
#                                 )
                                
#                                 fig.update_layout(
#                                     height=350,
#                                     margin=dict(t=30, b=30, l=80, r=30),
#                                 )
                                
#                                 st.plotly_chart(fig, use_container_width=True)
#                         else:
#                             # Just show request count
#                             fig = px.bar(
#                                 usage_by_hour,
#                                 x="hour",
#                                 y="request_count",
#                                 title="Request Count Over Time",
#                                 labels={"request_count": "Request Count", "hour": "Time"},
#                             )
                            
#                             fig.update_layout(
#                                 height=400,
#                                 margin=dict(t=30, b=30, l=80, r=30),
#                             )
                            
#                             st.plotly_chart(fig, use_container_width=True)
#                     else:
#                         st.info("No usage data available for this model.")
                
#                 # Benchmark
#                 st.subheader("Model Benchmark", anchor=False)
                
#                 col1, col2 = st.columns([3, 1])
                
#                 with col1:
#                     benchmark_text = st.text_area(
#                         "Benchmark Prompt",
#                         value="Explain the theory of relativity in simple terms.",
#                         height=100
#                     )
                
#                 with col2:
#                     num_runs = st.number_input("Number of Runs", min_value=1, max_value=10, value=3)
#                     run_benchmark = st.button("Run Benchmark", type="primary", use_container_width=True)
                
#                 if run_benchmark and benchmark_text:
#                     progress_bar = st.progress(0)
#                     results = []
                    
#                     for i in range(num_runs):
#                         with st.spinner(f"Running benchmark {i+1}/{num_runs}..."):
#                             start_time = time.time()
#                             result = await monitor.simulate_request(selected_model, benchmark_text)
#                             duration = time.time() - start_time
                            
#                             if result.get("success", False):
#                                 results.append({
#                                     "run": i+1,
#                                     "duration": duration,
#                                     "prompt_tokens": result.get("prompt_tokens", 0),
#                                     "completion_tokens": result.get("completion_tokens", 0),
#                                     "total_tokens": result.get("total_tokens", 0),
#                                     "tokens_per_second": result.get("total_tokens", 0) / max(duration, 0.001)
#                                 })
                        
#                         progress_bar.progress((i+1) / num_runs)
                    
#                     if results:
#                         # Calculate average metrics
#                         avg_duration = sum(r["duration"] for r in results) / len(results)
#                         avg_throughput = sum(r["tokens_per_second"] for r in results) / len(results)
                        
#                         # Display results
#                         st.success(f"Benchmark completed: Average response time {avg_duration:.2f}s")
                        
#                         # Display summary metrics
#                         col1, col2, col3 = st.columns(3)
                        
#                         with col1:
#                             st.metric("Avg. Latency", f"{avg_duration:.2f}s")
                        
#                         with col2:
#                             st.metric("Avg. Throughput", f"{avg_throughput:.1f} tokens/sec")
                        
#                         with col3:
#                             st.metric("Total Runs", num_runs)
                        
#                         # Display detailed results
#                         results_df = pd.DataFrame(results)
#                         st.dataframe(results_df, use_container_width=True)
                        
#                         # Create comparison chart
#                         fig = px.bar(
#                             results_df,
#                             x="run",
#                             y="duration",
#                             title="Response Time by Run",
#                             labels={"duration": "Duration (seconds)", "run": "Run"},
#                         )
                        
#                         fig.update_layout(
#                             height=300,
#                             margin=dict(t=30, b=30, l=80, r=30),
#                         )
                        
#                         st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.info(f"No request data available for model {selected_model} in the selected time range.")
#         else:
#             st.info(f"No request data available for model {selected_model}.")
#     else:
#         st.info("No request data available. Make some requests to Ollama to see model analytics.")


"""
Ollama dashboard module for displaying Ollama LLM metrics and performance.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import sys
import platform
import asyncio
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional, Tuple

from src.monitors.ollama_monitor import OllamaMonitor
from src.widgets.metric_cards import render_metric_card, render_metric_row
from src.widgets.charts import render_time_series, render_bar_chart
from src.utils.helpers import format_number, format_time
from src.config.settings import get_ollama_api_url
from src.core.database import db
from src.core.cache import cached

# Define color scheme for consistent visualization
COLORS = {
    'primary': '#3366CC',
    'secondary': '#DC3912',
    'tertiary': '#FF9900',
    'success': '#109618',
    'warning': '#FF9900',
    'danger': '#DC3912',
    'info': '#3366CC',
    'light': '#AAAAAA',
    'dark': '#333333',
    'background': '#F9F9F9',
    'grid': '#DDDDDD',
    'text': '#333333'
}

def time_range_to_filter(time_range: str) -> dict:
    """
    Convert a time range string to a filter dictionary for database queries.
    
    Args:
        time_range: Time range string (e.g., "Last hour", "Last 24 hours", etc.)
        
    Returns:
        Dictionary with filter parameters
    """
    now = datetime.now()
    
    if time_range == "Last 15 minutes":
        start_time = now - timedelta(minutes=15)
    elif time_range == "Last hour":
        start_time = now - timedelta(hours=1)
    elif time_range == "Last 3 hours":
        start_time = now - timedelta(hours=3)
    elif time_range == "Last 6 hours":
        start_time = now - timedelta(hours=6)
    elif time_range == "Last 12 hours":
        start_time = now - timedelta(hours=12)
    elif time_range == "Last 24 hours":
        start_time = now - timedelta(hours=24)
    elif time_range == "Last 7 days":
        start_time = now - timedelta(days=7)
    elif time_range == "Last 30 days":
        start_time = now - timedelta(days=30)
    else:
        # Default to Last hour if not recognized
        start_time = now - timedelta(hours=1)
    
    return {
        "start_time": start_time,
        "end_time": now
    }

def filter_dataframe_by_time(df: pd.DataFrame, time_range: str) -> pd.DataFrame:
    """
    Filter a DataFrame based on a time range string.
    
    Args:
        df: DataFrame to filter
        time_range: Time range string
        
    Returns:
        Filtered DataFrame
    """
    if "timestamp" not in df.columns:
        return df
    
    filter_params = time_range_to_filter(time_range)
    
    # Filter by timestamp
    return df[(df["timestamp"] >= filter_params["start_time"]) & 
              (df["timestamp"] <= filter_params["end_time"])]

def render_settings_tab(monitor: OllamaMonitor):
    """Render the settings tab for Ollama configuration."""
    st.subheader("Ollama Connection Settings", anchor=False)
    
    # Current connection info
    current_url = get_ollama_api_url()
    
    with st.form("ollama_settings_form"):
        new_url = st.text_input("Ollama API URL", value=current_url)
        
        col1, col2 = st.columns(2)
        with col1:
            test_button = st.form_submit_button("Test Connection")
        with col2:
            save_button = st.form_submit_button("Save Settings")
    
        if test_button:
            with st.spinner("Testing connection..."):
                # Create a temporary monitor with the new URL
                temp_monitor = OllamaMonitor(api_url=new_url)
                # Don't use asyncio.run() here!
                health = {"healthy": False, "error": "Connection testing not available in this view"}
                try:
                    # Try a simpler test
                    import requests
                    response = requests.get(f"{new_url}/api/tags", timeout=3)
                    if response.status_code == 200:
                        health = {"healthy": True}
                except Exception as e:
                    health = {"healthy": False, "error": str(e)}
                
                if health.get("healthy", False):
                    st.success(f"✅ Successfully connected to Ollama at {new_url}")
                else:
                    st.error(f"❌ Failed to connect to Ollama at {new_url}: {health.get('error', 'Unknown error')}")
        
        if save_button and new_url != current_url:
            # Update connection settings
            st.session_state.ollama_api_url = new_url
            st.success(f"Updated Ollama API URL to {new_url}")
            st.info("Reload the page to apply changes")
    
    # Data retention settings
    st.subheader("Data Retention Settings", anchor=False)
    
    with st.form("data_retention_form"):
        current_retention = st.session_state.get("ollama_data_retention_days", 7)
        retention_days = st.slider(
            "Data Retention Period (days)",
            min_value=1,
            max_value=90,
            value=current_retention
        )
        
        save_retention = st.form_submit_button("Save Retention Settings")
        
        if save_retention and retention_days != current_retention:
            st.session_state.ollama_data_retention_days = retention_days
            st.success(f"Updated data retention period to {retention_days} days")
    
    # Model management
    st.subheader("Model Management", anchor=False)
    
    # Get available models - don't use asyncio.run()!
    models = []
    try:
        # Use the cached models data that should be already available in the monitor
        models = monitor.models if hasattr(monitor, 'models') and monitor.models else []
        
        # If no cached data, display a message
        if not models:
            st.info("Loading models information not available in settings view. Please check the Models tab.")
    except Exception as e:
        st.error(f"Could not load model information: {str(e)}")
    
    if models:
        # Convert models list to DataFrame for display
        models_data = []
        for model in models:
            model_data = {
                "Name": model.get("name", "Unknown"),
                "Size": f"{model.get('size', 0) / (1024 * 1024 * 1024):.2f} GB" if model.get("size") else "Unknown",
                "Modified": model.get("modified_at", "Unknown"),
            }
            models_data.append(model_data)
        
        models_df = pd.DataFrame(models_data)
        
        # Display models with actions
        edited_df = st.data_editor(
            models_df,
            column_config={
                "Name": st.column_config.TextColumn("Model Name"),
                "Size": st.column_config.TextColumn("Size"),
                "Modified": st.column_config.TextColumn("Last Modified"),
                "Action": st.column_config.SelectboxColumn(
                    "Action",
                    options=["None", "Delete"],
                    required=True,
                    default="None"
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Handle model actions
        if st.button("Apply Actions"):
            for i, row in edited_df.iterrows():
                if row.get("Action") == "Delete":
                    model_name = row.get("Name")
                    st.warning(f"Would delete model: {model_name}")
                    # In a real implementation, you would call an API to delete the model
                    # monitor.delete_model(model_name)
    else:
        st.info("No models available in Ollama")
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        st.caption("These settings are for advanced users and may affect system performance.")
        
        monitoring_interval = st.session_state.get("ollama_monitoring_interval", 5)
        new_interval = st.slider(
            "Monitoring Interval (seconds)",
            min_value=1,
            max_value=60,
            value=monitoring_interval
        )
        
        if st.button("Save Advanced Settings"):
            st.session_state.ollama_monitoring_interval = new_interval
            st.success(f"Updated monitoring interval to {new_interval} seconds")
            st.info("Changes will take effect after restarting the application")
    
    # Debug information
    with st.expander("Debug Information"):
        st.code(f"""
        Ollama API URL: {current_url}
        Dashboard Version: 0.1.0
        Python Version: {sys.version}
        Operating System: {platform.platform()}
        """)


async def render_dashboard(monitor: Optional[OllamaMonitor] = None, time_range: str = "Last hour"):
    """
    Render the Ollama LLM dashboard.
    
    Args:
        monitor: Ollama monitor instance
        time_range: Time range filter for metrics
    """
    st.subheader("Ollama LLM Performance Dashboard", anchor=False)
    
    # Initialize monitor if not provided
    if monitor is None:
        monitor = OllamaMonitor(api_url=get_ollama_api_url())
    
    # Perform health check
    with st.spinner("Checking Ollama service..."):
        health = await monitor.perform_health_check()
    
    # Health status
    health_status = health.get("healthy", False)
    if health_status:
        st.success("✅ Ollama service is running", icon="✅")
    else:
        st.error(f"❌ Ollama service is not available: {health.get('error', 'Unknown error')}", icon="❌")
        
        # Show connection settings
        with st.expander("Connection Settings"):
            current_url = get_ollama_api_url()
            new_url = st.text_input("Ollama API URL", value=current_url, key="health_check_api_url")
            if st.button("Update Connection", key="health_check_update_btn") and new_url != current_url:
                # Update connection settings
                st.session_state.ollama_api_url = new_url
                st.success(f"Updated Ollama API URL to {new_url}")
                st.experimental_rerun()
        
        return
    
    # Tabs for different sections
    tabs = st.tabs(["Overview", "Models", "Performance", "Requests", "Settings"])
    
    # OVERVIEW TAB
    with tabs[0]:
        await render_overview_tab(monitor, time_range)
    
    # MODELS TAB
    with tabs[1]:
        await render_models_tab(monitor)
    
    # PERFORMANCE TAB
    with tabs[2]:
        await render_performance_tab(monitor, time_range)
    
    # REQUESTS TAB
    with tabs[3]:
        await render_requests_tab(monitor, time_range)
    
    # SETTINGS TAB
    with tabs[4]:
        render_settings_tab(monitor)


async def render_overview_tab(monitor: OllamaMonitor, time_range: str):
    """Render the overview tab with summary metrics."""
    # Collect metrics if needed
    with st.spinner("Collecting Ollama metrics..."):
        metrics = await monitor.collect_all_metrics()
        performance_stats = monitor.get_performance_stats(time_range_to_filter(time_range))
        token_stats = monitor.get_token_usage_stats(time_range_to_filter(time_range))
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card(
            title="Available Models",
            value=len(metrics.get("models", [])),
            description="Total models loaded in Ollama",
            icon="📚",
            delta=None
        )
    
    with col2:
        render_metric_card(
            title="Total Requests",
            value=performance_stats.get("total_requests", 0),
            description=f"Requests in {time_range.lower()}",
            icon="🔄",
            delta=None
        )
    
    with col3:
        render_metric_card(
            title="Avg. Latency",
            value=f"{performance_stats.get('average_latency', 0):.2f}s",
            description="Average response time",
            icon="⏱️",
            delta=None
        )
    
    with col4:
        render_metric_card(
            title="Success Rate",
            value=f"{performance_stats.get('success_rate', 0):.1f}%",
            description="Request success rate",
            icon="✅",
            delta=None
        )
    
    # Second metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card(
            title="Total Tokens",
            value=format_number(token_stats.get("total_tokens", 0)),
            description=f"Tokens in {time_range.lower()}",
            icon="🔤",
            delta=None
        )
    
    with col2:
        render_metric_card(
            title="Avg. Throughput",
            value=f"{performance_stats.get('average_throughput', 0):.1f}",
            description="Tokens per second",
            icon="⚡",
            delta=None
        )
    
    with col3:
        render_metric_card(
            title="Prompt Tokens",
            value=format_number(token_stats.get("prompt_tokens", 0)),
            description="Input tokens used",
            icon="📥",
            delta=None
        )
    
    with col4:
        render_metric_card(
            title="Completion Tokens",
            value=format_number(token_stats.get("completion_tokens", 0)),
            description="Output tokens generated",
            icon="📤",
            delta=None
        )
    
    # Get dataframes for charts
    dfs = monitor.to_dataframe()
    
    # Store metrics in database if enabled
    if hasattr(db, 'store_metric'):
        # Store overall stats
        db.store_metric('ollama', 'total_requests', performance_stats.get("total_requests", 0))
        db.store_metric('ollama', 'average_latency', performance_stats.get("average_latency", 0))
        db.store_metric('ollama', 'success_rate', performance_stats.get("success_rate", 0))
        db.store_metric('ollama', 'total_tokens', token_stats.get("total_tokens", 0))
        db.store_metric('ollama', 'average_throughput', performance_stats.get("average_throughput", 0))
    
    # Token usage chart
    st.subheader("Token Usage Over Time", anchor=False)
    
    if "token_throughput" in dfs and not dfs["token_throughput"].empty:
        df_tokens = dfs["token_throughput"]
        
        # Apply time range filter
        if "timestamp" in df_tokens.columns:
            df_tokens["timestamp"] = pd.to_datetime(df_tokens["timestamp"])
            df_tokens = filter_dataframe_by_time(df_tokens, time_range)
        
        if not df_tokens.empty:
            # Prepare data for plotting
            if "model" in df_tokens.columns and "tokens_per_second" in df_tokens.columns:
                fig = px.line(
                    df_tokens, 
                    x="timestamp", 
                    y="tokens_per_second",
                    color="model",
                    title="Token Throughput Over Time",
                    labels={"tokens_per_second": "Tokens/second", "timestamp": "Time"},
                )
                
                fig.update_layout(
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    margin=dict(t=30, b=50, l=80, r=30),
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No token usage data available for the selected time range.")
    else:
        st.info("No token usage data available yet. Make some requests to Ollama to see metrics.")
    
    # Latency chart
    st.subheader("Response Latency", anchor=False)
    
    if "latency" in dfs and not dfs["latency"].empty:
        df_latency = dfs["latency"]
        
        # Apply time range filter
        if "timestamp" in df_latency.columns:
            df_latency["timestamp"] = pd.to_datetime(df_latency["timestamp"])
            df_latency = filter_dataframe_by_time(df_latency, time_range)
        
        if not df_latency.empty:
            # Prepare data for plotting
            if "model" in df_latency.columns and "latency_seconds" in df_latency.columns:
                fig = px.line(
                    df_latency, 
                    x="timestamp", 
                    y="latency_seconds",
                    color="model",
                    title="Response Latency Over Time",
                    labels={"latency_seconds": "Latency (seconds)", "timestamp": "Time"},
                )
                
                fig.update_layout(
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    margin=dict(t=30, b=50, l=80, r=30),
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No latency data available for the selected time range.")
    else:
        st.info("No latency data available yet. Make some requests to Ollama to see metrics.")
    
    # Quick test section
    st.subheader("Quick Test", anchor=False)
    
    # Get available models
    models = [model["name"] for model in metrics.get("models", [])]
    
    if models:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            test_prompt = st.text_area("Test Prompt", value="Explain quantum computing in simple terms", height=100, key="overview_test_prompt")
        
        with col2:
            selected_model = st.selectbox("Model", options=models, key="overview_model_selector")
            test_button = st.button("Run Test", type="primary", use_container_width=True, key="overview_test_button")
        
        if test_button and test_prompt:
            with st.spinner(f"Testing {selected_model} with prompt..."):
                start_time = time.time()
                result = await monitor.simulate_request(selected_model, test_prompt)
                duration = time.time() - start_time
                
                if result.get("success", False):
                    st.success(f"Test completed in {duration:.2f} seconds")
                    
                    # Display metrics
                    metrics_cols = st.columns(3)
                    with metrics_cols[0]:
                        st.metric("Prompt Tokens", result.get("prompt_tokens", "N/A"))
                    with metrics_cols[1]:
                        st.metric("Completion Tokens", result.get("completion_tokens", "N/A"))
                    with metrics_cols[2]:
                        st.metric("Tokens/second", round(result.get("total_tokens", 0) / max(duration, 0.001), 1))
                    
                    # Display response in expandable panel
                    with st.expander("View Response", expanded=True):
                        st.text_area("Response", value=result.get("response", "No response"), height=200, disabled=True, key="overview_response_area")
                else:
                    st.error(f"Test failed: {result.get('error', 'Unknown error')}")
    else:
        st.warning("No models available. Please load models in Ollama first.")


async def render_models_tab(monitor: OllamaMonitor):
    """Render the models tab with detailed model information."""
    # Get available models
    with st.spinner("Loading model information..."):
        models = await monitor.get_models()
    
    if not models:
        st.warning("No models available in Ollama. Please load models first.")
        
        # Show instructions for loading models
        with st.expander("How to load models"):
            st.markdown("""
            To load models into Ollama, use the following command:
            ```bash
            ollama pull <model>
            ```
            
            Available models can be found on [Ollama Library](https://ollama.ai/library).
            """)
        
        return
    
    # Model selector
    selected_model = st.selectbox(
        "Select Model",
        options=[model["name"] for model in models],
        index=0,
        key="models_tab_model_selector"
    )
    
    # Get detailed information for the selected model
    with st.spinner(f"Loading details for {selected_model}..."):
        model_info = await monitor.get_model_info(selected_model)
    
    # Display model information
    if model_info:
        # Basic info columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            size_gb = model_info.get("size", 0) / (1024 * 1024 * 1024)
            render_metric_card(
                title="Model Size",
                value=f"{size_gb:.2f} GB",
                description="Disk space used",
                icon="💾"
            )
        
        with col2:
            render_metric_card(
                title="Parameters",
                value=model_info.get("parameter_size", "Unknown"),
                description="Model parameters",
                icon="🧮"
            )
        
        with col3:
            render_metric_card(
                title="Format",
                value=model_info.get("format", "Unknown"),
                description="Model format",
                icon="📄"
            )
        
        # Model family and details
        st.subheader("Model Information", anchor=False)
        
        # Display model metadata
        metadata = model_info.get("metadata", {})
        
        if metadata:
            info_cols = st.columns(2)
            
            with info_cols[0]:
                st.markdown("#### Model Details")
                details = {
                    "Family": metadata.get("family", "Unknown"),
                    "Template": metadata.get("template", "N/A"),
                    "Quantization": metadata.get("quantization", "N/A"),
                    "Quantization Level": metadata.get("quantization_level", "N/A"),
                }
                
                for key, value in details.items():
                    st.markdown(f"**{key}:** {value}")
            
            with info_cols[1]:
                st.markdown("#### System Requirements")
                requirements = {
                    "RAM Required": f"{metadata.get('ram_required', 'Unknown')} GB" if metadata.get('ram_required') else "Unknown",
                    "VRAM Required": f"{metadata.get('vram_required', 'Unknown')} GB" if metadata.get('vram_required') else "Unknown",
                    "Parameters": metadata.get("parameter_size", "Unknown"),
                    "Context Length": metadata.get("context_length", "Unknown"),
                }
                
                for key, value in details.items():
                    st.markdown(f"**{key}:** {value}")
        
        # Model parameters
        if "parameters" in model_info:
            st.markdown("#### Model Parameters")
            
            params = model_info.get("parameters", {})
            
            # Check if params is a dictionary
            if isinstance(params, dict) and params:
                param_df = pd.DataFrame({
                    "Parameter": list(params.keys()),
                    "Value": list(params.values())
                })
                
                st.dataframe(param_df, use_container_width=True, hide_index=True)
            # Check if params is a string
            elif isinstance(params, str) and params.strip():
                # Try to parse it as JSON in case it's a stringified JSON
                try:
                    parsed_params = json.loads(params)
                    if isinstance(parsed_params, dict):
                        param_df = pd.DataFrame({
                            "Parameter": list(parsed_params.keys()),
                            "Value": list(parsed_params.values())
                        })
                        st.dataframe(param_df, use_container_width=True, hide_index=True)
                    else:
                        st.text(params)
                except json.JSONDecodeError:
                    # If it's not valid JSON, just display as text
                    st.text(params)
            else:
                st.info("No parameter information available for this model.")
        
        # Model files
        if "modelfile" in model_info:
            with st.expander("View Modelfile"):
                st.code(model_info.get("modelfile", ""), language="dockerfile")
        
        # Model license
        if "license" in model_info:
            with st.expander("View License"):
                st.text(model_info.get("license", ""))
    else:
        st.error(f"Failed to load detailed information for {selected_model}")
    
    # Model performance metrics
    st.subheader("Performance Metrics", anchor=False)
    
    # Get request history for this model
    dfs = monitor.to_dataframe()
    
    if "requests" in dfs and not dfs["requests"].empty:
        df_requests = dfs["requests"]
        
        # Filter for selected model
        if "model" in df_requests.columns:
            df_model = df_requests[df_requests["model"] == selected_model]
            
            if not df_model.empty:
                # Calculate average metrics
                avg_latency = df_model["duration_seconds"].mean() if "duration_seconds" in df_model.columns else 0
                
                if "total_tokens" in df_model.columns and "duration_seconds" in df_model.columns:
                    df_model["tokens_per_second"] = df_model["total_tokens"] / df_model["duration_seconds"].clip(lower=0.001)
                    avg_throughput = df_model["tokens_per_second"].mean()
                else:
                    avg_throughput = 0
                
                # Display metrics
                metric_cols = st.columns(3)
                
                with metric_cols[0]:
                    st.metric("Average Latency", f"{avg_latency:.2f}s")
                
                with metric_cols[1]:
                    st.metric("Average Throughput", f"{avg_throughput:.1f} tokens/sec")
                
                with metric_cols[2]:
                    st.metric("Total Requests", len(df_model))
                
                # Latency chart
                if "timestamp" in df_model.columns and "duration_seconds" in df_model.columns:
                    df_model["timestamp"] = pd.to_datetime(df_model["timestamp"])
                    df_model = df_model.sort_values("timestamp")
                    
                    fig = px.line(
                        df_model, 
                        x="timestamp", 
                        y="duration_seconds",
                        title="Response Latency Over Time",
                        labels={"duration_seconds": "Latency (seconds)", "timestamp": "Time"},
                    )
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(t=30, b=30, l=80, r=30),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No performance data available for {selected_model} yet.")
        else:
            st.info("No model-specific performance data available.")
    else:
        st.info("No performance data available yet. Make some requests to see metrics.")
    
    # List of all models
    st.subheader("All Available Models", anchor=False)
    
    # Convert models list to DataFrame for display
    if models:
        models_data = []
        for model in models:
            model_data = {
                "Name": model.get("name", "Unknown"),
                "Size": f"{model.get('size', 0) / (1024 * 1024 * 1024):.2f} GB" if model.get("size") else "Unknown",
                "Modified": model.get("modified_at", "Unknown"),
                "Format": model.get("format", "Unknown"),
            }
            models_data.append(model_data)
        
        models_df = pd.DataFrame(models_data)
        st.dataframe(models_df, use_container_width=True, hide_index=True)
    else:
        st.info("No models available.")


async def render_performance_tab(monitor: OllamaMonitor, time_range: str):
    """Render the performance tab with detailed performance metrics."""
    # Get performance statistics
    with st.spinner("Loading performance metrics..."):
        performance_stats = monitor.get_performance_stats(time_range_to_filter(time_range))
        token_stats = monitor.get_token_usage_stats(time_range_to_filter(time_range))
        dfs = monitor.to_dataframe()
    
    # Overview metrics
    st.subheader("Performance Overview", anchor=False)
    
    # Display metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card(
            title="Avg. Latency",
            value=f"{performance_stats.get('average_latency', 0):.2f}s",
            description="Average response time",
            icon="⏱️"
        )
    
    with col2:
        render_metric_card(
            title="Avg. Throughput",
            value=f"{performance_stats.get('average_throughput', 0):.1f}",
            description="Tokens per second",
            icon="⚡"
        )
    
    with col3:
        render_metric_card(
            title="Success Rate",
            value=f"{performance_stats.get('success_rate', 0):.1f}%",
            description="Request success rate",
            icon="✅"
        )
    
    with col4:
        render_metric_card(
            title="Error Rate",
            value=f"{performance_stats.get('error_rate', 0):.1f}%",
            description="Request error rate",
            icon="❌"
        )
    
    # Performance charts
    st.subheader("Performance Metrics Over Time", anchor=False)
    
    chart_tabs = st.tabs(["Latency", "Throughput", "Token Usage", "Success Rate"])
    
    # Latency chart
    with chart_tabs[0]:
        if "latency" in dfs and not dfs["latency"].empty:
            df_latency = dfs["latency"]
            
            # Apply time range filter
            if "timestamp" in df_latency.columns:
                df_latency["timestamp"] = pd.to_datetime(df_latency["timestamp"])
                df_latency = filter_dataframe_by_time(df_latency, time_range)
            
            if not df_latency.empty:
                # Prepare data for plotting
                if "model" in df_latency.columns and "latency_seconds" in df_latency.columns:
                    # Overall latency trend
                    fig = px.line(
                        df_latency, 
                        x="timestamp", 
                        y="latency_seconds",
                        color="model",
                        title="Response Latency Over Time",
                        labels={"latency_seconds": "Latency (seconds)", "timestamp": "Time"},
                    )
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(t=30, b=50, l=80, r=30),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Latency histogram
                    fig = px.histogram(
                        df_latency,
                        x="latency_seconds",
                        color="model",
                        nbins=20,
                        title="Latency Distribution",
                        labels={"latency_seconds": "Latency (seconds)", "count": "Frequency"},
                    )
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(t=30, b=50, l=80, r=30),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No latency data available for the selected time range.")
        else:
            st.info("No latency data available yet.")
    
    # Throughput chart
    with chart_tabs[1]:
        if "token_throughput" in dfs and not dfs["token_throughput"].empty:
            df_throughput = dfs["token_throughput"]
            
            # Apply time range filter
            if "timestamp" in df_throughput.columns:
                df_throughput["timestamp"] = pd.to_datetime(df_throughput["timestamp"])
                df_throughput = filter_dataframe_by_time(df_throughput, time_range)
            
            if not df_throughput.empty:
                # Prepare data for plotting
                if "model" in df_throughput.columns and "tokens_per_second" in df_throughput.columns:
                    # Overall throughput trend
                    fig = px.line(
                        df_throughput, 
                        x="timestamp", 
                        y="tokens_per_second",
                        color="model",
                        title="Token Throughput Over Time",
                        labels={"tokens_per_second": "Tokens/second", "timestamp": "Time"},
                    )
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(t=30, b=50, l=80, r=30),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Throughput by model
                    fig = px.box(
                        df_throughput,
                        x="model",
                        y="tokens_per_second",
                        title="Token Throughput Distribution by Model",
                        labels={"tokens_per_second": "Tokens/second", "model": "Model"},
                    )
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(t=30, b=50, l=80, r=30),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No throughput data available for the selected time range.")
        else:
            st.info("No throughput data available yet.")
    
    # Token usage chart
    with chart_tabs[2]:
        if "requests" in dfs and not dfs["requests"].empty:
            df_requests = dfs["requests"]
            
            # Apply time range filter
            if "timestamp" in df_requests.columns:
                df_requests["timestamp"] = pd.to_datetime(df_requests["timestamp"])
                df_requests = filter_dataframe_by_time(df_requests, time_range)
            
            if not df_requests.empty:
                # Token usage by model
                if "model" in df_requests.columns and "total_tokens" in df_requests.columns:
                    token_by_model = df_requests.groupby("model")[["total_tokens", "prompt_tokens", "completion_tokens"]].sum().reset_index()
                    
                    fig = px.bar(
                        token_by_model,
                        x="model",
                        y=["prompt_tokens", "completion_tokens"],
                        title="Token Usage by Model",
                        labels={"value": "Tokens", "model": "Model", "variable": "Token Type"},
                    )
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(t=30, b=50, l=80, r=30),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cumulative token usage over time
                    if "timestamp" in df_requests.columns:
                        df_requests = df_requests.sort_values("timestamp")
                        df_requests["cumulative_tokens"] = df_requests["total_tokens"].cumsum()
                        
                        fig = px.line(
                            df_requests,
                            x="timestamp",
                            y="cumulative_tokens",
                            color="model",
                            title="Cumulative Token Usage Over Time",
                            labels={"cumulative_tokens": "Total Tokens", "timestamp": "Time"},
                        )
                        
                        fig.update_layout(
                            height=300,
                            margin=dict(t=30, b=50, l=80, r=30),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No token usage data available for the selected time range.")
        else:
            st.info("No token usage data available yet.")
    
    # Success rate chart
    with chart_tabs[3]:
        if "requests" in dfs and not dfs["requests"].empty:
            df_requests = dfs["requests"]
            
            # Apply time range filter
            if "timestamp" in df_requests.columns:
                df_requests["timestamp"] = pd.to_datetime(df_requests["timestamp"])
                df_requests = filter_dataframe_by_time(df_requests, time_range)
            
            if not df_requests.empty and "success" in df_requests.columns:
                # Success rate over time
                df_requests["success_int"] = df_requests["success"].astype(int)
                df_requests["hour"] = df_requests["timestamp"].dt.floor("H")
                
                success_by_hour = df_requests.groupby(["hour", "model"]).agg(
                    success_rate=("success_int", "mean"),
                    count=("success_int", "count")
                ).reset_index()
                
                success_by_hour["success_rate"] = success_by_hour["success_rate"] * 100
                
                fig = px.line(
                    success_by_hour,
                    x="hour",
                    y="success_rate",
                    color="model",
                    title="Success Rate Over Time",
                    labels={"success_rate": "Success Rate (%)", "hour": "Time"},
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(t=30, b=50, l=80, r=30),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Success rate by model
                success_by_model = df_requests.groupby("model").agg(
                    success_rate=("success_int", "mean"),
                    count=("success_int", "count")
                ).reset_index()
                
                success_by_model["success_rate"] = success_by_model["success_rate"] * 100
                
                fig = px.bar(
                    success_by_model,
                    x="model",
                    y="success_rate",
                    text="count",
                    title="Success Rate by Model",
                    labels={"success_rate": "Success Rate (%)", "model": "Model", "count": "Requests"},
                )
                
                fig.update_layout(
                    height=300,
                    yaxis_range=[0, 100],
                    margin=dict(t=30, b=50, l=80, r=30),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No success rate data available for the selected time range.")
        else:
            st.info("No success rate data available yet.")
    
    
    # Performance comparison
    st.subheader("Model Performance Comparison", anchor=False)

    if "requests" in dfs and not dfs["requests"].empty:
        df_requests = dfs["requests"]
        
        # Apply time range filter
        if "timestamp" in df_requests.columns:
            df_requests["timestamp"] = pd.to_datetime(df_requests["timestamp"])
            df_requests = filter_dataframe_by_time(df_requests, time_range)
        
        if not df_requests.empty and "model" in df_requests.columns:
            # Create a performance comparison table
            performance_by_model = df_requests.groupby("model").agg(
                avg_latency=("duration_seconds", "mean"),
                min_latency=("duration_seconds", "min"),
                max_latency=("duration_seconds", "max"),
                success_rate=("success", "mean"),
                request_count=("request_id", "count")
            )
            
            if "total_tokens" in df_requests.columns and "duration_seconds" in df_requests.columns:
                # Calculate tokens per second
                df_requests["tokens_per_second"] = df_requests["total_tokens"] / df_requests["duration_seconds"].clip(lower=0.001)
                
                tokens_by_model = df_requests.groupby("model").agg(
                    avg_throughput=("tokens_per_second", "mean"),
                    total_tokens=("total_tokens", "sum")
                )
                
                performance_by_model = performance_by_model.join(tokens_by_model)
            
            # Format the table
            performance_by_model["success_rate"] = performance_by_model["success_rate"] * 100
            performance_by_model = performance_by_model.round(2)
            
            # Display the table
            st.dataframe(performance_by_model, use_container_width=True)
        else:
            st.info("No performance comparison data available for the selected time range.")
    else:
        st.info("No performance comparison data available yet.")


async def render_requests_tab(monitor: OllamaMonitor, time_range: str):
    """Render the requests tab with request history and analytics."""
    # Get request history
    with st.spinner("Loading request history..."):
        dfs = monitor.to_dataframe()
    
    # Display request metrics
    st.subheader("Request Analytics", anchor=False)
    
    if "requests" in dfs and not dfs["requests"].empty:
        df_requests = dfs["requests"].copy()
        
        # Apply time range filter
        if "timestamp" in df_requests.columns:
            df_requests["timestamp"] = pd.to_datetime(df_requests["timestamp"])
            df_requests = filter_dataframe_by_time(df_requests, time_range)
        
        if not df_requests.empty:
            # Calculate summary metrics
            total_requests = len(df_requests)
            if "success" in df_requests.columns:
                success_count = df_requests["success"].sum()
                success_rate = (success_count / total_requests) * 100 if total_requests > 0 else 0
                error_count = total_requests - success_count
            else:
                success_count = 0
                success_rate = 0
                error_count = 0
            
            # Show metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                render_metric_card(
                    title="Total Requests",
                    value=total_requests,
                    description=f"In {time_range.lower()}",
                    icon="🔄"
                )
            
            with col2:
                render_metric_card(
                    title="Successful",
                    value=success_count,
                    description=f"{success_rate:.1f}% success rate",
                    icon="✅"
                )
            
            with col3:
                render_metric_card(
                    title="Errors",
                    value=error_count,
                    description=f"{100 - success_rate:.1f}% error rate",
                    icon="❌"
                )
            
            with col4:
                if "duration_seconds" in df_requests.columns:
                    avg_latency = df_requests["duration_seconds"].mean()
                    render_metric_card(
                        title="Avg. Latency",
                        value=f"{avg_latency:.2f}s",
                        description="Average response time",
                        icon="⏱️"
                    )
                else:
                    render_metric_card(
                        title="Avg. Latency",
                        value="N/A",
                        description="No latency data",
                        icon="⏱️"
                    )
            
            # Request volume over time
            st.subheader("Request Volume", anchor=False)
            
            if "timestamp" in df_requests.columns:
                # Group by hour and model
                df_requests["hour"] = df_requests["timestamp"].dt.floor("H")
                requests_by_hour = df_requests.groupby(["hour", "model"]).size().reset_index(name="count")
                
                fig = px.line(
                    requests_by_hour,
                    x="hour",
                    y="count",
                    color="model",
                    title="Requests Volume Over Time",
                    labels={"count": "Number of Requests", "hour": "Time"},
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(t=30, b=50, l=80, r=30),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Request history table
            st.subheader("Request History", anchor=False)
            
            # Add filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                models = ["All"] + sorted(df_requests["model"].unique().tolist())
                selected_model = st.selectbox("Filter by Model", models, key="requests_model_filter")
            
            with col2:
                if "success" in df_requests.columns:
                    status_options = ["All", "Success", "Error"]
                    selected_status = st.selectbox("Filter by Status", status_options, key="requests_status_filter")
                else:
                    selected_status = "All"
            
            with col3:
                sort_options = ["Newest First", "Oldest First", "Longest Duration", "Shortest Duration"]
                sort_option = st.selectbox("Sort by", sort_options, key="requests_sort_option")
            
            # Apply filters
            filtered_df = df_requests.copy()
            
            if selected_model != "All":
                filtered_df = filtered_df[filtered_df["model"] == selected_model]
            
            if selected_status != "All" and "success" in filtered_df.columns:
                if selected_status == "Success":
                    filtered_df = filtered_df[filtered_df["success"] == True]
                else:
                    filtered_df = filtered_df[filtered_df["success"] == False]
            
            # Apply sorting
            if sort_option == "Newest First":
                filtered_df = filtered_df.sort_values("timestamp", ascending=False)
            elif sort_option == "Oldest First":
                filtered_df = filtered_df.sort_values("timestamp", ascending=True)
            elif sort_option == "Longest Duration" and "duration_seconds" in filtered_df.columns:
                filtered_df = filtered_df.sort_values("duration_seconds", ascending=False)
            elif sort_option == "Shortest Duration" and "duration_seconds" in filtered_df.columns:
                filtered_df = filtered_df.sort_values("duration_seconds", ascending=True)
            
            # Display the filtered table
            if not filtered_df.empty:
                # Prepare display columns
                display_columns = []
                
                if "timestamp" in filtered_df.columns:
                    filtered_df["formatted_time"] = filtered_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
                    display_columns.append("formatted_time")
                
                if "model" in filtered_df.columns:
                    display_columns.append("model")
                
                if "request_type" in filtered_df.columns:
                    display_columns.append("request_type")
                
                if "duration_seconds" in filtered_df.columns:
                    filtered_df["duration_formatted"] = filtered_df["duration_seconds"].apply(lambda x: f"{x:.2f}s")
                    display_columns.append("duration_formatted")
                
                if "prompt_tokens" in filtered_df.columns:
                    display_columns.append("prompt_tokens")
                
                if "completion_tokens" in filtered_df.columns:
                    display_columns.append("completion_tokens")
                
                if "total_tokens" in filtered_df.columns:
                    display_columns.append("total_tokens")
                
                if "success" in filtered_df.columns:
                    display_columns.append("success")
                
                # Create display dataframe
                if display_columns:
                    display_df = filtered_df[display_columns].copy()
                    
                    # Rename columns for better display
                    column_rename = {
                        "formatted_time": "Time",
                        "model": "Model",
                        "request_type": "Type",
                        "duration_formatted": "Duration",
                        "prompt_tokens": "Prompt Tokens",
                        "completion_tokens": "Completion Tokens",
                        "total_tokens": "Total Tokens",
                        "success": "Success"
                    }
                    
                    display_df = display_df.rename(columns=column_rename)
                    
                    # Display the table
                    st.dataframe(display_df, use_container_width=True, height=400)
                    
                    # Add export option
                    if st.button("Export to CSV", key="export_requests_csv"):
                        csv = display_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"ollama_requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_requests_csv"
                        )
                else:
                    st.info("No columns available to display")
            else:
                st.info("No requests match the selected filters")
            
            # Request details
            with st.expander("View Request Details"):
                request_id = st.selectbox(
                    "Select a request to view details",
                    options=filtered_df["request_id"].tolist() if "request_id" in filtered_df.columns else [],
                    key="request_details_selector"
                )
                
                if request_id:
                    selected_request = filtered_df[filtered_df["request_id"] == request_id].iloc[0]
                    
                    # Display request details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Request Information")
                        
                        details = {
                            "Request ID": selected_request.get("request_id", "N/A"),
                            "Model": selected_request.get("model", "N/A"),
                            "Type": selected_request.get("request_type", "N/A"),
                            "Time": selected_request.get("formatted_time", "N/A"),
                            "Success": "✅ Yes" if selected_request.get("success", False) else "❌ No"
                        }
                        
                        for key, value in details.items():
                            st.markdown(f"**{key}:** {value}")
                    
                    with col2:
                        st.markdown("#### Performance Metrics")
                        
                        metrics = {
                            "Duration": f"{selected_request.get('duration_seconds', 0):.2f} seconds",
                            "Prompt Tokens": selected_request.get("prompt_tokens", "N/A"),
                            "Completion Tokens": selected_request.get("completion_tokens", "N/A"),
                            "Total Tokens": selected_request.get("total_tokens", "N/A"),
                            "Tokens/Second": f"{selected_request.get('tokens_per_second', 0):.1f}"
                        }
                        
                        for key, value in metrics.items():
                            st.markdown(f"**{key}:** {value}")
                    
                    # Display prompt and response if available
                    if "prompt" in selected_request:
                        st.markdown("#### Prompt")
                        st.text_area("", value=selected_request["prompt"], height=150, disabled=True, key="details_prompt_area")
                    
                    if "response" in selected_request:
                        st.markdown("#### Response")
                        st.text_area("", value=selected_request["response"], height=200, disabled=True, key="details_response_area")
                    
                    # Display error information if available
                    if "error" in selected_request and selected_request["error"]:
                        st.markdown("#### Error Information")
                        st.error(selected_request["error"])
        else:
            st.info("No request data available for the selected time range.")
    else:
        st.info("No request history available yet. Make some requests to Ollama to see data.")
    
    # Request simulator
    st.subheader("Request Simulator", anchor=False)
    
    # Get available models
    with st.spinner("Loading models..."):
        models = await monitor.get_models()
    
    if models:
        model_names = [model["name"] for model in models]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            sim_prompt = st.text_area("Prompt", height=100, key="simulator_prompt")
        
        with col2:
            sim_model = st.selectbox("Model", options=model_names, key="simulator_model")
            sim_button = st.button("Send Request", type="primary", use_container_width=True, key="simulator_button")
        
        if sim_button and sim_prompt:
            with st.spinner(f"Sending request to {sim_model}..."):
                start_time = time.time()
                result = await monitor.simulate_request(sim_model, sim_prompt)
                duration = time.time() - start_time
                
                if result.get("success", False):
                    st.success(f"Request completed in {duration:.2f} seconds")
                    
                    # Display metrics
                    metrics_cols = st.columns(3)
                    with metrics_cols[0]:
                        st.metric("Prompt Tokens", result.get("prompt_tokens", "N/A"))
                    with metrics_cols[1]:
                        st.metric("Completion Tokens", result.get("completion_tokens", "N/A"))
                    with metrics_cols[2]:
                        st.metric("Tokens/second", round(result.get("total_tokens", 0) / max(duration, 0.001), 1))
                    
                    # Display response
                    st.markdown("#### Response")
                    st.text_area("", value=result.get("response", "No response"), height=200, disabled=True, key="simulator_response_area")
                else:
                    st.error(f"Request failed: {result.get('error', 'Unknown error')}")
    else:
        st.warning("No models available for testing. Please load models in Ollama first.")


async def render_model_tab(monitor: OllamaMonitor, time_range: str):
    """
    Render the detailed model tab with specific model analytics and performance.
    
    Args:
        monitor: Ollama monitor instance
        time_range: Time range filter for metrics
    """
    st.subheader("Model Analytics", anchor=False)
    
    # Get available models
    with st.spinner("Loading model information..."):
        models = await monitor.get_models()
        dfs = monitor.to_dataframe()
    
    if not models:
        st.warning("No models available in Ollama. Please load models first.")
        return
    
    # Model selector
    model_names = [model["name"] for model in models]
    selected_model = st.selectbox(
        "Select Model",
        options=model_names,
        index=0,
        key="model_tab_model_selector"
    )
    
    # Get detailed information for the selected model
    with st.spinner(f"Loading details for {selected_model}..."):
        model_info = await monitor.get_model_info(selected_model)
    
    # Display model basic info
    if model_info:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            size_gb = model_info.get("size", 0) / (1024 * 1024 * 1024)
            render_metric_card(
                title="Model Size",
                value=f"{size_gb:.2f} GB",
                description="Disk space used",
                icon="💾"
            )
        
        with col2:
            param_size = model_info.get("parameter_size", "Unknown")
            if isinstance(param_size, (int, float)):
                param_display = f"{param_size / 1e9:.1f}B" if param_size >= 1e9 else param_size
            else:
                param_display = param_size
                
            render_metric_card(
                title="Parameters",
                value=param_display,
                description="Model parameters",
                icon="🧮"
            )
        
        with col3:
            render_metric_card(
                title="Format",
                value=model_info.get("format", "Unknown"),
                description="Model format",
                icon="📄"
            )
    
    # Filter data for selected model
    if "requests" in dfs and not dfs["requests"].empty:
        df_requests = dfs["requests"].copy()
        
        # Apply model filter
        if "model" in df_requests.columns:
            df_model = df_requests[df_requests["model"] == selected_model]
            
            # Apply time range filter
            if "timestamp" in df_model.columns:
                df_model["timestamp"] = pd.to_datetime(df_model["timestamp"])
                df_model = filter_dataframe_by_time(df_model, time_range)
            
            if not df_model.empty:
                # Performance metrics
                st.subheader("Performance Metrics", anchor=False)
                
                # Calculate metrics
                request_count = len(df_model)
                
                if "success" in df_model.columns:
                    success_count = df_model["success"].sum()
                    success_rate = (success_count / request_count) * 100 if request_count > 0 else 0
                else:
                    success_count = 0
                    success_rate = 0
                
                if "duration_seconds" in df_model.columns:
                    avg_latency = df_model["duration_seconds"].mean()
                    min_latency = df_model["duration_seconds"].min()
                    max_latency = df_model["duration_seconds"].max()
                    p95_latency = df_model["duration_seconds"].quantile(0.95)
                else:
                    avg_latency = min_latency = max_latency = p95_latency = 0
                
                if "total_tokens" in df_model.columns and "duration_seconds" in df_model.columns:
                    df_model["tokens_per_second"] = df_model["total_tokens"] / df_model["duration_seconds"].clip(lower=0.001)
                    avg_throughput = df_model["tokens_per_second"].mean()
                    total_tokens = df_model["total_tokens"].sum()
                    prompt_tokens = df_model["prompt_tokens"].sum() if "prompt_tokens" in df_model.columns else 0
                    completion_tokens = df_model["completion_tokens"].sum() if "completion_tokens" in df_model.columns else 0
                else:
                    avg_throughput = total_tokens = prompt_tokens = completion_tokens = 0
                
                # Display metrics in cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    render_metric_card(
                        title="Requests",
                        value=request_count,
                        description=f"In {time_range.lower()}",
                        icon="🔄"
                    )
                
                with col2:
                    render_metric_card(
                        title="Success Rate",
                        value=f"{success_rate:.1f}%",
                        description=f"{success_count} successful requests",
                        icon="✅"
                    )
                
                with col3:
                    render_metric_card(
                        title="Avg. Latency",
                        value=f"{avg_latency:.2f}s",
                        description=f"P95: {p95_latency:.2f}s",
                        icon="⏱️"
                    )
                
                with col4:
                    render_metric_card(
                        title="Avg. Throughput",
                        value=f"{avg_throughput:.1f}",
                        description="Tokens per second",
                        icon="⚡"
                    )
                
                # Second row of metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    render_metric_card(
                        title="Total Tokens",
                        value=format_number(total_tokens),
                        description=f"In {time_range.lower()}",
                        icon="🔤"
                    )
                
                with col2:
                    render_metric_card(
                        title="Prompt Tokens",
                        value=format_number(prompt_tokens),
                        description=f"{(prompt_tokens / total_tokens * 100):.1f}% of total" if total_tokens > 0 else "0% of total",
                        icon="📥"
                    )
                
                with col3:
                    render_metric_card(
                        title="Completion Tokens",
                        value=format_number(completion_tokens),
                        description=f"{(completion_tokens / total_tokens * 100):.1f}% of total" if total_tokens > 0 else "0% of total",
                        icon="📤"
                    )
                
                with col4:
                    if "duration_seconds" in df_model.columns:
                        total_duration = df_model["duration_seconds"].sum()
                        render_metric_card(
                            title="Total Duration",
                            value=f"{total_duration:.1f}s",
                            description=f"Avg: {avg_latency:.2f}s per request",
                            icon="⏰"
                        )
                    else:
                        render_metric_card(
                            title="Total Duration",
                            value="N/A",
                            description="No duration data available",
                            icon="⏰"
                        )
                
                # Performance charts
                st.subheader("Performance Charts", anchor=False)
                
                chart_tabs = st.tabs(["Latency", "Throughput", "Usage Over Time"])
                
                # Latency chart
                with chart_tabs[0]:
                    if "timestamp" in df_model.columns and "duration_seconds" in df_model.columns:
                        # Latency over time
                        fig = px.line(
                            df_model.sort_values("timestamp"),
                            x="timestamp",
                            y="duration_seconds",
                            title="Latency Over Time",
                            labels={"duration_seconds": "Latency (seconds)", "timestamp": "Time"},
                        )
                        
                        fig.update_layout(
                            height=350,
                            margin=dict(t=30, b=30, l=80, r=30),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Latency distribution
                        fig = px.histogram(
                            df_model,
                            x="duration_seconds",
                            nbins=20,
                            title="Latency Distribution",
                            labels={"duration_seconds": "Latency (seconds)", "count": "Frequency"},
                        )
                        
                        fig.update_layout(
                            height=300,
                            margin=dict(t=30, b=30, l=80, r=30),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No latency data available for this model.")
                
                # Throughput chart
                with chart_tabs[1]:
                    if "timestamp" in df_model.columns and "tokens_per_second" in df_model.columns:
                        # Throughput over time
                        fig = px.line(
                            df_model.sort_values("timestamp"),
                            x="timestamp",
                            y="tokens_per_second",
                            title="Token Throughput Over Time",
                            labels={"tokens_per_second": "Tokens/second", "timestamp": "Time"},
                        )
                        
                        fig.update_layout(
                            height=350,
                            margin=dict(t=30, b=30, l=80, r=30),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Throughput distribution
                        fig = px.histogram(
                            df_model,
                            x="tokens_per_second",
                            nbins=20,
                            title="Throughput Distribution",
                            labels={"tokens_per_second": "Tokens/second", "count": "Frequency"},
                        )
                        
                        fig.update_layout(
                            height=300,
                            margin=dict(t=30, b=30, l=80, r=30),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No throughput data available for this model.")
                
                # Usage over time
                with chart_tabs[2]:
                    if "timestamp" in df_model.columns:
                        # Group by hour
                        df_model["hour"] = df_model["timestamp"].dt.floor("H")
                        usage_by_hour = df_model.groupby("hour").agg(
                            request_count=("model", "count")
                        ).reset_index()
                        
                        if "total_tokens" in df_model.columns:
                            tokens_by_hour = df_model.groupby("hour").agg(
                                total_tokens=("total_tokens", "sum"),
                                prompt_tokens=("prompt_tokens", "sum") if "prompt_tokens" in df_model.columns else None,
                                completion_tokens=("completion_tokens", "sum") if "completion_tokens" in df_model.columns else None,
                            ).reset_index()
                            
                            # Create a figure with two y-axes
                            fig = make_subplots(specs=[[{"secondary_y": True}]])
                            
                            # Add request count trace on primary y-axis
                            fig.add_trace(
                                go.Scatter(
                                    x=usage_by_hour["hour"],
                                    y=usage_by_hour["request_count"],
                                    name="Requests",
                                    line=dict(color=COLORS['primary'], width=2),
                                ),
                                secondary_y=False,
                            )
                            
                            # Add token count trace on secondary y-axis
                        
                            fig.add_trace(
                                go.Scatter(
                                    x=tokens_by_hour["hour"],
                                    y=tokens_by_hour["total_tokens"],
                                    name="Tokens",
                                    line=dict(color=COLORS['secondary'], width=2),
                                ),
                                secondary_y=True,
                            )
                            
                            # Set titles
                            fig.update_layout(
                                title_text="Model Usage Over Time",
                                height=400,
                                margin=dict(t=30, b=30, l=80, r=80),
                            )
                            
                            # Set y-axes titles
                            fig.update_yaxes(title_text="Request Count", secondary_y=False)
                            fig.update_yaxes(title_text="Token Count", secondary_y=True)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Token breakdown over time
                            if "prompt_tokens" in tokens_by_hour.columns and "completion_tokens" in tokens_by_hour.columns:
                                fig = px.area(
                                    tokens_by_hour,
                                    x="hour",
                                    y=["prompt_tokens", "completion_tokens"],
                                    title="Token Usage Breakdown Over Time",
                                    labels={"value": "Token Count", "hour": "Time", "variable": "Token Type"},
                                )
                                
                                fig.update_layout(
                                    height=350,
                                    margin=dict(t=30, b=30, l=80, r=30),
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Just show request count
                            fig = px.bar(
                                usage_by_hour,
                                x="hour",
                                y="request_count",
                                title="Request Count Over Time",
                                labels={"request_count": "Request Count", "hour": "Time"},
                            )
                            
                            fig.update_layout(
                                height=400,
                                margin=dict(t=30, b=30, l=80, r=30),
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No usage data available for this model.")
                
                # Benchmark
                st.subheader("Model Benchmark", anchor=False)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    benchmark_text = st.text_area(
                        "Benchmark Prompt",
                        value="Explain the theory of relativity in simple terms.",
                        height=100,
                        key="benchmark_prompt"
                    )
                
                with col2:
                    num_runs = st.number_input("Number of Runs", min_value=1, max_value=10, value=3, key="benchmark_runs")
                    run_benchmark = st.button("Run Benchmark", type="primary", use_container_width=True, key="run_benchmark")
                
                if run_benchmark and benchmark_text:
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i in range(num_runs):
                        with st.spinner(f"Running benchmark {i+1}/{num_runs}..."):
                            start_time = time.time()
                            result = await monitor.simulate_request(selected_model, benchmark_text)
                            duration = time.time() - start_time
                            
                            if result.get("success", False):
                                results.append({
                                    "run": i+1,
                                    "duration": duration,
                                    "prompt_tokens": result.get("prompt_tokens", 0),
                                    "completion_tokens": result.get("completion_tokens", 0),
                                    "total_tokens": result.get("total_tokens", 0),
                                    "tokens_per_second": result.get("total_tokens", 0) / max(duration, 0.001)
                                })
                        
                        progress_bar.progress((i+1) / num_runs)
                    
                    if results:
                        # Calculate average metrics
                        avg_duration = sum(r["duration"] for r in results) / len(results)
                        avg_throughput = sum(r["tokens_per_second"] for r in results) / len(results)
                        
                        # Display results
                        st.success(f"Benchmark completed: Average response time {avg_duration:.2f}s")
                        
                        # Display summary metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Avg. Latency", f"{avg_duration:.2f}s")
                        
                        with col2:
                            st.metric("Avg. Throughput", f"{avg_throughput:.1f} tokens/sec")
                        
                        with col3:
                            st.metric("Total Runs", num_runs)
                        
                        # Display detailed results
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Create comparison chart
                        fig = px.bar(
                            results_df,
                            x="run",
                            y="duration",
                            title="Response Time by Run",
                            labels={"duration": "Duration (seconds)", "run": "Run"},
                        )
                        
                        fig.update_layout(
                            height=300,
                            margin=dict(t=30, b=30, l=80, r=30),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No request data available for model {selected_model} in the selected time range.")
        else:
            st.info(f"No request data available for model {selected_model}.")
    else:
        st.info("No request data available. Make some requests to Ollama to see model analytics.")