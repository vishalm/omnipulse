"""
Chart widgets for creating consistent visualizations.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple


def render_time_series(
    df: pd.DataFrame,
    x_column: str,
    y_columns: Union[str, List[str]],
    title: str = "",
    height: int = 300,
    color_sequence: Optional[List[str]] = None,
    show_legend: bool = True,
    stack: bool = False,
    range_slider: bool = False,
    custom_labels: Optional[Dict[str, str]] = None,
    y_axis_title: Optional[str] = None,
    area: bool = False
):
    """
    Render a time series chart with Plotly.
    
    Args:
        df: DataFrame containing the data
        x_column: Column name for x-axis (typically timestamp)
        y_columns: Column name(s) for y-axis
        title: Chart title
        height: Chart height in pixels
        color_sequence: Optional custom color sequence
        show_legend: Whether to show the legend
        stack: Whether to stack y values (for multiple y columns)
        range_slider: Whether to include a range slider
        custom_labels: Optional dictionary mapping column names to display labels
        y_axis_title: Optional title for y-axis
        area: Whether to use area chart instead of line chart
    """
    # Convert to list if single string
    if isinstance(y_columns, str):
        y_columns = [y_columns]
    
    # Safety check for empty data or missing columns
    if df is None or df.empty or x_column not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            height=height,
            annotations=[
                {
                    "text": "No data available",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 16
                    },
                    "x": 0.5,
                    "y": 0.5
                }
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
        return fig
    
    # Filter out y_columns that don't exist in the dataframe
    valid_y_columns = [col for col in y_columns if col in df.columns]
    
    if not valid_y_columns:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            height=height,
            annotations=[
                {
                    "text": "No valid Y columns found",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 16
                    },
                    "x": 0.5,
                    "y": 0.5
                }
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
        return fig
    
    # Sort data by x_column to ensure line continuity
    plot_df = df.sort_values(by=x_column)
    
    # Create figure
    if area:
        fig = px.area(
            plot_df,
            x=x_column,
            y=valid_y_columns,
            title=title,
            labels=custom_labels,
            color_discrete_sequence=color_sequence
        )
    else:
        fig = px.line(
            plot_df,
            x=x_column,
            y=valid_y_columns,
            title=title,
            labels=custom_labels,
            color_discrete_sequence=color_sequence
        )
    
    # Apply custom hover template for better readability
    for trace in fig.data:
        trace.hovertemplate = '%{y}<extra></extra>'
    
    # Update layout
    fig.update_layout(
        height=height,
        margin=dict(t=30, b=30 if not range_slider else 50, l=80, r=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3 if not range_slider else -0.4,
            xanchor="center",
            x=0.5
        ) if show_legend else dict(visible=False),
        hovermode="x unified"
    )
    
    # Update y-axis title if provided
    if y_axis_title:
        fig.update_yaxes(title_text=y_axis_title)
    
    # Enable stacking if requested
    if stack and len(valid_y_columns) > 1:
        fig.update_layout(stackgroup='one')
    
    # Add range slider if requested
    if range_slider:
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date" if pd.api.types.is_datetime64_any_dtype(plot_df[x_column]) else None
            )
        )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    return fig


def render_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_columns: Union[str, List[str]],
    title: str = "",
    height: int = 300,
    color_sequence: Optional[List[str]] = None,
    horizontal: bool = False,
    stacked: bool = False,
    custom_labels: Optional[Dict[str, str]] = None,
    text_auto: bool = False,
    sort_values: bool = False,
    sort_ascending: bool = False
):
    """
    Render a bar chart with Plotly.
    
    Args:
        df: DataFrame containing the data
        x_column: Column name for x-axis
        y_columns: Column name(s) for y-axis
        title: Chart title
        height: Chart height in pixels
        color_sequence: Optional custom color sequence
        horizontal: Whether to use horizontal bars
        stacked: Whether to stack bars (for multiple y columns)
        custom_labels: Optional dictionary mapping column names to display labels
        text_auto: Whether to show values on bars
        sort_values: Whether to sort bars by values
        sort_ascending: Whether to sort in ascending order (if sorting)
    """
    # Convert to list if single string
    if isinstance(y_columns, str):
        y_columns = [y_columns]
    
    # Safety check for empty data or missing columns
    if df is None or df.empty or x_column not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            height=height,
            annotations=[
                {
                    "text": "No data available",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 16
                    },
                    "x": 0.5,
                    "y": 0.5
                }
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
        return fig
    
    # Filter out y_columns that don't exist in the dataframe
    valid_y_columns = [col for col in y_columns if col in df.columns]
    
    if not valid_y_columns:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            height=height,
            annotations=[
                {
                    "text": "No valid Y columns found",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 16
                    },
                    "x": 0.5,
                    "y": 0.5
                }
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
        return fig
    
    # Sort dataframe if requested
    plot_df = df.copy()
    if sort_values and len(valid_y_columns) == 1:
        plot_df = plot_df.sort_values(by=valid_y_columns[0], ascending=sort_ascending)
    
    # Handle text display
    text_auto_value = text_auto
    if text_auto and not isinstance(text_auto, str):
        # Auto-format text based on value range
        max_val = plot_df[valid_y_columns[0]].max() if len(valid_y_columns) > 0 else 0
        if max_val > 10000:
            text_auto_value = '.1s'  # Abbreviated format for large numbers
        elif max_val > 100:
            text_auto_value = '.0f'  # No decimals for medium numbers
        else:
            text_auto_value = '.2f'  # 2 decimals for small numbers
    
    # Create figure
    if horizontal:
        fig = px.bar(
            plot_df,
            y=x_column,
            x=valid_y_columns,
            title=title,
            labels=custom_labels,
            color_discrete_sequence=color_sequence,
            orientation='h',
            barmode='stack' if stacked else 'group',
            text_auto=text_auto_value
        )
    else:
        fig = px.bar(
            plot_df,
            x=x_column,
            y=valid_y_columns,
            title=title,
            labels=custom_labels,
            color_discrete_sequence=color_sequence,
            barmode='stack' if stacked else 'group',
            text_auto=text_auto_value
        )
    
    # Fine-tune text presentation for better readability
    if text_auto:
        if horizontal:
            fig.update_traces(textposition='auto', textangle=0, insidetextanchor='middle')
        else:
            fig.update_traces(textposition='outside', textangle=0)
    
    # Update layout
    fig.update_layout(
        height=height,
        margin=dict(t=30, b=50, l=80, r=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        hovermode="closest"
    )
    
    # Rotate x-axis labels if needed for better readability
    if not horizontal and len(plot_df[x_column].unique()) > 5:
        fig.update_layout(xaxis_tickangle=-45)
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    return fig


def render_pie_chart(
    df: pd.DataFrame,
    names_column: str,
    values_column: str,
    title: str = "",
    height: int = 300,
    color_sequence: Optional[List[str]] = None,
    donut: bool = False,
    show_percentages: bool = True,
    show_legend: bool = True,
    sort_values: bool = True
):
    """
    Render a pie or donut chart with Plotly.
    
    Args:
        df: DataFrame containing the data
        names_column: Column name for slice labels
        values_column: Column name for slice values
        title: Chart title
        height: Chart height in pixels
        color_sequence: Optional custom color sequence
        donut: Whether to use a donut chart
        show_percentages: Whether to show percentages on slices
        show_legend: Whether to show the legend
        sort_values: Whether to sort slices by values
    """
    # Safety check for empty data or missing columns
    if df is None or df.empty or names_column not in df.columns or values_column not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            height=height,
            annotations=[
                {
                    "text": "No data available",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 16
                    },
                    "x": 0.5,
                    "y": 0.5
                }
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
        return fig
    
    # Drop rows with null values
    plot_df = df.dropna(subset=[names_column, values_column]).copy()
    
    # Check for any remaining data after filtering
    if plot_df.empty or plot_df[values_column].sum() == 0:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            height=height,
            annotations=[
                {
                    "text": "No valid data for pie chart",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 16
                    },
                    "x": 0.5,
                    "y": 0.5
                }
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
        return fig
    
    # Group by names and sum values in case of duplicates
    plot_df = plot_df.groupby(names_column)[values_column].sum().reset_index()
    
    # Sort dataframe if requested
    if sort_values:
        plot_df = plot_df.sort_values(by=values_column, ascending=False)
    
    # Limit the number of slices to improve readability
    max_slices = 8
    if len(plot_df) > max_slices:
        # Keep top N-1 values and group the rest as "Other"
        top_df = plot_df.iloc[:max_slices-1]
        other_value = plot_df.iloc[max_slices-1:][values_column].sum()
        
        # Only add "Other" category if it has a non-zero value
        if other_value > 0:
            other_df = pd.DataFrame({
                names_column: ["Other"],
                values_column: [other_value]
            })
            plot_df = pd.concat([top_df, other_df], ignore_index=True)
        else:
            plot_df = top_df
    
    # Create figure
    fig = px.pie(
        plot_df,
        names=names_column,
        values=values_column,
        title=title,
        color_discrete_sequence=color_sequence
    )
    
    # Update traces for donut chart
    if donut:
        fig.update_traces(hole=0.4)
    
    # Update text mode
    if show_percentages:
        fig.update_traces(textposition='inside', textinfo='percent+label')
    else:
        fig.update_traces(textposition='inside', textinfo='label+value')
    
    # Update layout
    fig.update_layout(
        height=height,
        margin=dict(t=30, b=30, l=30, r=30),
        showlegend=show_legend,
        legend=dict(
            orientation="h" if len(plot_df) <= 5 else "v",
            yanchor="top",
            y=1.1 if len(plot_df) <= 5 else 1.0,
            xanchor="center",
            x=0.5 if len(plot_df) <= 5 else 1.0
        )
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    return fig


def render_gauge_chart(
    value: float,
    title: str = "",
    min_value: float = 0,
    max_value: float = 100,
    threshold_values: Optional[List[float]] = None,
    threshold_colors: Optional[List[str]] = None,
    height: int = 200,
    show_current_value: bool = True,
    suffix: str = "",
    decimal_places: int = 1
):
    """
    Render a gauge chart with Plotly.
    
    Args:
        value: Current value to display
        title: Chart title
        min_value: Minimum value on the gauge
        max_value: Maximum value on the gauge
        threshold_values: Optional list of thresholds for color changes
        threshold_colors: Optional list of colors for each threshold
        height: Chart height in pixels
        show_current_value: Whether to show the current value in the gauge
        suffix: Suffix to add to the value (e.g., "%", "GB")
        decimal_places: Number of decimal places to show
    """
    # Default thresholds and colors if not provided
    if threshold_values is None:
        threshold_values = [max_value * 0.33, max_value * 0.66, max_value]
    
    if threshold_colors is None:
        threshold_colors = ["green", "yellow", "red"]
    
    # Calculate percentage
    percent = (value - min_value) / (max_value - min_value) * 100
    
    # Create figure
    fig = go.Figure(go.Indicator(
        mode="gauge+number" if show_current_value else "gauge",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [min_value, max_value], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_value, threshold_values[0]], 'color': threshold_colors[0]},
                {'range': [threshold_values[0], threshold_values[1]], 'color': threshold_colors[1]},
                {'range': [threshold_values[1], max_value], 'color': threshold_colors[2]}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        },
        number={'suffix': suffix, 'valueformat': f'.{decimal_places}f'}
    ))
    
    # Update layout
    fig.update_layout(
        height=height,
        margin=dict(t=30, b=0, l=30, r=30)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    return fig


def render_heat_map(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    z_column: str,
    title: str = "",
    height: int = 400,
    color_scale: Optional[Union[str, List[str]]] = None,
    show_values: bool = False,
    custom_labels: Optional[Dict[str, str]] = None
):
    """
    Render a heat map with Plotly.
    
    Args:
        df: DataFrame containing the data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        z_column: Column name for z values (color intensity)
        title: Chart title
        height: Chart height in pixels
        color_scale: Optional custom color scale
        show_values: Whether to show values in cells
        custom_labels: Optional dictionary mapping column names to display labels
    """
    # Create a pivot table if not already in the right format
    if z_column in df.columns:
        plot_df = df.pivot_table(
            values=z_column,
            index=y_column,
            columns=x_column,
            aggfunc='mean'
        )
    else:
        # Assume data is already pivoted
        plot_df = df
    
    # Create figure
    fig = px.imshow(
        plot_df, 
        title=title,
        color_continuous_scale=color_scale,
        labels=custom_labels
    )
    
    # Show values if requested
    if show_values:
        fig.update_traces(text=plot_df.values, texttemplate="%{text:.2f}")
    
    # Update layout
    fig.update_layout(
        height=height,
        margin=dict(t=30, b=50, l=80, r=30)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    return fig


def render_multi_chart(
    num_rows: int,
    num_cols: int,
    subplot_titles: List[str],
    height: int = 600,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False
):
    """
    Create a multi-chart layout with Plotly.
    
    Args:
        num_rows: Number of rows in the grid
        num_cols: Number of columns in the grid
        subplot_titles: List of titles for each subplot
        height: Overall chart height in pixels
        shared_xaxes: Whether to share x-axes across subplots
        shared_yaxes: Whether to share y-axes across subplots
        
    Returns:
        fig: Plotly figure object with multiple subplots
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes
    )
    
    # Update layout
    fig.update_layout(
        height=height,
        margin=dict(t=50, b=30, l=80, r=30)
    )
    
    return fig


def add_line_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    name: str,
    row: Optional[int] = None,
    col: Optional[int] = None,
    color: Optional[str] = None,
    line_dash: Optional[str] = None,
    mode: str = 'lines',
    fill: Optional[str] = None
):
    """
    Add a line trace to a Plotly figure.
    
    Args:
        fig: Plotly figure object
        df: DataFrame containing the data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        name: Name for the trace (appears in legend)
        row: Row index for subplot (1-based)
        col: Column index for subplot (1-based)
        color: Line color
        line_dash: Line style (e.g., 'solid', 'dash', 'dot')
        mode: Trace mode (e.g., 'lines', 'markers', 'lines+markers')
        fill: Fill style (e.g., 'tozeroy', 'tonexty')
    """
    trace = go.Scatter(
        x=df[x_column],
        y=df[y_column],
        name=name,
        mode=mode,
        line=dict(color=color, dash=line_dash) if color or line_dash else None,
        fill=fill
    )
    
    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)
    
    return fig


def add_bar_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    name: str,
    row: Optional[int] = None,
    col: Optional[int] = None,
    color: Optional[str] = None,
    text: Optional[List] = None,
    orientation: str = 'v'
):
    """
    Add a bar trace to a Plotly figure.
    
    Args:
        fig: Plotly figure object
        df: DataFrame containing the data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        name: Name for the trace (appears in legend)
        row: Row index for subplot (1-based)
        col: Column index for subplot (1-based)
        color: Bar color
        text: Optional text to display on bars
        orientation: Bar orientation ('v' for vertical, 'h' for horizontal)
    """
    trace = go.Bar(
        x=df[y_column] if orientation == 'h' else df[x_column],
        y=df[x_column] if orientation == 'h' else df[y_column],
        name=name,
        marker_color=color,
        text=text,
        orientation=orientation
    )
    
    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)
    
    return fig