"""
Visualization utilities for OmniPulse monitoring dashboard.

This module provides functions for creating charts, graphs, and other visual elements
used in the OmniPulse dashboard to display various metrics and analytics.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any, Tuple, Optional, Union, Callable

# Color schemes for consistent visualization
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
    'text': '#333333',
    'chart_palette': [
        '#3366CC', '#DC3912', '#FF9900', '#109618', '#990099',
        '#0099C6', '#DD4477', '#66AA00', '#B82E2E', '#316395'
    ]
}

# Define chart templates with consistent styling
CHART_TEMPLATE = {
    'layout': {
        'font': {'family': 'Roboto, Arial, sans-serif', 'color': COLORS['text']},
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'margin': {'t': 30, 'r': 10, 'b': 40, 'l': 50},
        'xaxis': {
            'gridcolor': COLORS['grid'],
            'zerolinecolor': COLORS['grid'],
            'tickfont': {'size': 10},
        },
        'yaxis': {
            'gridcolor': COLORS['grid'],
            'zerolinecolor': COLORS['grid'],
            'tickfont': {'size': 10},
        },
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1
        },
        'colorway': COLORS['chart_palette']
    }
}

# Apply global theme at import time
def apply_theme_to_figures():
    """
    Apply global theme settings to all Plotly figures.
    This function should be called at the beginning of the application
    to ensure consistent styling across all visualizations.
    """
    import plotly.io as pio
    
    # Create a custom template based on our styling
    template = go.layout.Template(
        layout=go.Layout(
            font=dict(family='Roboto, Arial, sans-serif', color=COLORS['text']),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, r=10, b=40, l=50),
            xaxis=dict(
                gridcolor=COLORS['grid'],
                zerolinecolor=COLORS['grid'],
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                gridcolor=COLORS['grid'],
                zerolinecolor=COLORS['grid'],
                tickfont=dict(size=10)
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            colorway=COLORS['chart_palette']
        )
    )
    
    # Register the template and set as default
    pio.templates["omnipulse"] = template
    pio.templates.default = "omnipulse"

# Apply the theme immediately upon module import
apply_theme_to_figures()

def format_number(value: float, precision: int = 1, unit: str = '') -> str:
    """
    Format a number with appropriate suffixes (K, M, G, T) and optional unit.
    
    Args:
        value: The number to format
        precision: Number of decimal places
        unit: Optional unit to append
        
    Returns:
        Formatted string representation of the number
    """
    if value is None:
        return "N/A"
    
    suffix = ""
    if abs(value) >= 1e12:
        value /= 1e12
        suffix = "T"
    elif abs(value) >= 1e9:
        value /= 1e9
        suffix = "G"
    elif abs(value) >= 1e6:
        value /= 1e6
        suffix = "M"
    elif abs(value) >= 1e3:
        value /= 1e3
        suffix = "K"
        
    if value == 0 or abs(value) >= 100:
        # No decimal places for large numbers
        formatted = f"{value:.0f}"
    else:
        # Use specified precision for smaller numbers
        formatted = f"{value:.{precision}f}"
        
    # Remove trailing zeros and decimal point if not needed
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.')
        
    if unit:
        return f"{formatted}{suffix} {unit}"
    else:
        return f"{formatted}{suffix}"

def format_time_delta(seconds: float) -> str:
    """
    Format a time duration in seconds to a human-readable string.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted time string (e.g., "3.5 min", "2.1 hr", "1.5 days")
    """
    if seconds is None:
        return "N/A"
    
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    elif seconds < 60:
        return f"{seconds:.1f} sec"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f} hr"
    else:
        return f"{seconds / 86400:.1f} days"

def format_bytes(bytes_value: int, precision: int = 1) -> str:
    """
    Format bytes value to human-readable string with appropriate units.
    
    Args:
        bytes_value: Value in bytes
        precision: Number of decimal places
        
    Returns:
        Formatted string with appropriate unit (B, KB, MB, GB, TB)
    """
    if bytes_value is None:
        return "N/A"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    
    if bytes_value == 0:
        return "0 B"
    
    i = 0
    while bytes_value >= 1024 and i < len(units) - 1:
        bytes_value /= 1024.0
        i += 1
        
    if bytes_value < 10:
        # More precision for small numbers
        return f"{bytes_value:.{precision+1}f} {units[i]}"
    else:
        return f"{bytes_value:.{precision}f} {units[i]}"

def create_line_chart(
    data: pd.DataFrame,
    x_column: str,
    y_columns: List[str],
    title: str = '',
    x_title: str = '',
    y_title: str = '',
    colors: List[str] = None,
    height: int = 300,
    mode: str = 'lines',
    line_shape: str = 'linear',
    area: bool = False,
    legend: bool = True,
    y_format: Callable = None,
    range_selector: bool = False
) -> go.Figure:
    """
    Create a line chart from pandas DataFrame.
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis
        y_columns: List of column names for y-axis
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        colors: List of colors for each line
        height: Chart height in pixels
        mode: Plotly mode ('lines', 'lines+markers', etc.)
        line_shape: Line shape ('linear', 'spline', etc.)
        area: Whether to fill area under the line
        legend: Whether to show legend
        y_format: Function to format y-axis tick labels
        range_selector: Whether to add time range selector
        
    Returns:
        Plotly figure object
    """
    if colors is None:
        colors = COLORS['chart_palette']
    
    # Ensure data is sorted by x_column for proper line rendering
    if x_column in data.columns:
        data = data.sort_values(by=x_column)
    
    fig = go.Figure()
    
    # Check if x axis is datetime
    is_datetime = False
    if x_column in data.columns and data[x_column].dtype in ['datetime64[ns]', '<M8[ns]']:
        is_datetime = True
    
    for i, y_column in enumerate(y_columns):
        if y_column not in data.columns:
            continue
            
        color = colors[i % len(colors)]
        
        if area:
            fig.add_trace(
                go.Scatter(
                    x=data[x_column],
                    y=data[y_column],
                    name=y_column.replace('_', ' ').title(),
                    mode=mode,
                    line=dict(width=2, color=color, shape=line_shape),
                    fill='tozeroy',
                    fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.2])}',
                    hovertemplate=f"{y_column.replace('_', ' ').title()}: %{{y}}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data[x_column],
                    y=data[y_column],
                    name=y_column.replace('_', ' ').title(),
                    mode=mode,
                    line=dict(width=2, color=color, shape=line_shape),
                    hovertemplate=f"{y_column.replace('_', ' ').title()}: %{{y}}<extra></extra>",
                )
            )
    
    # Apply template and customize
    fig.update_layout(
        title=title,
        height=height,
        showlegend=legend,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
    )
    
    # Format y-axis ticks if formatter provided
    if y_format:
        fig.update_yaxes(tickformat=y_format)
    
    # Add range selector for time series
    if range_selector and is_datetime:
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=15, label="15m", step="minute", stepmode="backward"),
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=6, label="6h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ]),
                    bgcolor=COLORS['background'],
                    activecolor=COLORS['primary'],
                    font=dict(color=COLORS['text'])
                ),
                rangeslider=dict(visible=False),
                type="date"
            )
        )
    
    return fig

def create_bar_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = '',
    x_title: str = '',
    y_title: str = '',
    color: str = None,
    height: int = 300,
    horizontal: bool = False,
    sort: bool = False,
    color_discrete_map: Dict = None
) -> go.Figure:
    """
    Create a bar chart from pandas DataFrame.
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        color: Color for bars or column name for color mapping
        height: Chart height in pixels
        horizontal: If True, creates a horizontal bar chart
        sort: Whether to sort bars by value
        color_discrete_map: Dictionary mapping categories to colors
        
    Returns:
        Plotly figure object
    """
    if color is None:
        color = COLORS['primary']
    
    # Make a copy to avoid modifying the original dataframe
    plot_data = data.copy()
    
    # Check if columns exist
    if x_column not in plot_data.columns or y_column not in plot_data.columns:
        # Handle missing columns by creating an empty dataframe with the required columns
        plot_data = pd.DataFrame({x_column: [], y_column: []})
    elif sort:
        plot_data = plot_data.sort_values(y_column)
    
    # For categorical data, ensure proper ordering
    if x_column in plot_data.columns and plot_data[x_column].dtype == 'object':
        # Use category dtype with defined order to prevent alphabetical sorting
        if not plot_data.empty:
            plot_data[x_column] = pd.Categorical(
                plot_data[x_column],
                categories=plot_data[x_column].unique()
            )
    
    if horizontal:
        fig = px.bar(
            plot_data, 
            y=x_column, 
            x=y_column, 
            title=title,
            color=color if isinstance(color, str) and color in plot_data.columns else None,
            color_discrete_map=color_discrete_map,
            height=height,
            text_auto='.2s' if not plot_data.empty and plot_data[y_column].max() > 1000 else True
        )
        
        # If color is a string but not a column name, set the bar color
        if isinstance(color, str) and color not in plot_data.columns:
            fig.update_traces(marker_color=color)
    else:
        fig = px.bar(
            plot_data, 
            x=x_column, 
            y=y_column, 
            title=title,
            color=color if isinstance(color, str) and color in plot_data.columns else None,
            color_discrete_map=color_discrete_map,
            height=height,
            text_auto='.2s' if not plot_data.empty and plot_data[y_column].max() > 1000 else True
        )
        
        # If color is a string but not a column name, set the bar color
        if isinstance(color, str) and color not in plot_data.columns:
            fig.update_traces(marker_color=color)
    
    # Customize layout
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        bargap=0.2,
        hovermode="closest"
    )
    
    # Improve text display when using text labels
    fig.update_traces(
        textposition='auto', 
        textangle=0,
        textfont_size=10
    )
    
    return fig

def create_gauge_chart(
    value: float,
    title: str = '',
    min_value: float = 0,
    max_value: float = 100,
    threshold_value: float = None,
    format_func: Callable = None,
    color_scheme: List[str] = None,
    height: int = 200
) -> go.Figure:
    """
    Create a gauge chart to display a single metric value.
    
    Args:
        value: Value to display
        title: Chart title
        min_value: Minimum value on gauge
        max_value: Maximum value on gauge
        threshold_value: Optional threshold value for warning/danger
        format_func: Function to format the displayed value
        color_scheme: List of colors for gauge segments
        height: Chart height in pixels
        
    Returns:
        Plotly figure object
    """
    if format_func is None:
        format_func = lambda x: f"{x:.1f}"
    
    if color_scheme is None:
        color_scheme = [
            COLORS['success'], 
            COLORS['warning'], 
            COLORS['danger']
        ]
    
    # Ensure value is within range
    display_value = max(min_value, min(max_value, value))
    
    # Create gauge steps based on threshold or evenly distributed
    if threshold_value is not None:
        # Use threshold to determine color
        if max_value == min_value:
            normalized_value = 0.5  # Avoid division by zero
        else:
            normalized_value = (display_value - min_value) / (max_value - min_value)
        normalized_threshold = (threshold_value - min_value) / (max_value - min_value) if max_value != min_value else 0.8
        
        if normalized_value < normalized_threshold * 0.75:
            # Good - below 75% of threshold
            gauge_color = color_scheme[0]
        elif normalized_value < normalized_threshold:
            # Warning - between 75% and 100% of threshold
            gauge_color = color_scheme[1]
        else:
            # Danger - above threshold
            gauge_color = color_scheme[2]
        
        steps = [
            {'range': [min_value, threshold_value * 0.75], 'color': color_scheme[0], 'thickness': 0.75},
            {'range': [threshold_value * 0.75, threshold_value], 'color': color_scheme[1], 'thickness': 0.75},
            {'range': [threshold_value, max_value], 'color': color_scheme[2], 'thickness': 0.75},
        ]
    else:
        # Evenly distribute colors
        step_size = (max_value - min_value) / len(color_scheme)
        steps = [
            {'range': [min_value + i * step_size, min_value + (i + 1) * step_size], 
             'color': color, 'thickness': 0.75}
            for i, color in enumerate(color_scheme)
        ]
        
        # Determine gauge color based on value position
        if max_value == min_value:
            gauge_color = color_scheme[0]
        else:
            normalized_value = (display_value - min_value) / (max_value - min_value)
            color_index = min(int(normalized_value * len(color_scheme)), len(color_scheme) - 1)
            gauge_color = color_scheme[color_index]
    
    # Create figure
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_value,
        title={'text': title, 'font': {'size': 14}},
        number={
            'font': {'size': 20, 'color': gauge_color},
            'valueformat': '',
            'suffix': '',
            'prefix': ''
        },
        gauge={
            'axis': {'range': [min_value, max_value], 'tickwidth': 1, 'tickcolor': COLORS['text']},
            'bar': {'color': gauge_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': steps,
        }
    ))
    
    # Format the displayed value
    formatted_value = format_func(display_value)
    fig.update_traces(number={'value': formatted_value, 'valueformat': ''})
    
    # Customize layout
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']}
    )
    
    return fig

def create_pie_chart(
    data: pd.DataFrame,
    value_column: str,
    name_column: str,
    title: str = '',
    height: int = 300,
    hole: float = 0.4,
    colors: List[str] = None,
    sort: bool = True
) -> go.Figure:
    """
    Create a pie/donut chart from pandas DataFrame.
    
    Args:
        data: DataFrame containing the data
        value_column: Column name for values
        name_column: Column name for labels
        title: Chart title
        height: Chart height in pixels
        hole: Size of the hole (0 for pie chart, 0.4 for donut chart)
        colors: List of colors for pie segments
        sort: Whether to sort segments by value
        
    Returns:
        Plotly figure object
    """
    if colors is None:
        colors = COLORS['chart_palette']
    
    if sort:
        data = data.sort_values(value_column, ascending=False)
    
    fig = px.pie(
        data,
        values=value_column,
        names=name_column,
        title=title,
        hole=hole,
        height=height,
        color_discrete_sequence=colors
    )
    
    # Customize layout
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=1))
    )
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Roboto, Arial, sans-serif', 'color': COLORS['text']},
        showlegend=False if len(data) <= 5 else True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
    )
    
    return fig

def create_heatmap(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    z_column: str,
    title: str = '',
    x_title: str = '',
    y_title: str = '',
    height: int = 400,
    colorscale: str = 'Viridis',
    showscale: bool = True
) -> go.Figure:
    """
    Create a heatmap from pandas DataFrame.
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        z_column: Column name for z-values (color intensity)
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        height: Chart height in pixels
        colorscale: Color scale name
        showscale: Whether to show color scale
        
    Returns:
        Plotly figure object
    """
    # Make a copy to avoid modifying the original dataframe
    plot_data = data.copy()
    
    # Check if necessary columns exist
    if not all(col in plot_data.columns for col in [x_column, y_column, z_column]):
        # Return an empty heatmap with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for heatmap",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=height, title=title)
        return fig
    
    # Pivot data for heatmap format if needed
    if len(plot_data[[x_column, y_column, z_column]].drop_duplicates()) == len(plot_data):
        # Data is in long format, need to pivot
        try:
            pivot_data = plot_data.pivot(index=y_column, columns=x_column, values=z_column)
        except ValueError:
            # Handle duplicate entries by taking the mean
            pivot_data = plot_data.pivot_table(
                index=y_column, 
                columns=x_column, 
                values=z_column,
                aggfunc='mean'
            )
    else:
        # Data is already in the right format
        pivot_data = plot_data
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=colorscale,
        showscale=showscale,
        hovertemplate='%{y} - %{x}: %{z}<extra></extra>',
        colorbar=dict(
            title=z_column.replace('_', ' ').title(),
            titleside='right',
            titlefont=dict(size=12),
            tickfont=dict(size=10)
        )
    ))
    
    # Customize layout
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title=x_title if x_title else x_column.replace('_', ' ').title(),
        yaxis_title=y_title if y_title else y_column.replace('_', ' ').title(),
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(
            side='bottom',
            tickangle=-45 if len(pivot_data.columns) > 5 else 0
        )
    )
    
    return fig

def create_metric_card(
    title: str,
    value: Any,
    previous_value: Any = None,
    change_formatter: Callable = None,
    value_formatter: Callable = None,
    is_positive_good: bool = True,
    show_sparkline: bool = False,
    sparkline_data: List[float] = None,
    prefix: str = '',
    suffix: str = '',
    help_text: str = None
) -> None:
    """
    Create a metric card with optional trend indicator and sparkline.
    
    Args:
        title: Metric title
        value: Current metric value
        previous_value: Previous metric value for trend calculation
        change_formatter: Function to format the change value
        value_formatter: Function to format the displayed value
        is_positive_good: Whether a positive change is good (green) or bad (red)
        show_sparkline: Whether to show a sparkline
        sparkline_data: List of historical values for sparkline
        prefix: Prefix for displayed value
        suffix: Suffix for displayed value
        help_text: Optional help text to display on hover
    """
    # Apply formatters
    if value_formatter:
        formatted_value = value_formatter(value)
    else:
        formatted_value = value
    
    # Calculate change if previous value is provided
    if previous_value is not None and previous_value != 0:
        change = (value - previous_value) / previous_value * 100
        
        if change_formatter:
            formatted_change = change_formatter(change)
        else:
            formatted_change = f"{change:.1f}%"
        
        # Determine color based on change direction and is_positive_good
        if (change > 0 and is_positive_good) or (change < 0 and not is_positive_good):
            change_color = COLORS['success']
            change_icon = "↑"
        elif (change < 0 and is_positive_good) or (change > 0 and not is_positive_good):
            change_color = COLORS['danger']
            change_icon = "↓"
        else:
            change_color = COLORS['light']
            change_icon = "→"
    else:
        change = None
        formatted_change = ""
        change_color = COLORS['light']
        change_icon = ""
    
    # Create card with custom CSS
    with st.container():
        st.markdown(f"""
        <div style="
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="color: #666; font-size: 14px; margin-bottom: 5px;">{title}</p>
                    <h3 style="font-size: 24px; margin: 0; font-weight: 500;">
                        {prefix}{formatted_value}{suffix}
                    </h3>
                </div>
                <div>
                    {f'<p style="color: {change_color}; font-size: 16px; margin: 0;">{change_icon} {formatted_change}</p>' if change is not None else ''}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if help_text:
            with st.expander("More info"):
                st.caption(help_text)
    
    # Add sparkline if enabled
    if show_sparkline and sparkline_data and len(sparkline_data) > 1:
        sparkline_df = pd.DataFrame({
            'index': range(len(sparkline_data)),
            'value': sparkline_data
        })
        
        # Create a small sparkline chart
        spark_chart = create_line_chart(
            sparkline_df,
            'index',
            ['value'],
            height=50,
            area=True,
            legend=False
        )
        
        # Customize for sparkline appearance
        spark_chart.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        )
        
        st.plotly_chart(spark_chart, use_container_width=True, config={'displayModeBar': False})

def create_status_indicator(
    status: str,
    show_text: bool = True,
    size: str = "medium"
) -> str:
    """
    Create a colored status indicator.
    
    Args:
        status: Status string ('healthy', 'warning', 'critical', 'unknown')
        show_text: Whether to show status text
        size: Size of indicator ('small', 'medium', 'large')
        
    Returns:
        HTML string with status indicator
    """
    sizes = {
        "small": {"dot": 8, "font": 12},
        "medium": {"dot": 12, "font": 14},
        "large": {"dot": 16, "font": 16}
    }
    
    dot_size = sizes.get(size, sizes["medium"])["dot"]
    font_size = sizes.get(size, sizes["medium"])["font"]
    
    status_colors = {
        'healthy': COLORS['success'],
        'warning': COLORS['warning'],
        'critical': COLORS['danger'],
        'unknown': COLORS['light']
    }
    
    color = status_colors.get(status.lower(), COLORS['light'])
    
    if show_text:
        return f"""
        <div style="display: flex; align-items: center;">
            <div style="
                width: {dot_size}px;
                height: {dot_size}px;
                background-color: {color};
                border-radius: 50%;
                margin-right: 6px;
            "></div>
            <span style="font-size: {font_size}px;">{status.title()}</span>
        </div>
        """
    else:
        return f"""
        <div style="
            width: {dot_size}px;
            height: {dot_size}px;
            background-color: {color};
            border-radius: 50%;
        "></div>
        """

def create_time_series_chart(
    data: pd.DataFrame,
    timestamp_column: str,
    metric_columns: List[str],
    title: str = '',
    height: int = 400,
    time_range: str = 'all',
    resample: str = None,
    agg_func: str = 'mean',
    show_range_selector: bool = True
) -> go.Figure:
    """
    Create a time series chart with time range selection and optional resampling.
    
    Args:
        data: DataFrame containing time series data
        timestamp_column: Column name with timestamps
        metric_columns: List of column names for metrics to plot
        title: Chart title
        height: Chart height in pixels
        time_range: Time range to display ('15m', '1h', '6h', '1d', 'all')
        resample: Pandas resample rule (e.g., '1min', '5min', '1h')
        agg_func: Aggregation function for resampling ('mean', 'sum', 'max', etc.)
        show_range_selector: Whether to show time range selector
        
    Returns:
        Plotly figure object
    """
    # Ensure timestamp column is datetime type
    if data[timestamp_column].dtype != 'datetime64[ns]':
        data = data.copy()
        data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    
    # Filter data based on time range
    if time_range != 'all':
        now = pd.Timestamp.now()
        
        if time_range == '15m':
            start_time = now - pd.Timedelta(minutes=15)
        elif time_range == '1h':
            start_time = now - pd.Timedelta(hours=1)
        elif time_range == '6h':
            start_time = now - pd.Timedelta(hours=6)
        elif time_range == '1d':
            start_time = now - pd.Timedelta(days=1)
        else:
            start_time = data[timestamp_column].min()
            
        filtered_data = data[data[timestamp_column] >= start_time]
    else:
        filtered_data = data
    
    # Resample data if specified
    if resample and filtered_data.shape[0] > 0:
        # Set timestamp as index for resampling
        filtered_data = filtered_data.set_index(timestamp_column)
        
        # Select aggregation function
        if agg_func == 'mean':
            agg_fn = np.mean
        elif agg_func == 'sum':
            agg_fn = np.sum
        elif agg_func == 'max':
            agg_fn = np.max
        elif agg_func == 'min':
            agg_fn = np.min
        elif agg_func == 'median':
            agg_fn = np.median
        else:
            agg_fn = np.mean
        
        # Resample
        resampled = filtered_data[metric_columns].resample(resample).agg(agg_fn)
        
        # Reset index to get timestamp as column again
        filtered_data = resampled.reset_index()
        timestamp_column = filtered_data.columns[0]  # First column is the timestamp
    
    # Create the chart
    fig = create_line_chart(
        filtered_data,
        timestamp_column,
        metric_columns,
        title=title,
        height=height,
        range_selector=show_range_selector,
        mode='lines+markers' if filtered_data.shape[0] < 100 else 'lines'
    )
    
    return fig

def create_comparison_chart(
    current_data: pd.DataFrame,
    historical_data: pd.DataFrame,
    timestamp_column: str,
    metric_column: str,
    title: str = '',
    height: int = 400,
    current_label: str = 'Current',
    historical_label: str = 'Historical',
    show_anomalies: bool = False,
    anomaly_threshold: float = 2.0
) -> go.Figure:
    """
    Create a chart comparing current metrics with historical baseline.
    
    Args:
        current_data: DataFrame containing current time series data
        historical_data: DataFrame containing historical time series data
        timestamp_column: Column name with timestamps
        metric_column: Column name for metric to compare
        title: Chart title
        height: Chart height in pixels
        current_label: Label for current data series
        historical_label: Label for historical data series
        show_anomalies: Whether to highlight anomalies
        anomaly_threshold: Z-score threshold for anomaly detection
        
    Returns:
        Plotly figure object
    """
    # Create figure with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Add current data trace
    fig.add_trace(
        go.Scatter(
            x=current_data[timestamp_column],
            y=current_data[metric_column],
            name=current_label,
            line=dict(color=COLORS['primary'], width=2),
            mode='lines'
        )
    )
    
    # Add historical data trace
    fig.add_trace(
        go.Scatter(
            x=historical_data[timestamp_column],
            y=historical_data[metric_column],
            name=historical_label,
            line=dict(color=COLORS['secondary'], width=2, dash='dash'),
            mode='lines'
        )
    )
    
    # Add anomalies if enabled
    if show_anomalies and len(current_data) > 0:
        # Calculate mean and standard deviation of historical data
        historical_mean = historical_data[metric_column].mean()
        historical_std = historical_data[metric_column].std()
        
        if historical_std > 0:  # Avoid division by zero
            # Calculate z-scores
            current_data['z_score'] = (current_data[metric_column] - historical_mean) / historical_std
            
            # Identify anomalies
            anomalies = current_data[abs(current_data['z_score']) > anomaly_threshold]
            
            if len(anomalies) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=anomalies[timestamp_column],
                        y=anomalies[metric_column],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(
                            color=COLORS['danger'],
                            size=10,
                            symbol='circle',
                            line=dict(width=2, color='white')
                        )
                    )
                )
    
    # Customize layout
    fig.update_layout(
        title=title,
        height=height,
        template=CHART_TEMPLATE,
        xaxis_title="Time",
        yaxis_title=metric_column.replace('_', ' ').title(),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_grid_dashboard(
    charts: List[Tuple[str, go.Figure]],
    n_columns: int = 2,
    height_ratios: List[float] = None,
    container_height: str = None
) -> None:
    """
    Create a grid layout dashboard with multiple charts.
    
    Args:
        charts: List of (chart_id, chart_figure) tuples
        n_columns: Number of columns in the grid
        height_ratios: Optional list of relative heights for each row
        container_height: Optional height for each chart container (CSS value)
    """
    # Safety check
    if not charts:
        st.warning("No charts provided to display in grid dashboard.")
        return
    
    # Calculate number of rows needed
    n_charts = len(charts)
    n_rows = (n_charts + n_columns - 1) // n_columns  # Ceiling division
    
    # Validate height_ratios
    if height_ratios and len(height_ratios) != n_rows:
        st.warning(f"Number of height ratios ({len(height_ratios)}) doesn't match number of rows ({n_rows}). Using equal heights.")
        height_ratios = None
    
    # Normalize height ratios if provided
    if height_ratios:
        total = sum(height_ratios)
        height_ratios = [h/total for h in height_ratios]
    
    # Create CSS for fixed height containers if specified
    if container_height:
        st.markdown(f"""
        <style>
        .chart-container {{
            height: {container_height};
            overflow: hidden;
            margin-bottom: 1rem;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    # Create rows
    for i in range(n_rows):
        cols = st.columns(n_columns)
        
        # Add charts to this row
        for j in range(n_columns):
            chart_idx = i * n_columns + j
            
            if chart_idx < n_charts:
                chart_id, chart = charts[chart_idx]
                
                with cols[j]:
                    # If container height specified, wrap in a div
                    if container_height:
                        st.markdown(f'<div class="chart-container">', unsafe_allow_html=True)
                        st.plotly_chart(chart, use_container_width=True, key=chart_id)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.plotly_chart(chart, use_container_width=True, key=chart_id)

def create_multi_axis_chart(
    data: pd.DataFrame,
    x_column: str,
    y_columns: List[str],
    y_axis_map: Dict[str, int] = None,
    title: str = '',
    height: int = 400,
    colors: List[str] = None,
    axis_titles: List[str] = None
) -> go.Figure:
    """
    Create a chart with multiple y-axes for different metrics.
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis
        y_columns: List of column names for y-axes
        y_axis_map: Dictionary mapping column names to axis indices (1 or 2)
        title: Chart title
        height: Chart height in pixels
        colors: List of colors for each line
        axis_titles: List of titles for y-axes
        
    Returns:
        Plotly figure object
    """
    if colors is None:
        colors = COLORS['chart_palette']
    
    if y_axis_map is None:
        # By default, first metric on primary y-axis, others on secondary
        y_axis_map = {col: 1 if i == 0 else 2 for i, col in enumerate(y_columns)}
    
    # Create figure with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces for each metric
    for i, y_column in enumerate(y_columns):
        color = colors[i % len(colors)]
        use_secondary_y = (y_axis_map.get(y_column, 1) == 2)
        
        fig.add_trace(
            go.Scatter(
                x=data[x_column],
                y=data[y_column],
                name=y_column.replace('_', ' ').title(),
                line=dict(width=2, color=color),
                mode='lines'
            ),
            secondary_y=use_secondary_y
        )
    
    # Set y-axis titles
    if axis_titles and len(axis_titles) >= 2:
        fig.update_yaxes(title_text=axis_titles[0], secondary_y=False)
        fig.update_yaxes(title_text=axis_titles[1], secondary_y=True)
    else:
        primary_cols = [col for col, axis in y_axis_map.items() if axis == 1]
        secondary_cols = [col for col, axis in y_axis_map.items() if axis == 2]
        
        if primary_cols:
            primary_title = primary_cols[0].replace('_', ' ').title()
            if len(primary_cols) > 1:
                primary_title += " & Related"
            fig.update_yaxes(title_text=primary_title, secondary_y=False)
        
        if secondary_cols:
            secondary_title = secondary_cols[0].replace('_', ' ').title()
            if len(secondary_cols) > 1:
                secondary_title += " & Related"
            fig.update_yaxes(title_text=secondary_title, secondary_y=True)
    
    # Customize layout
    fig.update_layout(
        title=title,
        height=height,
        template=CHART_TEMPLATE,
        xaxis_title="Time" if data[x_column].dtype in ['datetime64[ns]', '<M8[ns]'] else x_column.replace('_', ' ').title(),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_stacked_bar_chart(
    data: pd.DataFrame,
    x_column: str,
    y_columns: List[str],
    title: str = '',
    x_title: str = '',
    y_title: str = '',
    colors: List[str] = None,
    height: int = 400,
    horizontal: bool = False
) -> go.Figure:
    """
    Create a stacked bar chart from pandas DataFrame.
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis
        y_columns: List of column names for stacked values
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        colors: List of colors for each stack
        height: Chart height in pixels
        horizontal: If True, creates a horizontal stacked bar chart
        
    Returns:
        Plotly figure object
    """
    if colors is None:
        colors = COLORS['chart_palette']
    
    fig = go.Figure()
    
    for i, y_column in enumerate(y_columns):
        color = colors[i % len(colors)]
        
        if horizontal:
            fig.add_trace(
                go.Bar(
                    y=data[x_column],
                    x=data[y_column],
                    name=y_column.replace('_', ' ').title(),
                    marker_color=color,
                    orientation='h'
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=data[x_column],
                    y=data[y_column],
                    name=y_column.replace('_', ' ').title(),
                    marker_color=color
                )
            )
    
    # Set barmode to stack
    fig.update_layout(barmode='stack')
    
    # Customize layout
    fig.update_layout(
        title=title,
        height=height,
        template=CHART_TEMPLATE,
        xaxis_title=x_title if x_title else x_column.replace('_', ' ').title(),
        yaxis_title=y_title if y_title else ', '.join([col.replace('_', ' ').title() for col in y_columns]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_datatable(
    data: pd.DataFrame,
    title: str = None,
    height: int = None,
    precision: int = 2,
    column_config: Dict = None,
    hide_index: bool = True,
    use_container_width: bool = True
) -> None:
    """
    Create an interactive data table with formatting.
    
    Args:
        data: DataFrame containing the data
        title: Optional table title
        height: Optional table height
        precision: Number of decimal places for float columns
        column_config: Dictionary of column configurations
        hide_index: Whether to hide the index column
        use_container_width: Whether to use full container width
    """
    # Display title if provided
    if title:
        st.subheader(title)
    
    # Format numbers for better display
    display_data = data.copy()
    
    # Apply number formatting
    for col in display_data.select_dtypes(include=['float']).columns:
        display_data[col] = display_data[col].round(precision)
    
    # Create default column config if not provided
    if column_config is None:
        column_config = {}
        
        # Configure datetime columns
        for col in display_data.select_dtypes(include=['datetime']).columns:
            column_config[col] = st.column_config.DatetimeColumn(
                format="MMM DD, YYYY, hh:mm:ss a",
                label=col.replace('_', ' ').title()
            )
        
        # Configure number columns
        for col in display_data.select_dtypes(include=['float']).columns:
            column_config[col] = st.column_config.NumberColumn(
                format=f"%.{precision}f",
                label=col.replace('_', ' ').title()
            )
    
    # Display the table
    st.dataframe(
        display_data,
        column_config=column_config,
        hide_index=hide_index,
        height=height,
        use_container_width=use_container_width
    )

def create_distribution_chart(
    data: pd.DataFrame,
    value_column: str,
    title: str = '',
    height: int = 400,
    nbins: int = 20,
    color: str = None,
    show_stats: bool = True
) -> go.Figure:
    """
    Create a histogram showing value distribution with optional statistics.
    
    Args:
        data: DataFrame containing the data
        value_column: Column name for values to analyze
        title: Chart title
        height: Chart height in pixels
        nbins: Number of histogram bins
        color: Color for histogram bars
        show_stats: Whether to show statistics annotations
        
    Returns:
        Plotly figure object
    """
    if color is None:
        color = COLORS['primary']
    
    # Make a copy to avoid modifying the original dataframe
    plot_data = data.copy()
    
    # Check if value column exists
    if value_column not in plot_data.columns or plot_data.empty:
        # Return an empty histogram with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for distribution chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=height, title=title)
        return fig
    
    # Drop NaNs from the value column
    plot_data = plot_data.dropna(subset=[value_column])
    
    # Determine appropriate bin size
    if len(plot_data) > 0:
        values = plot_data[value_column].values
        # Use Freedman-Diaconis rule to determine bin width if we have enough data
        if len(values) > 30:
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                bin_width = 2 * iqr / (len(values) ** (1/3))
                data_range = values.max() - values.min()
                if data_range > 0 and bin_width > 0:
                    nbins = int(min(nbins, np.ceil(data_range / bin_width)))
    
    # Create histogram
    fig = px.histogram(
        plot_data,
        x=value_column,
        nbins=nbins,
        title=title,
        height=height,
        color_discrete_sequence=[color],
        histnorm='',  # Can be 'percent', 'probability', 'density', 'probability density'
        opacity=0.8
    )
    
    # Calculate statistics
    if show_stats and len(plot_data) > 0:
        values = plot_data[value_column].dropna()
        
        if len(values) > 0:
            stats = {
                'Mean': values.mean(),
                'Median': values.median(),
                'Std Dev': values.std(),
                'Min': values.min(),
                'Max': values.max(),
                'Count': len(values)
            }
            
            # Add vertical lines for mean and median
            fig.add_vline(
                x=stats['Mean'],
                line_dash="solid",
                line_color=COLORS['secondary'],
                annotation_text="Mean",
                annotation_position="top right"
            )
            
            fig.add_vline(
                x=stats['Median'],
                line_dash="dash",
                line_color=COLORS['info'],
                annotation_text="Median",
                annotation_position="top left"
            )
            
            # Add KDE curve for smooth distribution visualization
            try:
                from scipy import stats as scipy_stats
                if len(values) > 3:  # Need at least 3 points for KDE
                    kde_x = np.linspace(values.min(), values.max(), 100)
                    kde = scipy_stats.gaussian_kde(values)
                    kde_y = kde(kde_x) * len(values) * (values.max() - values.min()) / nbins
                    
                    fig.add_trace(
                        go.Scatter(
                            x=kde_x,
                            y=kde_y,
                            mode='lines',
                            line=dict(color=COLORS['tertiary'], width=2),
                            name='Density'
                        )
                    )
            except (ImportError, ValueError):
                # Skip KDE if scipy not available or if KDE fails
                pass
            
            # Add stats annotation
            stats_text = "<br>".join([f"{k}: {v:.2f}" for k, v in stats.items() if k != 'Count'])
            stats_text += f"<br>Count: {stats['Count']}"
            
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.98,
                text=stats_text,
                showarrow=False,
                font=dict(size=10),
                align="right",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=COLORS['light'],
                borderwidth=1,
                borderpad=4
            )
    
    # Customize layout
    fig.update_layout(
        xaxis_title=value_column.replace('_', ' ').title(),
        yaxis_title="Frequency",
        bargap=0.1
    )
    
    return fig