"""
Metric card widgets for displaying metrics in a consistent way.
"""

import streamlit as st
from typing import Any, Optional, List, Dict, Union


def render_metric_card(
    title: str,
    value: Any,
    description: str = "",
    icon: str = "",
    delta: Optional[Union[int, float]] = None,
    delta_prefix: str = "",
    progress_value: Optional[float] = None,
    color: str = "blue",
    size: str = "medium"
):
    """
    Render a metric card with consistent styling.
    
    Args:
        title: Title of the metric card
        value: Value to display (main focus)
        description: Optional description text
        icon: Optional emoji icon
        delta: Optional delta value for change indication
        delta_prefix: Prefix for delta value (e.g., "+", "-")
        progress_value: Optional progress bar value (0-1)
        color: Card accent color (blue, green, red, yellow, purple)
        size: Card size (small, medium, large)
    """
    # Apply CSS for the metric card
    card_css = f"""
    <style>
    .metric-card {{
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid var(--accent-color, #4e8df5);
    }}
    .metric-card.blue {{ --accent-color: #4e8df5; }}
    .metric-card.green {{ --accent-color: #4CAF50; }}
    .metric-card.red {{ --accent-color: #F44336; }}
    .metric-card.yellow {{ --accent-color: #FFC107; }}
    .metric-card.purple {{ --accent-color: #9C27B0; }}
    
    .metric-card .icon {{
        font-size: 1.5rem;
        margin-right: 0.5rem;
        opacity: 0.8;
    }}
    
    .metric-card .title {{
        color: #555;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }}
    
    .metric-card .value {{
        color: #111;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }}
    
    .metric-card.small .value {{ font-size: 1.2rem; }}
    .metric-card.large .value {{ font-size: 1.8rem; }}
    
    .metric-card .description {{
        color: #777;
        font-size: 0.8rem;
    }}
    
    .metric-card .delta {{
        margin-left: 0.3rem;
        font-size: 0.8rem;
        font-weight: 500;
    }}
    
    .metric-card .delta.positive {{ color: #4CAF50; }}
    .metric-card .delta.negative {{ color: #F44336; }}
    
    .metric-card .progress {{
        margin-top: 0.5rem;
        height: 4px;
        background-color: #f0f0f0;
        border-radius: 2px;
        overflow: hidden;
    }}
    
    .metric-card .progress-bar {{
        height: 100%;
        background-color: var(--accent-color, #4e8df5);
    }}
    </style>
    """
    
    # Start building the HTML
    icon_html = f'<span class="icon">{icon}</span> ' if icon else ''
    title_html = f'<div class="title">{icon_html}{title}</div>'
    
    # Format delta if provided
    delta_html = ''
    if delta is not None:
        delta_class = 'positive' if delta >= 0 else 'negative'
        delta_sign = '+' if delta > 0 else ''
        delta_html = f'<span class="delta {delta_class}">{delta_prefix}{delta_sign}{delta}</span>'
    
    # Format value
    value_html = f'<div class="value">{value}{delta_html}</div>'
    
    # Description
    description_html = f'<div class="description">{description}</div>' if description else ''
    
    # Progress bar
    progress_html = ''
    if progress_value is not None:
        # Ensure progress value is between 0 and 1
        progress = max(0, min(1, progress_value))
        progress_percentage = int(progress * 100)
        progress_html = f'''
        <div class="progress">
            <div class="progress-bar" style="width: {progress_percentage}%;"></div>
        </div>
        '''
    
    # Combine all elements
    html = f'''
    {card_css}
    <div class="metric-card {color} {size}">
        {title_html}
        {value_html}
        {description_html}
        {progress_html}
    </div>
    '''
    
    # Render the HTML
    st.markdown(html, unsafe_allow_html=True)


def render_metric_row(metrics: List[Dict[str, Any]]):
    """
    Render a row of metric cards.
    
    Args:
        metrics: List of dictionaries with metric card parameters
    """
    cols = st.columns(len(metrics))
    
    for i, metric in enumerate(metrics):
        with cols[i]:
            render_metric_card(**metric)


def render_stat_comparison(
    title: str,
    current_value: Any,
    previous_value: Any,
    change_type: str = "percent",
    is_inverted: bool = False,
    label_current: str = "Current",
    label_previous: str = "Previous",
    description: str = "",
    icon: str = ""
):
    """
    Render a metric card with a comparison between current and previous values.
    
    Args:
        title: Title of the comparison
        current_value: Current value
        previous_value: Previous value for comparison
        change_type: Type of change calculation ('percent', 'absolute')
        is_inverted: Whether a decrease is positive (e.g., for error rates)
        label_current: Label for current value
        label_previous: Label for previous value
        description: Optional description text
        icon: Optional emoji icon
    """
    # Calculate change
    if change_type == "percent" and previous_value != 0:
        change = (current_value - previous_value) / previous_value * 100
        change_text = f"{change:.1f}%"
    else:
        change = current_value - previous_value
        change_text = f"{change:+.2f}"
    
    # Determine if change is positive or negative (considering inversion)
    is_positive = (change > 0 and not is_inverted) or (change < 0 and is_inverted)
    change_color = "green" if is_positive else "red"
    
    # Apply CSS for the comparison card
    card_css = f"""
    <style>
    .comparison-card {{
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }}
    
    .comparison-card .title {{
        color: #555;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }}
    
    .comparison-card .icon {{
        font-size: 1.2rem;
        margin-right: 0.5rem;
        opacity: 0.8;
    }}
    
    .comparison-card .values {{
        display: flex;
        align-items: flex-end;
        margin-bottom: 0.5rem;
    }}
    
    .comparison-card .current {{
        color: #111;
        font-size: 1.5rem;
        font-weight: 600;
    }}
    
    .comparison-card .previous {{
        color: #777;
        font-size: 1rem;
        margin-left: 0.5rem;
    }}
    
    .comparison-card .change {{
        font-size: 0.9rem;
        font-weight: 500;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 0.5rem;
    }}
    
    .comparison-card .change.positive {{ 
        background-color: rgba(76, 175, 80, 0.1);
        color: #4CAF50; 
    }}
    
    .comparison-card .change.negative {{ 
        background-color: rgba(244, 67, 54, 0.1);
        color: #F44336; 
    }}
    
    .comparison-card .labels {{
        display: flex;
        font-size: 0.8rem;
        color: #777;
    }}
    
    .comparison-card .label-current {{
        margin-right: 1rem;
    }}
    
    .comparison-card .description {{
        margin-top: 0.5rem;
        color: #777;
        font-size: 0.8rem;
    }}
    </style>
    """
    
    # Build the HTML
    icon_html = f'<span class="icon">{icon}</span>' if icon else ''
    
    html = f'''
    {card_css}
    <div class="comparison-card">
        <div class="title">
            {icon_html} {title}
        </div>
        <div class="values">
            <div class="current">{current_value}</div>
            <div class="previous">{previous_value}</div>
            <div class="change {change_color if change != 0 else ''}">{change_text}</div>
        </div>
        <div class="labels">
            <div class="label-current">{label_current}</div>
            <div class="label-previous">{label_previous}</div>
        </div>
        {f'<div class="description">{description}</div>' if description else ''}
    </div>
    '''
    
    # Render the HTML
    st.markdown(html, unsafe_allow_html=True)
    """
    Metric card widgets for displaying metrics in a consistent way.
    """

import streamlit as st
from typing import Any, Optional, List, Dict, Union

"""
Metric card widgets for displaying metrics in a consistent way.
"""

import streamlit as st
from typing import Any, Optional, List, Dict, Union


def render_metric_card(
    title: str,
    value: Any,
    description: str = "",
    icon: str = "",
    delta: Optional[Union[int, float]] = None,
    delta_prefix: str = "",
    progress_value: Optional[float] = None,
    color: str = "blue",
    size: str = "medium"
):
    """
    Render a metric card with consistent styling.
    
    Args:
        title: Title of the metric card
        value: Value to display (main focus)
        description: Optional description text
        icon: Optional emoji icon
        delta: Optional delta value for change indication
        delta_prefix: Prefix for delta value (e.g., "+", "-")
        progress_value: Optional progress bar value (0-1)
        color: Card accent color (blue, green, red, yellow, purple)
        size: Card size (small, medium, large)
    """
    # Apply CSS for the metric card
    card_css = f"""
    <style>
    .metric-card {{
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid var(--accent-color, #4e8df5);
    }}
    .metric-card.blue {{ --accent-color: #4e8df5; }}
    .metric-card.green {{ --accent-color: #4CAF50; }}
    .metric-card.red {{ --accent-color: #F44336; }}
    .metric-card.yellow {{ --accent-color: #FFC107; }}
    .metric-card.purple {{ --accent-color: #9C27B0; }}
    
    .metric-card .icon {{
        font-size: 1.5rem;
        margin-right: 0.5rem;
        opacity: 0.8;
    }}
    
    .metric-card .title {{
        color: #555;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }}
    
    .metric-card .value {{
        color: #111;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }}
    
    .metric-card.small .value {{ font-size: 1.2rem; }}
    .metric-card.large .value {{ font-size: 1.8rem; }}
    
    .metric-card .description {{
        color: #777;
        font-size: 0.8rem;
    }}
    
    .metric-card .delta {{
        margin-left: 0.3rem;
        font-size: 0.8rem;
        font-weight: 500;
    }}
    
    .metric-card .delta.positive {{ color: #4CAF50; }}
    .metric-card .delta.negative {{ color: #F44336; }}
    
    .metric-card .progress {{
        margin-top: 0.5rem;
        height: 4px;
        background-color: #f0f0f0;
        border-radius: 2px;
        overflow: hidden;
    }}
    
    .metric-card .progress-bar {{
        height: 100%;
        background-color: var(--accent-color, #4e8df5);
    }}
    </style>
    """
    
    # Start building the HTML
    icon_html = f'<span class="icon">{icon}</span> ' if icon else ''
    title_html = f'<div class="title">{icon_html}{title}</div>'
    
    # Format delta if provided
    delta_html = ''
    if delta is not None:
        delta_class = 'positive' if delta >= 0 else 'negative'
        delta_sign = '+' if delta > 0 else ''
        delta_html = f'<span class="delta {delta_class}">{delta_prefix}{delta_sign}{delta}</span>'
    
    # Format value
    value_html = f'<div class="value">{value}{delta_html}</div>'
    
    # Description
    description_html = f'<div class="description">{description}</div>' if description else ''
    
    # Progress bar
    progress_html = ''
    if progress_value is not None:
        # Ensure progress value is between 0 and 1
        progress = max(0, min(1, progress_value))
        progress_percentage = int(progress * 100)
        progress_html = f'''
        <div class="progress">
            <div class="progress-bar" style="width: {progress_percentage}%;"></div>
        </div>
        '''
    
    # Combine all elements
    html = f'''
    {card_css}
    <div class="metric-card {color} {size}">
        {title_html}
        {value_html}
        {description_html}
        {progress_html}
    </div>
    '''
    
    # Render the HTML
    st.markdown(html, unsafe_allow_html=True)


def render_metric_row(metrics: List[Dict[str, Any]]):
    """
    Render a row of metric cards.
    
    Args:
        metrics: List of dictionaries with metric card parameters
    """
    cols = st.columns(len(metrics))
    
    for i, metric in enumerate(metrics):
        with cols[i]:
            render_metric_card(**metric)


def render_stat_comparison(
    title: str,
    current_value: Any,
    previous_value: Any,
    change_type: str = "percent",
    is_inverted: bool = False,
    label_current: str = "Current",
    label_previous: str = "Previous",
    description: str = "",
    icon: str = ""
):
    """
    Render a metric card with a comparison between current and previous values.
    
    Args:
        title: Title of the comparison
        current_value: Current value
        previous_value: Previous value for comparison
        change_type: Type of change calculation ('percent', 'absolute')
        is_inverted: Whether a decrease is positive (e.g., for error rates)
        label_current: Label for current value
        label_previous: Label for previous value
        description: Optional description text
        icon: Optional emoji icon
    """
    # Calculate change
    if change_type == "percent" and previous_value != 0:
        change = (current_value - previous_value) / previous_value * 100
        change_text = f"{change:.1f}%"
    else:
        change = current_value - previous_value
        change_text = f"{change:+.2f}"
    
    # Determine if change is positive or negative (considering inversion)
    is_positive = (change > 0 and not is_inverted) or (change < 0 and is_inverted)
    change_color = "green" if is_positive else "red"
    
    # Apply CSS for the comparison card
    card_css = f"""
    <style>
    .comparison-card {{
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }}
    
    .comparison-card .title {{
        color: #555;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }}
    
    .comparison-card .icon {{
        font-size: 1.2rem;
        margin-right: 0.5rem;
        opacity: 0.8;
    }}
    
    .comparison-card .values {{
        display: flex;
        align-items: flex-end;
        margin-bottom: 0.5rem;
    }}
    
    .comparison-card .current {{
        color: #111;
        font-size: 1.5rem;
        font-weight: 600;
    }}
    
    .comparison-card .previous {{
        color: #777;
        font-size: 1rem;
        margin-left: 0.5rem;
    }}
    
    .comparison-card .change {{
        font-size: 0.9rem;
        font-weight: 500;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 0.5rem;
    }}
    
    .comparison-card .change.positive {{ 
        background-color: rgba(76, 175, 80, 0.1);
        color: #4CAF50; 
    }}
    
    .comparison-card .change.negative {{ 
        background-color: rgba(244, 67, 54, 0.1);
        color: #F44336; 
    }}
    
    .comparison-card .labels {{
        display: flex;
        font-size: 0.8rem;
        color: #777;
    }}
    
    .comparison-card .label-current {{
        margin-right: 1rem;
    }}
    
    .comparison-card .description {{
        margin-top: 0.5rem;
        color: #777;
        font-size: 0.8rem;
    }}
    </style>
    """
    
    # Build the HTML
    icon_html = f'<span class="icon">{icon}</span>' if icon else ''
    
    html = f'''
    {card_css}
    <div class="comparison-card">
        <div class="title">
            {icon_html} {title}
        </div>
        <div class="values">
            <div class="current">{current_value}</div>
            <div class="previous">{previous_value}</div>
            <div class="change {change_color if change != 0 else ''}">{change_text}</div>
        </div>
        <div class="labels">
            <div class="label-current">{label_current}</div>
            <div class="label-previous">{label_previous}</div>
        </div>
        {f'<div class="description">{description}</div>' if description else ''}
    </div>
    '''
    
    # Render the HTML
    st.markdown(html, unsafe_allow_html=True)