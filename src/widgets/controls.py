"""
Dashboard control widgets for user interaction.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime, timedelta


def render_time_range_picker(
    key: str = "time_range",
    default_option: str = "Last hour"
) -> str:
    """
    Render a time range picker dropdown.
    
    Args:
        key: Session state key for the control
        default_option: Default selected option
        
    Returns:
        Selected time range option
    """
    options = [
        "Last 15 minutes",
        "Last hour",
        "Last 3 hours",
        "Last day",
        "Last week",
        "Custom range"
    ]
    
    selected_option = st.selectbox(
        "Time Range",
        options=options,
        index=options.index(default_option) if default_option in options else 1,
        key=key
    )
    
    # If custom range selected, show date/time pickers
    if selected_option == "Custom range":
        col1, col2 = st.columns(2)
        
        # Calculate default start/end times
        now = datetime.now()
        default_start = now - timedelta(hours=24)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start.date(),
                key=f"{key}_start_date"
            )
            start_time = st.time_input(
                "Start Time",
                value=default_start.time(),
                key=f"{key}_start_time"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=now.date(),
                key=f"{key}_end_date"
            )
            end_time = st.time_input(
                "End Time",
                value=now.time(),
                key=f"{key}_end_time"
            )
        
        # Store custom range in session state
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
        
        st.session_state[f"{key}_start"] = start_datetime
        st.session_state[f"{key}_end"] = end_datetime
        
        # Display selected range
        st.write(f"Selected range: {start_datetime.strftime('%Y-%m-%d %H:%M')} to {end_datetime.strftime('%Y-%m-%d %H:%M')}")
    
    return selected_option


def render_refresh_control(
    interval: int = 60,
    key: str = "auto_refresh",
    min_value: int = 0,
    max_value: int = 3600
) -> int:
    """
    Render auto-refresh controls.
    
    Args:
        interval: Default refresh interval in seconds
        key: Session state key for the control
        min_value: Minimum refresh interval
        max_value: Maximum refresh interval
        
    Returns:
        Selected refresh interval in seconds
    """
    st.subheader("Auto Refresh", anchor=False)
    
    # Enable/disable auto refresh
    enable_refresh = st.checkbox(
        "Enable Auto Refresh",
        value=interval > 0,
        key=f"{key}_enabled"
    )
    
    # Refresh interval slider
    if enable_refresh:
        refresh_interval = st.slider(
            "Refresh Every (seconds)",
            min_value=max(min_value, 5),  # Minimum 5 seconds
            max_value=max_value,
            value=max(5, interval),
            step=5,
            key=f"{key}_interval"
        )
    else:
        refresh_interval = 0
    
    # Manual refresh button
    if st.button("Refresh Now", key=f"{key}_manual"):
        st.experimental_rerun()
    
    return refresh_interval


def render_filter_controls(
    filters: Dict[str, List[str]],
    key_prefix: str = "filter"
) -> Dict[str, Any]:
    """
    Render multiple filter controls.
    
    Args:
        filters: Dictionary of filter name to list of options
        key_prefix: Prefix for session state keys
        
    Returns:
        Dictionary of selected filter values
    """
    st.subheader("Filters", anchor=False)
    
    selections = {}
    
    for filter_name, options in filters.items():
        # Add "All" option
        all_options = ["All"] + options
        
        # Create filter control
        selected = st.selectbox(
            filter_name.capitalize(),
            options=all_options,
            key=f"{key_prefix}_{filter_name}"
        )
        
        # Store selection
        selections[filter_name] = None if selected == "All" else selected
    
    return selections


def render_pagination_controls(
    total_items: int,
    items_per_page: int = 10,
    key: str = "pagination"
) -> Dict[str, int]:
    """
    Render pagination controls.
    
    Args:
        total_items: Total number of items
        items_per_page: Number of items per page
        key: Session state key for the control
        
    Returns:
        Dictionary with start and end indices
    """
    # Calculate total pages
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    
    # Initialize current page in session state if not exists
    if f"{key}_page" not in st.session_state:
        st.session_state[f"{key}_page"] = 1
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("Previous", key=f"{key}_prev", disabled=st.session_state[f"{key}_page"] <= 1):
            st.session_state[f"{key}_page"] -= 1
    
    with col2:
        st.write(f"Page {st.session_state[f"{key}_page"]} of {total_pages}")
    
    with col3:
        if st.button("Next", key=f"{key}_next", disabled=st.session_state[f"{key}_page"] >= total_pages):
            st.session_state[f"{key}_page"] += 1
    
    # Calculate start and end indices
    start_idx = (st.session_state[f"{key}_page"] - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    return {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "current_page": st.session_state[f"{key}_page"],
        "total_pages": total_pages
    }


def render_items_per_page_control(
    key: str = "items_per_page",
    default: int = 10,
    options: List[int] = [5, 10, 25, 50, 100]
) -> int:
    """
    Render a control for selecting items per page.
    
    Args:
        key: Session state key for the control
        default: Default number of items per page
        options: Available options for items per page
        
    Returns:
        Selected number of items per page
    """
    return st.selectbox(
        "Items per page",
        options=options,
        index=options.index(default) if default in options else 1,
        key=key
    )


def render_collapsible_section(
    title: str,
    content_function: Callable,
    key: str,
    default_open: bool = False
):
    """
    Render a collapsible section with custom content.
    
    Args:
        title: Section title
        content_function: Function to call to render content
        key: Unique key for this section
        default_open: Whether the section is open by default
    """
    is_open = st.checkbox(
        title,
        value=default_open,
        key=f"collapsible_{key}"
    )
    
    if is_open:
        content_function()


def render_tabs_control(
    tabs: List[str],
    content_functions: List[Callable],
    key: str = "tabs"
):
    """
    Render a tabbed interface with custom content.
    
    Args:
        tabs: List of tab names
        content_functions: List of functions to call for each tab content
        key: Session state key for the control
    """
    # Create tabs
    st_tabs = st.tabs(tabs)
    
    # Render content for each tab
    for i, tab in enumerate(st_tabs):
        with tab:
            content_functions[i]()


def render_search_box(
    placeholder: str = "Search...",
    key: str = "search"
) -> str:
    """
    Render a search box.
    
    Args:
        placeholder: Placeholder text
        key: Session state key for the control
        
    Returns:
        Search query string
    """
    return st.text_input(
        "Search",
        placeholder=placeholder,
        key=key
    )


def render_sort_control(
    options: List[str],
    default_option: str,
    key: str = "sort"
) -> str:
    """
    Render a sort control.
    
    Args:
        options: List of sort options
        default_option: Default sort option
        key: Session state key for the control
        
    Returns:
        Selected sort option
    """
    return st.selectbox(
        "Sort by",
        options=options,
        index=options.index(default_option) if default_option in options else 0,
        key=key
    )


def render_toggle_button(
    label: str,
    key: str,
    default: bool = False
) -> bool:
    """
    Render a toggle button.
    
    Args:
        label: Button label
        key: Session state key for the control
        default: Default state
        
    Returns:
        Current button state (True if toggled on)
    """
    # Initialize state if needed
    if key not in st.session_state:
        st.session_state[key] = default
    
    # Define button colors based on state
    if st.session_state[key]:
        button_label = f"✓ {label}"
        button_type = "primary"
    else:
        button_label = label
        button_type = "secondary"
    
    # Render the button and toggle state when clicked
    if st.button(button_label, type=button_type, key=f"{key}_button"):
        st.session_state[key] = not st.session_state[key]
        # Force a rerun to update the button appearance
        st.experimental_rerun()
    
    return st.session_state[key]


def render_dashboard_controls(
    available_widgets: List[str],
    key_prefix: str = "dashboard"
) -> Dict[str, Any]:
    """
    Render controls for customizing a dashboard.
    
    Args:
        available_widgets: List of available widget names
        key_prefix: Prefix for session state keys
        
    Returns:
        Dictionary of dashboard configuration
    """
    st.subheader("Dashboard Controls", anchor=False)
    
    # Dashboard title
    dashboard_title = st.text_input(
        "Dashboard Title",
        value="Custom Dashboard",
        key=f"{key_prefix}_title"
    )
    
    # Add widget section
    st.subheader("Add Widget", anchor=False)
    
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        widget_type = st.selectbox(
            "Widget Type",
            options=available_widgets,
            key=f"{key_prefix}_widget_type"
        )
    
    with col2:
        widget_size = st.selectbox(
            "Size",
            options=["Small", "Medium", "Large"],
            key=f"{key_prefix}_widget_size"
        )
    
    with col3:
        add_button = st.button("Add Widget", key=f"{key_prefix}_add_widget")
    
    # Return configuration
    return {
        "title": dashboard_title,
        "add_widget": add_button,
        "widget_type": widget_type,
        "widget_size": widget_size.lower()
    }