"""
Theme configuration for OmniPulse dashboard.
"""

import streamlit as st
from typing import Dict, Any, Optional

# Theme definitions
THEMES = {
    "light": {
        "background_color": "#f5f7fa",
        "secondary_background_color": "#ffffff",
        "text_color": "#333333",
        "secondary_text_color": "#666666",
        "accent_color": "#4e8df5",
        "success_color": "#4CAF50",
        "warning_color": "#FFC107",
        "error_color": "#F44336",
        "info_color": "#9C27B0",
        "border_color": "#e0e0e0",
        "shadow": "0 1px 3px rgba(0, 0, 0, 0.1)",
        "card_background": "#ffffff",
        "chart_colors": [
            "#4e8df5", "#4CAF50", "#F44336", "#FFC107", "#9C27B0",
            "#00BCD4", "#FF9800", "#607D8B", "#8BC34A", "#E91E63"
        ],
        "font_family": "sans-serif"
    },
    "dark": {
        "background_color": "#121212",
        "secondary_background_color": "#1e1e1e",
        "text_color": "#e0e0e0",
        "secondary_text_color": "#b0b0b0",
        "accent_color": "#4e8df5",
        "success_color": "#4CAF50",
        "warning_color": "#FFC107",
        "error_color": "#F44336", 
        "info_color": "#9C27B0",
        "border_color": "#333333",
        "shadow": "0 1px 3px rgba(0, 0, 0, 0.3)",
        "card_background": "#1e1e1e",
        "chart_colors": [
            "#4e8df5", "#4CAF50", "#F44336", "#FFC107", "#9C27B0",
            "#00BCD4", "#FF9800", "#607D8B", "#8BC34A", "#E91E63"
        ],
        "font_family": "sans-serif"
    },
    "blue": {
        "background_color": "#f0f5ff",
        "secondary_background_color": "#ffffff",
        "text_color": "#333333",
        "secondary_text_color": "#666666",
        "accent_color": "#1e88e5",
        "success_color": "#43a047",
        "warning_color": "#ffb300",
        "error_color": "#e53935",
        "info_color": "#8e24aa",
        "border_color": "#d0e1ff",
        "shadow": "0 1px 3px rgba(0, 0, 0, 0.1)",
        "card_background": "#ffffff",
        "chart_colors": [
            "#1e88e5", "#43a047", "#e53935", "#ffb300", "#8e24aa",
            "#00acc1", "#fb8c00", "#546e7a", "#7cb342", "#d81b60"
        ],
        "font_family": "sans-serif"
    },
    "green": {
        "background_color": "#f0fff5",
        "secondary_background_color": "#ffffff",
        "text_color": "#333333",
        "secondary_text_color": "#666666",
        "accent_color": "#43a047",
        "success_color": "#43a047",
        "warning_color": "#ffb300",
        "error_color": "#e53935",
        "info_color": "#1e88e5",
        "border_color": "#d0ffd0",
        "shadow": "0 1px 3px rgba(0, 0, 0, 0.1)",
        "card_background": "#ffffff",
        "chart_colors": [
            "#43a047", "#1e88e5", "#e53935", "#ffb300", "#8e24aa",
            "#00acc1", "#fb8c00", "#546e7a", "#7cb342", "#d81b60"
        ],
        "font_family": "sans-serif"
    },
    "high_contrast": {
        "background_color": "#ffffff",
        "secondary_background_color": "#f0f0f0",
        "text_color": "#000000",
        "secondary_text_color": "#333333",
        "accent_color": "#0000ff",
        "success_color": "#008000",
        "warning_color": "#ff8000",
        "error_color": "#ff0000", 
        "info_color": "#800080",
        "border_color": "#000000",
        "shadow": "0 1px 3px rgba(0, 0, 0, 0.3)",
        "card_background": "#ffffff",
        "chart_colors": [
            "#0000ff", "#008000", "#ff0000", "#ff8000", "#800080",
            "#008080", "#ff8000", "#404040", "#808000", "#800000"
        ],
        "font_family": "sans-serif"
    },
    "terminal": {
        "background_color": "#0d1117",
        "secondary_background_color": "#161b22",
        "text_color": "#c9d1d9",
        "secondary_text_color": "#8b949e",
        "accent_color": "#58a6ff",
        "success_color": "#3fb950",
        "warning_color": "#d29922",
        "error_color": "#f85149",
        "info_color": "#a371f7",
        "border_color": "#30363d",
        "shadow": "0 1px 3px rgba(0, 0, 0, 0.4)",
        "card_background": "#161b22",
        "chart_colors": [
            "#58a6ff", "#3fb950", "#f85149", "#d29922", "#a371f7",
            "#2ea043", "#f0883e", "#8b949e", "#56d364", "#ff7b72"
        ],
        "font_family": "monospace"
    }
}


def apply_theme(theme_name: str = "light", custom_theme: Optional[Dict[str, Any]] = None):
    """
    Apply the selected theme to the Streamlit app.
    
    Args:
        theme_name: Name of the predefined theme to use
        custom_theme: Optional custom theme overrides
    """
    # Get the theme definition
    if theme_name in THEMES:
        theme = THEMES[theme_name].copy()
    else:
        theme = THEMES["light"].copy()
    
    # Apply custom theme overrides if provided
    if custom_theme:
        theme.update(custom_theme)
    
    # Apply the CSS
    st.markdown(f"""
    <style>
    :root {{
        --background-color: {theme["background_color"]};
        --secondary-background-color: {theme["secondary_background_color"]};
        --text-color: {theme["text_color"]};
        --secondary-text-color: {theme["secondary_text_color"]};
        --accent-color: {theme["accent_color"]};
        --success-color: {theme["success_color"]};
        --warning-color: {theme["warning_color"]};
        --error-color: {theme["error_color"]};
        --info-color: {theme["info_color"]};
        --border-color: {theme["border_color"]};
        --shadow: {theme["shadow"]};
        --card-background: {theme["card_background"]};
        --font-family: {theme["font_family"]};
    }}
    
    /* Base styles */
    .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    
    .stApp {{
        background-color: var(--background-color);
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-color);
        font-family: var(--font-family);
    }}
    
    p, ol, ul, dl {{
        color: var(--text-color);
        font-family: var(--font-family);
    }}
    
    /* Metric cards */
    .metric-card {{
        background-color: var(--card-background);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: var(--shadow);
        border-left: 4px solid var(--accent-color);
        margin-bottom: 1rem;
    }}
    
    .metric-card .title {{
        color: var(--secondary-text-color);
        font-family: var(--font-family);
    }}
    
    .metric-card .value {{
        color: var(--text-color);
        font-family: var(--font-family);
    }}
    
    .metric-card .description {{
        color: var(--secondary-text-color);
        font-family: var(--font-family);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        font-family: var(--font-family);
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background-color: var(--secondary-background-color);
        border-radius: 0px 4px 4px 4px;
        padding: 1rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-color);
        font-family: var(--font-family);
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        background-color: var(--accent-color);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-family: var(--font-family);
    }
    
    div.stButton > button:hover {
        background-color: {theme.get("accent_hover_color", theme["accent_color"])};
        color: white;
        border: none;
    }
    
    /* Links */
    [data-testid="StyledLinkIconContainer"] {
        color: var(--secondary-text-color);
    }
    
    a {
        color: var(--accent-color);
    }
    
    /* Headers */
    h1 {
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    h3 {
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Dashboard header and footer */
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 1rem;
    }
    
    .dashboard-footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-color);
        color: var(--secondary-text-color);
        font-size: 0.8rem;
        text-align: center;
    }
    
    /* Inputs */
    [data-testid="stTextInput"] > div > div > input {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border-color: var(--border-color);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-color);
    }
    
    /* Selects */
    [data-testid="stSelectbox"] {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
    }
    
    /* Dataframes and tables */
    [data-testid="stDataFrame"] {
        background-color: var(--card-background);
        border-radius: 8px;
        overflow: hidden;
    }
    
    [data-testid="stDataFrame"] th {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        font-weight: 600;
    }
    
    [data-testid="stDataFrame"] td {
        color: var(--text-color);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border-radius: 4px;
    }
    
    .streamlit-expanderContent {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border-bottom-left-radius: 4px;
        border-bottom-right-radius: 4px;
        padding: 1rem;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: var(--secondary-background-color);
        border-radius: 8px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: var(--accent-color);
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background-color: var(--card-background);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: var(--shadow);
    }
    
    [data-testid="stMetric"] label {
        color: var(--secondary-text-color);
    }
    
    [data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--text-color);
    }
    
    [data-testid="stMetricDelta"] {
        color: var(--success-color);
    }
    
    [data-testid="stMetricDelta"].negative {
        color: var(--error-color);
    }
    
    /* Info, warning, error, success styles */
    .stAlert {
        border-radius: 8px;
    }
    
    [data-baseweb="notification"] {
        border-radius: 8px;
    }
    
    .element-container [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
        padding: 0.5rem 0;
    }
    
    /* Adjust specific alert types */
    [data-icon="alert-circle-sharp"] + div {
        background-color: rgba(var(--info-color-rgb), 0.1);
        color: var(--info-color);
    }
    
    [data-icon="alert-sharp"] + div {
        background-color: rgba(var(--warning-color-rgb), 0.1);
        color: var(--warning-color);
    }
    
    [data-icon="close-sharp"] + div {
        background-color: rgba(var(--error-color-rgb), 0.1);
        color: var(--error-color);
    }
    
    [data-icon="check-sharp"] + div {
        background-color: rgba(var(--success-color-rgb), 0.1);
        color: var(--success-color);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Store the theme in session state for other components to use
    st.session_state.current_theme = theme
    st.session_state.current_theme_name = theme_name


def get_theme_options() -> Dict[str, str]:
    """
    Get available theme options with display names.
    
    Returns:
        Dictionary of theme key -> display name
    """
    return {
        "light": "Light Mode",
        "dark": "Dark Mode",
        "blue": "Blue Theme",
        "green": "Green Theme",
        "high_contrast": "High Contrast",
        "terminal": "Terminal Theme"
    }


def get_current_theme() -> Dict[str, Any]:
    """
    Get the current theme configuration.
    
    Returns:
        Dictionary with theme configuration
    """
    if "current_theme" in st.session_state:
        return st.session_state.current_theme
    else:
        # Default to light theme
        return THEMES["light"]


def render_theme_selector():
    """
    Render a theme selector widget.
    """
    theme_options = get_theme_options()
    
    # Get current theme name
    current_theme_name = st.session_state.get("current_theme_name", "light")
    
    # Create theme selector
    selected_theme = st.selectbox(
        "Dashboard Theme",
        options=list(theme_options.keys()),
        format_func=lambda x: theme_options[x],
        index=list(theme_options.keys()).index(current_theme_name)
    )
    
    # Apply theme if changed
    if selected_theme != current_theme_name:
        apply_theme(selected_theme)
        st.success(f"Applied theme: {theme_options[selected_theme]}")
        st.experimental_rerun()


def add_custom_colors(theme: Dict[str, Any]):
    """
    Add RGB variants of colors to a theme for use in CSS with transparency.
    
    Args:
        theme: Theme dictionary to modify
    """
    # Helper function to convert hex to RGB
    def hex_to_rgb(hex_color: str) -> str:
        hex_color = hex_color.lstrip('#')
        return f"{int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}"
    
    # Add RGB variants for each color
    for key, value in list(theme.items()):
        if key.endswith('_color') and value.startswith('#'):
            theme[f"{key}_rgb"] = hex_to_rgb(value)
    
    # Add hover variants for buttons
    if "accent_color" in theme and "accent_hover_color" not in theme:
        # Simple darkening for hover state
        hex_color = theme["accent_color"].lstrip('#')
        r = max(0, int(hex_color[0:2], 16) - 20)
        g = max(0, int(hex_color[2:4], 16) - 20)
        b = max(0, int(hex_color[4:6], 16) - 20)
        theme["accent_hover_color"] = f"#{r:02x}{g:02x}{b:02x}"
    
    return theme