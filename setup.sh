#!/bin/bash

# OmniPulse: Enterprise LLM & System Monitoring Hub
# Setup Script

set -e  # Exit on error

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}  Setting up OmniPulse: Enterprise LLM & System Monitor${NC}"
echo -e "${BLUE}======================================================${NC}"

# Create project directory
PROJECT_DIR="omnipulse"
echo -e "${GREEN}Creating project directory structure...${NC}"

# Create main directories
mkdir -p $PROJECT_DIR
mkdir -p $PROJECT_DIR/src/{config,core,dashboards,monitors,utils,widgets}
mkdir -p $PROJECT_DIR/tests

# Create empty Python files with basic imports
create_python_file() {
    local file_path=$1
    local file_name=$(basename "$file_path")
    local module_name="${file_name%.py}"
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$file_path")"
    
    if [[ "$file_name" == "__init__.py" ]]; then
        touch "$file_path"
    else
        echo "\"\"\"
$module_name module for OmniPulse monitoring dashboard.
\"\"\"" > "$file_path"
    fi
    
    echo -e "${GREEN}Created${NC} $file_path"
}

# Create __init__.py files in all directories
find $PROJECT_DIR -type d -exec touch {}"/__init__.py" \;

# Create source files
create_python_file "$PROJECT_DIR/src/app.py"
create_python_file "$PROJECT_DIR/src/config/settings.py"
create_python_file "$PROJECT_DIR/src/config/defaults.py"
create_python_file "$PROJECT_DIR/src/config/themes.py"
create_python_file "$PROJECT_DIR/src/core/auth.py"
create_python_file "$PROJECT_DIR/src/core/cache.py"
create_python_file "$PROJECT_DIR/src/core/database.py"
create_python_file "$PROJECT_DIR/src/dashboards/ollama_dashboard.py"
create_python_file "$PROJECT_DIR/src/dashboards/system_dashboard.py"
create_python_file "$PROJECT_DIR/src/dashboards/python_dashboard.py"
create_python_file "$PROJECT_DIR/src/dashboards/custom_dashboard.py"
create_python_file "$PROJECT_DIR/src/monitors/ollama_monitor.py"
create_python_file "$PROJECT_DIR/src/monitors/system_monitor.py"
create_python_file "$PROJECT_DIR/src/monitors/python_monitor.py"
create_python_file "$PROJECT_DIR/src/monitors/custom_monitor.py"
create_python_file "$PROJECT_DIR/src/utils/visualization.py"
create_python_file "$PROJECT_DIR/src/utils/alerts.py"
create_python_file "$PROJECT_DIR/src/utils/helpers.py"
create_python_file "$PROJECT_DIR/src/widgets/metric_cards.py"
create_python_file "$PROJECT_DIR/src/widgets/charts.py"
create_python_file "$PROJECT_DIR/src/widgets/controls.py"

# Create test files
create_python_file "$PROJECT_DIR/tests/test_monitors.py"
create_python_file "$PROJECT_DIR/tests/test_dashboards.py"
create_python_file "$PROJECT_DIR/tests/test_utils.py"

# Create README.md
cat > $PROJECT_DIR/README.md << 'EOF'
# OmniPulse: Enterprise LLM & System Monitoring Hub

A comprehensive real-time monitoring dashboard for Ollama LLM services, Python applications, and system telemetry.

## Features

- Real-time monitoring of Ollama LLM usage, performance, and health
- System telemetry tracking (CPU, memory, disk usage, network)
- Python application performance monitoring
- Customizable dashboards and visualization widgets
- Alert configuration and notification system
- Dark/light theme support
- Responsive design for desktop and mobile viewing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/omnipulse.git
cd omnipulse

# Set up the environment
./setup.sh

# Start the dashboard
./run.sh
```

## Configuration

Edit the `.env` file to configure your monitoring settings:

```
OLLAMA_API_URL=http://localhost:11434
REFRESH_INTERVAL=5
ENABLE_AUTHENTICATION=false
```

## Custom Dashboards

OmniPulse supports custom dashboard creation through the UI or by editing JSON configuration files located in `config/dashboards/`.

## License

MIT
EOF

# Create requirements.txt
cat > $PROJECT_DIR/requirements.txt << 'EOF'
streamlit>=1.27.0
plotly>=5.16.0
pandas>=2.0.0
psutil>=5.9.0
requests>=2.28.0
python-dotenv>=1.0.0
watchdog>=3.0.0
streamlit-extras>=0.3.0
streamlit-autorefresh>=0.1.0
APScheduler>=3.10.0
httpx>=0.24.0
pydantic>=2.0.0
streamlit-authenticator>=0.2.2
EOF

# Create .env.example
cat > $PROJECT_DIR/.env.example << 'EOF'
# OmniPulse Configuration

# Ollama Settings
OLLAMA_API_URL=http://localhost:11434

# Monitoring Settings
REFRESH_INTERVAL=5  # seconds
HISTORY_RETENTION=24  # hours

# System Monitor Settings
ENABLE_CPU_MONITORING=true
ENABLE_MEMORY_MONITORING=true
ENABLE_DISK_MONITORING=true
ENABLE_NETWORK_MONITORING=true

# Authentication (Optional)
ENABLE_AUTHENTICATION=false
#AUTH_USERNAME=admin
#AUTH_PASSWORD_HASH=  # Generate with password_hasher.py

# Alerts
ENABLE_ALERTS=true
ALERT_EMAIL=  # Optional email for notifications
EOF

# Create run.sh
cat > $PROJECT_DIR/run.sh << 'EOF'
#!/bin/bash

cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || echo "No virtual environment found, using system Python"
streamlit run src/app.py "$@"
EOF
chmod +x $PROJECT_DIR/run.sh

echo -e "${GREEN}Creating initial Streamlit app...${NC}"

# Create initial app.py with basic content
cat > $PROJECT_DIR/src/app.py << 'EOF'
"""
OmniPulse: Enterprise LLM & System Monitoring Hub
Main Streamlit application entry point
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import time
from datetime import datetime
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from src.config import settings
from src.dashboards import (
    ollama_dashboard,
    system_dashboard,
    python_dashboard,
    custom_dashboard
)
from src.utils import helpers, alerts

# Set page configuration
st.set_page_config(
    page_title="OmniPulse Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom theme
def apply_custom_styles():
    """Apply custom CSS styling to the dashboard"""
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp header {
        background-color: rgba(255, 255, 255, 0.95);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
    }
    .metric-card {
        background-color: white;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# Auto refresh dashboard (configurable in settings)
refresh_interval = settings.get_refresh_interval()
if refresh_interval > 0:
    st_autorefresh(interval=refresh_interval * 1000, key="data_refresh")

# Dashboard header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🚀 OmniPulse Dashboard")
    st.caption("Enterprise LLM & System Monitoring Hub")
with col2:
    st.text(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("Refresh Now"):
        st.experimental_rerun()

# Main navigation
tabs = st.tabs(["Ollama LLM", "System", "Python Apps", "Custom"])

# Ollama LLM Dashboard
with tabs[0]:
    st.header("Ollama LLM Monitoring")
    # This will be implemented in the ollama_dashboard module
    st.info("Ollama metrics will appear here. Configure Ollama API endpoint in settings.")
    
# System Dashboard    
with tabs[1]:
    st.header("System Monitoring")
    # This will be implemented in the system_dashboard module
    st.info("System metrics will appear here, including CPU, memory, disk and network stats.")

# Python Apps Dashboard
with tabs[2]:
    st.header("Python Applications")
    # This will be implemented in the python_dashboard module
    st.info("Python application metrics will appear here.")

# Custom Dashboard
with tabs[3]:
    st.header("Custom Dashboard")
    # This will be implemented in the custom_dashboard module
    st.info("Build your custom monitoring dashboard here by adding widgets.")

# Sidebar for configuration and filtering
with st.sidebar:
    st.title("OmniPulse")
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/master/examples/data/logo.png", width=100)
    
    st.subheader("Settings")
    time_range = st.selectbox("Time Range", ["Last 15 minutes", "Last hour", "Last 3 hours", "Last day"])
    
    st.subheader("Filters")
    # These filters will be populated based on available data
    
    st.subheader("About")
    st.info("OmniPulse v0.1.0\nMonitoring Ollama and system resources in real-time.")
    
    if st.button("View Documentation"):
        st.markdown("[OmniPulse Documentation](https://github.com/yourusername/omnipulse)")

# Footer
st.markdown("---")
st.caption("OmniPulse - Enterprise Monitoring Dashboard © 2025")
EOF

# Make the bash scripts executable
chmod +x $PROJECT_DIR/setup.sh
chmod +x $PROJECT_DIR/run.sh

# Display completion message
echo -e "${GREEN}OmniPulse project structure has been created!${NC}"
echo -e "${BLUE}Next steps:${NC}"
echo -e "1. Create a virtual environment: ${GREEN}cd $PROJECT_DIR && python -m venv venv${NC}"
echo -e "2. Activate the virtual environment: ${GREEN}source venv/bin/activate${NC}"
echo -e "3. Install dependencies: ${GREEN}pip install -r requirements.txt${NC}"
echo -e "4. Start the dashboard: ${GREEN}./run.sh${NC}"
echo ""
echo -e "${BLUE}Happy monitoring!${NC}"