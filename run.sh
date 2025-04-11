#!/bin/bash

# OmniPulse Dashboard: Run Script
# This script starts the OmniPulse dashboard

set -e  # Exit on error

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}   Starting OmniPulse: Enterprise Monitoring Hub     ${NC}"
echo -e "${BLUE}======================================================${NC}"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${BLUE}No virtual environment found.${NC}"
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python -m venv venv
    source venv/bin/activate
    
    echo -e "${GREEN}Installing dependencies...${NC}"
    pip install -r requirements.txt
fi

# Check if Streamlit is installed
if ! pip show streamlit > /dev/null; then
    echo -e "${BLUE}Streamlit not found. Installing...${NC}"
    pip install streamlit
fi
sed -i '' 's/from distutils import spawn/import shutil/; s/spawn.find_executable/shutil.which/g' venv/lib/python3.12/site-packages/GPUtil/GPUtil.py

# Start dashboard
echo -e "${GREEN}Starting OmniPulse dashboard...${NC}"

streamlit run src/app.py "$@"