#!/bin/bash


set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_requirements() {
    echo -e "${YELLOW}Checking system requirements...${NC}"
    
    if ! python3 -c 'import sys; exit(1) if sys.version_info < (3,8) else exit(0)'; then
        echo -e "${RED}Error: Python 3.8 or later is required${NC}"
        exit 1
    fi

    
    for cmd in pip3 virtualenv; do
        if ! command -v $cmd &> /dev/null; then
            echo -e "${RED}Error: $cmd not found. Please install it first.${NC}"
            exit 1
        fi
    done
}

#!/bin/bash

# Adaptive MVGNAS Installation Script
# Author: Project Team
# ------------------------------------

set -e  # Exit immediately on error

# Colors for status messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_requirements() {
    echo -e "${YELLOW}Checking system requirements...${NC}"
    
    # Verify Python version
    if ! python3 -c 'import sys; exit(1) if sys.version_info < (3,8) else exit(0)'; then
        echo -e "${RED}Error: Python 3.8 or later is required${NC}"
        exit 1
    fi

    # Check for essential build tools
    for cmd in pip3 virtualenv; do
        if ! command -v $cmd &> /dev/null; then
            echo -e "${RED}Error: $cmd not found. Please install it first.${NC}"
            exit 1
        fi
    done
}

# Main installation process
install_system() {
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv mvgnas_env || {
        echo -e "${RED}Failed to create virtual environment${NC}";
        exit 1;
    }

    echo -e "${YELLOW}Activating environment...${NC}"
    source mvgnas_env/bin/activate || {
        echo -e "${RED}Failed to activate virtual environment${NC}";
        exit 1;
    }

    echo -e "${YELLOW}Upgrading base packages...${NC}"
    pip3 install --upgrade pip wheel setuptools || {
        echo -e "${RED}Failed to upgrade base packages${NC}";
        exit 1;
    }

    echo -e "${YELLOW}Installing requirements...${NC}"
    pip3 install -r requirements.txt || {
        echo -e "${RED}Failed to install Python dependencies${NC}";
        exit 1;
    }

    echo -e "${YELLOW}Downloading NLP models...${NC}"
    python3 -m spacy download en_core_web_lg || {
        echo -e "${RED}Failed to download spaCy model${NC}";
        exit 1;
    }

    echo -e "${GREEN}Installation completed successfully!${NC}"
}

# Post-installation check
verify_installation() {
    echo -e "${YELLOW}Verifying installation...${NC}"
    
    # Check critical packages
    for pkg in "torch" "torch_geometric" "spacy"; do
        if ! python3 -c "import $pkg" 2>/dev/null; then
            echo -e "${RED}Error: $pkg not installed correctly${NC}"
            exit 1
        fi
    done
    
    echo -e "${GREEN}All components verified successfully!${NC}"
}

main() {
    check_requirements
    install_system
    verify_installation
    
    echo -e "\n${YELLOW}Starting Adaptive MVGNAS system...${NC}"
    python3 main.py "$@"  # Pass any command-line arguments to main.py
    
    deactivate
    echo -e "${GREEN}Execution completed. Virtual environment deactivated.${NC}"
}


main "$@"# Main installation process
install_system() {
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv mvgnas_env || {
        echo -e "${RED}Failed to create virtual environment${NC}";
        exit 1;
    }

    echo -e "${YELLOW}Activating environment...${NC}"
    source mvgnas_env/bin/activate || {
        echo -e "${RED}Failed to activate virtual environment${NC}";
        exit 1;
    }

    echo -e "${YELLOW}Upgrading base packages...${NC}"
    pip3 install --upgrade pip wheel setuptools || {
        echo -e "${RED}Failed to upgrade base packages${NC}";
        exit 1;
    }

    echo -e "${YELLOW}Installing requirements...${NC}"
    pip3 install -r requirements.txt || {
        echo -e "${RED}Failed to install Python dependencies${NC}";
        exit 1;
    }

    echo -e "${YELLOW}Downloading NLP models...${NC}"
    python3 -m spacy download en_core_web_lg || {
        echo -e "${RED}Failed to download spaCy model${NC}";
        exit 1;
    }

    echo -e "${GREEN}Installation completed successfully!${NC}"
}


verify_installation() {
    echo -e "${YELLOW}Verifying installation...${NC}"
    

    for pkg in "torch" "torch_geometric" "spacy"; do
        if ! python3 -c "import $pkg" 2>/dev/null; then
            echo -e "${RED}Error: $pkg not installed correctly${NC}"
            exit 1
        fi
    done
    
    echo -e "${GREEN}All components verified successfully!${NC}"
}


main() {
    check_requirements
    install_system
    verify_installation
    
    echo -e "\n${YELLOW}Starting Adaptive MVGNAS system...${NC}"
    python3 main.py "$@"
    
    deactivate
    echo -e "${GREEN}Execution completed. Virtual environment deactivated.${NC}"
}
main "$@"