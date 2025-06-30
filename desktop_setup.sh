#!/bin/bash
# Desktop Implementation Setup Script
# Implements all laptop developments on desktop according to DESKTOP_IMPLEMENTATION.mf

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_FILE="${SCRIPT_DIR}/DESKTOP_IMPLEMENTATION.mf"
LOG_FILE="${SCRIPT_DIR}/desktop_setup.log"

# Desktop-specific paths (modify as needed)
DESKTOP_USER="${USER}"
DESKTOP_HOME="${HOME}"
DESKTOP_WORKSPACE="${HOME}/Desktop/TradeKnowledge"
CLAUDE_CONFIG_DIR="${HOME}/.claude"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}TradeKnowledge Desktop Implementation Setup${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# Function to log and print
log_and_print() {
    echo -e "$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to generate checksums
generate_checksums() {
    local file_list="$1"
    local checksum_file="${SCRIPT_DIR}/desktop_checksums.txt"
    
    log_and_print "${YELLOW}📝 Generating file checksums...${NC}"
    
    # Clear existing checksums
    > "$checksum_file"
    
    # Generate checksums for all files in manifest
    while IFS= read -r line; do
        if [[ "$line" =~ ^\./ ]] && [[ "$line" =~ -> ]]; then
            local src_file=$(echo "$line" | cut -d' ' -f1)
            if [[ -f "$src_file" ]]; then
                local checksum=$(md5sum "$src_file" | cut -d' ' -f1)
                echo "$checksum  $src_file" >> "$checksum_file"
            fi
        fi
    done < "$MANIFEST_FILE"
    
    log_and_print "${GREEN}✅ Checksums generated: $checksum_file${NC}"
}

# Function to verify prerequisites
verify_prerequisites() {
    log_and_print "${YELLOW}🔍 Verifying prerequisites...${NC}"
    
    # Check Python version
    if command_exists python3; then
        local python_version=$(python3 --version | cut -d' ' -f2)
        log_and_print "${GREEN}✅ Python version: $python_version${NC}"
    else
        log_and_print "${RED}❌ Python 3 not found${NC}"
        exit 1
    fi
    
    # Check Git version
    if command_exists git; then
        local git_version=$(git --version | cut -d' ' -f3)
        log_and_print "${GREEN}✅ Git version: $git_version${NC}"
    else
        log_and_print "${RED}❌ Git not found${NC}"
        exit 1
    fi
    
    # Check disk space
    local available_space=$(df -h "$HOME" | awk 'NR==2 {print $4}')
    log_and_print "${GREEN}✅ Available disk space: $available_space${NC}"
    
    # Check if we're in TradeKnowledge directory
    if [[ ! -f "$MANIFEST_FILE" ]]; then
        log_and_print "${RED}❌ Manifest file not found. Run this script from TradeKnowledge directory.${NC}"
        exit 1
    fi
    
    log_and_print "${GREEN}✅ All prerequisites verified${NC}"
}

# Function to backup existing setup
backup_existing() {
    log_and_print "${YELLOW}💾 Creating backup of existing setup...${NC}"
    
    if [[ -d "$CLAUDE_CONFIG_DIR" ]]; then
        cp -r "$CLAUDE_CONFIG_DIR" "${CLAUDE_CONFIG_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
        log_and_print "${GREEN}✅ Claude config backed up${NC}"
    fi
    
    if [[ -d "$DESKTOP_WORKSPACE" ]]; then
        cp -r "$DESKTOP_WORKSPACE" "${DESKTOP_WORKSPACE}.backup.$(date +%Y%m%d_%H%M%S)"
        log_and_print "${GREEN}✅ Workspace backed up${NC}"
    fi
}

# Function to setup directories
setup_directories() {
    log_and_print "${YELLOW}📁 Setting up directories...${NC}"
    
    # Create required directories
    local directories=(
        "$CLAUDE_CONFIG_DIR"
        "$CLAUDE_CONFIG_DIR/logs"
        "$DESKTOP_WORKSPACE/src/core"
        "$DESKTOP_WORKSPACE/src/integrations"
        "$DESKTOP_WORKSPACE/fastapi_auth"
        "$DESKTOP_WORKSPACE/agents/executor"
        "$DESKTOP_WORKSPACE/agents/mastermind"
        "$DESKTOP_WORKSPACE/agents/researcher"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log_and_print "${GREEN}✅ Created: $dir${NC}"
    done
}

# Function to update paths in files
update_paths() {
    log_and_print "${YELLOW}🔧 Updating paths for desktop environment...${NC}"
    
    # Files that need path updates
    local files_to_update=(
        ".env.desktop"
        "${CLAUDE_CONFIG_DIR}/startup_services.sh"
        "src/integrations/superclaude_sync.py"
    )
    
    for file in "${files_to_update[@]}"; do
        if [[ -f "$file" ]]; then
            # Update user paths
            sed -i "s|/home/scottschweizer|${DESKTOP_HOME}|g" "$file"
            # Update workspace paths  
            sed -i "s|~/TradeKnowledge|${DESKTOP_WORKSPACE}|g" "$file"
            # Update ports (8765 -> 8766)
            sed -i "s|8765|8766|g" "$file"
            log_and_print "${GREEN}✅ Updated paths in: $file${NC}"
        fi
    done
}

# Function to set permissions
set_permissions() {
    log_and_print "${YELLOW}🔐 Setting file permissions...${NC}"
    
    local executable_files=(
        "${CLAUDE_CONFIG_DIR}/startup_services.sh"
        "src/core/llmlingua_service_runner.py"
        "src/integrations/superclaude_sync.py"
        "desktop_setup.sh"
    )
    
    for file in "${executable_files[@]}"; do
        if [[ -f "$file" ]]; then
            chmod +x "$file"
            log_and_print "${GREEN}✅ Made executable: $file${NC}"
        fi
    done
}

# Function to setup git configuration
setup_git() {
    log_and_print "${YELLOW}🔧 Setting up Git configuration...${NC}"
    
    # Ensure we're in git repository
    if [[ ! -d ".git" ]]; then
        log_and_print "${RED}❌ Not in a git repository${NC}"
        return 1
    fi
    
    # Check current branch
    local current_branch=$(git branch --show-current)
    log_and_print "${BLUE}📍 Current branch: $current_branch${NC}"
    
    # Create desktop-dev branch if it doesn't exist
    if ! git show-ref --verify --quiet refs/heads/desktop-dev; then
        log_and_print "${YELLOW}🌿 Creating desktop-dev branch...${NC}"
        git checkout -b desktop-dev
        git push -u origin desktop-dev
    else
        log_and_print "${BLUE}📍 Switching to desktop-dev branch...${NC}"
        git checkout desktop-dev
    fi
    
    # Configure git for desktop
    git config --local branch.desktop-dev.device desktop
    git config --local alias.sync-main '!git checkout main && git pull origin main'
    git config --local alias.sync-desktop '!git checkout desktop-dev && git merge main'
    
    log_and_print "${GREEN}✅ Git configuration complete${NC}"
}

# Function to install dependencies
install_dependencies() {
    log_and_print "${YELLOW}📦 Installing dependencies...${NC}"
    
    # Check if requirements.txt exists
    if [[ -f "requirements.txt" ]]; then
        log_and_print "${BLUE}📦 Installing main requirements...${NC}"
        pip install -r requirements.txt
    fi
    
    # Install FastAPI auth requirements
    if [[ -f "fastapi_auth/requirements.txt" ]]; then
        log_and_print "${BLUE}📦 Installing FastAPI auth requirements...${NC}"
        cd fastapi_auth
        pip install -r requirements.txt
        cd ..
    fi
    
    log_and_print "${GREEN}✅ Dependencies installed${NC}"
}

# Function to run health checks
run_health_checks() {
    log_and_print "${YELLOW}🏥 Running health checks...${NC}"
    
    local checks_passed=0
    local total_checks=5
    
    # Test 1: LLMLingua service runner
    if python -c "from src.core.llmlingua_service_runner import main; print('LLMLingua import: OK')" 2>/dev/null; then
        log_and_print "${GREEN}✅ LLMLingua service runner import test passed${NC}"
        ((checks_passed++))
    else
        log_and_print "${RED}❌ LLMLingua service runner import test failed${NC}"
    fi
    
    # Test 2: SuperClaude integration
    if python -c "from src.integrations.superclaude_sync import SuperClaudeSync; print('SuperClaude import: OK')" 2>/dev/null; then
        log_and_print "${GREEN}✅ SuperClaude integration import test passed${NC}"
        ((checks_passed++))
    else
        log_and_print "${RED}❌ SuperClaude integration import test failed${NC}"
    fi
    
    # Test 3: FastAPI auth system
    if python -c "import sys; sys.path.append('fastapi_auth'); from app.main import app; print('FastAPI import: OK')" 2>/dev/null; then
        log_and_print "${GREEN}✅ FastAPI auth system import test passed${NC}"
        ((checks_passed++))
    else
        log_and_print "${RED}❌ FastAPI auth system import test failed${NC}"
    fi
    
    # Test 4: Agent configurations
    if [[ -f "agents/executor/CLAUDE_ENHANCED.md" && -f "agents/mastermind/CLAUDE_ENHANCED.md" && -f "agents/researcher/CLAUDE_ENHANCED.md" ]]; then
        log_and_print "${GREEN}✅ Enhanced agent configurations present${NC}"
        ((checks_passed++))
    else
        log_and_print "${RED}❌ Enhanced agent configurations missing${NC}"
    fi
    
    # Test 5: Startup script
    if [[ -x "${CLAUDE_CONFIG_DIR}/startup_services.sh" ]]; then
        log_and_print "${GREEN}✅ Startup script executable${NC}"
        ((checks_passed++))
    else
        log_and_print "${RED}❌ Startup script not executable${NC}"
    fi
    
    log_and_print "${BLUE}📊 Health check results: $checks_passed/$total_checks passed${NC}"
    
    if [[ $checks_passed -eq $total_checks ]]; then
        log_and_print "${GREEN}🎉 All health checks passed!${NC}"
        return 0
    else
        log_and_print "${YELLOW}⚠️  Some health checks failed. Review the issues above.${NC}"
        return 1
    fi
}

# Function to display completion summary
show_completion_summary() {
    log_and_print "${BLUE}============================================================================${NC}"
    log_and_print "${GREEN}🎉 Desktop Implementation Complete!${NC}"
    log_and_print "${BLUE}============================================================================${NC}"
    echo ""
    log_and_print "${GREEN}✅ Services Available:${NC}"
    log_and_print "${BLUE}   • LLMLingua: http://localhost:8766${NC}"
    log_and_print "${BLUE}   • LLMLingua Health: http://localhost:8766/health${NC}"
    log_and_print "${BLUE}   • SuperClaude Sync: python -m src.integrations.superclaude_sync${NC}"
    log_and_print "${BLUE}   • FastAPI Auth: cd fastapi_auth && uvicorn app.main:app --port 8001${NC}"
    echo ""
    log_and_print "${GREEN}✅ Quick Start Commands:${NC}"
    log_and_print "${BLUE}   • Start services: ${CLAUDE_CONFIG_DIR}/startup_services.sh${NC}"
    log_and_print "${BLUE}   • Test LLMLingua: curl http://localhost:8766/health${NC}"
    log_and_print "${BLUE}   • Setup SuperClaude: python -m src.integrations.superclaude_sync --setup${NC}"
    log_and_print "${BLUE}   • Git sync: git sync-main && git sync-desktop${NC}"
    echo ""
    log_and_print "${GREEN}✅ Documentation:${NC}"
    log_and_print "${BLUE}   • FastAPI Auth: ./fastapi_auth/README.md${NC}"
    log_and_print "${BLUE}   • SuperClaude: ./SUPERCLAUDE_INTEGRATION.md${NC}"
    log_and_print "${BLUE}   • Setup Log: $LOG_FILE${NC}"
    echo ""
}

# Main execution
main() {
    # Initialize log file
    echo "Desktop Implementation Setup - $(date)" > "$LOG_FILE"
    
    log_and_print "${BLUE}🚀 Starting desktop implementation setup...${NC}"
    
    # Run setup steps
    verify_prerequisites
    backup_existing
    generate_checksums
    setup_directories
    update_paths
    set_permissions
    setup_git
    install_dependencies
    
    # Run health checks
    if run_health_checks; then
        show_completion_summary
        log_and_print "${GREEN}✅ Setup completed successfully!${NC}"
        exit 0
    else
        log_and_print "${RED}❌ Setup completed with issues. Check the log for details.${NC}"
        exit 1
    fi
}

# Run main function
main "$@"