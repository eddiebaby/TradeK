#!/bin/bash

# FIRE Command Aliases and Integration
# Source this file to get convenient FIRE aliases

# Base FIRE function with environment auto-loading
fire() {
    local fire_dir="/home/scott/TradeKnowledge"
    
    # Change to project directory
    cd "$fire_dir" || {
        echo "❌ Error: Could not change to TradeKnowledge directory"
        return 1
    }
    
    # Auto-load environment if .env exists
    if [ -f ".env" ]; then
        set -a
        source .env
        set +a
    fi
    
    # Execute FIRE command
    python3 fire "$@"
}

# Convenient aliases
alias fire-status='fire --status'
alias fire-health='fire --health-check'
alias fire-interactive='fire --interactive'

# Quick task aliases
alias fire-api='fire --stack fastapi --deploy docker'
alias fire-ml='fire --stack pytorch --deploy k8s'
alias fire-web='fire --stack react --deploy docker'
alias fire-django='fire --stack django --deploy k8s'

# Quality level aliases
alias fire-dev='fire --quality development'
alias fire-staging='fire --quality staging'
alias fire-prod='fire --quality production'

# Deployment target aliases
alias fire-docker='fire --deploy docker'
alias fire-k8s='fire --deploy k8s'
alias fire-serverless='fire --deploy serverless'

# Combined convenience functions
fire-fastapi() {
    fire "$1" --stack fastapi --deploy docker --quality production
}

fire-ml-pipeline() {
    fire "$1" --stack pytorch --deploy k8s --quality production
}

fire-react-app() {
    fire "$1" --stack react --deploy docker --quality production
}

fire-django-app() {
    fire "$1" --stack django --deploy k8s --quality production
}

# Environment setup helper
fire-setup() {
    echo "🔥 Setting up FIRE environment..."
    
    local fire_dir="/home/scott/TradeKnowledge"
    cd "$fire_dir" || {
        echo "❌ Error: Could not change to TradeKnowledge directory"
        return 1
    }
    
    # Load environment
    if [ -f "load_env.sh" ]; then
        source load_env.sh
    else
        echo "⚠️  Warning: load_env.sh not found"
    fi
    
    # Run health check
    if [ -x "fire" ]; then
        fire --health-check
    else
        echo "❌ Error: FIRE command not found or not executable"
        return 1
    fi
    
    echo "✅ FIRE environment ready!"
}

# Help function
fire-help() {
    echo "🔥 FIRE Command Help"
    echo "==================="
    echo ""
    echo "Basic Commands:"
    echo "  fire 'task description'        - Execute task with defaults"
    echo "  fire-status                    - Show system status"
    echo "  fire-health                    - Run health check"
    echo "  fire-interactive               - Interactive mode"
    echo "  fire-setup                     - Setup environment"
    echo ""
    echo "Quick Task Commands:"
    echo "  fire-api 'description'         - FastAPI + Docker + Production"
    echo "  fire-ml 'description'          - PyTorch + K8s + Production" 
    echo "  fire-web 'description'         - React + Docker + Production"
    echo "  fire-django 'description'      - Django + K8s + Production"
    echo ""
    echo "Quality Levels:"
    echo "  fire-dev 'task'                - Development quality"
    echo "  fire-staging 'task'            - Staging quality"
    echo "  fire-prod 'task'               - Production quality"
    echo ""
    echo "Deployment Targets:"
    echo "  fire-docker 'task'             - Docker deployment"
    echo "  fire-k8s 'task'                - Kubernetes deployment"
    echo "  fire-serverless 'task'         - Serverless deployment"
    echo ""
    echo "Examples:"
    echo "  fire 'build trading API'"
    echo "  fire-api 'add authentication'"
    echo "  fire-ml 'implement recommendation system'"
    echo "  fire 'build dashboard' --stack react --deploy k8s"
    echo ""
    echo "For full options: fire --help"
}

# Auto-completion for common tasks
_fire_completion() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local tasks=(
        "build API"
        "add authentication"
        "implement tests"
        "create dashboard"
        "build ML pipeline"
        "add database"
        "implement caching"
        "add monitoring"
        "build microservice"
        "create REST API"
        "implement GraphQL"
        "add websockets"
        "build trading bot"
        "implement AI model"
        "create web app"
    )
    
    COMPREPLY=($(compgen -W "${tasks[*]}" -- "$cur"))
}

# Register completion
complete -F _fire_completion fire
complete -F _fire_completion fire-api
complete -F _fire_completion fire-ml
complete -F _fire_completion fire-web
complete -F _fire_completion fire-django

echo "🔥 FIRE aliases loaded!"
echo "   Type 'fire-help' for usage guide"
echo "   Type 'fire-setup' to initialize environment"