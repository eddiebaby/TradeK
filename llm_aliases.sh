# TradeKnowledge Local LLM Aliases and Functions
# Add this to your ~/.bashrc: source ~/TradeKnowledge/llm_aliases.sh

# Quick commands for local LLM
alias qwen="snap run ollama run qwen2.5-coder:7b"
alias qwen-chat="snap run ollama run qwen2.5-coder:7b"
alias llm-start="snap run ollama serve &"
alias llm-stop="pkill -f 'ollama serve'"
alias llm-status="pgrep -f 'ollama serve' && echo 'Ollama running' || echo 'Ollama stopped'"
alias llm-models="snap run ollama list"
alias llm-setup="cd ~/TradeKnowledge && ./setup_local_llm.sh"

# Function for coding assistance
code_assist() {
    if [ -z "$1" ]; then
        echo "Usage: code_assist 'your coding question or task'"
        return 1
    fi
    echo "🤖 Qwen2.5-Coder is thinking..."
    snap run ollama run qwen2.5-coder:7b "$1"
}

# Function for code review
code_review() {
    if [ -z "$1" ]; then
        echo "Usage: code_review 'path/to/file.py' or code_review 'paste your code here'"
        return 1
    fi
    
    if [ -f "$1" ]; then
        local code_content=$(cat "$1")
        echo "🔍 Reviewing file: $1"
        snap run ollama run qwen2.5-coder:7b "Please review this code for best practices, potential bugs, and improvements:\n\n$code_content"
    else
        echo "🔍 Reviewing provided code..."
        snap run ollama run qwen2.5-coder:7b "Please review this code for best practices, potential bugs, and improvements:\n\n$1"
    fi
}

# Function for TradeKnowledge integration
tk_code() {
    if [ -z "$1" ]; then
        echo "Usage: tk_code 'describe what you want to implement in TradeKnowledge'"
        return 1
    fi
    echo "💰 TradeKnowledge coding assistant..."
    snap run ollama run qwen2.5-coder:7b "You are a coding assistant for TradeKnowledge, a financial intelligence platform using Python, FastAPI, SQLite, and AI agents. Help with: $1"
}

echo "🚀 TradeKnowledge LLM aliases loaded!"
echo "Available commands: qwen, code_assist, code_review, tk_code, llm-start, llm-stop, llm-status"