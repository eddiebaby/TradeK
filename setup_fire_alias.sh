#!/bin/bash
# 🔥 FIRE Command Alias Setup
# Source this file to enable /fire command in your shell

# Set up the fire command alias
alias fire='python3 /home/scott/TradeKnowledge/fire_command.py'
alias /fire='python3 /home/scott/TradeKnowledge/fire_command.py'

# Also create a direct executable
export PATH="/home/scott/TradeKnowledge:$PATH"

echo "🔥 FIRE command aliases activated!"
echo "Usage:"
echo "  fire 'your task'"
echo "  /fire 'your task'"
echo "  fire --status"
echo "  fire --interactive"