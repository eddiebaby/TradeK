# Desktop Quick Start Guide

## 🚀 Implementing Laptop Developments on Desktop

This guide helps you quickly implement all laptop developments made since 6am on your desktop environment.

### 📋 What You're Getting

All developments from the laptop including:
- **LLMLingua Auto-Start Integration** (50-70% token compression)
- **FastAPI JWT Authentication System** (Production-ready with 95%+ test coverage)
- **SuperClaude Integration** (Web ↔ Local sync)
- **Enhanced SPARC Agent Configurations**
- **Git Branch Strategy** for multi-device development

---

## ⚡ One-Command Setup

**On Desktop:**
```bash
# 1. Clone or sync the repository
git clone https://github.com/eddiebaby/TradeK.git TradeKnowledge
cd TradeKnowledge

# 2. Run the automated setup
./desktop_setup.sh
```

**That's it!** The script handles everything automatically.

---

## 🔧 Manual Setup (If Needed)

### Step 1: Prepare Environment
```bash
# Ensure you're in the TradeKnowledge directory
cd /path/to/TradeKnowledge

# Check git status
git status
```

### Step 2: Review the Manifest
```bash
# Review what will be implemented
cat DESKTOP_IMPLEMENTATION.mf
```

### Step 3: Run Setup Script
```bash
# Make executable and run
chmod +x desktop_setup.sh
./desktop_setup.sh
```

---

## ✅ Verification

After setup, verify everything works:

### Test LLMLingua Service
```bash
# Start services
~/.claude/startup_services.sh

# Test compression service
curl http://localhost:8766/health
```

### Test SuperClaude Integration
```bash
# Setup SuperClaude sync
python -m src.integrations.superclaude_sync --setup

# Test integration
python -m src.integrations.superclaude_sync --test
```

### Test FastAPI Authentication
```bash
# Test imports
cd fastapi_auth
python -c "from app.main import app; print('FastAPI ready!')"

# Run development server
uvicorn app.main:app --port 8001 --reload
```

### Test Git Configuration
```bash
# Check branch setup
git branch -a
git config --list | grep desktop

# Test sync aliases
git sync-main
git sync-desktop
```

---

## 🎯 Quick Service Overview

### LLMLingua Compression
- **Port**: 8766 (desktop-specific)
- **Health**: http://localhost:8766/health
- **Test**: http://localhost:8766/test
- **Auto-starts** with Claude Code

### SuperClaude Sync
- **Export Dir**: `~/Downloads/superclaude_exports`
- **Setup**: `python -m src.integrations.superclaude_sync --setup`
- **Auto-sync**: Monitors for new conversations every 5 minutes

### FastAPI Authentication
- **Port**: 8001 (desktop-specific)
- **Docs**: http://localhost:8001/docs
- **Features**: JWT auth, user management, 95%+ test coverage

### Enhanced SPARC Agents
- **RESEARCHER**: `agents/researcher/CLAUDE_ENHANCED.md`
- **MASTERMIND**: `agents/mastermind/CLAUDE_ENHANCED.md`
- **EXECUTOR**: `agents/executor/CLAUDE_ENHANCED.md`

---

## 🔀 Git Workflow

### Daily Development
```bash
# Work on desktop branch
git checkout desktop-dev

# Sync with latest changes
git sync-main
git sync-desktop

# Make desktop-specific changes
# ... your work ...

# Commit changes
git add .
git commit -m "Desktop: specific changes"
git push
```

### Sync with Laptop
```bash
# Pull laptop changes
git pull origin laptop-dev

# Merge or cherry-pick changes
git merge laptop-dev  # or cherry-pick specific commits
```

---

## 🛠️ Device-Specific Settings

### Desktop Configuration
- **User**: scott (adjust in `.env.desktop`)
- **Workspace**: `/home/scott/Desktop/TradeKnowledge`
- **LLMLingua Port**: 8766
- **FastAPI Port**: 8001
- **Performance**: Optimized for desktop resources

### Environment File
```bash
# Use desktop-specific settings
cp .env.desktop .env

# Or manually set:
export DEVICE_NAME=desktop
export LLMLINGUA_PORT=8766
```

---

## 🆘 Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check logs
tail -f ~/.claude/logs/llmlingua.log

# Check ports
netstat -tulpn | grep 8766
```

**Import errors:**
```bash
# Check Python environment
python --version
pip list | grep fastapi

# Reinstall dependencies
pip install -r requirements.txt
```

**Git conflicts:**
```bash
# Use device-specific branches
git checkout desktop-dev
git stash  # if needed
```

**Path issues:**
```bash
# Update paths in files
sed -i 's|/home/scottschweizer|/home/scott|g' .env.desktop
```

### Getting Help

1. **Check logs**: `~/.claude/logs/`
2. **Review setup log**: `desktop_setup.log`
3. **Verify manifest**: `DESKTOP_IMPLEMENTATION.mf`
4. **Test health**: All services have health check endpoints

---

## 📚 Documentation

- **FastAPI Auth**: `fastapi_auth/README.md`
- **SuperClaude**: `SUPERCLAUDE_INTEGRATION.md`
- **Implementation Details**: `DESKTOP_IMPLEMENTATION.mf`
- **FIRE Workflow**: `FIRE_PRODUCTION_READINESS_ASSESSMENT.md`

---

## 🎉 Success!

Once setup is complete, you'll have:
- ✅ Auto-starting LLMLingua compression (50-70% token savings)
- ✅ Production-ready FastAPI authentication system
- ✅ SuperClaude web-to-local integration
- ✅ Enhanced SPARC agent configurations
- ✅ Multi-device git workflow
- ✅ Desktop-optimized performance settings

Your desktop environment will now match your laptop development setup with device-specific optimizations!