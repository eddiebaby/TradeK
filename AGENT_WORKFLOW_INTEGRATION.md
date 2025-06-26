# TradeKnowledge Agent Trio Workflow Integration

## Overview
This document establishes the permanent integration of the specialized agent trio into the TradeKnowledge project workflow. The framework implements context isolation where each agent has domain-specific expertise without access to global project memory.

## Agent Architecture

### 🔍 RESEARCHER Agent
- **Purpose**: Knowledge Architect & Intelligence Synthesizer
- **Domain**: Research, intelligence gathering, trend analysis
- **Context File**: `/agents/researcher/CLAUDE.md`
- **Interaction Script**: `/agents/ask_researcher.py`

**Core Capabilities:**
- Multi-source intelligence gathering
- Security landscape research 
- Performance benchmarking analysis
- Best practice identification
- Technology trend prediction
- Evidence-based insight synthesis

### 🧠 MASTERMIND Agent  
- **Purpose**: Strategic Architect & Quality Orchestrator
- **Domain**: Strategic planning, architectural design, quality strategy
- **Context File**: `/agents/mastermind/CLAUDE.md`
- **Interaction Script**: `/agents/ask_mastermind.py`

**Core Capabilities:**
- Strategic analysis with deep contemplation
- Architectural design and pattern selection
- Quality strategy orchestration (TDD, testing pyramids)
- Risk assessment and mitigation planning
- Technical debt evaluation
- Performance bottleneck prediction

### ⚡ EXECUTOR Agent
- **Purpose**: Implementation Virtuoso & Operational Expert
- **Domain**: Code implementation, testing, DevOps automation
- **Context File**: `/agents/executor/CLAUDE.md`
- **Interaction Script**: `/agents/ask_executor.py`

**Core Capabilities:**
- TDD workflow implementation (Red-Green-Refactor)
- Comprehensive test suite creation (unit, integration, mutation, property, chaos)
- Performance optimization and monitoring
- Security implementation and hardening
- CI/CD pipeline automation
- Deployment and operational excellence

## Workflow Integration Commands

### Daily Development Workflow

#### Research Phase
```bash
cd /home/scottschweizer/TradeKnowledge/agents
python ask_researcher.py
# Research topics like:
# - "API security best practices for financial systems"
# - "Performance optimization for real-time trading"
# - "Microservices patterns for financial applications"
```

#### Strategic Planning Phase
```bash
cd /home/scottschweizer/TradeKnowledge/agents
python ask_mastermind.py
# Strategic analysis like:
# - "Design architecture for high-frequency trading system"
# - "Create testing strategy for financial data processing"
# - "Assess technical debt in current trading platform"
```

#### Implementation Phase
```bash
cd /home/scottschweizer/TradeKnowledge/agents
python ask_executor.py
# Implementation tasks like:
# - "Implement TDD workflow for trading algorithm"
# - "Create comprehensive test suite for API endpoints"
# - "Setup CI/CD pipeline with quality gates"
```

### Trio Collaboration Workflows

#### Full SPARC Development Cycle
```bash
cd /home/scottschweizer/TradeKnowledge/agents
python sparc_trio_demo.py
```

#### Interactive Multi-Agent Session
```bash
cd /home/scottschweizer/TradeKnowledge/agents
python easy_start.py
```

## Claude Code Integration

### Context Isolation with Claude Code
Use the `--add-dir` flag to work with individual agent contexts:

```bash
# Research sessions (isolated research context)
claude-code --add-dir /home/scottschweizer/TradeKnowledge/agents/researcher/

# Strategic planning sessions (isolated strategy context)
claude-code --add-dir /home/scottschweizer/TradeKnowledge/agents/mastermind/

# Implementation sessions (isolated implementation context)  
claude-code --add-dir /home/scottschweizer/TradeKnowledge/agents/executor/
```

### Benefits of Context Isolation
- **Focused Expertise**: Each agent operates in its specialized domain
- **No Cross-Contamination**: Prevents global context from diluting specialized knowledge
- **Clean Separation**: Research, strategy, and implementation remain distinct
- **Scalable Architecture**: Can add more specialized agents without conflicts

## Project Integration Points

### TradeKnowledge API Development
1. **RESEARCHER** investigates API security patterns and performance benchmarks
2. **MASTERMIND** designs API architecture and quality strategy
3. **EXECUTOR** implements with TDD and comprehensive testing

### Vector Search Enhancement
1. **RESEARCHER** researches vector database optimization techniques
2. **MASTERMIND** architects search performance improvements
3. **EXECUTOR** implements optimizations with monitoring

### Financial Data Processing
1. **RESEARCHER** analyzes real-time processing patterns and compliance requirements
2. **MASTERMIND** designs scalable data pipeline architecture
3. **EXECUTOR** implements with fault tolerance and observability

## Quality Standards Integration

### Research Quality (RESEARCHER)
- Minimum 85% research accuracy, target 92%
- Minimum 80% insight relevance, target 90%
- Cross-reference 3+ sources, target 5+ sources

### Strategic Quality (MASTERMIND)  
- Minimum 85% strategic accuracy, target 92%
- 100% architectural compliance to design principles
- Comprehensive risk assessment coverage

### Implementation Quality (EXECUTOR)
- Minimum 90% test coverage, target 95%
- Minimum 80% mutation score, target 90%
- Zero security vulnerabilities tolerance
- Sub-100ms response time targets

## Permanent Memory Integration

### Global Project Memory (CLAUDE.local.md)
- Contains agent workflow commands and integration patterns
- References isolated agent contexts
- Maintains project-wide architectural decisions

### Agent-Specific Memory
- **RESEARCHER**: Research history and intelligence database
- **MASTERMIND**: Architectural decisions and strategic insights
- **EXECUTOR**: Implementation artifacts and quality metrics

### Cross-Agent Handoffs
Use trio orchestration for complex tasks requiring multiple agent expertise:
```bash
cd /home/scottschweizer/TradeKnowledge/agents
python trio_orchestrator.py
```

## Quick Reference Integration

### File Locations
```bash
# Agent interaction scripts
ls /home/scottschweizer/TradeKnowledge/agents/ask_*.py

# Agent context files  
find /home/scottschweizer/TradeKnowledge/agents/*/CLAUDE.md

# Usage documentation
cat /home/scottschweizer/TradeKnowledge/agents/AGENT_USAGE_GUIDE.md
```

### Development Shortcuts
```bash
# Quick research task
cd /home/scottschweizer/TradeKnowledge/agents && python ask_researcher.py

# Quick strategic analysis
cd /home/scottschweizer/TradeKnowledge/agents && python ask_mastermind.py

# Quick implementation task
cd /home/scottschweizer/TradeKnowledge/agents && python ask_executor.py
```

## Future Enhancements

### Planned Integrations
- **Knowledge Base Integration**: Connect RESEARCHER to TradeKnowledge vector database
- **API Integration**: Direct integration with TradeKnowledge search and ingestion APIs
- **Monitoring Integration**: EXECUTOR integration with project monitoring systems
- **Automated Handoffs**: Enhanced trio communication and task delegation

### Scaling Considerations
- Additional specialized agents (SECURITY, PERFORMANCE, COMPLIANCE)
- Agent federation for complex multi-domain tasks
- Integration with external knowledge sources
- Automated quality gate enforcement

This workflow integration provides a structured, scalable approach to leveraging specialized AI expertise throughout the TradeKnowledge development lifecycle.