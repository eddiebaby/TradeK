# Trio Agent Usage Guide

This guide shows how to interact with each agent in the TradeKnowledge trio, following the isolated context pattern where each agent has its own specialized domain.

## 🎯 Agent Framework Overview

### The Trio
- **🔍 RESEARCHER** - Knowledge Architect & Intelligence Synthesizer
- **🧠 MASTERMIND** - Strategic Architect & Quality Orchestrator  
- **⚡ EXECUTOR** - Implementation Virtuoso & Operational Expert

### Context Isolation Pattern
Each agent operates in its own isolated context with specialized CLAUDE.md files:
```
agents/
├── researcher/
│   ├── CLAUDE.md          # Research-specific context
│   └── researcher_agent.py
├── mastermind/
│   ├── CLAUDE.md          # Strategy-specific context  
│   └── mastermind_agent.py
└── executor/
    ├── CLAUDE.md          # Implementation-specific context
    └── executor_agent.py
```

## 🔍 Using RESEARCHER Agent

### Interactive Mode
```bash
cd /home/scottschweizer/TradeKnowledge/agents
python ask_researcher.py
```

### Direct Python Usage
```python
import asyncio
from researcher.researcher_agent import ResearcherAgent

async def research_example():
    researcher = ResearcherAgent()
    
    research_spec = {
        "domains": ["technical_deep_dive", "security_intelligence"],
        "focus_areas": ["API authentication best practices"],
        "depth": "comprehensive",
        "target_format": "strategy"  # or "implementation"
    }
    
    result = await researcher.targeted_research(research_spec)
    return result

# Run research
result = asyncio.run(research_example())
```

### Research Capabilities
- **Technical Deep Dive**: Architecture patterns, implementation strategies
- **Security Intelligence**: Threat analysis, vulnerability research
- **Performance Benchmarking**: Optimization techniques, scaling strategies
- **Best Practices**: Industry standards, proven methodologies
- **Trend Analysis**: Emerging technologies, future predictions
- **Market Intelligence**: Competitive analysis, technology evaluation

### Example Research Tasks
```bash
# In ask_researcher.py interactive mode:
"security best practices for API authentication"
"performance optimization techniques for FastAPI"
"microservices architecture patterns for trading systems"
"testing strategies for real-time data processing"
"emerging trends in AI-powered development"
```

## 🧠 Using MASTERMIND Agent

### Interactive Mode
```bash
cd /home/scottschweizer/TradeKnowledge/agents
python ask_mastermind.py
```

### Direct Python Usage
```python
import asyncio
from mastermind.mastermind_agent import MastermindAgent

async def strategy_example():
    mastermind = MastermindAgent()
    
    # Strategic analysis
    result = await mastermind.strategic_analysis(
        "Design a microservices architecture for real-time trading"
    )
    
    return result

# Run strategic analysis
result = asyncio.run(strategy_example())
```

### Strategic Capabilities
- **Architectural Analysis**: System design and pattern selection
- **Quality Strategy Design**: TDD orchestration and test strategies
- **Risk Assessment**: Failure prediction and mitigation planning
- **Technical Debt Assessment**: Code quality and maintainability analysis
- **Performance Prediction**: Bottleneck identification and scaling plans
- **Security Threat Modeling**: Threat analysis and security architecture

### Example Strategic Tasks
```bash
# In ask_mastermind.py interactive mode:
"Design a microservices architecture for a trading platform"
"Create a testing strategy for real-time data processing"
"Analyze performance bottlenecks in our current API"
"Design authentication system for multi-tenant SaaS"
"Assess technical debt and prioritize refactoring"
```

## ⚡ Using EXECUTOR Agent

### Interactive Mode
```bash
cd /home/scottschweizer/TradeKnowledge/agents
python ask_executor.py
```

### Quick Demo
```bash
cd /home/scottschweizer/TradeKnowledge/agents
python ask_executor.py demo
```

### Direct Python Usage
```python
import asyncio
from executor.executor_agent import ExecutorAgent

async def implementation_example():
    executor = ExecutorAgent()
    
    # Create task context
    task_context = executor.create_task_context(
        description="Create REST API endpoint with authentication",
        requirements={
            "implementation_type": "tdd",
            "test_coverage": 95,
            "mutation_score": 85
        },
        performance_targets={
            "response_time": "< 100ms",
            "throughput": "> 1000 rps"
        },
        success_criteria={
            "all_tests_pass": True,
            "coverage_target_met": True
        }
    )
    
    result = await executor.process_task(task_context)
    return result

# Run implementation
result = asyncio.run(implementation_example())
```

### Implementation Capabilities
- **TDD Workflow**: Red-Green-Refactor cycle with quality gates
- **Comprehensive Testing**: Unit, integration, property, mutation, chaos tests
- **Performance Optimization**: Benchmark-driven improvement cycles
- **Security Implementation**: Defense-in-depth coding practices
- **CI/CD Pipeline Creation**: Automated deployment and quality gates
- **Monitoring Setup**: Observability and alerting configuration

### Example Implementation Tasks
```bash
# In ask_executor.py interactive mode:
"implement a REST API endpoint for user authentication"
"create a caching layer with Redis integration"
"build a real-time data processing pipeline"
"implement TDD workflow for trading algorithm"
"create comprehensive test suite for search API"
"setup CI/CD pipeline with quality gates"
```

## 🤝 Trio Collaboration Patterns

### SPARC Framework
The agents follow the SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) framework:

1. **🔍 RESEARCHER** - Gathers requirements and validates specifications
2. **🧠 MASTERMIND** - Designs architecture and creates strategic plans
3. **⚡ EXECUTOR** - Implements with TDD and comprehensive testing

### Running Full Trio Demos
```bash
cd /home/scottschweizer/TradeKnowledge/agents

# SPARC trio demonstration
python sparc_trio_demo.py

# Intelligence-driven development
python demo_trio_analysis.py

# Interactive trio collaboration
python easy_start.py
```

## 📁 Context Isolation Benefits

### Independent Operation
- Each agent has specialized knowledge domain
- No cross-contamination of context
- Focused expertise without distractions
- Clean separation of concerns

### Claude Code Integration
To use with Claude Code's --add-dir flag:
```bash
# Work with individual agents
claude-code --add-dir agents/researcher
claude-code --add-dir agents/mastermind  
claude-code --add-dir agents/executor

# Or work with specific agent during session
claude-code --add-dir agents/researcher/
# Then use ask_researcher.py for interactions
```

### Agent-Specific Memory
Each agent maintains its own:
- Research history (RESEARCHER)
- Architectural decisions (MASTERMIND)
- Implementation artifacts (EXECUTOR)

## 🎛️ Advanced Usage

### Programmatic Agent Orchestration
```python
from trio_orchestrator import trio_orchestrator

# Execute full intelligence-driven development
results = await trio_orchestrator.execute_intelligence_driven_development(
    requirement="Build a high-performance trading API",
    project_context={"stack": "FastAPI + PostgreSQL"},
    quality_requirements={"test_coverage": 95}
)
```

### Custom Research Specifications
```python
# Targeted research for specific use cases
research_spec = {
    "domains": ["security_intelligence", "performance_benchmarking"],
    "focus_areas": ["OAuth implementation", "Redis caching"],
    "depth": "comprehensive",
    "context": {"current_stack": "FastAPI", "requirements": "sub-100ms"},
    "target_format": "implementation"  # Format for EXECUTOR
}
```

### Quality-Driven Implementation
```python
# High-quality implementation with comprehensive testing
task_context = executor.create_task_context(
    description="Implement trading algorithm",
    requirements={
        "implementation_type": "tdd",
        "test_types": ["unit", "integration", "property", "mutation"],
        "coverage_target": 95,
        "mutation_score": 90
    },
    performance_targets={
        "latency": "< 50ms",
        "throughput": "> 5000 ops/sec"
    }
)
```

## 🚀 Getting Started

1. **Choose Your Agent**: Based on your task type (research, strategy, implementation)
2. **Use Interactive Mode**: Run the appropriate `ask_*.py` script
3. **Or Use Programmatically**: Import and use the agent classes directly
4. **Leverage Context Isolation**: Each agent focuses on its expertise
5. **Combine for Complex Tasks**: Use trio orchestration for full development cycles

The trio agents provide specialized expertise while maintaining clean separation of concerns through context isolation.