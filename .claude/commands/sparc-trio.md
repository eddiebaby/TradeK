# SPARC Trio Agent Workflow

Invoke the specialized SPARC agent trio for comprehensive analysis and implementation.

## Usage

**Agent Selection:**
- `researcher` - Knowledge architect and intelligence synthesizer  
- `mastermind` - Strategic architect and quality orchestrator
- `executor` - Implementation virtuoso and operational expert
- `trio` - Full collaborative workflow (all three agents)

## Commands

### Individual Agent Invocation
```bash
cd /home/scott/TradeKnowledge/agents

# Research & Intelligence Gathering
python ask_researcher.py

# Strategic Analysis & Architecture 
python ask_mastermind.py

# Implementation & Testing
python ask_executor.py
```

### Trio Collaboration Workflows
```bash
cd /home/scott/TradeKnowledge/agents

# Full SPARC demonstration with handoffs
python sparc_trio_demo.py

# Interactive trio collaboration
python easy_start.py

# Specific trio analysis patterns
python demo_trio_analysis.py
```

## Workflow Patterns

### Sequential SPARC Pattern
1. **RESEARCHER**: Gather intelligence, analyze requirements, identify best practices
2. **MASTERMIND**: Design architecture, create strategy, define quality gates
3. **EXECUTOR**: Implement with TDD, create tests, deploy with monitoring

### Parallel Consultation Pattern
- Consult multiple agents simultaneously for different perspectives
- Synthesize recommendations across domains
- Validate decisions through cross-agent review

## Context Isolation Rules

- Each agent maintains **isolated context** - no access to global project memory
- Agents focus on their specialized domains only
- Communication happens through structured handoffs and shared artifacts
- Quality gates enforce validation between agent transitions

## Output Integration

Agent outputs should be integrated as follows:
1. **Research Findings** → Strategic planning input
2. **Strategic Plans** → Implementation roadmaps  
3. **Implementation** → Quality metrics and deployment artifacts
4. **Cross-validation** → Continuous improvement feedback

## Example Usage

**Research Phase:**
```
Ask RESEARCHER: "Analyze London School TDD best practices for financial trading systems"
```

**Strategy Phase:**
```  
Ask MASTERMIND: "Design TDD architecture for real-time market data processing with risk management"
```

**Implementation Phase:**
```
Ask EXECUTOR: "Implement London School TDD tests for market data validation with full coverage"
```

**Full Trio:**
```
Run trio workflow: "Design and implement a secure API endpoint for stock analysis with comprehensive testing"
```