---
allowed-tools: [Bash, Read, Edit, Write, MultiEdit, Task, TodoWrite, TodoRead, LS, Glob, Grep]
description: "Execute production-grade SPARC trio workflow for intelligent development with comprehensive quality gates"
---

# 🔥 FIRE Command - Production SPARC Trio Workflow

Execute the complete SPARC trio workflow (RESEARCHER → MASTERMIND → EXECUTOR) for production-grade intelligent development with comprehensive quality gates, testing, and deployment preparation.

## Workflow Overview

The FIRE command orchestrates a six-phase development workflow:

1. **🔍 Environment Validation**: Check dependencies, API keys, agent health
2. **🤖 SPARC Trio Initialization**: Activate real RESEARCHER, MASTERMIND, EXECUTOR agents
3. **📝 Task Analysis & Planning**: Intelligence gathering + strategic architecture
4. **⚡ Implementation**: TDD workflow with quality gates
5. **🛡️ Quality Gates**: Testing, security, performance validation
6. **🚀 Deployment Preparation**: Container, monitoring, security configuration

## Execution Strategy

### Primary Approach: Direct Agent Integration
Use the existing fire script infrastructure for real SPARC trio agent execution:

```bash
# Execute the production-ready fire script
cd /home/scott/TradeKnowledge
python3 fire_command.py "USER_TASK_DESCRIPTION"
```

### Fallback Approach: Manual SPARC Workflow
If the fire script encounters issues, execute the SPARC workflow manually:

#### Phase 1: RESEARCHER Intelligence Gathering
```bash
cd /home/scott/TradeKnowledge/agents
python ask_researcher.py
```
**Prompt**: "Analyze [TASK] for technical requirements, best practices, security considerations, and implementation patterns. Provide comprehensive intelligence for strategic planning."

#### Phase 2: MASTERMIND Strategic Architecture
```bash
cd /home/scott/TradeKnowledge/agents
python ask_mastermind.py
```
**Prompt**: "Based on the research intelligence, design strategic architecture for [TASK]. Include quality gates, risk assessment, and implementation roadmap."

#### Phase 3: EXECUTOR Implementation
```bash
cd /home/scott/TradeKnowledge/agents
python ask_executor.py
```
**Prompt**: "Implement [TASK] using TDD workflow with the strategic guidance. Include comprehensive tests, security measures, and deployment configuration."

## Quality Gates Integration

Apply the established quality standards from `/project:quality-gates`:

### Code Quality Requirements
- **Test Coverage**: Minimum 90%, Target 95%
- **Mutation Score**: Minimum 80%, Target 90%
- **Security Score**: 9.8/10 minimum
- **Performance**: <100ms response time, >1000 rps throughput
- **Compliance**: SOX, FINRA standards

### Automated Quality Validation
```bash
# Run comprehensive quality checks
cd /home/scott/TradeKnowledge

# Code quality pipeline
ruff check src/ --fix
ruff format src/
mypy src/ --strict --ignore-missing-imports
bandit -r src/ -f json -o security-report.json
pytest tests/ --cov=src --cov-report=html --cov-fail-under=90

# Performance and integration tests
pytest tests/performance/ --benchmark-only --benchmark-min-rounds=10
pytest tests/integration/ -v --timeout=30
```

## Technology Stack Options

### Supported Stacks
- **FastAPI**: Modern Python API framework (default)
- **Django**: Full-featured web framework
- **Flask**: Lightweight web framework
- **PyTorch**: Machine learning and neural networks
- **TensorFlow**: Deep learning and AI
- **React**: Frontend JavaScript framework
- **Vue**: Progressive JavaScript framework
- **Next.js**: React-based full-stack framework

### Deployment Targets
- **Docker**: Containerized deployment (default)
- **Kubernetes**: Orchestrated container deployment
- **Serverless**: AWS Lambda/Azure Functions
- **Bare-metal**: Direct server deployment

## Task Complexity Assessment

The FIRE workflow automatically assesses task complexity:

- **Simple**: Basic CRUD, simple APIs (2-4 hours, 95% success)
- **Moderate**: Integration services, dashboards (1-2 days, 85% success)
- **Complex**: ML models, microservices (1-2 weeks, 75% success)
- **Enterprise**: Scalable systems, compliance (2-4 weeks, 65% success)

## FIRE Command Usage Patterns

### Quick Development Task
```
/project:fire
Task: "Create a FastAPI endpoint for user authentication with JWT tokens"
```

### Complex ML Project
```
/project:fire
Task: "Implement a neural network for stock price prediction using PyTorch with real-time data ingestion"
```

### Enterprise System
```
/project:fire
Task: "Design and implement a scalable microservices architecture for financial trading with full compliance and monitoring"
```

### Interactive Mode
```
/project:fire
Task: "interactive" - This will prompt for task details, tech stack, deployment target, and quality level
```

## Expected Deliverables

### Research Phase Output
- Technical analysis and requirements
- Best practices and implementation patterns
- Security considerations and risk assessment
- Industry standards and compliance requirements

### Strategic Phase Output
- System architecture design
- Implementation roadmap
- Quality gates and success criteria
- Risk mitigation strategies
- Technology recommendations

### Implementation Phase Output
- Production-ready code with comprehensive tests
- TDD test suite (unit, integration, end-to-end)
- Security implementation and vulnerability remediation
- Performance optimization and benchmarking
- Documentation and operational runbooks

### Quality Validation Output
- Test coverage reports (>90%)
- Security scan results (9.8/10+)
- Performance benchmarks (<100ms response)
- Compliance validation results
- Deployment readiness assessment

### Deployment Preparation Output
- Container configuration (Dockerfile, docker-compose)
- Kubernetes manifests (if selected)
- Monitoring and alerting setup
- Security hardening configuration
- CI/CD pipeline templates

## Integration with Existing Systems

### Agent Coordination
- Leverages existing SPARC trio agents in `/agents/`
- Maintains context isolation between agent domains
- Uses structured handoffs and shared artifacts
- Enables cross-agent validation and review

### Quality System Integration
- Integrates with `/project:quality-gates` standards
- Uses established pre-commit hooks and CI/CD pipelines
- Maintains compliance with financial system requirements
- Provides automated quality reporting and dashboards

### Development Workflow Integration
- Compatible with existing development patterns
- Supports both greenfield and brownfield projects
- Integrates with current testing and deployment infrastructure
- Maintains consistency with project conventions

## Error Handling and Fallbacks

### Agent Availability Issues
- Falls back to simulation mode if real agents unavailable
- Provides intelligent simulation based on task complexity
- Maintains workflow continuity with degraded capabilities
- Reports agent status and recommended actions

### Dependency Problems
- Validates environment before execution
- Provides clear error messages and resolution steps
- Supports partial execution with available components
- Offers alternative approaches for missing dependencies

### Quality Gate Failures
- Reports specific quality issues with remediation guidance
- Supports iterative improvement cycles
- Provides graduated quality levels (development/staging/production)
- Maintains quality history and trends

## Success Metrics

### Workflow Efficiency
- **Analysis Completeness**: Research coverage and depth
- **Strategic Accuracy**: Architecture alignment with requirements
- **Implementation Quality**: Code quality and test coverage
- **Trio Synergy**: Cross-agent collaboration effectiveness

### Quality Achievement
- **Test Coverage**: 90%+ coverage with 80%+ mutation score
- **Security Rating**: 9.8/10+ security score
- **Performance**: <100ms response time, >1000 rps throughput
- **Compliance**: Full SOX/FINRA compliance validation

### Delivery Success
- **Time to Production**: Reduced development cycle time
- **Quality Consistency**: Reproducible high-quality outputs
- **Risk Mitigation**: Proactive security and compliance
- **Operational Excellence**: Production-ready deployment artifacts