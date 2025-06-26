# 🔥 FIRE Command Usage Guide

The FIRE command is your production-ready gateway to the SPARC trio agents, providing instant access to intelligent development workflows.

## Quick Start

```bash
# Simple task execution
./fire "build trading API with authentication"

# Interactive mode
./fire --interactive

# Check system status
./fire --status

# Run health check
./fire --health-check
```

## Command Overview

The FIRE command activates the SPARC trio:
- 🔍 **RESEARCHER** - Intelligence synthesis and analysis
- 🧠 **MASTERMIND** - Strategic architecture and planning  
- ⚡ **EXECUTOR** - Implementation virtuosity and testing

## Usage Examples

### 1. Basic Task Execution
```bash
# Load environment and execute task
source ./load_env.sh && ./fire "build FastAPI trading bot"
```

### 2. Technology Stack Selection
```bash
./fire "implement ML pipeline" --stack pytorch --deploy k8s
```

### 3. Interactive Mode
```bash
./fire --interactive
# Follow guided prompts for task definition
```

### 4. Production Deployment
```bash
./fire "build microservices architecture" --quality production --deploy k8s
```

### 5. Development Workflow
```bash
./fire "add authentication to API" --quality development --stack fastapi
```

## Command Options

| Option | Values | Description |
|--------|--------|-------------|
| `--stack` | `fastapi`, `django`, `flask`, `pytorch`, `tensorflow`, `react`, `vue`, `nextjs` | Technology stack |
| `--deploy` | `docker`, `k8s`, `serverless`, `bare-metal` | Deployment target |
| `--quality` | `development`, `staging`, `production` | Quality gates level |
| `--interactive` | flag | Interactive guided mode |
| `--status` | flag | Show system and agent status |
| `--health-check` | flag | Run comprehensive health check |
| `--config` | path | Custom configuration file |
| `--output` | path | Output directory for artifacts |

## Production Quality Gates

When `--quality production` is used, FIRE applies strict quality gates:

### ✅ Quality Standards
- **Test Coverage**: ≥98%
- **Mutation Score**: ≥90% 
- **Security Score**: ≥9.8/10
- **Response Time**: <50ms
- **Throughput**: >10,000 RPS
- **Availability**: 99.99%

### 🔒 Security Features
- Input validation and sanitization
- Authentication middleware
- Rate limiting and DDoS protection
- CORS configuration
- Security headers (HSTS, CSP, etc.)
- Vulnerability scanning

### 📊 Performance Optimizations
- Async/await patterns
- Database query optimization
- Caching strategies
- Resource monitoring
- Auto-scaling configuration

## SPARC Workflow Phases

### Phase 1: Environment Validation
- Check API keys and dependencies
- Validate MCP server connectivity
- Verify trio agent readiness

### Phase 2: Trio Initialization
- Activate RESEARCHER agent
- Initialize MASTERMIND agent
- Start EXECUTOR agent
- Establish blackboard communication

### Phase 3: Task Analysis & Planning
- RESEARCHER gathers intelligence
- MASTERMIND develops strategy
- Complexity assessment
- Effort estimation

### Phase 4: Implementation
- EXECUTOR implements with TDD
- Red-Green-Refactor cycles
- Comprehensive test generation
- Code quality optimization

### Phase 5: Quality Gates
- Test coverage validation
- Security vulnerability scanning
- Performance benchmarking
- Compliance checking

### Phase 6: Deployment Preparation
- Container configuration
- Kubernetes manifests
- Monitoring setup
- Security hardening

## Technology Stack Configurations

### FastAPI (Default)
```bash
./fire "build REST API" --stack fastapi
```
- Production-ready ASGI server
- Automatic OpenAPI documentation
- Type hints and validation
- Async/await support

### Django
```bash
./fire "build web application" --stack django
```
- Enterprise-grade web framework
- Built-in admin interface
- ORM and migrations
- Security features

### PyTorch/ML
```bash
./fire "implement neural network" --stack pytorch
```
- Deep learning framework
- GPU acceleration
- Model training pipeline
- Deployment optimization

### React/Frontend
```bash
./fire "build dashboard" --stack react
```
- Modern UI framework
- Component-based architecture
- State management
- Performance optimization

## Deployment Targets

### Docker
```bash
./fire "containerize application" --deploy docker
```
- Multi-stage builds
- Security hardening
- Minimal base images
- Health checks

### Kubernetes
```bash
./fire "deploy to cloud" --deploy k8s
```
- Production manifests
- Auto-scaling configuration
- Resource limits
- Service mesh ready

### Serverless
```bash
./fire "build lambda function" --deploy serverless
```
- AWS Lambda optimization
- Cold start minimization
- Event-driven architecture
- Cost optimization

## Environment Setup

### Prerequisites
```bash
# Ensure environment is loaded
source ./load_env.sh

# Verify MCP servers
./check_mcp_health.sh

# Check FIRE status
./fire --status
```

### Required Environment Variables
- `GEMINI_API_KEY` - For Zen MCP server
- `PERPLEXITY_API_KEY` - For research capabilities
- `GITHUB_PERSONAL_ACCESS_TOKEN` - For code management
- `OPENAI_API_KEY` - For AI capabilities
- `ANTHROPIC_API_KEY` - For Claude integration

## Output and Results

### Task Execution Results
FIRE generates comprehensive results including:
- Implementation artifacts
- Test suites and coverage reports
- Deployment configurations
- Monitoring setup
- Security configurations

### Result Storage
```bash
# Save results to file
./fire "build API" --output ./results/

# Results saved as JSON
cat ./results/fire_result_<session_id>.json
```

## Interactive Mode Guide

```bash
./fire --interactive
```

Interactive mode provides guided prompts for:
1. **Task Definition** - Describe what you want to build
2. **Technology Stack** - Select appropriate framework
3. **Deployment Target** - Choose deployment method
4. **Quality Level** - Set quality gates
5. **Confirmation** - Review and execute

## Status and Health Monitoring

### System Status
```bash
./fire --status
```
Shows:
- Environment variables status
- SPARC trio agent readiness
- Dependency availability
- Overall system health

### Health Check
```bash
./fire --health-check
```
Performs:
- Comprehensive environment validation
- MCP server connectivity tests
- Performance benchmarks
- Security validations

## Troubleshooting

### Common Issues

1. **Environment Variables Missing**
   ```bash
   source ./load_env.sh
   ```

2. **MCP Servers Not Ready**
   ```bash
   ./initialize_mcp_servers.sh
   ```

3. **Python Dependencies Missing**
   ```bash
   pip install click asyncio requests
   ```

4. **Trio Agents Not Found**
   - Verify `agents/` directory exists
   - Check agent CLAUDE.md files

### Error Resolution

| Error | Solution |
|-------|----------|
| `No module named 'pytest'` | Install with `pip install pytest` |
| `Environment not ready` | Run `source ./load_env.sh` |
| `Trio agents missing` | Check `agents/` directory structure |
| `MCP servers offline` | Run `./initialize_mcp_servers.sh` |

## Advanced Usage

### Custom Configuration
```bash
./fire --config fire-config.yaml "build enterprise API"
```

### Batch Processing
```bash
# Multiple tasks
for task in "build auth" "add tests" "deploy production"; do
    ./fire "$task" --quality production
done
```

### CI/CD Integration
```bash
# In pipeline
source ./load_env.sh
./fire "$BUILD_TASK" --stack $TECH_STACK --deploy $DEPLOY_TARGET --output ./artifacts/
```

## Best Practices

1. **Always load environment first**: `source ./load_env.sh`
2. **Use production quality for important tasks**: `--quality production`
3. **Specify appropriate tech stack**: `--stack <framework>`
4. **Save results for review**: `--output ./results/`
5. **Run health checks regularly**: `./fire --health-check`

---

**🎉 Ready to FIRE!** The SPARC trio is ready for production-grade intelligent development.