# Quality Gates Integration

Automated quality checks and gates for TradeKnowledge financial system development.

## Quality Standards

### Code Quality Requirements
- **Test Coverage**: Minimum 90%, Target 95%
- **Mutation Score**: Minimum 80%, Target 90% for business logic
- **Code Complexity**: Maximum 10 cyclomatic complexity per method
- **Method Length**: Maximum 20 lines per method
- **Security Vulnerabilities**: Zero tolerance for HIGH/CRITICAL issues
- **Performance**: < 100ms response time, > 1000 rps throughput

### Financial System Specific Standards
- **Calculation Accuracy**: 100% accuracy for financial calculations (no rounding errors)
- **Data Integrity**: Complete validation of all financial data inputs
- **Audit Logging**: All financial operations must be logged
- **Risk Controls**: All trading decisions must pass risk validation
- **Regulatory Compliance**: Code must meet SOX, FINRA compliance standards

## Pre-commit Quality Checks

### Automated Code Quality Pipeline
```bash
# Run complete quality gate check
cd /home/scott/TradeKnowledge

# 1. Code formatting and linting
ruff check src/ --fix
ruff format src/

# 2. Type checking
mypy src/ --strict --ignore-missing-imports

# 3. Security scanning
bandit -r src/ -f json -o security-report.json

# 4. Test execution with coverage
pytest tests/ --cov=src --cov-report=html --cov-fail-under=90

# 5. Performance testing
pytest tests/performance/ --benchmark-only --benchmark-min-rounds=10

# 6. Integration tests
pytest tests/integration/ -v --timeout=30
```

### Financial Calculation Validation
```bash
# Specialized financial accuracy tests
cd /home/scott/TradeKnowledge

# Test financial calculation precision
python -c "
from tests.validation.financial_accuracy import FinancialAccuracyValidator
validator = FinancialAccuracyValidator()

# Validate calculation precision
precision_results = validator.validate_calculation_precision()
print('Calculation Precision:', precision_results)

# Validate rounding behavior
rounding_results = validator.validate_rounding_behavior()
print('Rounding Validation:', rounding_results)

# Check for floating point errors
float_errors = validator.check_floating_point_errors()
print('Floating Point Errors:', float_errors)
"

# Test data integrity validation
python -c "
from tests.validation.data_integrity import DataIntegrityValidator
validator = DataIntegrityValidator()

# Validate market data integrity
market_data_results = validator.validate_market_data()
print('Market Data Integrity:', market_data_results)

# Validate portfolio calculations
portfolio_results = validator.validate_portfolio_calculations()
print('Portfolio Calculation Integrity:', portfolio_results)
"
```

## London School TDD Quality Gates

### Interaction Testing Validation
```bash
# Validate London School TDD compliance
python -c "
from tests.validation.tdd_compliance import TDDComplianceValidator
validator = TDDComplianceValidator()

# Check mock usage patterns
mock_compliance = validator.validate_mock_usage()
print('Mock Usage Compliance:', mock_compliance)

# Validate interaction testing
interaction_tests = validator.validate_interaction_testing()
print('Interaction Test Coverage:', interaction_tests)

# Check test isolation
isolation_results = validator.validate_test_isolation()
print('Test Isolation Score:', isolation_results)
"
```

### Test Quality Metrics
```bash
# Advanced test quality analysis
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=95

# Mutation testing for test quality
mutmut run --paths-to-mutate=src/ --tests-dir=tests/

# Test performance analysis
pytest tests/ --durations=10 --timeout=5

# Test flakiness detection
pytest tests/ --count=5 --timeout=10
```

## Continuous Integration Quality Gates

### GitHub Actions Integration
```yaml
# .github/workflows/quality-gates.yml
name: Quality Gates
on: [push, pull_request]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
          
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
          
      - name: Code formatting check
        run: ruff check src/ tests/
        
      - name: Type checking
        run: mypy src/ --strict
        
      - name: Security scan
        run: bandit -r src/
        
      - name: Test with coverage
        run: pytest tests/ --cov=src --cov-fail-under=90
        
      - name: Financial accuracy tests
        run: pytest tests/validation/financial_accuracy.py -v
        
      - name: Performance tests
        run: pytest tests/performance/ --benchmark-only
```

### Pre-commit Hooks Configuration
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: [-r, src/]

  - repo: local
    hooks:
      - id: financial-accuracy-tests
        name: Financial Accuracy Tests
        entry: pytest tests/validation/financial_accuracy.py
        language: system
        pass_filenames: false
```

## Performance Quality Gates

### Response Time Validation
```bash
# API performance testing
python -c "
from tests.performance.api_performance import APIPerformanceValidator
validator = APIPerformanceValidator()

# Test API response times
response_times = validator.test_api_response_times()
print('API Response Times:', response_times)

# Test database query performance
db_performance = validator.test_database_performance()
print('Database Performance:', db_performance)

# Test calculation performance
calc_performance = validator.test_calculation_performance()
print('Calculation Performance:', calc_performance)
"
```

### Memory and Resource Usage
```bash
# Memory usage validation
python -c "
from tests.performance.resource_monitor import ResourceMonitor
monitor = ResourceMonitor()

# Monitor memory usage during analysis
memory_usage = monitor.monitor_memory_usage()
print('Memory Usage Profile:', memory_usage)

# Check for memory leaks
leak_detection = monitor.detect_memory_leaks()
print('Memory Leak Detection:', leak_detection)

# Resource utilization analysis
resource_usage = monitor.analyze_resource_utilization()
print('Resource Utilization:', resource_usage)
"
```

## Security Quality Gates

### Financial Data Protection
```bash
# Security validation for financial systems
python -c "
from tests.security.financial_security import FinancialSecurityValidator
validator = FinancialSecurityValidator()

# Validate data encryption
encryption_check = validator.validate_data_encryption()
print('Data Encryption Status:', encryption_check)

# Check API key security
api_security = validator.validate_api_key_security()
print('API Key Security:', api_security)

# Validate access controls
access_controls = validator.validate_access_controls()
print('Access Control Validation:', access_controls)
"
```

### Dependency Security Scanning
```bash
# Dependency vulnerability scanning
safety check --json --output security-deps.json

# Docker security scanning (if using containers)
docker scout cves tradeknowledge:latest

# Infrastructure security checks
python -c "
from tests.security.infrastructure_security import InfrastructureSecurityValidator
validator = InfrastructureSecurityValidator()

# Check network security
network_security = validator.validate_network_security()
print('Network Security:', network_security)

# Validate container security
container_security = validator.validate_container_security()
print('Container Security:', container_security)
"
```

## Compliance Quality Gates

### Regulatory Compliance Validation
```bash
# SOX compliance validation
python -c "
from tests.compliance.sox_compliance import SOXComplianceValidator
validator = SOXComplianceValidator()

# Validate audit logging
audit_compliance = validator.validate_audit_logging()
print('Audit Logging Compliance:', audit_compliance)

# Check data retention policies
retention_compliance = validator.validate_data_retention()
print('Data Retention Compliance:', retention_compliance)

# Validate segregation of duties
duties_compliance = validator.validate_segregation_of_duties()
print('Segregation of Duties:', duties_compliance)
"

# FINRA compliance validation
python -c "
from tests.compliance.finra_compliance import FINRAComplianceValidator
validator = FINRAComplianceValidator()

# Validate trade reporting
trade_reporting = validator.validate_trade_reporting()
print('Trade Reporting Compliance:', trade_reporting)

# Check record keeping
record_keeping = validator.validate_record_keeping()
print('Record Keeping Compliance:', record_keeping)
"
```

## Quality Metrics Dashboard

### Real-time Quality Monitoring
```bash
# Generate quality metrics dashboard
python -c "
from src.monitoring.quality_dashboard import QualityDashboard
dashboard = QualityDashboard()

# Generate comprehensive quality report
quality_report = dashboard.generate_quality_report()
print('Quality Report:', quality_report)

# Get quality trends
quality_trends = dashboard.get_quality_trends(days=30)
print('Quality Trends:', quality_trends)

# Generate alerts for quality issues
quality_alerts = dashboard.check_quality_alerts()
print('Quality Alerts:', quality_alerts)
"
```

### Automated Quality Reporting
```bash
# Generate weekly quality report
python -c "
from src.reporting.quality_reporter import QualityReporter
reporter = QualityReporter()

# Generate comprehensive weekly report
weekly_report = reporter.generate_weekly_report()
print(weekly_report)

# Export quality metrics
reporter.export_quality_metrics('quality-metrics.json')
reporter.export_quality_dashboard('quality-dashboard.html')
"
```

## Quality Gate Commands

### Development Workflow Integration
```bash
# Pre-development quality check
make quality-check-pre

# Post-development quality validation
make quality-check-post

# Release readiness validation
make quality-check-release

# Emergency hotfix quality validation
make quality-check-hotfix
```

### Makefile Integration
```makefile
# Makefile quality targets
.PHONY: quality-check-pre quality-check-post quality-check-release

quality-check-pre:
	ruff check src/ tests/
	mypy src/ --strict
	pytest tests/unit/ --cov=src --cov-fail-under=90

quality-check-post:
	pytest tests/ --cov=src --cov-fail-under=95
	bandit -r src/
	safety check
	pytest tests/performance/ --benchmark-only

quality-check-release:
	pytest tests/ --cov=src --cov-fail-under=95
	mutmut run --paths-to-mutate=src/
	pytest tests/integration/ -v
	pytest tests/validation/ -v
	bandit -r src/ -f json -o security-report.json
	safety check --json --output deps-security.json
```