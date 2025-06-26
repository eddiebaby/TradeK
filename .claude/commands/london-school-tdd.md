# London School TDD for Financial Systems

Implement Test-Driven Development using London School (Mockist) approach optimized for financial trading systems.

## Core Principles

### London School TDD Characteristics
- **Interaction-based testing**: Focus on behavior and object collaboration
- **Heavy mocking**: Mock all dependencies and external services  
- **Outside-in development**: Start from API/UI and work inward
- **Fast feedback loops**: Isolated, fast unit tests

### Financial System Applications
- **Trading Algorithm Testing**: Mock market data feeds, test decision logic
- **API Integration Testing**: Mock external APIs (Schwab, IEX, Polygon)
- **Risk Management Testing**: Mock market conditions, test risk calculations
- **Event-Driven Testing**: Test event publishing/subscribing in agent systems

## Implementation Workflow

### 1. Red Phase - Write Failing Test
```python
# Example: Test trading decision interaction
def test_trading_algorithm_makes_buy_decision_when_signals_align():
    # Arrange - Mock all dependencies
    market_data_mock = Mock(spec=MarketDataService)
    risk_calculator_mock = Mock(spec=RiskCalculator) 
    portfolio_mock = Mock(spec=Portfolio)
    
    # Configure mocks for expected behavior
    market_data_mock.get_latest_price.return_value = 150.0
    risk_calculator_mock.calculate_position_size.return_value = 100
    portfolio_mock.can_afford.return_value = True
    
    algorithm = TradingAlgorithm(market_data_mock, risk_calculator_mock, portfolio_mock)
    
    # Act
    decision = algorithm.evaluate_buy_signal("AAPL")
    
    # Assert - Verify interactions occurred
    market_data_mock.get_latest_price.assert_called_once_with("AAPL")
    risk_calculator_mock.calculate_position_size.assert_called_once_with("AAPL", 150.0)
    portfolio_mock.can_afford.assert_called_once_with(15000.0)  # 100 shares * $150
    
    assert decision.action == "BUY"
    assert decision.quantity == 100
```

### 2. Green Phase - Make Test Pass
```python
class TradingAlgorithm:
    def __init__(self, market_data, risk_calculator, portfolio):
        self._market_data = market_data
        self._risk_calculator = risk_calculator  
        self._portfolio = portfolio
    
    def evaluate_buy_signal(self, symbol):
        price = self._market_data.get_latest_price(symbol)
        position_size = self._risk_calculator.calculate_position_size(symbol, price)
        total_cost = position_size * price
        
        if self._portfolio.can_afford(total_cost):
            return TradingDecision("BUY", position_size)
        
        return TradingDecision("HOLD", 0)
```

### 3. Refactor Phase - Improve Design
- Extract command patterns for trading decisions
- Introduce strategy patterns for different algorithms
- Add builder patterns for complex financial instruments

## Testing Patterns for Financial Systems

### Market Data Integration Testing
```python
def test_market_data_processor_handles_price_updates():
    # Mock external data feed
    data_feed_mock = Mock(spec=MarketDataFeed)
    price_validator_mock = Mock(spec=PriceValidator)
    event_publisher_mock = Mock(spec=EventPublisher)
    
    processor = MarketDataProcessor(data_feed_mock, price_validator_mock, event_publisher_mock)
    
    # Simulate price update
    price_data = {"symbol": "AAPL", "price": 150.0, "timestamp": datetime.now()}
    processor.process_price_update(price_data)
    
    # Verify interactions
    price_validator_mock.validate.assert_called_once_with(price_data)
    event_publisher_mock.publish.assert_called_once_with("price_updated", price_data)
```

### Risk Management Testing  
```python
def test_risk_manager_prevents_excessive_position_size():
    portfolio_mock = Mock(spec=Portfolio)
    risk_rules_mock = Mock(spec=RiskRules)
    
    # Setup portfolio constraints
    portfolio_mock.get_total_value.return_value = 100000.0
    risk_rules_mock.max_position_percent.return_value = 0.05  # 5% max
    
    risk_manager = RiskManager(portfolio_mock, risk_rules_mock)
    
    # Test position size calculation
    max_position = risk_manager.calculate_max_position("AAPL", 150.0)
    
    # Verify risk rules were consulted
    portfolio_mock.get_total_value.assert_called_once()
    risk_rules_mock.max_position_percent.assert_called_once()
    
    assert max_position == 33  # $5000 / $150 = 33 shares (rounded down)
```

## Quality Gates

### Test Coverage Requirements
- **Unit Tests**: 95% coverage minimum for financial calculations
- **Integration Tests**: 90% coverage for API interactions
- **Mutation Testing**: 85% mutation score for risk management logic

### Performance Requirements
- **Unit Tests**: < 10ms execution time each
- **Integration Tests**: < 100ms execution time each  
- **Test Suite**: Complete run in < 30 seconds

### Security Testing
- **Input Validation**: All financial inputs validated and sanitized
- **API Security**: All external API calls include authentication and rate limiting
- **Data Protection**: Sensitive financial data mocked in tests, never real data

## Commands

### Run London School TDD Workflow
```bash
# Start TDD session with financial system focus
cd /home/scott/TradeKnowledge

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-fail-under=95

# Run mutation testing
mutmut run --paths-to-mutate=src/

# Performance testing
pytest tests/ --benchmark-only

# Security testing
bandit -r src/ -f json -o security-report.json
```

### Mock Generation Helpers
```bash
# Generate mocks for financial services
python -c "
from unittest.mock import Mock, create_autospec
from src.services import MarketDataService, RiskCalculator

# Auto-generate spec-based mocks
market_data_mock = create_autospec(MarketDataService, spec_set=True)
risk_calc_mock = create_autospec(RiskCalculator, spec_set=True)

print('Generated mocks with full interface specifications')
"
```

## Best Practices

1. **Mock External Dependencies**: Always mock APIs, databases, file systems
2. **Test Behavior, Not Implementation**: Focus on what the code does, not how
3. **One Assertion Per Test**: Keep tests focused and clear
4. **Descriptive Test Names**: Use business language in test names
5. **Arrange-Act-Assert**: Structure tests clearly
6. **Fast Feedback**: Tests should run quickly and provide immediate feedback