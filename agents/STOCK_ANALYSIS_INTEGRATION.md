# Stock Analysis Integration Guide

## CrewAI-Inspired Multi-Agent Stock Analysis for RESEARCHER Agent

This integration extends the existing RESEARCHER agent with comprehensive stock analysis capabilities inspired by CrewAI's multi-agent framework, optimized for Ollama local models.

## Overview

### Key Components Analyzed from CrewAI Example

1. **Multi-Agent Architecture**: Financial Analyst, Research Analyst, Investment Advisor, and SEC Specialist agents
2. **Data Sources**: SEC API, web scraping, news aggregation, financial metrics
3. **Analysis Methodologies**: Sequential task processing with agent specialization
4. **Output Formats**: Structured investment recommendations with supporting evidence
5. **Local Model Integration**: Optimized for Ollama with llama3.1

### Integration Benefits

- 🔍 **Comprehensive Analysis**: Multi-domain stock research covering financials, sentiment, technical, and risk factors
- 🤖 **Local AI Models**: Uses Ollama for privacy and cost-effective analysis
- 🏗️ **Modular Design**: Extends existing RESEARCHER agent without disruption
- 📊 **Structured Output**: Consistent investment recommendations with confidence scoring
- 🔄 **Agent Integration**: Seamlessly works with MASTERMIND and EXECUTOR agents

## Architecture

### Extended RESEARCHER Agent Capabilities

```python
# New stock-specific capabilities added to existing RESEARCHER agent
stock_capabilities = [
    "stock_financial_analysis",
    "market_sentiment_research", 
    "sec_filing_analysis",
    "technical_pattern_recognition",
    "competitive_intelligence",
    "investment_risk_assessment"
]
```

### Analysis Domains

1. **Financial Metrics Analysis**
   - P/E ratio, EPS growth, ROE, debt ratios
   - Revenue and profit margin analysis
   - Cash flow and balance sheet strength

2. **Market Sentiment Research**
   - News sentiment aggregation
   - Analyst ratings synthesis
   - Social media monitoring
   - Options flow analysis

3. **SEC Filing Analysis**
   - 10-K annual report deep dive
   - 10-Q quarterly analysis
   - Management discussion interpretation
   - Risk factor assessment

4. **Technical Analysis**
   - Price pattern recognition
   - Technical indicators (RSI, MACD, moving averages)
   - Support/resistance levels
   - Volume analysis

5. **Competitive Intelligence**
   - Market position assessment
   - Peer comparison analysis
   - Competitive advantages identification
   - Industry trend analysis

6. **Investment Risk Assessment**
   - Multi-factor risk scoring
   - Portfolio allocation recommendations
   - Risk mitigation strategies
   - Stress testing scenarios

## Installation and Setup

### 1. Install Additional Dependencies

```bash
cd /home/scottschweizer/TradeKnowledge/agents/researcher
pip install -r requirements_stock_analysis.txt
```

### 2. Configure Environment Variables

Create or update your `.env` file:

```bash
# SEC API (required for filing analysis)
SEC_API_KEY=your_sec_api_key_here

# Financial Data API (optional but recommended)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
FINANCIAL_DATA_API_KEY=your_financial_api_key_here

# News API (for sentiment analysis)
NEWS_API_KEY=your_news_api_key_here

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

### 3. Verify Ollama Setup

Ensure Ollama is running with the required model:

```bash
# Start Ollama service
ollama serve

# Pull required model
ollama pull llama3.1

# Test model availability
curl http://localhost:11434/api/tags
```

## Usage

### Basic Stock Analysis

```bash
# Interactive stock analysis
cd /home/scottschweizer/TradeKnowledge/agents
python ask_stock_researcher.py

# Quick demo
python ask_stock_researcher.py --demo
```

### Programmatic Usage

```python
from researcher.stock_analysis_extension import StockAnalysisResearcher

# Initialize researcher
researcher = StockAnalysisResearcher()

# Define analysis specification
analysis_spec = {
    "ticker_symbol": "AAPL",
    "domains": [
        "financial_metrics",
        "market_sentiment", 
        "sec_filings",
        "technical_analysis",
        "risk_assessment"
    ],
    "depth": "comprehensive",
    "time_horizon": "medium_term",
    "context": {"user_query": "Investment analysis for AAPL"},
    "priority": 1
}

# Conduct analysis
result = await researcher.analyze_stock(analysis_spec)

# Access results
print(f"Recommendation: {result.investment_recommendation['recommendation']}")
print(f"Overall Score: {result.investment_recommendation['overall_score']}/10")
print(f"Risk Score: {result.risk_assessment['overall_risk_score']}/10")
```

### Integration with Other Agents

#### For MASTERMIND Strategic Planning

```python
# Format results for strategic planning
strategy_format = await researcher.format_stock_analysis_for_strategy(result)

# Strategic insights include:
# - Architecture recommendations
# - Technology evaluation
# - Risk assessment
# - Trend implications
# - Competitive landscape
```

#### For EXECUTOR Implementation

```python
# Format results for implementation
impl_format = await researcher.format_stock_analysis_for_implementation(result)

# Implementation guidance includes:
# - Execution strategy
# - Position sizing
# - Risk controls
# - Monitoring specifications
# - Automation opportunities
```

## Analysis Output Structure

### Comprehensive Stock Analysis Result

```python
StockAnalysisResult {
    analysis_id: str
    ticker_symbol: str
    financial_metrics: FinancialMetrics
    market_sentiment: Dict[str, Any]
    sec_insights: List[Dict[str, Any]]
    technical_signals: Dict[str, Any]
    competitive_position: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    investment_recommendation: Dict[str, Any]
    confidence_score: float
    analysis_timestamp: float
}
```

### Investment Recommendation Format

```python
investment_recommendation = {
    "recommendation": "Buy | Hold | Sell",
    "action": "Specific action to take",
    "overall_score": 7.5,  # 1-10 scale
    "target_price": 425.50,
    "time_horizon": "6-12 months",
    "portfolio_allocation": {
        "recommended_allocation": "2-3%",
        "risk_level": "Moderate",
        "position_sizing": "Conservative approach"
    },
    "key_catalysts": ["Earnings", "Product launches", "..."],
    "exit_conditions": ["Score < 5.0", "Risk > 8.0", "..."],
    "confidence": 0.92
}
```

## Performance Optimization

### Ollama Configuration for Financial Analysis

```yaml
ollama:
  model: "llama3.1"
  temperature: 0.1  # Low temperature for consistent analysis
  max_tokens: 2048
  timeout: 120
  context_length: 4096
```

### Analysis Depth Options

1. **Quick Analysis** (~30 seconds)
   - Financial metrics + market sentiment
   - Basic recommendation

2. **Standard Analysis** (~60 seconds)
   - Financial + sentiment + technical + risk
   - Comprehensive recommendation

3. **Comprehensive Analysis** (~120 seconds)
   - All domains + competitive + SEC filings
   - Full investment thesis

## Testing

### Run Integration Tests

```bash
cd /home/scottschweizer/TradeKnowledge/agents
python test_stock_analysis.py
```

### Test Scenarios Covered

1. ✅ **Integration Test**: Verify capabilities are added to existing agent
2. ✅ **Domain-Specific Test**: Test individual analysis domains
3. ✅ **Comprehensive Analysis Test**: Full multi-domain analysis
4. ✅ **Performance Benchmark Test**: Timing and efficiency tests

## API Integration Guide

### SEC API Setup

1. Sign up at [sec-api.io](https://sec-api.io)
2. Get API key from dashboard
3. Add to environment variables
4. Test with sample ticker

### Financial Data APIs (Optional)

Choose one or more:

- **Alpha Vantage**: Free tier available, good for basic metrics
- **IEX Cloud**: Comprehensive financial data, paid plans
- **Polygon.io**: Real-time and historical data
- **Yahoo Finance (yfinance)**: Free but rate-limited

### News Sentiment APIs

- **NewsAPI**: Free tier for news aggregation
- **Polygon News**: Financial news with sentiment
- **Alpha Vantage News**: Integrated with financial data

## Configuration Management

### Domain-Specific Configuration

```yaml
# stock_analysis_config.yaml
domains:
  financial_metrics:
    priority: high
    confidence_threshold: 0.85
    key_metrics: [pe_ratio, eps_growth, roe, debt_to_equity]
  
  market_sentiment:
    priority: high
    confidence_threshold: 0.80
    data_sources: [news_sentiment, analyst_ratings, social_media]
```

### Risk Management Parameters

```yaml
risk_management:
  max_position_size: 0.05  # 5% of portfolio
  stop_loss_threshold: 0.15  # 15% loss
  profit_taking_threshold: 0.25  # 25% gain
  rebalancing_frequency: "monthly"
```

## Best Practices

### 1. Data Quality Management

- Validate data freshness (< 24 hours for market data)
- Cross-validate across multiple sources
- Implement outlier detection
- Use confidence scoring

### 2. Risk Management

- Always include risk assessment
- Set position size limits
- Implement stop-loss conditions
- Regular portfolio rebalancing

### 3. Performance Monitoring

- Track analysis accuracy over time
- Monitor model performance
- Adjust confidence thresholds
- Optimize for speed vs. accuracy

### 4. Compliance and Disclaimers

- ⚠️ **Not Investment Advice**: Analysis is for research purposes only
- 📋 **Regulatory Compliance**: Ensure compliance with local regulations
- 🔒 **Data Privacy**: Use local models to protect sensitive information
- 📝 **Documentation**: Maintain audit trail of recommendations

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check Ollama service
   curl http://localhost:11434/api/tags
   
   # Restart if needed
   ollama serve
   ```

2. **SEC API Rate Limits**
   ```python
   # Implement rate limiting
   import asyncio
   await asyncio.sleep(0.1)  # 10 requests per second max
   ```

3. **Model Performance Issues**
   ```yaml
   # Adjust Ollama configuration
   temperature: 0.1  # More deterministic
   max_tokens: 1024  # Reduce for speed
   ```

4. **Memory Usage**
   ```python
   # Clear analysis history periodically
   researcher.research_history = researcher.research_history[-10:]
   ```

### Error Handling

The system includes comprehensive error handling:

- API timeouts and retries
- Missing data graceful degradation
- Confidence scoring for reliability
- Fallback to cached data when available

## Future Enhancements

### Planned Features

1. **Real-time Monitoring**: Continuous stock tracking with alerts
2. **Portfolio Management**: Multi-stock portfolio analysis
3. **Backtesting**: Historical performance validation
4. **Advanced Models**: Integration with specialized financial models
5. **Custom Indicators**: User-defined technical indicators

### Integration Opportunities

1. **Trading Platforms**: API integration for automated trading
2. **Portfolio Tools**: Export to portfolio management software
3. **Reporting**: Automated report generation
4. **Alerts**: Email/SMS notifications for key events

## Conclusion

This integration successfully brings CrewAI-inspired multi-agent stock analysis capabilities to your existing RESEARCHER agent, providing:

- 🔍 **Comprehensive Analysis**: 6 specialized analysis domains
- 🤖 **Local AI Processing**: Privacy-focused with Ollama
- 🏗️ **Seamless Integration**: Works with existing agent architecture
- 📊 **Structured Output**: Consistent, actionable recommendations
- 🔄 **Agent Coordination**: Ready for MASTERMIND and EXECUTOR handoffs

The system is production-ready with comprehensive testing, error handling, and performance optimization for reliable financial research and analysis.

## Support and Maintenance

For questions or issues:

1. Check the test suite results
2. Review configuration files
3. Validate API keys and permissions
4. Monitor Ollama performance
5. Check system logs for detailed error information

The integration maintains the high-quality standards of your existing agent system while adding powerful new capabilities for financial intelligence gathering and investment research.