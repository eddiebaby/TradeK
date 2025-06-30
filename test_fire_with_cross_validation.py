#!/usr/bin/env python3
"""
Test FIRE command with OpenAI cross-validation
"""

import sys
import json
from datetime import datetime, timezone
from fire_cross_validation import FireCrossValidator

def test_complete_workflow():
    """Test the complete FIRE workflow with cross-validation"""
    
    print("🔥 Testing FIRE Command with OpenAI Cross-Validation")
    print("=" * 60)
    
    # Initialize the cross-validator
    validator = FireCrossValidator()
    
    if not validator.available:
        print("❌ Cross-validation not available - missing API keys")
        return False
    
    print(f"✅ Cross-validation available using OpenAI models")
    
    # Simulate SPARC trio outputs (similar to what FIRE command would produce)
    mock_sparc_results = {
        "RESEARCHER": """
        Research Analysis: Stock Market Trading Algorithm
        
        Based on comprehensive analysis of market data and algorithmic trading patterns,
        I recommend implementing a momentum-based strategy with the following components:
        
        1. Technical Indicators: RSI, MACD, Bollinger Bands
        2. Risk Management: 2% max risk per trade, stop-loss at 1.5%
        3. Market Conditions: Focus on trending markets, avoid choppy conditions
        4. Timeframes: 5-minute and 15-minute charts for entry signals
        
        The research indicates this approach has a 65% win rate in backtesting
        with a Sharpe ratio of 1.8 over the past 3 years.
        """,
        
        "MASTERMIND": """
        Strategic Architecture: Trading System Design
        
        System Architecture Recommendations:
        
        1. Modular Design: Separate data ingestion, signal generation, and execution
        2. Microservices: Independent services for each trading strategy
        3. Database: InfluxDB for time-series data, PostgreSQL for metadata
        4. API Gateway: Rate limiting and authentication for external connections
        5. Risk Engine: Real-time position monitoring and automatic cutoffs
        
        Technology Stack:
        - Python with asyncio for concurrent processing
        - FastAPI for REST endpoints
        - Redis for caching and pub/sub
        - Docker for containerization
        - Kubernetes for orchestration
        
        Expected performance: <50ms latency, 99.9% uptime, handle 1000+ requests/second
        """,
        
        "EXECUTOR": """
        Implementation Plan: Trading Algorithm TDD Development
        
        Phase 1: Core Infrastructure (Week 1-2)
        - Set up FastAPI application with proper logging
        - Implement database models and migrations
        - Create authentication and authorization middleware
        - Build market data ingestion pipeline
        
        Phase 2: Strategy Engine (Week 3-4)
        - Implement technical indicator calculations
        - Create signal generation logic with proper validation
        - Build risk management components
        - Add comprehensive unit tests (target: 95% coverage)
        
        Phase 3: Trading Execution (Week 5-6)
        - Integrate with broker APIs (Schwab, Interactive Brokers)
        - Implement order management system
        - Add real-time monitoring and alerting
        - Create integration tests and stress testing
        
        Quality Gates:
        - All tests must pass before deployment
        - Code coverage minimum 90%
        - Performance tests show <50ms response time
        - Security scan with zero critical vulnerabilities
        """
    }
    
    print("\n🤖 SPARC Trio Analysis Complete")
    print("📊 Running Cross-Validation with OpenAI o4-mini...")
    
    # Run cross-validation on the complete workflow
    validation_results = validator.validate_fire_output(mock_sparc_results)
    
    print("\n" + "=" * 60)
    print("🔍 CROSS-VALIDATION RESULTS")
    print("=" * 60)
    
    # Display summary
    summary = validation_results["cross_validation_summary"]
    print(f"Status: {'✅ ENABLED' if summary['enabled'] else '❌ DISABLED'}")
    print(f"Model Used: {summary.get('model_used', 'N/A')}")
    print(f"Timestamp: {summary.get('timestamp', 'N/A')}")
    print(f"Cost Estimate: {summary.get('total_cost_estimate', 'N/A')}")
    
    if summary['enabled']:
        print(f"Overall Score: {summary.get('overall_score', 'N/A')}/10")
        print(f"Quality Grade: {summary.get('quality_grade', 'N/A')}")
    
    # Display individual agent validations
    print(f"\n📋 Individual Agent Validations:")
    for agent_name, validation in validation_results.get("agent_validations", {}).items():
        print(f"\n{agent_name}:")
        print(f"  Status: {validation.get('status', 'unknown')}")
        if validation.get('status') == 'success':
            print(f"  Score: {validation.get('validation_score', 'N/A')}/10")
            print(f"  Feedback: {validation.get('feedback', 'No feedback')}")
        else:
            print(f"  Error: {validation.get('message', 'Unknown error')}")
    
    # Success criteria
    success_criteria = [
        summary.get('enabled', False),
        summary.get('overall_score', 0) >= 7.0,
        len(validation_results.get("agent_validations", {})) == 3
    ]
    
    overall_success = all(success_criteria)
    
    print(f"\n{'✅' if overall_success else '❌'} Overall Test Result: {'PASS' if overall_success else 'FAIL'}")
    
    if overall_success:
        print("\n🎉 FIRE Command with Cross-Validation is WORKING!")
        print("💡 You now have:")
        print("   - OpenAI o4-mini cost-effective cross-validation")
        print("   - Real-time quality scoring of SPARC trio outputs") 
        print("   - Integrated feedback for continuous improvement")
        print("   - Estimated cost: ~$0.0003 per full validation")
    
    return overall_success

if __name__ == "__main__":
    success = test_complete_workflow()
    sys.exit(0 if success else 1)