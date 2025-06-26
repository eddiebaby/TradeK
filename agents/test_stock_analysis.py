#!/usr/bin/env python3
"""
Stock Analysis Integration Test

Tests the CrewAI-inspired stock analysis capabilities integrated into 
the RESEARCHER agent with Ollama local model optimization.
"""
import asyncio
import sys
import json
import time
from pathlib import Path

# Add agents directory to path
sys.path.append(str(Path(__file__).parent))

sys.path.append(str(Path(__file__).parent / "researcher"))
from stock_analysis_extension import StockAnalysisResearcher, StockAnalysisDomain
from core.agent_base import TaskContext

async def test_comprehensive_stock_analysis():
    """Test comprehensive stock analysis functionality."""
    
    print("🧪 Testing Comprehensive Stock Analysis")
    print("=" * 50)
    
    researcher = StockAnalysisResearcher()
    
    # Test analysis specification
    test_spec = {
        "ticker_symbol": "AAPL",
        "domains": [
            "financial_metrics",
            "market_sentiment", 
            "sec_filings",
            "technical_analysis",
            "competitive_analysis",
            "risk_assessment"
        ],
        "depth": "comprehensive",
        "time_horizon": "medium_term",
        "context": {"test_run": True, "integration_test": True},
        "priority": 1
    }
    
    print(f"📊 Testing analysis for: {test_spec['ticker_symbol']}")
    print(f"🔍 Domains: {len(test_spec['domains'])} analysis areas")
    print(f"⏱️  Depth: {test_spec['depth']}")
    
    start_time = time.time()
    
    try:
        # Run comprehensive analysis
        result = await researcher.analyze_stock(test_spec)
        
        analysis_time = time.time() - start_time
        
        print(f"\n✅ Analysis completed in {analysis_time:.2f} seconds")
        print("-" * 50)
        
        # Validate results structure
        assert result.ticker_symbol == "AAPL"
        assert result.analysis_id is not None
        assert result.confidence_score > 0
        
        print("🎯 Validation Results:")
        print(f"   ✓ Ticker Symbol: {result.ticker_symbol}")
        print(f"   ✓ Analysis ID: {result.analysis_id[:20]}...")
        print(f"   ✓ Confidence Score: {result.confidence_score:.2%}")
        
        # Test financial metrics
        metrics = result.financial_metrics
        print(f"\n💰 Financial Metrics Test:")
        print(f"   ✓ P/E Ratio: {metrics.pe_ratio}")
        print(f"   ✓ EPS Growth: {metrics.eps_growth}%")
        print(f"   ✓ ROE: {metrics.roe}%")
        print(f"   ✓ Debt/Equity: {metrics.debt_to_equity}")
        
        # Test market sentiment
        sentiment = result.market_sentiment
        print(f"\n📊 Market Sentiment Test:")
        print(f"   ✓ Overall Sentiment: {sentiment.get('overall_sentiment', 'N/A')}")
        print(f"   ✓ Sentiment Score: {sentiment.get('sentiment_score', 0):.2f}")
        print(f"   ✓ Analyst Ratings Available: {bool(sentiment.get('analyst_ratings'))}")
        
        # Test investment recommendation
        recommendation = result.investment_recommendation
        print(f"\n🎯 Investment Recommendation Test:")
        print(f"   ✓ Recommendation: {recommendation.get('recommendation', 'N/A')}")
        print(f"   ✓ Overall Score: {recommendation.get('overall_score', 0):.1f}/10")
        print(f"   ✓ Target Price: ${recommendation.get('target_price', 'N/A')}")
        print(f"   ✓ Action Plan: {recommendation.get('action', 'N/A')[:50]}...")
        
        # Test risk assessment
        risk = result.risk_assessment
        print(f"\n⚠️ Risk Assessment Test:")
        print(f"   ✓ Overall Risk Score: {risk.get('overall_risk_score', 0):.1f}/10")
        print(f"   ✓ Risk Categories: {len(risk.get('risk_categories', {}))}")
        print(f"   ✓ Risk Mitigation: {len(risk.get('risk_mitigation', []))} strategies")
        
        # Test formatting for other agents
        print(f"\n🔄 Agent Integration Test:")
        
        # Test strategy formatting
        strategy_format = await researcher.format_stock_analysis_for_strategy(result)
        print(f"   ✓ Strategy Format: {len(strategy_format)} sections")
        assert "strategic_investment_insights" in strategy_format
        assert "portfolio_strategy" in strategy_format
        
        # Test implementation formatting
        impl_format = await researcher.format_stock_analysis_for_implementation(result)
        print(f"   ✓ Implementation Format: {len(impl_format)} sections")
        assert "implementation_guidance" in impl_format
        assert "monitoring_specifications" in impl_format
        
        print(f"\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_domain_specific_analysis():
    """Test individual domain analysis capabilities."""
    
    print("\n🔬 Testing Domain-Specific Analysis")
    print("=" * 50)
    
    researcher = StockAnalysisResearcher()
    
    # Test each domain individually
    domains_to_test = [
        ("financial_metrics", "Financial metrics calculation and analysis"),
        ("market_sentiment", "News and sentiment aggregation"),
        ("risk_assessment", "Multi-factor risk evaluation")
    ]
    
    for domain, description in domains_to_test:
        print(f"\n🎯 Testing {domain}")
        print(f"   Description: {description}")
        
        test_spec = {
            "ticker_symbol": "MSFT",
            "domains": [domain],
            "depth": "standard",
            "time_horizon": "short_term",
            "context": {"domain_test": domain},
            "priority": 1
        }
        
        try:
            result = await researcher.analyze_stock(test_spec)
            print(f"   ✅ {domain} analysis completed")
            print(f"   📊 Confidence: {result.confidence_score:.2%}")
            
        except Exception as e:
            print(f"   ❌ {domain} test failed: {e}")
            return False
    
    print(f"\n✅ All domain tests completed successfully!")
    return True

async def test_performance_benchmarks():
    """Test performance and timing benchmarks."""
    
    print("\n⏱️ Testing Performance Benchmarks")
    print("=" * 50)
    
    researcher = StockAnalysisResearcher()
    
    # Performance test scenarios
    scenarios = [
        {"name": "Quick Analysis", "depth": "quick", "domains": 2},
        {"name": "Standard Analysis", "depth": "standard", "domains": 4},
        {"name": "Comprehensive Analysis", "depth": "comprehensive", "domains": 6}
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n🚀 {scenario['name']} Performance Test")
        
        test_spec = {
            "ticker_symbol": "GOOGL",
            "domains": ["financial_metrics", "market_sentiment", "risk_assessment"][:scenario["domains"]],
            "depth": scenario["depth"],
            "time_horizon": "medium_term",
            "context": {"performance_test": True},
            "priority": 1
        }
        
        start_time = time.time()
        
        try:
            result = await researcher.analyze_stock(test_spec)
            elapsed_time = time.time() - start_time
            
            performance_result = {
                "scenario": scenario["name"],
                "elapsed_time": elapsed_time,
                "confidence": result.confidence_score,
                "domains_analyzed": len(test_spec["domains"]),
                "success": True
            }
            
            results.append(performance_result)
            
            print(f"   ⏱️ Elapsed Time: {elapsed_time:.2f} seconds")
            print(f"   📊 Confidence: {result.confidence_score:.2%}")
            print(f"   🎯 Status: ✅ Success")
            
        except Exception as e:
            performance_result = {
                "scenario": scenario["name"],
                "elapsed_time": 0,
                "confidence": 0,
                "domains_analyzed": 0,
                "success": False,
                "error": str(e)
            }
            
            results.append(performance_result)
            print(f"   🎯 Status: ❌ Failed - {e}")
    
    # Performance summary
    print(f"\n📈 Performance Summary:")
    print("-" * 30)
    
    successful_tests = [r for r in results if r["success"]]
    if successful_tests:
        avg_time = sum(r["elapsed_time"] for r in successful_tests) / len(successful_tests)
        avg_confidence = sum(r["confidence"] for r in successful_tests) / len(successful_tests)
        
        print(f"   Average Analysis Time: {avg_time:.2f} seconds")
        print(f"   Average Confidence: {avg_confidence:.2%}")
        print(f"   Success Rate: {len(successful_tests)}/{len(results)} ({len(successful_tests)/len(results)*100:.1f}%)")
    else:
        print("   No successful tests to analyze")
    
    return len(successful_tests) == len(results)

async def test_integration_with_existing_agent():
    """Test integration with existing RESEARCHER agent capabilities."""
    
    print("\n🔗 Testing Integration with Existing Agent")
    print("=" * 50)
    
    researcher = StockAnalysisResearcher()
    
    # Test that stock capabilities are added to existing capabilities
    capabilities = researcher.get_capabilities()
    
    stock_capabilities = [
        "stock_financial_analysis",
        "market_sentiment_research", 
        "sec_filing_analysis",
        "technical_pattern_recognition",
        "competitive_intelligence",
        "investment_risk_assessment"
    ]
    
    print(f"📋 Checking capabilities integration:")
    
    for capability in stock_capabilities:
        if capability in capabilities:
            print(f"   ✅ {capability}")
        else:
            print(f"   ❌ {capability} - Missing!")
            return False
    
    # Test research modes integration
    research_modes = researcher.get_research_modes()
    
    stock_modes = [
        "financial_deep_dive",
        "market_intelligence",
        "regulatory_analysis",
        "technical_research",
        "competitive_research",
        "risk_intelligence"
    ]
    
    print(f"\n🎯 Checking research modes integration:")
    
    for mode in stock_modes:
        if mode in research_modes:
            print(f"   ✅ {mode}")
        else:
            print(f"   ❌ {mode} - Missing!")
            return False
    
    print(f"\n✅ Integration test passed!")
    return True

async def run_all_tests():
    """Run all stock analysis tests."""
    
    print("🧪 Stock Analysis Integration Test Suite")
    print("=" * 60)
    print("Testing CrewAI-inspired multi-agent stock analysis")
    print("Optimized for Ollama local models")
    print("=" * 60)
    
    test_results = []
    
    # Run all test suites
    tests = [
        ("Integration Test", test_integration_with_existing_agent),
        ("Domain-Specific Test", test_domain_specific_analysis),
        ("Comprehensive Analysis Test", test_comprehensive_stock_analysis),
        ("Performance Benchmark Test", test_performance_benchmarks)
    ]
    
    for test_name, test_function in tests:
        print(f"\n🔍 Running {test_name}...")
        
        try:
            result = await test_function()
            test_results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            test_results.append((test_name, False))
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 Test Suite Summary")
    print("-" * 30)
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print("-" * 30)
    print(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    print(f"🎯 Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Stock analysis integration is ready.")
    else:
        print("\n⚠️ Some tests failed. Review implementation before deployment.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Test suite interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test suite error: {e}")
        sys.exit(1)