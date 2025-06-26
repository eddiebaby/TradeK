#!/usr/bin/env python3
"""
Stock Analysis Demo - Quick demonstration of agent capabilities for stock analysis
"""

import asyncio
import sys
from pathlib import Path

# Fix imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "researcher"))

from researcher.stock_analysis_extension import StockAnalysisResearcher, StockAnalysisDomain
from core.agent_base import TaskContext

async def quick_stock_demo(ticker="AAPL"):
    """Quick stock analysis demonstration."""
    
    print(f"📈 STOCK ANALYSIS DEMO - {ticker}")
    print("=" * 50)
    
    try:
        # Create the enhanced researcher
        researcher = StockAnalysisResearcher()
        print(f"✅ Created enhanced RESEARCHER agent with stock capabilities")
        
        # Show stock analysis capabilities
        capabilities = researcher.get_capabilities()
        stock_capabilities = [cap for cap in capabilities if 'stock' in cap.lower()]
        
        print(f"\n🎯 Stock-specific capabilities:")
        for cap in stock_capabilities:
            print(f"   • {cap}")
        
        # Create a simple analysis request
        analysis_spec = {
            "ticker_symbol": ticker,
            "domains": ["financial_analysis", "market_intelligence", "risk_evaluation"],  # Use correct enum values
            "depth": "standard",
            "time_horizon": "medium_term",
            "context": {"demo": True, "user_request": True},
            "priority": 1
        }
        
        print(f"\n🔍 Analyzing {ticker} with domains:")
        for domain in analysis_spec["domains"]:
            print(f"   • {domain}")
        
        # Run the analysis
        print(f"\n⚡ Starting analysis...")
        result = await researcher.analyze_stock(analysis_spec)
        
        print(f"\n✅ Analysis completed!")
        print(f"📊 Ticker: {result.ticker_symbol}")
        print(f"🎯 Confidence Score: {result.confidence_score:.2%}")
        print(f"📈 Analysis ID: {result.analysis_id[:20]}...")
        
        # Show financial metrics if available
        if hasattr(result, 'financial_metrics') and result.financial_metrics:
            metrics = result.financial_metrics
            print(f"\n💰 Financial Metrics:")
            print(f"   • P/E Ratio: {getattr(metrics, 'pe_ratio', 'N/A')}")
            print(f"   • EPS Growth: {getattr(metrics, 'eps_growth', 'N/A')}%")
            print(f"   • ROE: {getattr(metrics, 'roe', 'N/A')}%")
            print(f"   • Debt/Equity: {getattr(metrics, 'debt_to_equity', 'N/A')}")
        
        # Show investment recommendation if available
        if hasattr(result, 'investment_recommendation') and result.investment_recommendation:
            rec = result.investment_recommendation
            print(f"\n🎯 Investment Recommendation:")
            print(f"   • Recommendation: {rec.get('recommendation', 'N/A')}")
            print(f"   • Overall Score: {rec.get('overall_score', 0):.1f}/10")
            print(f"   • Target Price: ${rec.get('target_price', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def show_available_domains():
    """Show available stock analysis domains."""
    
    print("\n🔬 AVAILABLE STOCK ANALYSIS DOMAINS")
    print("=" * 50)
    
    print("Available domains for analysis:")
    for domain in StockAnalysisDomain:
        print(f"   • {domain.name}: {domain.value}")
        
    print(f"\nYou can analyze stocks using any combination of these domains.")
    print(f"Example: financial_analysis + market_intelligence + risk_evaluation")

async def trio_stock_workflow_demo():
    """Demonstrate how the trio would work together for stock analysis."""
    
    print(f"\n🤖 TRIO WORKFLOW FOR STOCK ANALYSIS")
    print("=" * 50)
    
    print("How the agent trio collaborates for stock analysis:")
    print()
    
    print("1. 🔍 RESEARCHER Agent:")
    print("   • Gathers financial data and market intelligence")
    print("   • Analyzes SEC filings and technical indicators")
    print("   • Provides risk assessment and competitive analysis")
    print()
    
    print("2. 🧠 MASTERMIND Agent:")
    print("   • Creates strategic investment thesis")
    print("   • Designs portfolio allocation strategy")
    print("   • Plans risk management approach")
    print()
    
    print("3. ⚡ EXECUTOR Agent:")
    print("   • Implements trading algorithms")
    print("   • Sets up monitoring and alerts")
    print("   • Creates automated testing framework")
    print()
    
    print("💡 This creates a complete research → strategy → implementation pipeline!")

def show_usage_options():
    """Show different ways to use the stock analysis system."""
    
    print(f"\n🚀 USAGE OPTIONS FOR STOCK ANALYSIS")
    print("=" * 50)
    
    print("OPTION 1: 🐍 Direct Python (what we just demonstrated)")
    print("   ```python")
    print("   from researcher.stock_analysis_extension import StockAnalysisResearcher")
    print("   researcher = StockAnalysisResearcher()")
    print("   result = await researcher.analyze_stock(analysis_spec)")
    print("   ```")
    print()
    
    print("OPTION 2: 📱 Interactive Scripts")
    print("   python ask_stock_researcher.py    # Interactive stock research")
    print("   python easy_start.py              # Full trio collaboration")
    print()
    
    print("OPTION 3: 🌐 API Server")
    print("   python -m uvicorn src.api.main:app --reload")
    print("   POST /api/v1/agents/research with stock analysis request")
    print()
    
    print("OPTION 4: 🎯 Specific Stock Scripts")
    print("   python stock_demo.py              # This demo")
    print("   python test_stock_analysis.py     # Comprehensive tests")

async def main():
    """Run the complete stock analysis demonstration."""
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print("🎯 TradeKnowledge Stock Analysis Demo")
    print("Demonstrating multi-agent stock analysis capabilities")
    print("=" * 60)
    
    # Show available domains
    await show_available_domains()
    
    # Run a quick demo
    success = await quick_stock_demo(ticker)
    
    if success:
        print(f"\n🎉 Stock analysis demo completed successfully!")
    else:
        print(f"\n⚠️ Demo had issues - check the logs above")
    
    # Show trio workflow
    await trio_stock_workflow_demo()
    
    # Show usage options
    show_usage_options()
    
    print(f"\n" + "=" * 60)
    print("📈 READY FOR STOCK ANALYSIS!")
    print("Choose any of the usage options above to start analyzing stocks.")

if __name__ == "__main__":
    # Allow passing ticker as command line argument
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    if ticker.lower() == "help":
        print("Usage: python stock_demo.py [TICKER]")
        print("Example: python stock_demo.py MSFT")
        sys.exit(0)
    
    asyncio.run(main())