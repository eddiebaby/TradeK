#!/usr/bin/env python3
"""
Direct WMT stock analysis using the stock researcher agent
"""
import asyncio
import sys
from pathlib import Path

# Add agents directory to path
sys.path.append(str(Path(__file__).parent / "agents"))
sys.path.append(str(Path(__file__).parent / "agents" / "researcher"))

try:
    from stock_analysis_extension import StockAnalysisResearcher, StockAnalysisDomain
    from core.agent_base import TaskContext
except ImportError as e:
    print(f"Import error: {e}")
    print("Using fallback stock analysis...")
    
    # Fallback: Use web search for basic analysis
    import requests
    
    def analyze_wmt_basic():
        print("📈 WMT (Walmart Inc.) Basic Analysis")
        print("=" * 50)
        
        print("\n🏪 Company Overview:")
        print("• Walmart Inc. - Multinational retail corporation")
        print("• Largest company by revenue globally")
        print("• Operating hypermarkets, discount department stores, and grocery stores")
        
        print("\n💰 Key Business Segments:")
        print("• Walmart U.S. - Largest segment with physical stores and e-commerce")
        print("• Walmart International - Operations in 19 countries")
        print("• Sam's Club - Membership-only warehouse clubs")
        
        print("\n📊 Recent Performance Indicators:")
        print("• Strong omnichannel growth strategy")
        print("• E-commerce expansion and digital transformation")
        print("• Supply chain optimization initiatives")
        print("• Healthcare services expansion")
        
        print("\n🎯 Investment Considerations:")
        print("✅ Strengths:")
        print("  • Market leadership and scale advantages")
        print("  • Strong cash flow generation")
        print("  • Dividend aristocrat status")
        print("  • E-commerce growth momentum")
        
        print("\n⚠️ Risk Factors:")
        print("  • Intense retail competition")
        print("  • Labor cost pressures")
        print("  • Economic sensitivity")
        print("  • Regulatory scrutiny")
        
        print("\n🔍 Recommendation:")
        print("• Consider for defensive portfolio allocation")
        print("• Monitor quarterly earnings for e-commerce progress")
        print("• Watch for margin improvement initiatives")
        
        print("\n⚠️ This is educational analysis only, not investment advice")
    
    analyze_wmt_basic()
    sys.exit(0)

async def analyze_wmt():
    """Analyze WMT stock using the stock researcher agent"""
    
    print("📈 Analyzing WMT (Walmart Inc.)")
    print("=" * 50)
    
    researcher = StockAnalysisResearcher()
    
    # Comprehensive analysis specification for WMT
    analysis_spec = {
        "ticker_symbol": "WMT",
        "domains": [
            "financial_metrics",
            "market_sentiment", 
            "technical_analysis",
            "competitive_analysis",
            "risk_assessment"
        ],
        "depth": "comprehensive",
        "time_horizon": "medium_term",
        "context": {"user_query": "Comprehensive investment analysis of Walmart Inc."},
        "priority": 1
    }
    
    try:
        print("\n🔍 Conducting multi-domain analysis...")
        print("⏳ This may take a moment...")
        
        analysis_result = await researcher.analyze_stock(analysis_spec)
        
        print("\n✅ WMT Analysis Complete!")
        print("=" * 60)
        
        # Executive Summary
        print("📈 Walmart Inc. (WMT) Investment Analysis")
        print("-" * 45)
        
        recommendation = analysis_result.investment_recommendation
        print(f"🎯 Recommendation: {recommendation.get('recommendation', 'Hold')}")
        print(f"📊 Overall Score: {recommendation.get('overall_score', 7.0):.1f}/10")
        print(f"🎪 Target Price: ${recommendation.get('target_price', 'TBD')}")
        print(f"⚖️ Risk Level: {analysis_result.risk_assessment.get('overall_risk_score', 5.0):.1f}/10")
        print(f"🔍 Confidence: {analysis_result.confidence_score:.1%}")
        
        # Key Financial Metrics
        metrics = analysis_result.financial_metrics
        print(f"\n💰 Key Financial Metrics:")
        print(f"   • P/E Ratio: {metrics.pe_ratio or 'N/A'}")
        print(f"   • Revenue Growth: {metrics.revenue_growth or 'N/A'}%")
        print(f"   • ROE: {metrics.roe or 'N/A'}%")
        print(f"   • Debt/Equity: {metrics.debt_to_equity or 'N/A'}")
        
        # Investment Thesis
        print(f"\n💡 Investment Thesis:")
        print(f"   {recommendation.get('thesis', 'Walmart remains a defensive retail play with e-commerce growth potential')}")
        
        # Key Catalysts
        catalysts = recommendation.get('key_catalysts', [
            "E-commerce and digital transformation progress",
            "Healthcare services expansion",
            "Supply chain automation benefits",
            "International market growth"
        ])
        print(f"\n🚀 Key Catalysts:")
        for catalyst in catalysts[:4]:
            print(f"   • {catalyst}")
        
        print("\n" + "=" * 60)
        print("⚠️ This analysis is for research purposes only")
        print("💡 Please consult a financial advisor for investment decisions")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Falling back to basic analysis...")
        
        # Basic WMT analysis if agent fails
        print("\n📈 WMT Basic Analysis")
        print("• Strong retail fundamentals")
        print("• E-commerce growth trajectory") 
        print("• Dividend reliability")
        print("• Market share leadership")

if __name__ == "__main__":
    asyncio.run(analyze_wmt())