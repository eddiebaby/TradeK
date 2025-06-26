#!/usr/bin/env python3
"""
Interactive Stock Analysis Script for RESEARCHER Agent

Provides easy access to comprehensive stock analysis capabilities inspired by
CrewAI's multi-agent approach, optimized for Ollama local models.
"""
import asyncio
import sys
import json
from pathlib import Path

# Add agents directory to path
sys.path.append(str(Path(__file__).parent))

sys.path.append(str(Path(__file__).parent / "researcher"))
from stock_analysis_extension import StockAnalysisResearcher, StockAnalysisDomain
from core.agent_base import TaskContext

async def main():
    print("📈 RESEARCHER Stock Analysis Agent")
    print("=" * 55)
    print("Powered by CrewAI-inspired multi-agent analysis")
    print("Optimized for Ollama local models")
    print("-" * 55)
    
    researcher = StockAnalysisResearcher()
    
    print("\n🔍 Stock Analysis Capabilities:")
    stock_capabilities = [
        "Financial Metrics Analysis (P/E, EPS, ROE, Debt ratios)",
        "Market Sentiment Research (News, analyst ratings, social)",
        "SEC Filing Analysis (10-K, 10-Q deep dive)",
        "Technical Pattern Recognition (RSI, MACD, trends)",
        "Competitive Intelligence (Market position, peer analysis)",
        "Investment Risk Assessment (Multi-factor risk scoring)"
    ]
    
    for i, capability in enumerate(stock_capabilities, 1):
        print(f"   {i}. {capability}")
    
    print("\n📊 Analysis Domains Available:")
    for domain in StockAnalysisDomain:
        print(f"   • {domain.value.replace('_', ' ').title()}")
    
    while True:
        print("\n" + "="*70)
        print("Enter a stock ticker for comprehensive analysis")
        print("(Type 'quit' to exit, 'help' for examples, 'demo' for AAPL demo)")
        
        user_input = input("📈 Stock ticker (e.g., AAPL, MSFT, TSLA): ").strip().upper()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        if user_input.lower() == 'help':
            print("\n📝 Example Stock Analysis Requests:")
            print("   • 'AAPL' - Comprehensive Apple Inc. analysis")
            print("   • 'MSFT' - Microsoft Corporation deep dive")
            print("   • 'TSLA' - Tesla Inc. with volatility assessment")
            print("   • 'GOOGL' - Alphabet Inc. competitive analysis")
            print("   • 'AMZN' - Amazon.com multi-domain research")
            continue
        
        if user_input.lower() == 'demo':
            user_input = 'AAPL'
            print("🍎 Running demo analysis for Apple Inc. (AAPL)")
            
        if not user_input or len(user_input) > 6:
            print("❌ Please enter a valid stock ticker (1-6 characters)")
            continue
            
        print(f"\n🔍 RESEARCHER analyzing {user_input}")
        print("⏳ Conducting multi-domain stock research...")
        print("   📊 Financial metrics analysis")
        print("   📰 Market sentiment research") 
        print("   📋 SEC filing deep dive")
        print("   📈 Technical pattern recognition")
        print("   🏢 Competitive intelligence")
        print("   ⚠️  Investment risk assessment")
        
        try:
            # Create comprehensive analysis specification
            analysis_spec = {
                "ticker_symbol": user_input,
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
                "context": {"user_query": f"Comprehensive analysis of {user_input}"},
                "priority": 1
            }
            
            # Conduct stock analysis
            analysis_result = await researcher.analyze_stock(analysis_spec)
            
            print(f"\n✅ RESEARCHER Analysis Complete for {user_input}!")
            print("=" * 60)
            
            # Display executive summary
            print(f"📈 {user_input} Investment Analysis Summary")
            print("-" * 45)
            
            recommendation = analysis_result.investment_recommendation
            print(f"🎯 Recommendation: {recommendation.get('recommendation', 'N/A')}")
            print(f"📊 Overall Score: {recommendation.get('overall_score', 0):.1f}/10")
            print(f"🎪 Target Price: ${recommendation.get('target_price', 'N/A')}")
            print(f"⚖️ Risk Level: {analysis_result.risk_assessment.get('overall_risk_score', 0):.1f}/10")
            print(f"🔍 Confidence: {analysis_result.confidence_score:.1%}")
            
            # Display key financial metrics
            metrics = analysis_result.financial_metrics
            print(f"\n💰 Key Financial Metrics:")
            print(f"   • P/E Ratio: {metrics.pe_ratio or 'N/A'}")
            print(f"   • EPS Growth: {metrics.eps_growth or 'N/A'}%")
            print(f"   • Revenue Growth: {metrics.revenue_growth or 'N/A'}%")
            print(f"   • ROE: {metrics.roe or 'N/A'}%")
            print(f"   • Debt/Equity: {metrics.debt_to_equity or 'N/A'}")
            
            # Display market sentiment
            sentiment = analysis_result.market_sentiment
            if sentiment and not sentiment.get("error"):
                print(f"\n📊 Market Sentiment:")
                print(f"   • Overall: {sentiment.get('overall_sentiment', 'N/A').title()}")
                print(f"   • Sentiment Score: {sentiment.get('sentiment_score', 0):.2f}")
                
                ratings = sentiment.get('analyst_ratings', {})
                if ratings:
                    print(f"   • Analyst Ratings: {ratings.get('buy', 0)} Buy, " +
                          f"{ratings.get('hold', 0)} Hold, {ratings.get('sell', 0)} Sell")
            
            # Display investment action
            print(f"\n⚡ Recommended Action:")
            print(f"   {recommendation.get('action', 'Monitor position')}")
            
            # Display portfolio allocation
            allocation = recommendation.get('portfolio_allocation', {})
            if allocation:
                print(f"\n💼 Portfolio Allocation:")
                print(f"   • Recommended: {allocation.get('recommended_allocation', 'N/A')}")
                print(f"   • Risk Level: {allocation.get('risk_level', 'N/A')}")
            
            # Display key catalysts
            catalysts = recommendation.get('key_catalysts', [])
            if catalysts:
                print(f"\n🚀 Key Catalysts to Watch:")
                for catalyst in catalysts[:3]:
                    print(f"   • {catalyst}")
            
            # Display exit conditions
            exit_conditions = recommendation.get('exit_conditions', [])
            if exit_conditions:
                print(f"\n🚪 Exit Conditions:")
                for condition in exit_conditions[:3]:
                    print(f"   • {condition}")
            
            # Display competitive position
            competitive = analysis_result.competitive_position
            if competitive and not competitive.get("error"):
                position = competitive.get('market_position', 'N/A')
                print(f"\n🏢 Competitive Position: {position}")
                
                advantages = competitive.get('competitive_advantages', [])
                if advantages:
                    print("   Key Advantages:")
                    for advantage in advantages[:3]:
                        print(f"     • {advantage}")
            
            # Display technical signals
            technical = analysis_result.technical_signals
            if technical and not technical.get("error"):
                trend = technical.get('trend_analysis', {})
                print(f"\n📈 Technical Analysis:")
                print(f"   • Short-term: {trend.get('short_term', 'N/A').title()}")
                print(f"   • Medium-term: {trend.get('medium_term', 'N/A').title()}")
                print(f"   • Long-term: {trend.get('long_term', 'N/A').title()}")
                
                indicators = technical.get('technical_indicators', {})
                if indicators:
                    print(f"   • RSI: {indicators.get('rsi', 'N/A')}")
                    print(f"   • MACD: {indicators.get('macd', 'N/A')}")
            
            # Display SEC insights summary
            sec_insights = analysis_result.sec_insights
            if sec_insights and not any(insight.get("error") for insight in sec_insights):
                print(f"\n📋 SEC Filing Insights:")
                for insight in sec_insights[:2]:
                    print(f"   • {insight.get('description', 'N/A')}")
                    print(f"     Confidence: {insight.get('confidence', 0):.1%}")
            
            print("\n" + "=" * 60)
            print("💡 Analysis complete! Consider consulting a financial advisor.")
            print("⚠️  This is research only, not investment advice.")
            
        except Exception as e:
            print(f"❌ Analysis Error: {e}")
            import traceback
            traceback.print_exc()

async def quick_analysis_demo():
    """Run a quick demo of stock analysis capabilities."""
    
    print("🚀 Quick Stock Analysis Demo")
    print("=" * 40)
    
    researcher = StockAnalysisResearcher()
    
    # Demo analysis for AAPL
    demo_spec = {
        "ticker_symbol": "AAPL",
        "domains": ["financial_analysis", "market_intelligence", "risk_evaluation"],
        "depth": "standard",
        "time_horizon": "medium_term",
        "context": {"demo": True},
        "priority": 1
    }
    
    print("📱 Analyzing Apple Inc. (AAPL)...")
    result = await researcher.analyze_stock(demo_spec)
    
    print(f"\n✅ Demo Analysis Results:")
    print(f"   Recommendation: {result.investment_recommendation.get('recommendation')}")
    print(f"   Overall Score: {result.investment_recommendation.get('overall_score', 0):.1f}/10")
    print(f"   Risk Score: {result.risk_assessment.get('overall_risk_score', 0):.1f}/10")
    print(f"   Confidence: {result.confidence_score:.1%}")
    
    return result

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--demo":
            asyncio.run(quick_analysis_demo())
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Stock analysis session ended.")
    except Exception as e:
        print(f"❌ Error: {e}")