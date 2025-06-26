#!/usr/bin/env python3
"""
Comprehensive Stock Analysis - Full RESEARCHER capabilities
This recreates the institutional-quality analysis like the CELH report
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Fix imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "researcher"))

from researcher.stock_analysis_extension import StockAnalysisResearcher, StockAnalysisDomain
from core.agent_base import TaskContext

async def comprehensive_stock_analysis(ticker="LQDA"):
    """Run comprehensive institutional-quality stock analysis."""
    
    print(f"📊 {ticker.upper()} - Comprehensive Stock Analysis")
    print("=" * 60)
    print("🔍 RESEARCHER Agent - Professional Equity Research")
    print("=" * 60)
    
    try:
        # Create the enhanced researcher
        researcher = StockAnalysisResearcher()
        
        # Create comprehensive analysis specification
        analysis_spec = {
            "ticker_symbol": ticker.upper(),
            "domains": [
                "financial_analysis",      # Financial metrics, ratios, performance
                "market_intelligence",     # Market sentiment, news, analyst ratings
                "regulatory_analysis",     # SEC filings, compliance, governance
                "price_patterns",          # Technical analysis, charts, indicators
                "industry_comparison",     # Competitive analysis, peers
                "risk_evaluation"          # Investment risks, volatility
            ],
            "depth": "comprehensive",
            "time_horizon": "medium_term",
            "context": {
                "analysis_type": "institutional_research",
                "report_format": "professional",
                "include_recommendations": True,
                "include_price_targets": True,
                "confidence_required": 85
            },
            "priority": 1
        }
        
        print(f"🎯 Company Overview")
        print(f"Analyzing: {ticker.upper()}")
        print(f"Analysis Depth: {analysis_spec['depth']}")
        print(f"Time Horizon: {analysis_spec['time_horizon']}")
        print(f"Domains: {len(analysis_spec['domains'])} comprehensive areas")
        print()
        
        for i, domain in enumerate(analysis_spec['domains'], 1):
            domain_name = domain.replace('_', ' ').title()
            print(f"   {i}. {domain_name}")
        
        print(f"\n⚡ Starting comprehensive analysis...")
        print("=" * 40)
        
        # Run the comprehensive analysis
        result = await researcher.analyze_stock(analysis_spec)
        
        print(f"\n✅ Analysis Completed!")
        print(f"📊 Ticker: {result.ticker_symbol}")
        print(f"🎯 Confidence Score: {result.confidence_score:.2%}")
        print(f"📈 Analysis ID: {result.analysis_id}")
        print()
        
        # Generate comprehensive report sections
        await generate_executive_summary(result)
        await generate_financial_analysis(result)
        await generate_market_position(result)
        await generate_risk_assessment(result)
        await generate_technical_analysis(result)
        await generate_investment_thesis(result)
        await generate_recommendations(result)
        
        return result
        
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def generate_executive_summary(result):
    """Generate executive summary section."""
    print("🎯 Executive Summary")
    print("-" * 40)
    
    # This would typically pull from the analysis result
    print(f"Company: {result.ticker_symbol}")
    print(f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}")
    print(f"Overall Confidence: {result.confidence_score:.1%}")
    
    if hasattr(result, 'company_overview'):
        print(f"Sector: {result.company_overview.get('sector', 'N/A')}")
        print(f"Industry: {result.company_overview.get('industry', 'N/A')}")
        print(f"Market Cap: {result.company_overview.get('market_cap', 'N/A')}")
    
    print()

async def generate_financial_analysis(result):
    """Generate detailed financial analysis."""
    print("💰 Financial Performance")
    print("-" * 40)
    
    if hasattr(result, 'financial_metrics') and result.financial_metrics:
        metrics = result.financial_metrics
        
        print("Key Financial Metrics:")
        print(f"   • P/E Ratio: {getattr(metrics, 'pe_ratio', 'N/A')}")
        print(f"   • EPS Growth: {getattr(metrics, 'eps_growth', 'N/A')}%")
        print(f"   • Revenue Growth: {getattr(metrics, 'revenue_growth', 'N/A')}%")
        print(f"   • ROE: {getattr(metrics, 'roe', 'N/A')}%")
        print(f"   • ROA: {getattr(metrics, 'roa', 'N/A')}%")
        print(f"   • Debt/Equity: {getattr(metrics, 'debt_to_equity', 'N/A')}")
        print(f"   • Current Ratio: {getattr(metrics, 'current_ratio', 'N/A')}")
        print(f"   • Gross Margin: {getattr(metrics, 'gross_margin', 'N/A')}%")
        print(f"   • Operating Margin: {getattr(metrics, 'operating_margin', 'N/A')}%")
        print(f"   • Free Cash Flow: {getattr(metrics, 'free_cash_flow', 'N/A')}")
    else:
        print("Financial metrics analysis in progress...")
    
    print()

async def generate_market_position(result):
    """Generate market position and sentiment analysis."""
    print("📊 Market Position & Sentiment")
    print("-" * 40)
    
    if hasattr(result, 'market_sentiment') and result.market_sentiment:
        sentiment = result.market_sentiment
        
        print("Market Intelligence:")
        print(f"   • Overall Sentiment: {sentiment.get('overall_sentiment', 'N/A')}")
        print(f"   • Sentiment Score: {sentiment.get('sentiment_score', 0):.2f}/10")
        print(f"   • Analyst Coverage: {sentiment.get('analyst_count', 'N/A')} analysts")
        print(f"   • Average Rating: {sentiment.get('avg_rating', 'N/A')}")
        print(f"   • Price Target Consensus: ${sentiment.get('avg_price_target', 'N/A')}")
        
        if 'news_sentiment' in sentiment:
            print(f"   • Recent News Sentiment: {sentiment['news_sentiment']}")
        
        if 'social_sentiment' in sentiment:
            print(f"   • Social Media Buzz: {sentiment['social_sentiment']}")
    else:
        print("Market sentiment analysis in progress...")
    
    print()

async def generate_risk_assessment(result):
    """Generate comprehensive risk assessment."""
    print("⚠️ Risk Assessment")
    print("-" * 40)
    
    if hasattr(result, 'risk_assessment') and result.risk_assessment:
        risk = result.risk_assessment
        
        print("Investment Risk Analysis:")
        print(f"   • Overall Risk Score: {risk.get('overall_risk_score', 0):.1f}/10")
        print(f"   • Volatility Rating: {risk.get('volatility_rating', 'N/A')}")
        print(f"   • Beta: {risk.get('beta', 'N/A')}")
        print(f"   • 52-Week Range: ${risk.get('week_52_low', 'N/A')} - ${risk.get('week_52_high', 'N/A')}")
        
        risk_categories = risk.get('risk_categories', {})
        if risk_categories:
            print("\n   Risk Breakdown:")
            for category, score in risk_categories.items():
                print(f"      • {category.replace('_', ' ').title()}: {score}/10")
        
        risk_factors = risk.get('risk_factors', [])
        if risk_factors:
            print(f"\n   Key Risk Factors:")
            for factor in risk_factors[:5]:  # Show top 5
                print(f"      • {factor}")
    else:
        print("Risk assessment analysis in progress...")
    
    print()

async def generate_technical_analysis(result):
    """Generate technical analysis section."""
    print("📈 Technical Analysis")
    print("-" * 40)
    
    if hasattr(result, 'technical_analysis') and result.technical_analysis:
        technical = result.technical_analysis
        
        print("Technical Indicators:")
        print(f"   • Current Price: ${technical.get('current_price', 'N/A')}")
        print(f"   • 50-Day MA: ${technical.get('ma_50', 'N/A')}")
        print(f"   • 200-Day MA: ${technical.get('ma_200', 'N/A')}")
        print(f"   • RSI (14): {technical.get('rsi', 'N/A')}")
        print(f"   • MACD: {technical.get('macd', 'N/A')}")
        print(f"   • Volume (Avg): {technical.get('avg_volume', 'N/A')}")
        
        trend = technical.get('trend_analysis', {})
        if trend:
            print(f"\n   Trend Analysis:")
            print(f"      • Short-term Trend: {trend.get('short_term', 'N/A')}")
            print(f"      • Medium-term Trend: {trend.get('medium_term', 'N/A')}")
            print(f"      • Long-term Trend: {trend.get('long_term', 'N/A')}")
        
        support_resistance = technical.get('support_resistance', {})
        if support_resistance:
            print(f"\n   Support & Resistance:")
            print(f"      • Support Levels: {support_resistance.get('support', 'N/A')}")
            print(f"      • Resistance Levels: {support_resistance.get('resistance', 'N/A')}")
    else:
        print("Technical analysis in progress...")
    
    print()

async def generate_investment_thesis(result):
    """Generate investment thesis section."""
    print("🎯 Investment Thesis")
    print("-" * 40)
    
    if hasattr(result, 'investment_thesis') and result.investment_thesis:
        thesis = result.investment_thesis
        
        bull_case = thesis.get('bull_case', [])
        if bull_case:
            print("Bull Case:")
            for point in bull_case:
                print(f"   • {point}")
        
        bear_case = thesis.get('bear_case', [])
        if bear_case:
            print(f"\nBear Case:")
            for point in bear_case:
                print(f"   • {point}")
        
        catalysts = thesis.get('catalysts', [])
        if catalysts:
            print(f"\nKey Catalysts:")
            for catalyst in catalysts:
                print(f"   • {catalyst}")
    else:
        print("Investment thesis analysis in progress...")
    
    print()

async def generate_recommendations(result):
    """Generate final recommendations and price targets."""
    print("🔮 Investment Recommendation")
    print("-" * 40)
    
    if hasattr(result, 'investment_recommendation') and result.investment_recommendation:
        rec = result.investment_recommendation
        
        print(f"Rating: {rec.get('recommendation', 'N/A')} ⭐")
        print(f"Overall Score: {rec.get('overall_score', 0):.1f}/10")
        print(f"Target Price: ${rec.get('target_price', 'N/A')}")
        print(f"Time Horizon: {rec.get('time_horizon', 'N/A')}")
        
        if 'price_targets' in rec:
            targets = rec['price_targets']
            print(f"\nPrice Target Range:")
            print(f"   • Bear Case: ${targets.get('bear', 'N/A')}")
            print(f"   • Base Case: ${targets.get('base', 'N/A')}")
            print(f"   • Bull Case: ${targets.get('bull', 'N/A')}")
        
        if 'rationale' in rec:
            print(f"\nRationale:")
            print(f"   {rec['rationale']}")
        
        action_plan = rec.get('action_plan', '')
        if action_plan:
            print(f"\nAction Plan:")
            print(f"   {action_plan}")
    else:
        print("Investment recommendation in progress...")
    
    print()
    print("=" * 60)
    print("📈 ANALYSIS COMPLETE")
    print(f"Report generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}")
    print("=" * 60)

async def main():
    """Run comprehensive analysis for specified ticker."""
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "LQDA"
    
    if ticker.lower() == "help":
        print("Usage: python comprehensive_stock_analysis.py [TICKER]")
        print("Example: python comprehensive_stock_analysis.py LQDA")
        sys.exit(0)
    
    result = await comprehensive_stock_analysis(ticker)
    
    if result:
        print(f"\n✅ Comprehensive analysis for {ticker.upper()} completed successfully!")
    else:
        print(f"\n❌ Analysis failed - check logs above")

if __name__ == "__main__":
    asyncio.run(main())