#!/usr/bin/env python3
"""
LQDA stock analysis using the sophisticated RESEARCHER agent
"""
import asyncio
import sys
from pathlib import Path

# Add agents directory to path
sys.path.append(str(Path(__file__).parent / "agents" / "researcher"))

from stock_analysis_extension import StockAnalysisResearcher

async def analyze_lqda():
    """Analyze LQDA stock using the comprehensive researcher agent"""
    
    print("📈 LQDA (Liquidia Corporation) - Comprehensive Analysis")
    print("=" * 60)
    print("Powered by CrewAI-inspired multi-agent research framework")
    print("-" * 60)
    
    researcher = StockAnalysisResearcher()
    
    # Comprehensive analysis specification for LQDA
    analysis_spec = {
        "ticker_symbol": "LQDA",
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
        "context": {"user_query": "Comprehensive investment analysis of Liquidia Corporation"},
        "priority": 1
    }
    
    try:
        print("\n🔍 RESEARCHER conducting multi-domain analysis...")
        print("⏳ Performing comprehensive research across all domains...")
        
        analysis_result = await researcher.analyze_stock(analysis_spec)
        
        print("\n✅ LQDA Analysis Complete!")
        print("=" * 70)
        
        # Executive Summary
        print("📈 Liquidia Corporation (LQDA) Investment Analysis")
        print("-" * 50)
        
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
        print(f"   • EPS Growth: {metrics.eps_growth or 'N/A'}%")
        print(f"   • Revenue Growth: {metrics.revenue_growth or 'N/A'}%")
        print(f"   • ROE: {metrics.roe or 'N/A'}%")
        print(f"   • Debt/Equity: {metrics.debt_to_equity or 'N/A'}")
        print(f"   • Market Cap: ${metrics.market_cap or 'N/A':,}")
        
        # Market Sentiment Analysis
        sentiment = analysis_result.market_sentiment
        if sentiment and not sentiment.get("error"):
            print(f"\n📊 Market Sentiment:")
            print(f"   • Overall: {sentiment.get('overall_sentiment', 'N/A').title()}")
            print(f"   • Sentiment Score: {sentiment.get('sentiment_score', 0):.2f}")
            
            ratings = sentiment.get('analyst_ratings', {})
            if ratings:
                print(f"   • Analyst Ratings: {ratings.get('buy', 0)} Buy, " +
                      f"{ratings.get('hold', 0)} Hold, {ratings.get('sell', 0)} Sell")
                print(f"   • Current Price: ${ratings.get('current_price', 'N/A')}")
                print(f"   • Average Target: ${ratings.get('average_target', 'N/A')}")
        
        # Investment Thesis
        print(f"\n💡 Investment Thesis:")
        thesis = recommendation.get('thesis', 'Biotech company with innovative pulmonary therapeutics platform')
        print(f"   {thesis}")
        
        # Recommended Action
        print(f"\n⚡ Recommended Action:")
        print(f"   {recommendation.get('action', 'Monitor position and upcoming catalysts')}")
        
        # Portfolio Allocation
        allocation = recommendation.get('portfolio_allocation', {})
        if allocation:
            print(f"\n💼 Portfolio Allocation:")
            print(f"   • Recommended: {allocation.get('recommended_allocation', 'N/A')}")
            print(f"   • Risk Level: {allocation.get('risk_level', 'N/A')}")
            print(f"   • Position Sizing: {allocation.get('position_sizing', 'Conservative')}")
        
        # Key Catalysts
        catalysts = recommendation.get('key_catalysts', [
            "Clinical trial milestones and data readouts",
            "FDA regulatory approvals and submissions", 
            "Partnership and licensing agreements",
            "Commercial launch execution"
        ])
        print(f"\n🚀 Key Catalysts to Watch:")
        for i, catalyst in enumerate(catalysts[:4], 1):
            print(f"   {i}. {catalyst}")
        
        # Exit Conditions
        exit_conditions = recommendation.get('exit_conditions', [
            "Clinical trial failures or setbacks",
            "Regulatory delays or rejections",
            "Competitive threats emerging",
            "Funding concerns or dilution risk"
        ])
        print(f"\n🚪 Exit Conditions:")
        for i, condition in enumerate(exit_conditions[:4], 1):
            print(f"   {i}. {condition}")
        
        # Competitive Position
        competitive = analysis_result.competitive_position
        if competitive and not competitive.get("error"):
            print(f"\n🏢 Competitive Position:")
            position = competitive.get('market_position', 'Emerging biotechnology company')
            print(f"   • Market Position: {position}")
            
            advantages = competitive.get('competitive_advantages', [])
            if advantages:
                print("   • Key Advantages:")
                for advantage in advantages[:3]:
                    print(f"     • {advantage}")
        
        # Technical Analysis
        technical = analysis_result.technical_signals
        if technical and not technical.get("error"):
            trend = technical.get('trend_analysis', {})
            print(f"\n📈 Technical Analysis:")
            print(f"   • Short-term Trend: {trend.get('short_term', 'N/A').title()}")
            print(f"   • Medium-term Trend: {trend.get('medium_term', 'N/A').title()}")
            print(f"   • Long-term Trend: {trend.get('long_term', 'N/A').title()}")
            
            indicators = technical.get('technical_indicators', {})
            if indicators:
                print(f"   • RSI: {indicators.get('rsi', 'N/A')}")
                print(f"   • MACD: {indicators.get('macd', 'N/A')}")
                
            support_resistance = technical.get('support_resistance', {})
            if support_resistance:
                supports = support_resistance.get('support_levels', [])
                resistances = support_resistance.get('resistance_levels', [])
                if supports:
                    print(f"   • Support Levels: ${', $'.join(map(str, supports[:3]))}")
                if resistances:
                    print(f"   • Resistance Levels: ${', $'.join(map(str, resistances[:3]))}")
        
        # Risk Assessment Details
        risk = analysis_result.risk_assessment
        if risk and not risk.get("error"):
            print(f"\n⚠️ Risk Assessment:")
            print(f"   • Overall Risk Score: {risk.get('overall_risk_score', 5.0):.1f}/10")
            
            risk_categories = risk.get('risk_categories', {})
            for category, details in risk_categories.items():
                if isinstance(details, dict):
                    score = details.get('score', 5.0)
                    print(f"   • {category.replace('_', ' ').title()}: {score:.1f}/10")
        
        # SEC Filing Insights
        sec_insights = analysis_result.sec_insights
        if sec_insights and not any(insight.get("error") for insight in sec_insights):
            print(f"\n📋 SEC Filing Insights:")
            for i, insight in enumerate(sec_insights[:3], 1):
                description = insight.get('description', 'N/A')
                confidence = insight.get('confidence', 0)
                print(f"   {i}. {description}")
                print(f"      Confidence: {confidence:.1%}")
        
        print("\n" + "=" * 70)
        print("💡 COMPREHENSIVE ANALYSIS COMPLETE!")
        print("⚠️  This is AI-powered research for educational purposes only")
        print("📞 Please consult a licensed financial advisor for investment decisions")
        print("🔍 Consider additional due diligence before making investment choices")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Falling back to basic biotech analysis...")
        
        # Basic LQDA analysis if agent fails
        print("\n📈 LQDA (Liquidia Corporation) Basic Analysis")
        print("=" * 50)
        print("🧬 Company Profile:")
        print("• Liquidia Corporation - Biopharmaceutical company")
        print("• Focus: Pulmonary arterial hypertension (PAH) treatments")
        print("• Platform: PRINT technology for drug particle engineering")
        print("• Lead Product: YUTREPIA (treprostinil) inhalation powder")
        
        print("\n💊 Pipeline & Products:")
        print("• YUTREPIA: FDA-approved PAH treatment")
        print("• LIQ861: Inhaled treprostinil for PAH")
        print("• LIQ865: Backup formulation programs")
        
        print("\n📊 Investment Considerations:")
        print("✅ Strengths:")
        print("  • FDA-approved product in commercial stage")
        print("  • Proprietary PRINT particle engineering technology")
        print("  • Addressing significant unmet medical need in PAH")
        print("  • Experienced management team with pharma background")
        
        print("\n⚠️ Risk Factors:")
        print("  • Small biotech with limited revenue diversification")
        print("  • Competitive PAH market with established players")
        print("  • Regulatory and clinical development risks")
        print("  • Potential dilution from funding needs")
        
        print("\n🔍 Key Catalysts to Monitor:")
        print("  • YUTREPIA commercial launch progress and uptake")
        print("  • Clinical data and regulatory milestones")
        print("  • Partnership opportunities and business development")
        print("  • Pipeline advancement and new indications")
        
        print("\n💡 Investment Perspective:")
        print("• Speculative biotech play with commercial-stage asset")
        print("• Monitor commercial execution and market penetration")
        print("• Consider as small position in diversified biotech portfolio")
        
        print("\n⚠️ This is educational analysis only, not investment advice")

if __name__ == "__main__":
    asyncio.run(analyze_lqda())