#!/usr/bin/env python3
"""
Enhanced Researcher Agent with Stock Analysis Capabilities
Integrates CrewAI-style stock analysis with our optimized Ollama models
"""

import asyncio
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

# Add modules to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "modules"))

from core.ollama_integration import researcher_completion
from modules.stock_analysis import analyze_stock_for_researcher, StockAnalysisRequest
from influx_blackboard import write_task, update_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedStockResearcher:
    """Enhanced Researcher Agent with comprehensive stock analysis capabilities"""
    
    def __init__(self):
        self.session_stats = {
            "analyses_performed": 0,
            "total_tokens_saved": 0.0,
            "avg_response_time": 0.0
        }
    
    async def analyze_stock(self, symbol: str, analysis_type: str = "comprehensive") -> dict:
        """Perform comprehensive stock analysis"""
        logger.info(f"🔍 Starting stock analysis for {symbol}")
        
        start_time = datetime.now()
        
        try:
            # Perform comprehensive stock analysis
            analysis_results = await analyze_stock_for_researcher(symbol, analysis_type)
            
            # Enhance with AI-powered insights
            enhanced_results = await self._enhance_with_ai_insights(analysis_results)
            
            # Update session stats
            self.session_stats["analyses_performed"] += 1
            response_time = (datetime.now() - start_time).total_seconds()
            self.session_stats["avg_response_time"] = (
                (self.session_stats["avg_response_time"] * (self.session_stats["analyses_performed"] - 1) + response_time) 
                / self.session_stats["analyses_performed"]
            )
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Stock analysis failed for {symbol}: {e}")
            raise
    
    async def _enhance_with_ai_insights(self, analysis_results: dict) -> dict:
        """Enhance analysis with additional AI insights"""
        try:
            # Extract key data for AI enhancement
            symbol = analysis_results["symbol"]
            current_price = analysis_results["current_price"]
            price_change_percent = analysis_results["price_change_percent"]
            
            # Get technical and fundamental summaries
            technical_summary = analysis_results.get("analysis_components", {}).get("technical", {})
            fundamental_summary = analysis_results.get("analysis_components", {}).get("fundamental", {})
            
            # Create enhancement prompt
            enhancement_prompt = f"""
Based on the comprehensive analysis of {symbol}, provide strategic investment insights:

CURRENT SITUATION:
- Price: ${current_price:.2f} ({price_change_percent:+.2f}% change)
- Technical Analysis: {technical_summary.get('summary', 'Not available')}
- Fundamental Ratios: {fundamental_summary.get('ratios', 'Not available')}

Please provide:
1. Executive Summary (2-3 key points)
2. Investment Recommendation (Buy/Hold/Sell with confidence level)
3. Key Catalysts to Watch
4. Risk Factors
5. Price Targets (if applicable)

Focus on actionable insights for investment decision-making.
"""
            
            # Get AI enhancement
            ai_enhancement = await researcher_completion(
                prompt=enhancement_prompt,
                operation="stock_analysis_enhancement",
                max_tokens=800,
                temperature=0.2
            )
            
            # Add enhancement to results
            analysis_results["analysis_components"]["strategic_insights"] = {
                "ai_enhancement": ai_enhancement.get("content", "Enhancement not available"),
                "model_used": ai_enhancement.get("model", "unknown"),
                "tokens_used": ai_enhancement.get("tokens_used", 0),
                "cost_savings": ai_enhancement.get("cost_savings", 0)
            }
            
            # Track token savings
            self.session_stats["total_tokens_saved"] += ai_enhancement.get("cost_savings", 0)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"AI enhancement failed: {e}")
            # Return original results if enhancement fails
            return analysis_results
    
    async def compare_stocks(self, symbols: list, comparison_focus: str = "investment_potential") -> dict:
        """Compare multiple stocks across various metrics"""
        logger.info(f"📊 Comparing stocks: {', '.join(symbols)}")
        
        comparison_results = {
            "comparison_date": datetime.now().isoformat(),
            "symbols": symbols,
            "focus": comparison_focus,
            "individual_analyses": {},
            "comparative_analysis": {}
        }
        
        # Analyze each stock individually
        for symbol in symbols:
            try:
                analysis = await self.analyze_stock(symbol, "comprehensive")
                comparison_results["individual_analyses"][symbol] = analysis
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                comparison_results["individual_analyses"][symbol] = {"error": str(e)}
        
        # Generate comparative insights
        comparison_results["comparative_analysis"] = await self._generate_comparative_analysis(
            comparison_results["individual_analyses"], 
            comparison_focus
        )
        
        return comparison_results
    
    async def _generate_comparative_analysis(self, analyses: dict, focus: str) -> dict:
        """Generate AI-powered comparative analysis"""
        try:
            # Prepare comparison data
            comparison_data = []
            for symbol, analysis in analyses.items():
                if "error" not in analysis:
                    comparison_data.append({
                        "symbol": symbol,
                        "current_price": analysis.get("current_price", 0),
                        "price_change_percent": analysis.get("price_change_percent", 0),
                        "ai_insights": analysis.get("analysis_components", {}).get("ai_insights", {}).get("ai_analysis", "")[:200]
                    })
            
            # Create comparison prompt
            comparison_prompt = f"""
Compare the following stocks for {focus}:

STOCK COMPARISON DATA:
"""
            
            for stock in comparison_data:
                comparison_prompt += f"""
{stock['symbol']}: ${stock['current_price']:.2f} ({stock['price_change_percent']:+.2f}%)
Key Insights: {stock['ai_insights']}...

"""
            
            comparison_prompt += f"""
Based on the analysis, provide:
1. Ranking (Best to Worst for {focus})
2. Key Differentiators
3. Risk vs Reward Assessment
4. Portfolio Allocation Suggestions
5. Market Outlook Impact

Focus on practical investment implications.
"""
            
            # Get comparative analysis
            comparative_result = await researcher_completion(
                prompt=comparison_prompt,
                operation="stock_comparison",
                max_tokens=1000,
                temperature=0.3
            )
            
            return {
                "comparative_insights": comparative_result.get("content", "Comparison not available"),
                "model_used": comparative_result.get("model", "unknown"),
                "analysis_quality": "AI-powered comparative analysis"
            }
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            return {"error": f"Comparative analysis failed: {str(e)}"}
    
    async def sector_analysis(self, sector: str, top_stocks: list = None) -> dict:
        """Analyze an entire sector"""
        logger.info(f"🏭 Analyzing sector: {sector}")
        
        # Default stocks for common sectors
        sector_stocks = {
            "technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
            "finance": ["JPM", "BAC", "WFC", "GS", "C"],
            "healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
            "energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
            "retail": ["AMZN", "WMT", "TGT", "COST", "HD"]
        }
        
        stocks_to_analyze = top_stocks or sector_stocks.get(sector.lower(), [])
        
        if not stocks_to_analyze:
            return {"error": f"No stocks defined for sector: {sector}"}
        
        # Perform sector comparison
        sector_results = await self.compare_stocks(stocks_to_analyze, f"{sector}_sector_analysis")
        
        # Add sector-specific insights
        sector_prompt = f"""
Analyze the {sector} sector based on the following stock analysis:

SECTOR: {sector.upper()}
STOCKS ANALYZED: {', '.join(stocks_to_analyze)}

Provide sector-specific insights:
1. Sector Health and Trends
2. Leading vs Lagging Companies
3. Sector-Specific Risks and Opportunities
4. Economic Sensitivity Analysis
5. Investment Strategy for the Sector

Focus on sector dynamics and investment implications.
"""
        
        sector_insights = await researcher_completion(
            prompt=sector_prompt,
            operation="sector_analysis",
            max_tokens=1200,
            temperature=0.3
        )
        
        sector_results["sector_insights"] = {
            "sector_overview": sector_insights.get("content", "Sector analysis not available"),
            "analysis_scope": f"{len(stocks_to_analyze)} stocks analyzed"
        }
        
        return sector_results
    
    def get_session_summary(self) -> dict:
        """Get session performance summary"""
        return {
            "analyses_performed": self.session_stats["analyses_performed"],
            "total_tokens_saved": f"${self.session_stats['total_tokens_saved']:.4f}",
            "avg_response_time": f"{self.session_stats['avg_response_time']:.2f}s",
            "cost_efficiency": "Local processing with Ollama models"
        }

async def interactive_stock_research():
    """Interactive stock research session"""
    researcher = EnhancedStockResearcher()
    
    print("🔍 Enhanced Stock Researcher Agent")
    print("=" * 50)
    print("Commands:")
    print("  analyze <SYMBOL>           - Comprehensive stock analysis")
    print("  compare <SYM1,SYM2,SYM3>   - Compare multiple stocks")
    print("  sector <SECTOR_NAME>       - Analyze entire sector")
    print("  stats                      - Show session statistics")
    print("  quit                       - Exit")
    print()
    
    while True:
        try:
            command = input("Stock Research > ").strip().lower()
            
            if command == "quit" or command == "exit":
                summary = researcher.get_session_summary()
                print(f"\n📊 Session Summary:")
                print(f"   Analyses: {summary['analyses_performed']}")
                print(f"   Cost Savings: {summary['total_tokens_saved']}")
                print(f"   Avg Time: {summary['avg_response_time']}")
                break
            
            elif command.startswith("analyze "):
                symbol = command.split(" ", 1)[1].upper()
                print(f"\n🔄 Analyzing {symbol}...")
                
                result = await researcher.analyze_stock(symbol)
                
                print(f"\n📈 {symbol} Analysis Results:")
                print(f"   Current Price: ${result['current_price']:.2f}")
                print(f"   Price Change: {result['price_change_percent']:+.2f}%")
                
                # Show AI insights
                ai_insights = result.get("analysis_components", {}).get("ai_insights", {})
                if ai_insights and "ai_analysis" in ai_insights:
                    print(f"\n🤖 AI Analysis Preview:")
                    preview = ai_insights["ai_analysis"][:300]
                    print(f"   {preview}...")
                
                strategic_insights = result.get("analysis_components", {}).get("strategic_insights", {})
                if strategic_insights and "ai_enhancement" in strategic_insights:
                    print(f"\n🎯 Strategic Insights Preview:")
                    preview = strategic_insights["ai_enhancement"][:300]
                    print(f"   {preview}...")
            
            elif command.startswith("compare "):
                symbols_str = command.split(" ", 1)[1].upper()
                symbols = [s.strip() for s in symbols_str.split(",")]
                
                print(f"\n🔄 Comparing {', '.join(symbols)}...")
                
                result = await researcher.compare_stocks(symbols)
                
                print(f"\n📊 Comparison Results:")
                for symbol in symbols:
                    analysis = result["individual_analyses"].get(symbol, {})
                    if "error" not in analysis:
                        price = analysis.get("current_price", 0)
                        change = analysis.get("price_change_percent", 0)
                        print(f"   {symbol}: ${price:.2f} ({change:+.2f}%)")
                
                comparative = result.get("comparative_analysis", {})
                if "comparative_insights" in comparative:
                    print(f"\n🎯 Comparative Analysis Preview:")
                    preview = comparative["comparative_insights"][:300]
                    print(f"   {preview}...")
            
            elif command.startswith("sector "):
                sector = command.split(" ", 1)[1]
                print(f"\n🔄 Analyzing {sector} sector...")
                
                result = await researcher.sector_analysis(sector)
                
                if "error" not in result:
                    print(f"\n🏭 {sector.title()} Sector Analysis:")
                    stocks = result.get("symbols", [])
                    print(f"   Stocks Analyzed: {', '.join(stocks)}")
                    
                    sector_insights = result.get("sector_insights", {})
                    if "sector_overview" in sector_insights:
                        print(f"\n📋 Sector Overview Preview:")
                        preview = sector_insights["sector_overview"][:300]
                        print(f"   {preview}...")
                else:
                    print(f"❌ Error: {result['error']}")
            
            elif command == "stats":
                summary = researcher.get_session_summary()
                print(f"\n📊 Session Statistics:")
                for key, value in summary.items():
                    print(f"   {key.replace('_', ' ').title()}: {value}")
            
            else:
                print("❌ Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Stock Researcher Agent")
    parser.add_argument("--symbol", help="Stock symbol to analyze")
    parser.add_argument("--compare", help="Comma-separated symbols to compare")
    parser.add_argument("--sector", help="Sector to analyze")
    parser.add_argument("--interactive", action="store_true", help="Start interactive session")
    
    args = parser.parse_args()
    
    researcher = EnhancedStockResearcher()
    
    if args.interactive or not any([args.symbol, args.compare, args.sector]):
        await interactive_stock_research()
    
    elif args.symbol:
        result = await researcher.analyze_stock(args.symbol.upper())
        print(json.dumps(result, indent=2, default=str))
    
    elif args.compare:
        symbols = [s.strip().upper() for s in args.compare.split(",")]
        result = await researcher.compare_stocks(symbols)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.sector:
        result = await researcher.sector_analysis(args.sector)
        print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())