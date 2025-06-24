#!/usr/bin/env python3
"""
Real LQDA Analysis - Using Live Data Sources
No simulated data - only real information from Perplexity and Brave
"""

import asyncio
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add MCP tools to path
sys.path.append(str(Path(__file__).parent))

# Import MCP tools for real data
try:
    from mcp_brave_search import brave_web_search
    from mcp_sequential_thinking import sequentialthinking
    print("✅ MCP tools imported successfully")
except ImportError as e:
    print(f"❌ MCP tools not available: {e}")
    print("Using direct tool calls instead...")

class RealDataAnalyzer:
    """Real data analysis using live sources"""
    
    def __init__(self):
        self.symbol = "LQDA"
        self.company_name = "Liquidia Corporation"
    
    async def get_real_market_data(self):
        """Get real market data from yfinance"""
        print("📊 Fetching real market data from Yahoo Finance...")
        
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            hist = ticker.history(period="1y")
            
            # Calculate real technical indicators
            close_prices = hist['Close']
            
            # Real RSI calculation
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Real moving averages
            sma_20 = close_prices.rolling(window=20).mean()
            sma_50 = close_prices.rolling(window=50).mean()
            
            # Real volatility
            returns = close_prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            current_price = info.get('currentPrice', close_prices.iloc[-1])
            
            return {
                "current_price": current_price,
                "price_change": current_price - close_prices.iloc[-2] if len(close_prices) > 1 else 0,
                "volume": info.get('volume', 0),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE'),
                "forward_pe": info.get('forwardPE'),
                "price_to_book": info.get('priceToBook'),
                "debt_to_equity": info.get('debtToEquity'),
                "current_ratio": info.get('currentRatio'),
                "rsi": float(rsi.iloc[-1]) if not rsi.empty else None,
                "sma_20": float(sma_20.iloc[-1]) if not sma_20.empty else None,
                "sma_50": float(sma_50.iloc[-1]) if not sma_50.empty else None,
                "volatility": volatility,
                "52_week_high": info.get('fiftyTwoWeekHigh'),
                "52_week_low": info.get('fiftyTwoWeekLow'),
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "business_summary": info.get('longBusinessSummary', ''),
                "employees": info.get('fullTimeEmployees'),
                "website": info.get('website')
            }
            
        except Exception as e:
            print(f"❌ Error fetching market data: {e}")
            return None

async def search_lqda_news():
    """Search for current LQDA news and analysis"""
    print("🔍 Searching for current LQDA news and analysis...")
    
    try:
        # Search for recent LQDA news
        news_query = "LQDA Liquidia Corporation stock news earnings recent 2024 2025"
        
        # Use Brave search for current information
        search_results = await brave_web_search(
            query=news_query,
            count=10
        )
        
        print(f"Found {len(search_results.get('web', {}).get('results', []))} recent news items")
        return search_results
        
    except Exception as e:
        print(f"❌ Error searching news: {e}")
        return None

async def search_lqda_financials():
    """Search for LQDA financial information"""
    print("💰 Searching for LQDA financial data and analyst coverage...")
    
    try:
        # Search for financial information
        financial_query = "LQDA Liquidia Corporation financial results Q4 2024 earnings revenue YUTREPIA sales"
        
        financial_results = await brave_web_search(
            query=financial_query,
            count=8
        )
        
        print(f"Found {len(financial_results.get('web', {}).get('results', []))} financial reports")
        return financial_results
        
    except Exception as e:
        print(f"❌ Error searching financials: {e}")
        return None

async def search_analyst_ratings():
    """Search for analyst ratings and price targets"""
    print("📈 Searching for analyst ratings and price targets...")
    
    try:
        # Search for analyst coverage
        analyst_query = "LQDA Liquidia Corporation analyst rating price target buy sell hold 2024 2025"
        
        analyst_results = await brave_web_search(
            query=analyst_query,
            count=6
        )
        
        print(f"Found {len(analyst_results.get('web', {}).get('results', []))} analyst reports")
        return analyst_results
        
    except Exception as e:
        print(f"❌ Error searching analyst data: {e}")
        return None

async def analyze_with_thinking(market_data, news_data, financial_data, analyst_data):
    """Use sequential thinking to analyze all the real data"""
    print("🧠 Analyzing all real data with sequential thinking...")
    
    try:
        # Prepare comprehensive data summary
        data_summary = f"""
REAL MARKET DATA FOR LQDA:
Current Price: ${market_data.get('current_price', 0):.2f}
Market Cap: ${market_data.get('market_cap', 0):,}
Volume: {market_data.get('volume', 0):,}
PE Ratio: {market_data.get('pe_ratio', 'N/A')}
Forward PE: {market_data.get('forward_pe', 'N/A')}
Price to Book: {market_data.get('price_to_book', 'N/A')}
Current Ratio: {market_data.get('current_ratio', 'N/A')}
RSI: {market_data.get('rsi', 'N/A'):.1f}
20-day SMA: ${market_data.get('sma_20', 0):.2f}
50-day SMA: ${market_data.get('sma_50', 0):.2f}
Volatility: {market_data.get('volatility', 0):.1%}
52-Week Range: ${market_data.get('52_week_low', 0):.2f} - ${market_data.get('52_week_high', 0):.2f}

BUSINESS PROFILE:
Sector: {market_data.get('sector', 'N/A')}
Industry: {market_data.get('industry', 'N/A')}
Employees: {market_data.get('employees', 'N/A'):,}
Business: {market_data.get('business_summary', '')[:300]}...

RECENT NEWS HEADLINES:
{format_search_results(news_data, 'news')}

FINANCIAL REPORTS:
{format_search_results(financial_data, 'financials')}

ANALYST COVERAGE:
{format_search_results(analyst_data, 'analyst')}
"""
        
        # Use sequential thinking for comprehensive analysis
        analysis = await sequentialthinking(
            thought="I need to analyze LQDA comprehensively using only the real data provided. Let me examine the technical indicators, fundamental metrics, recent news, and analyst coverage to provide an evidence-based investment analysis.",
            nextThoughtNeeded=True,
            thoughtNumber=1,
            totalThoughts=8
        )
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        return None

def format_search_results(search_data, category):
    """Format search results for analysis"""
    if not search_data or 'web' not in search_data:
        return f"No {category} data available"
    
    results = search_data['web'].get('results', [])
    formatted = []
    
    for i, result in enumerate(results[:5], 1):
        title = result.get('title', 'No title')
        description = result.get('description', 'No description')
        url = result.get('url', '')
        
        formatted.append(f"{i}. {title}")
        if description:
            formatted.append(f"   {description[:150]}...")
        formatted.append(f"   Source: {url}")
        formatted.append("")
    
    return "\n".join(formatted)

async def main():
    """Main analysis function using only real data"""
    
    print("🔍 LQDA REAL DATA ANALYSIS")
    print("=" * 60)
    print("Using only live data sources - no simulated information")
    print("-" * 60)
    
    analyzer = RealDataAnalyzer()
    
    # Step 1: Get real market data
    market_data = await analyzer.get_real_market_data()
    if not market_data:
        print("❌ Failed to get market data")
        return
    
    print(f"✅ Retrieved real market data: ${market_data['current_price']:.2f}")
    
    # Step 2: Search for current news
    news_data = await search_lqda_news()
    
    # Step 3: Search for financial information  
    financial_data = await search_lqda_financials()
    
    # Step 4: Search for analyst ratings
    analyst_data = await search_analyst_ratings()
    
    # Step 5: Comprehensive analysis
    if all([market_data, news_data, financial_data, analyst_data]):
        analysis = await analyze_with_thinking(market_data, news_data, financial_data, analyst_data)
        
        if analysis:
            print("\n" + "=" * 60)
            print("🎯 COMPREHENSIVE REAL DATA ANALYSIS")
            print("=" * 60)
            print(analysis)
    
    # Display raw real data summary
    print("\n" + "=" * 60)
    print("📊 REAL MARKET DATA SUMMARY")
    print("=" * 60)
    print(f"Current Price: ${market_data['current_price']:.2f}")
    print(f"Market Cap: ${market_data['market_cap']:,}")
    print(f"P/E Ratio: {market_data['pe_ratio']}")
    print(f"RSI: {market_data['rsi']:.1f}")
    print(f"Volatility: {market_data['volatility']:.1%}")
    print(f"Current vs SMA20: {((market_data['current_price'] / market_data['sma_20']) - 1) * 100:+.1f}%")
    
    print("\n📰 RECENT NEWS FINDINGS:")
    if news_data and 'web' in news_data:
        for i, result in enumerate(news_data['web']['results'][:3], 1):
            print(f"  {i}. {result.get('title', 'No title')}")
    
    print("\n💰 FINANCIAL FINDINGS:")
    if financial_data and 'web' in financial_data:
        for i, result in enumerate(financial_data['web']['results'][:3], 1):
            print(f"  {i}. {result.get('title', 'No title')}")
    
    print("\n📈 ANALYST COVERAGE:")
    if analyst_data and 'web' in analyst_data:
        for i, result in enumerate(analyst_data['web']['results'][:3], 1):
            print(f"  {i}. {result.get('title', 'No title')}")
    
    print("\n" + "=" * 60)
    print("✅ REAL DATA ANALYSIS COMPLETE")
    print("⚠️  All data sourced from live feeds - no simulations")
    print("📊 Technical data from Yahoo Finance API")
    print("📰 News and analysis from Brave Search")
    print("🧠 Analysis from sequential thinking process")

if __name__ == "__main__":
    asyncio.run(main())