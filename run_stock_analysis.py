#!/usr/bin/env python3
"""
Stock analysis using the RESEARCHER agent
"""
import os
import sys

# Add agents directory to path
agents_dir = os.path.join(os.path.dirname(__file__), 'agents')
sys.path.insert(0, agents_dir)

from ask_researcher import ResearcherAgent

def analyze_stock(symbol):
    """Analyze a stock using the researcher agent"""
    
    agent = ResearcherAgent()
    
    query = f"""
    Conduct comprehensive market intelligence research on {symbol} stock including:
    
    1. Current financial health and key metrics
    2. Recent earnings performance and guidance
    3. Industry position and competitive analysis
    4. Market sentiment and analyst ratings
    5. Technical analysis indicators
    6. Risk factors and growth catalysts
    7. Investment thesis and recommendation
    
    Focus on actionable insights for trading decisions.
    """
    
    print(f"🔍 Analyzing {symbol} stock...")
    print("="*50)
    
    try:
        response = agent.research(query, mode="market_intelligence")
        print(response)
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "WMT"
    analyze_stock(symbol)