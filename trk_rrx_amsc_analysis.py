#!/usr/bin/env python3
"""
📊 TRK, RRX & AMSC - Comprehensive Stock Analysis
Premium analysis for three distinct investment opportunities
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

async def analyze_stock_comprehensive(symbol, company_name, sector_info):
    """Generate comprehensive analysis for individual stock"""
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1y")
        
        if hist.empty:
            print(f"❌ No data available for {symbol}")
            return
        
        # Calculate technical indicators
        close_prices = hist['Close']
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        sma_20 = close_prices.rolling(window=20).mean()
        sma_50 = close_prices.rolling(window=50).mean()
        
        returns = close_prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        current_price = info.get('currentPrice', close_prices.iloc[-1])
        ytd_start = hist.iloc[0]['Close'] if len(hist) > 100 else close_prices.iloc[0]
        ytd_return = ((current_price - ytd_start) / ytd_start) * 100
        
        # ATR for stop placement
        high_prices = hist['High']
        low_prices = hist['Low']
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift(1))
        tr3 = abs(low_prices - close_prices.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        current_atr = atr.iloc[-1]
        
        print(f"📊 {company_name} ({symbol}) - Comprehensive Analysis")
        print("=" * 80)
        print()
        
        print("🏢 Company Overview")
        print("-" * 18)
        print(f"{sector_info}")
        print()
        
        print("💰 Financial Performance")
        print("-" * 25)
        print()
        print("Current Market Metrics")
        print()
        print(f"  - Current Price: ${current_price:.2f}")
        print(f"  - YTD Return: {ytd_return:+.1f}%")
        print(f"  - Market Cap: ${info.get('marketCap', 0):,}")
        print(f"  - Volume: {info.get('volume', 0):,}")
        print(f"  - 52-Week Range: ${info.get('fiftyTwoWeekLow', 0):.2f} - ${info.get('fiftyTwoWeekHigh', 0):.2f}")
        print()
        
        print("Key Financial Ratios")
        print()
        print(f"  - P/E Ratio: {info.get('trailingPE', 'N/A')}")
        print(f"  - Forward P/E: {info.get('forwardPE', 'N/A')}")
        print(f"  - Price/Book: {info.get('priceToBook', 'N/A')}")
        print(f"  - Debt/Equity: {info.get('debtToEquity', 'N/A')}")
        print(f"  - ROE: {info.get('returnOnEquity', 'N/A')}")
        print(f"  - Profit Margin: {info.get('profitMargins', 'N/A')}")
        print()
        
        print("📈 Technical Analysis")
        print("-" * 20)
        print()
        print(f"  - RSI (14): {float(rsi.iloc[-1]):.1f} {'🔴 OVERSOLD' if float(rsi.iloc[-1]) < 30 else '🟡 NEUTRAL' if float(rsi.iloc[-1]) < 70 else '🔴 OVERBOUGHT'}")
        print(f"  - 20-Day SMA: ${float(sma_20.iloc[-1]):.2f}")
        print(f"  - 50-Day SMA: ${float(sma_50.iloc[-1]):.2f}")
        print(f"  - Price vs SMA20: {((current_price / float(sma_20.iloc[-1])) - 1) * 100:+.1f}%")
        print(f"  - Volatility: {volatility:.1%} annualized")
        print(f"  - ATR (14): ${current_atr:.2f}")
        print()
        
        print("🎯 Entry Points & Risk Management")
        print("-" * 33)
        print()
        
        # Entry strategies based on current technical position
        current_rsi_val = float(rsi.iloc[-1])
        sma20_val = float(sma_20.iloc[-1])
        sma50_val = float(sma_50.iloc[-1])
        
        print("BULLISH ENTRY STRATEGY")
        print("=" * 22)
        if current_rsi_val < 40:
            print(f"🟢 OVERSOLD BOUNCE SETUP")
            entry_price = current_price - (current_atr * 0.5)
            stop_price = current_price - (current_atr * 2)
            target_price = current_price + (current_atr * 2)
        else:
            print(f"📊 PULLBACK TO SUPPORT")
            entry_price = sma20_val
            stop_price = sma50_val - (current_atr * 1)
            target_price = current_price + (current_atr * 2)
        
        print(f"Entry: ${entry_price:.2f}")
        print(f"Stop Loss: ${stop_price:.2f}")
        print(f"Target: ${target_price:.2f}")
        print(f"Risk: {((entry_price - stop_price) / entry_price) * 100:.1f}%")
        print(f"Reward: {((target_price - entry_price) / entry_price) * 100:.1f}%")
        print(f"R/R Ratio: {((target_price - entry_price) / (entry_price - stop_price)):.1f}:1")
        print()
        
        print("BEARISH ENTRY STRATEGY")
        print("=" * 22)
        if current_rsi_val > 70:
            print(f"🔴 OVERBOUGHT FADE SETUP")
            entry_price = current_price + (current_atr * 0.5)
            stop_price = current_price + (current_atr * 2)
            target_price = current_price - (current_atr * 2)
        else:
            print(f"📉 BREAKDOWN TRADE")
            entry_price = sma50_val - (current_atr * 0.5)
            stop_price = sma20_val + (current_atr * 1)
            target_price = sma50_val - (current_atr * 3)
        
        print(f"Entry: ${entry_price:.2f}")
        print(f"Stop Loss: ${stop_price:.2f}")
        print(f"Target: ${target_price:.2f}")
        print(f"Risk: {((stop_price - entry_price) / entry_price) * 100:.1f}%")
        print(f"Reward: {((entry_price - target_price) / entry_price) * 100:.1f}%")
        print(f"R/R Ratio: {((entry_price - target_price) / (stop_price - entry_price)):.1f}:1")
        print()
        
        # Investment rating based on technical and fundamental factors
        pe_ratio = info.get('trailingPE', 0)
        market_cap = info.get('marketCap', 0)
        
        print("🎯 Investment Rating")
        print("-" * 19)
        
        # Simple scoring system
        score = 0
        if current_rsi_val < 30:
            score += 2  # Oversold
        elif current_rsi_val < 50:
            score += 1  # Neutral-bullish
        
        if current_price > sma20_val:
            score += 1  # Above short-term trend
        if current_price > sma50_val:
            score += 1  # Above medium-term trend
            
        if pe_ratio and 10 < pe_ratio < 25:
            score += 1  # Reasonable valuation
        
        if market_cap > 1e9:
            score += 1  # Large/mid cap stability
            
        if score >= 5:
            rating = "STRONG BUY 🚀"
        elif score >= 4:
            rating = "BUY 📈"
        elif score >= 3:
            rating = "HOLD ⚖️"
        elif score >= 2:
            rating = "WEAK HOLD ⚠️"
        else:
            rating = "SELL 📉"
            
        print(f"Rating: {rating}")
        print(f"Score: {score}/6")
        print()
        
        print("⚠️ Key Risk Factors")
        print("-" * 19)
        print(f"• Volatility: {volatility:.1%} (High volatility = higher risk)")
        print(f"• Liquidity: {info.get('volume', 0):,} avg volume")
        print(f"• Sector Risk: {info.get('sector', 'Unknown')} sector exposure")
        print()
        
        print("=" * 80)
        print()
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        print()

async def main():
    """Analyze all three stocks comprehensively"""
    
    print("📊 COMPREHENSIVE STOCK ANALYSIS - TRK, RRX & AMSC")
    print("=" * 60)
    print("🎯 Entry Points, Stop Losses & Investment Ratings")
    print("-" * 60)
    print()
    
    # Note: TRK might be TDF (Templeton Dragon Fund) based on search results
    await analyze_stock_comprehensive(
        "TDF", 
        "Templeton Dragon Fund", 
        "Closed-end fund focused on Chinese equity investments through a diversified\nportfolio of Greater China companies. Provides exposure to China's economic\ngrowth with professional management and regular distributions."
    )
    
    await analyze_stock_comprehensive(
        "RRX", 
        "Regal Rexnord Corporation", 
        "Industrial technology company providing motion control and power transmission\nsolutions. Serves diverse end markets including aerospace, food & beverage,\nenergy, and general industrial applications through two primary segments:\nMotion Control and Power & Motion."
    )
    
    await analyze_stock_comprehensive(
        "AMSC", 
        "American Superconductor Corporation", 
        "Technology company providing megawatt-scale power resiliency solutions.\nFocuses on wind energy systems, grid interconnection solutions, and\nsuperconductor technologies for power grid optimization and renewable\nenergy integration."
    )
    
    print("🎯 PORTFOLIO ALLOCATION RECOMMENDATIONS")
    print("=" * 42)
    print()
    print("DIVERSIFICATION STRATEGY")
    print("=" * 23)
    print("• TDF: 2-5% allocation (International/China exposure)")
    print("• RRX: 3-7% allocation (Industrial dividend play)")
    print("• AMSC: 1-3% allocation (High-growth tech speculation)")
    print("• Total: 6-15% combined allocation maximum")
    print()
    
    print("RISK MANAGEMENT")
    print("=" * 15)
    print("• Never risk more than 2% per individual position")
    print("• Use ATR-based stops for all positions")
    print("• Monitor correlations during market stress")
    print("• Rebalance quarterly or on 20% moves")
    print()
    
    print("KEY MONITORING POINTS")
    print("=" * 21)
    print("• TDF: China economic data, regulatory changes")
    print("• RRX: Industrial demand, commodity costs")
    print("• AMSC: Renewable energy policies, grid infrastructure spending")
    print("• All: Fed policy, market breadth, sector rotation")
    print()
    
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
    print("⚠️  Use proper position sizing and risk management")
    print("📊 Monitor technical levels and fundamental developments")

if __name__ == "__main__":
    asyncio.run(main())