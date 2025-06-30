#!/usr/bin/env python3
"""
IWM Technical and Fundamental Analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_iwm():
    """Perform comprehensive analysis of IWM ETF"""
    
    print('🔍 IWM (iShares Russell 2000 ETF) - Technical & Fundamental Analysis')
    print('=' * 80)
    
    try:
        # Get IWM data
        iwm = yf.Ticker('IWM')
        info = iwm.info
        
        # Basic Fund Information
        print('📊 FUND OVERVIEW')
        print('-' * 40)
        print(f'Fund Name: {info.get("longName", "iShares Russell 2000 ETF")}')
        print(f'Current Price: ${info.get("currentPrice", info.get("regularMarketPrice", "N/A"))}')
        print(f'52-Week High: ${info.get("fiftyTwoWeekHigh", "N/A")}')
        print(f'52-Week Low: ${info.get("fiftyTwoWeekLow", "N/A")}')
        print(f'Total Assets: ${info.get("totalAssets", "N/A")}')
        print(f'Average Volume: {info.get("averageVolume", "N/A"):,}')
        print(f'Expense Ratio: {info.get("annualReportExpenseRatio", info.get("expenseRatio", "N/A"))}')
        print(f'Dividend Yield: {info.get("dividendYield", "N/A")}')
        print(f'Beta: {info.get("beta", "N/A")}')
        
        # Get historical data
        hist_6m = iwm.history(period='6mo')
        hist_1y = iwm.history(period='1y')
        hist_2y = iwm.history(period='2y')
        
        if not hist_6m.empty:
            current_price = hist_6m['Close'].iloc[-1]
            
            print('\n📈 TECHNICAL ANALYSIS')
            print('-' * 40)
            
            # Moving Averages
            sma20 = hist_6m['Close'].rolling(20).mean().iloc[-1]
            sma50 = hist_6m['Close'].rolling(50).mean().iloc[-1]
            
            if not hist_1y.empty:
                sma200 = hist_1y['Close'].rolling(200).mean().iloc[-1]
            else:
                sma200 = None
            
            print(f'Current Price: ${current_price:.2f}')
            print(f'20-day SMA: ${sma20:.2f}')
            print(f'50-day SMA: ${sma50:.2f}')
            if sma200:
                print(f'200-day SMA: ${sma200:.2f}')
            
            # Price vs moving averages
            vs_sma20 = ((current_price - sma20) / sma20) * 100
            vs_sma50 = ((current_price - sma50) / sma50) * 100
            
            print(f'\nPrice vs Moving Averages:')
            print(f'vs 20-day SMA: {vs_sma20:+.1f}%')
            print(f'vs 50-day SMA: {vs_sma50:+.1f}%')
            
            if sma200:
                vs_sma200 = ((current_price - sma200) / sma200) * 100
                print(f'vs 200-day SMA: {vs_sma200:+.1f}%')
            
            # Volatility Analysis
            returns_6m = hist_6m['Close'].pct_change().dropna()
            volatility_6m = returns_6m.std() * (252**0.5) * 100
            
            if not hist_1y.empty:
                returns_1y = hist_1y['Close'].pct_change().dropna()
                volatility_1y = returns_1y.std() * (252**0.5) * 100
            else:
                volatility_1y = None
            
            print(f'\nVolatility:')
            print(f'6-Month Annualized: {volatility_6m:.1f}%')
            if volatility_1y:
                print(f'1-Year Annualized: {volatility_1y:.1f}%')
            
            # Performance
            perf_1m = ((current_price - hist_6m['Close'].iloc[-22]) / hist_6m['Close'].iloc[-22]) * 100 if len(hist_6m) >= 22 else None
            perf_3m = ((current_price - hist_6m['Close'].iloc[-66]) / hist_6m['Close'].iloc[-66]) * 100 if len(hist_6m) >= 66 else None
            perf_6m = ((current_price - hist_6m['Close'].iloc[0]) / hist_6m['Close'].iloc[0]) * 100
            
            if not hist_1y.empty:
                perf_1y = ((current_price - hist_1y['Close'].iloc[0]) / hist_1y['Close'].iloc[0]) * 100
            else:
                perf_1y = None
                
            if not hist_2y.empty:
                perf_2y = ((current_price - hist_2y['Close'].iloc[0]) / hist_2y['Close'].iloc[0]) * 100
            else:
                perf_2y = None
            
            print(f'\nPerformance:')
            if perf_1m: print(f'1-Month: {perf_1m:+.1f}%')
            if perf_3m: print(f'3-Month: {perf_3m:+.1f}%')
            print(f'6-Month: {perf_6m:+.1f}%')
            if perf_1y: print(f'1-Year: {perf_1y:+.1f}%')
            if perf_2y: print(f'2-Year: {perf_2y:+.1f}%')
            
            # RSI Calculation
            def calculate_rsi(prices, period=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            rsi = calculate_rsi(hist_6m['Close']).iloc[-1]
            print(f'\nTechnical Indicators:')
            print(f'RSI (14): {rsi:.1f}')
            
            # Support/Resistance levels
            recent_high = hist_6m['High'].rolling(20).max().iloc[-1]
            recent_low = hist_6m['Low'].rolling(20).min().iloc[-1]
            print(f'20-day High: ${recent_high:.2f}')
            print(f'20-day Low: ${recent_low:.2f}')
        
        print('\n📊 FUNDAMENTAL ANALYSIS (ETF Characteristics)')
        print('-' * 40)
        print(f'Index Tracked: Russell 2000')
        print(f'Asset Class: Small-Cap U.S. Equities')
        print(f'Sector Diversification: Broad small-cap exposure')
        print(f'Holdings Count: ~2000 companies')
        print(f'Methodology: Market-cap weighted')
        
        # Market Context
        print(f'\n🌍 MARKET CONTEXT')
        print('-' * 40)
        print(f'ETF Type: Passive Index Fund')
        print(f'Primary Benchmark: Russell 2000 Index')
        print(f'Risk Profile: Higher volatility than large-cap')
        print(f'Economic Sensitivity: High (small-caps sensitive to economic cycles)')
        
        # Technical Summary
        print(f'\n🎯 TECHNICAL SUMMARY')
        print('-' * 40)
        
        # Trend analysis
        if sma20 > sma50:
            short_trend = "Bullish (SMA20 > SMA50)"
        else:
            short_trend = "Bearish (SMA20 < SMA50)"
        
        print(f'Short-term Trend: {short_trend}')
        
        if rsi > 70:
            rsi_signal = "Overbought"
        elif rsi < 30:
            rsi_signal = "Oversold"
        else:
            rsi_signal = "Neutral"
        
        print(f'RSI Signal: {rsi_signal}')
        
        # Volume analysis
        recent_volume = hist_6m['Volume'].rolling(10).mean().iloc[-1]
        avg_volume = hist_6m['Volume'].mean()
        volume_ratio = recent_volume / avg_volume
        
        if volume_ratio > 1.2:
            volume_signal = "Above Average (High Interest)"
        elif volume_ratio < 0.8:
            volume_signal = "Below Average (Low Interest)"
        else:
            volume_signal = "Normal"
        
        print(f'Volume Trend: {volume_signal}')
        
        print(f'\n✅ Analysis completed successfully!')
        
    except Exception as e:
        print(f'❌ Error during analysis: {e}')

if __name__ == "__main__":
    analyze_iwm()