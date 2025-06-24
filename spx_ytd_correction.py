#!/usr/bin/env python3
"""
📊 S&P 500 (SPX) - YTD 2025 Performance Correction
Getting accurate year-to-date performance from January 1, 2025
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, date

def get_real_ytd_performance():
    """Calculate accurate YTD performance for SPX"""
    
    # Get S&P 500 data
    ticker = yf.Ticker("^GSPC")
    
    # Get data from start of 2025
    hist = ticker.history(start="2025-01-01", end=datetime.now().strftime("%Y-%m-%d"))
    
    if hist.empty:
        print("❌ No data available")
        return
    
    # Calculate YTD performance
    start_2025_price = hist['Close'].iloc[0]  # First trading day of 2025
    current_price = hist['Close'].iloc[-1]    # Latest close
    
    ytd_return = ((current_price - start_2025_price) / start_2025_price) * 100
    
    print("📊 S&P 500 (SPX) - CORRECTED YTD 2025 PERFORMANCE")
    print("=" * 55)
    print()
    print(f"🗓️ Start of 2025: {hist.index[0].strftime('%B %d, %Y')}")
    print(f"📈 Opening Price (Jan 1, 2025): {start_2025_price:,.2f}")
    print(f"📊 Current Price: {current_price:,.2f}")
    print(f"📈 YTD Return: {ytd_return:+.1f}%")
    print()
    
    # Additional context
    if ytd_return > 0:
        print(f"✅ SPX is UP {ytd_return:.1f}% year-to-date in 2025")
    else:
        print(f"❌ SPX is DOWN {abs(ytd_return):.1f}% year-to-date in 2025")
    
    print()
    print("📊 Monthly Breakdown (2025):")
    print("-" * 30)
    
    # Calculate monthly returns
    monthly_data = hist.groupby(hist.index.to_period('M'))['Close'].last()
    prev_price = start_2025_price
    
    for month, price in monthly_data.items():
        monthly_return = ((price - prev_price) / prev_price) * 100
        print(f"{month}: {monthly_return:+.1f}%")
        prev_price = price
    
    print()
    print("⚠️ NOTE: This is the actual YTD performance from January 1, 2025")
    print("📊 Previous analysis incorrectly showed 1-year performance (+10.6%)")
    
    return ytd_return, current_price

if __name__ == "__main__":
    get_real_ytd_performance()