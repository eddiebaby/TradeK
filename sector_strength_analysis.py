#!/usr/bin/env python3
"""
📊 S&P 500 SECTOR STRENGTH/WEAKNESS & CYCLE ANALYSIS
Comprehensive sector rotation and momentum analysis with real-time data
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Sector ETF mapping for comprehensive analysis
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials', 
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services'
}

async def calculate_sector_metrics(symbol, sector_name):
    """Calculate comprehensive metrics for sector analysis"""
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get different timeframes for cycle analysis
        hist_1y = ticker.history(period="1y")
        hist_6m = ticker.history(period="6mo")
        hist_3m = ticker.history(period="3mo")
        hist_1m = ticker.history(period="1mo")
        hist_5d = ticker.history(period="5d")
        
        if hist_1y.empty:
            print(f"❌ No data available for {symbol}")
            return None
        
        current_price = hist_1y['Close'].iloc[-1]
        
        # Performance calculations across timeframes
        ytd_start = hist_1y.iloc[0]['Close'] if len(hist_1y) > 100 else hist_1y['Close'].iloc[0]
        
        perf_1y = ((current_price - hist_1y['Close'].iloc[0]) / hist_1y['Close'].iloc[0]) * 100
        perf_6m = ((current_price - hist_6m['Close'].iloc[0]) / hist_6m['Close'].iloc[0]) * 100
        perf_3m = ((current_price - hist_3m['Close'].iloc[0]) / hist_3m['Close'].iloc[0]) * 100
        perf_1m = ((current_price - hist_1m['Close'].iloc[0]) / hist_1m['Close'].iloc[0]) * 100
        perf_5d = ((current_price - hist_5d['Close'].iloc[0]) / hist_5d['Close'].iloc[0]) * 100
        
        # Technical indicators
        close_prices = hist_1y['Close']
        
        # RSI calculation
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Moving averages
        sma_20 = close_prices.rolling(window=20).mean()
        sma_50 = close_prices.rolling(window=50).mean()
        sma_200 = close_prices.rolling(window=200).mean()
        
        # Trend analysis
        trend_short = "BULLISH" if current_price > sma_20.iloc[-1] else "BEARISH"
        trend_medium = "BULLISH" if current_price > sma_50.iloc[-1] else "BEARISH"
        trend_long = "BULLISH" if current_price > sma_200.iloc[-1] else "BEARISH"
        
        # Momentum analysis
        momentum_score = 0
        if perf_5d > 0: momentum_score += 1
        if perf_1m > 0: momentum_score += 1
        if perf_3m > 0: momentum_score += 1
        if current_rsi > 50: momentum_score += 1
        if current_price > sma_20.iloc[-1]: momentum_score += 1
        if current_price > sma_50.iloc[-1]: momentum_score += 1
        
        # Volatility
        returns = close_prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Volume analysis
        avg_volume = hist_1m['Volume'].mean()
        recent_volume = hist_5d['Volume'].iloc[-1]
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        return {
            'symbol': symbol,
            'sector': sector_name,
            'current_price': current_price,
            'perf_1y': perf_1y,
            'perf_6m': perf_6m,
            'perf_3m': perf_3m,
            'perf_1m': perf_1m,
            'perf_5d': perf_5d,
            'rsi': current_rsi,
            'trend_short': trend_short,
            'trend_medium': trend_medium,
            'trend_long': trend_long,
            'momentum_score': momentum_score,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'sma_20': sma_20.iloc[-1],
            'sma_50': sma_50.iloc[-1],
            'sma_200': sma_200.iloc[-1]
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        return None

async def analyze_sector_rotation():
    """Comprehensive sector strength/weakness analysis"""
    
    print("📊 S&P 500 SECTOR STRENGTH/WEAKNESS & CYCLE ANALYSIS")
    print("=" * 80)
    print("🎯 Real-time sector rotation and momentum analysis")
    print("-" * 80)
    print()
    
    # Collect all sector data
    sector_data = []
    
    print("📈 Fetching sector data...")
    for symbol, sector in SECTOR_ETFS.items():
        print(f"  Analyzing {sector} ({symbol})...")
        data = await calculate_sector_metrics(symbol, sector)
        if data:
            sector_data.append(data)
    
    if not sector_data:
        print("❌ No sector data available")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(sector_data)
    
    print()
    print("📊 SECTOR PERFORMANCE SUMMARY")
    print("=" * 50)
    print()
    
    # Performance rankings
    print("🏆 PERFORMANCE RANKINGS")
    print("-" * 25)
    print()
    
    timeframes = [
        ('5d', '5-Day'),
        ('1m', '1-Month'),
        ('3m', '3-Month'),
        ('6m', '6-Month'),
        ('1y', '1-Year')
    ]
    
    for period, label in timeframes:
        col = f'perf_{period}'
        df_sorted = df.sort_values(col, ascending=False)
        
        print(f"{label} Performance Leaders:")
        print()
        for i, row in df_sorted.head(3).iterrows():
            perf = row[col]
            trend_emoji = "🟢" if perf > 0 else "🔴"
            print(f"  {trend_emoji} {row['sector']:25} ({row['symbol']}): {perf:+.1f}%")
        
        print()
        print(f"{label} Performance Laggards:")
        print()
        for i, row in df_sorted.tail(3).iterrows():
            perf = row[col]
            trend_emoji = "🟢" if perf > 0 else "🔴"
            print(f"  {trend_emoji} {row['sector']:25} ({row['symbol']}): {perf:+.1f}%")
        print()
    
    print("🎯 MOMENTUM & STRENGTH ANALYSIS")
    print("=" * 35)
    print()
    
    # Momentum ranking
    df_momentum = df.sort_values('momentum_score', ascending=False)
    
    print("⚡ MOMENTUM LEADERS (Score 0-6)")
    print("-" * 30)
    print()
    for i, row in df_momentum.iterrows():
        score = row['momentum_score']
        if score >= 5:
            strength = "🔥 VERY STRONG"
        elif score >= 4:
            strength = "💪 STRONG"
        elif score >= 3:
            strength = "⚖️ NEUTRAL"
        elif score >= 2:
            strength = "⚠️ WEAK"
        else:
            strength = "❌ VERY WEAK"
            
        print(f"  {strength:15} {row['sector']:25} (Score: {score}/6)")
    
    print()
    print("📊 TECHNICAL TREND ANALYSIS")
    print("=" * 30)
    print()
    
    # Trend analysis by timeframe
    trend_summary = {
        'BULLISH_ALL': [],
        'BULLISH_MAJORITY': [],
        'MIXED': [],
        'BEARISH_MAJORITY': [],
        'BEARISH_ALL': []
    }
    
    for i, row in df.iterrows():
        bullish_count = sum([
            row['trend_short'] == 'BULLISH',
            row['trend_medium'] == 'BULLISH', 
            row['trend_long'] == 'BULLISH'
        ])
        
        if bullish_count == 3:
            trend_summary['BULLISH_ALL'].append(row)
        elif bullish_count == 2:
            trend_summary['BULLISH_MAJORITY'].append(row)
        elif bullish_count == 1:
            trend_summary['MIXED'].append(row)
        elif bullish_count == 0:
            trend_summary['BEARISH_ALL'].append(row)
    
    print("🟢 BULLISH ACROSS ALL TIMEFRAMES")
    print("-" * 35)
    for row in trend_summary['BULLISH_ALL']:
        rsi_status = "Overbought" if row['rsi'] > 70 else "Oversold" if row['rsi'] < 30 else "Neutral"
        print(f"  • {row['sector']:25} RSI: {row['rsi']:.1f} ({rsi_status})")
    
    print()
    print("🟡 MIXED TREND SIGNALS")
    print("-" * 22)
    for row in trend_summary['MIXED'] + trend_summary['BULLISH_MAJORITY']:
        print(f"  • {row['sector']:25} Short: {row['trend_short']:7} Med: {row['trend_medium']:7} Long: {row['trend_long']}")
    
    print()
    print("🔴 BEARISH ACROSS ALL TIMEFRAMES") 
    print("-" * 35)
    for row in trend_summary['BEARISH_ALL']:
        rsi_status = "Overbought" if row['rsi'] > 70 else "Oversold" if row['rsi'] < 30 else "Neutral"
        print(f"  • {row['sector']:25} RSI: {row['rsi']:.1f} ({rsi_status})")
    
    print()
    print("🔄 SECTOR ROTATION ANALYSIS")
    print("=" * 30)
    print()
    
    # Economic cycle analysis
    defensive_sectors = ['XLU', 'XLP', 'XLV', 'XLRE']
    cyclical_sectors = ['XLK', 'XLY', 'XLI', 'XLB', 'XLE']
    financial_sectors = ['XLF']
    
    defensive_perf = df[df['symbol'].isin(defensive_sectors)]['perf_3m'].mean()
    cyclical_perf = df[df['symbol'].isin(cyclical_sectors)]['perf_3m'].mean()
    financial_perf = df[df['symbol'].isin(financial_sectors)]['perf_3m'].mean()
    
    print("📈 ECONOMIC CYCLE POSITIONING")
    print("-" * 32)
    print()
    print(f"  Cyclical Sectors (3M Avg):   {cyclical_perf:+.1f}%")
    print(f"  Financial Sector (3M):       {financial_perf:+.1f}%") 
    print(f"  Defensive Sectors (3M Avg):  {defensive_perf:+.1f}%")
    print()
    
    if cyclical_perf > defensive_perf + 2:
        cycle_phase = "🚀 RISK-ON (Growth/Expansion)"
        cycle_desc = "Market favoring growth and cyclical sectors"
    elif defensive_perf > cyclical_perf + 2:
        cycle_phase = "🛡️ RISK-OFF (Defensive/Contraction)"
        cycle_desc = "Market seeking safety in defensive sectors"
    else:
        cycle_phase = "⚖️ BALANCED (Transition)"
        cycle_desc = "Mixed signals, potential sector rotation in progress"
    
    print(f"Current Market Cycle: {cycle_phase}")
    print(f"Interpretation: {cycle_desc}")
    print()
    
    print("🎯 SECTOR TRADING OPPORTUNITIES")
    print("=" * 35)
    print()
    
    # Identify trading opportunities
    opportunities = []
    
    for i, row in df.iterrows():
        signals = []
        
        # Oversold opportunities
        if row['rsi'] < 30 and row['perf_5d'] < -2:
            signals.append("OVERSOLD BOUNCE")
        
        # Momentum breakouts
        if row['momentum_score'] >= 5 and row['perf_5d'] > 1:
            signals.append("MOMENTUM BREAKOUT")
        
        # Relative strength
        if row['perf_1m'] > df['perf_1m'].quantile(0.75):
            signals.append("RELATIVE STRENGTH")
        
        # Mean reversion
        if row['rsi'] > 70 and row['perf_5d'] > 3:
            signals.append("OVERBOUGHT - FADE")
        
        if signals:
            opportunities.append({
                'sector': row['sector'],
                'symbol': row['symbol'],
                'signals': signals,
                'current_price': row['current_price'],
                'rsi': row['rsi'],
                'perf_5d': row['perf_5d']
            })
    
    print("🟢 BULLISH OPPORTUNITIES")
    print("-" * 23)
    for opp in opportunities:
        bullish_signals = [s for s in opp['signals'] if s in ['OVERSOLD BOUNCE', 'MOMENTUM BREAKOUT', 'RELATIVE STRENGTH']]
        if bullish_signals:
            print(f"  • {opp['sector']:25} ({opp['symbol']})")
            for signal in bullish_signals:
                print(f"    └─ {signal}")
            print(f"    RSI: {opp['rsi']:.1f} | 5D: {opp['perf_5d']:+.1f}%")
            print()
    
    print("🔴 BEARISH OPPORTUNITIES")
    print("-" * 23)
    for opp in opportunities:
        bearish_signals = [s for s in opp['signals'] if 'FADE' in s]
        if bearish_signals:
            print(f"  • {opp['sector']:25} ({opp['symbol']})")
            for signal in bearish_signals:
                print(f"    └─ {signal}")
            print(f"    RSI: {opp['rsi']:.1f} | 5D: {opp['perf_5d']:+.1f}%")
            print()
    
    print("📊 VOLATILITY & RISK ANALYSIS")
    print("=" * 32)
    print()
    
    df_vol = df.sort_values('volatility', ascending=False)
    
    print("⚡ HIGHEST VOLATILITY (Risk)")
    print("-" * 28)
    for i, row in df_vol.head(3).iterrows():
        print(f"  • {row['sector']:25} Volatility: {row['volatility']:.1f}%")
    
    print()
    print("🛡️ LOWEST VOLATILITY (Stability)")
    print("-" * 32)
    for i, row in df_vol.tail(3).iterrows():
        print(f"  • {row['sector']:25} Volatility: {row['volatility']:.1f}%")
    
    print()
    print("📈 VOLUME ANALYSIS")
    print("=" * 20)
    print()
    
    df_vol_ratio = df.sort_values('volume_ratio', ascending=False)
    
    print("🔥 ABOVE AVERAGE VOLUME (Interest)")
    print("-" * 35)
    for i, row in df_vol_ratio.iterrows():
        if row['volume_ratio'] > 1.2:
            print(f"  • {row['sector']:25} Volume Ratio: {row['volume_ratio']:.1f}x")
    
    print()
    print("🎯 ACTIONABLE INVESTMENT STRATEGY")
    print("=" * 35)
    print()
    
    # Strategy recommendations
    top_momentum = df_momentum.head(3)
    worst_momentum = df_momentum.tail(2)
    
    print("RECOMMENDED ACTIONS:")
    print("-" * 20)
    print()
    print("✅ OVERWEIGHT POSITIONS:")
    for i, row in top_momentum.iterrows():
        if row['momentum_score'] >= 4:
            print(f"  • {row['sector']:25} ({row['symbol']}) - Strong momentum")
    
    print()
    print("⚠️ UNDERWEIGHT POSITIONS:")
    for i, row in worst_momentum.iterrows():
        if row['momentum_score'] <= 2:
            print(f"  • {row['sector']:25} ({row['symbol']}) - Weak momentum")
    
    print()
    print("🔄 ROTATION PLAYS:")
    if cyclical_perf > defensive_perf + 2:
        print("  • Continue favoring cyclical/growth sectors")
        print("  • Technology, Consumer Discretionary, Industrials")
    elif defensive_perf > cyclical_perf + 2:
        print("  • Rotate into defensive sectors")
        print("  • Utilities, Consumer Staples, Healthcare")
    else:
        print("  • Balanced approach during transition period")
        print("  • Monitor for clear directional signals")
    
    print()
    print("=" * 80)
    print("✅ COMPREHENSIVE SECTOR ANALYSIS COMPLETE")
    print("⚠️  Monitor daily for rotation signals and momentum shifts")
    print("📊 Use sector ETFs for efficient diversified exposure")
    print("🎯 Combine with individual stock picking within strong sectors")

if __name__ == "__main__":
    asyncio.run(analyze_sector_rotation())