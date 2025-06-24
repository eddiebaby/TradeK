#!/usr/bin/env python3
"""
📊 S&P 500 SECTOR STRENGTH/WEAKNESS & CYCLE ANALYSIS (FIXED)
Comprehensive sector rotation and momentum analysis with robust data handling
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

def safe_performance_calc(current_price, start_price):
    """Safely calculate performance with NaN handling"""
    try:
        if pd.isna(current_price) or pd.isna(start_price) or start_price == 0:
            return np.nan
        return ((current_price - start_price) / start_price) * 100
    except:
        return np.nan

def get_safe_price_at_index(hist, index, fallback_days=5):
    """Safely get price at index with fallback"""
    try:
        if len(hist) > abs(index):
            return hist['Close'].iloc[index]
        elif len(hist) > fallback_days:
            return hist['Close'].iloc[-fallback_days]
        else:
            return hist['Close'].iloc[0]
    except:
        return np.nan

async def calculate_sector_metrics(symbol, sector_name):
    """Calculate comprehensive metrics for sector analysis with robust error handling"""
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get different timeframes for cycle analysis
        hist_1y = ticker.history(period="1y")
        
        if hist_1y.empty or len(hist_1y) < 5:
            print(f"❌ Insufficient data for {symbol}")
            return None
        
        current_price = hist_1y['Close'].iloc[-1]
        
        # Performance calculations with safe indexing
        # 5-day performance
        perf_5d = safe_performance_calc(current_price, get_safe_price_at_index(hist_1y, -5))
        
        # 1-month performance (approximately 20 trading days)
        perf_1m = safe_performance_calc(current_price, get_safe_price_at_index(hist_1y, -20, 10))
        
        # 3-month performance (approximately 60 trading days)
        perf_3m = safe_performance_calc(current_price, get_safe_price_at_index(hist_1y, -60, 30))
        
        # 6-month performance (approximately 120 trading days)
        perf_6m = safe_performance_calc(current_price, get_safe_price_at_index(hist_1y, -120, 60))
        
        # 1-year performance
        perf_1y = safe_performance_calc(current_price, hist_1y['Close'].iloc[0])
        
        # Technical indicators
        close_prices = hist_1y['Close']
        
        # RSI calculation with error handling
        try:
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            current_rsi = 50.0
        
        # Moving averages with error handling
        try:
            sma_20 = close_prices.rolling(window=20).mean()
            sma_50 = close_prices.rolling(window=50).mean()
            sma_200 = close_prices.rolling(window=200).mean()
            
            sma_20_val = sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else current_price
            sma_50_val = sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else current_price
            sma_200_val = sma_200.iloc[-1] if not pd.isna(sma_200.iloc[-1]) else current_price
        except:
            sma_20_val = sma_50_val = sma_200_val = current_price
        
        # Trend analysis
        trend_short = "BULLISH" if current_price > sma_20_val else "BEARISH"
        trend_medium = "BULLISH" if current_price > sma_50_val else "BEARISH"
        trend_long = "BULLISH" if current_price > sma_200_val else "BEARISH"
        
        # Momentum analysis (handle NaN values)
        momentum_score = 0
        if not pd.isna(perf_5d) and perf_5d > 0: momentum_score += 1
        if not pd.isna(perf_1m) and perf_1m > 0: momentum_score += 1
        if not pd.isna(perf_3m) and perf_3m > 0: momentum_score += 1
        if current_rsi > 50: momentum_score += 1
        if current_price > sma_20_val: momentum_score += 1
        if current_price > sma_50_val: momentum_score += 1
        
        # Volatility
        try:
            returns = close_prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            if pd.isna(volatility):
                volatility = 20.0  # default
        except:
            volatility = 20.0
        
        # Volume analysis
        try:
            if 'Volume' in hist_1y.columns:
                recent_volume = hist_1y['Volume'].iloc[-1]
                avg_volume = hist_1y['Volume'].tail(20).mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                if pd.isna(volume_ratio):
                    volume_ratio = 1.0
            else:
                volume_ratio = 1.0
        except:
            volume_ratio = 1.0
        
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
            'sma_20': sma_20_val,
            'sma_50': sma_50_val,
            'sma_200': sma_200_val
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        return None

def format_performance(value, show_emoji=True):
    """Format performance values with NaN handling"""
    if pd.isna(value):
        return "N/A"
    
    if show_emoji:
        emoji = "🟢" if value > 0 else "🔴"
        return f"{emoji} {value:+.1f}%"
    else:
        return f"{value:+.1f}%"

async def analyze_sector_rotation():
    """Comprehensive sector strength/weakness analysis with NaN handling"""
    
    print("📊 S&P 500 SECTOR STRENGTH/WEAKNESS & CYCLE ANALYSIS (FIXED)")
    print("=" * 85)
    print("🎯 Real-time sector rotation and momentum analysis")
    print("-" * 85)
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
        
        # Filter out NaN values for ranking
        df_valid = df[~pd.isna(df[col])]
        
        if df_valid.empty:
            print(f"{label} Performance: No valid data available")
            print()
            continue
        
        df_sorted = df_valid.sort_values(col, ascending=False)
        
        print(f"{label} Performance Leaders:")
        print()
        for idx, row in df_sorted.head(3).iterrows():
            perf = row[col]
            print(f"  {format_performance(perf)} {row['sector']:25} ({row['symbol']})")
        
        print()
        print(f"{label} Performance Laggards:")
        print()
        for idx, row in df_sorted.tail(3).iterrows():
            perf = row[col]
            print(f"  {format_performance(perf)} {row['sector']:25} ({row['symbol']})")
        print()
    
    print("🎯 MOMENTUM & STRENGTH ANALYSIS")
    print("=" * 35)
    print()
    
    # Momentum ranking
    df_momentum = df.sort_values('momentum_score', ascending=False)
    
    print("⚡ MOMENTUM LEADERS (Score 0-6)")
    print("-" * 30)
    print()
    for idx, row in df_momentum.iterrows():
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
    
    for idx, row in df.iterrows():
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
        rsi = row['rsi']
        rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        print(f"  • {row['sector']:25} RSI: {rsi:.1f} ({rsi_status})")
    
    print()
    print("🟡 MIXED TREND SIGNALS")
    print("-" * 22)
    for row in trend_summary['MIXED'] + trend_summary['BULLISH_MAJORITY']:
        print(f"  • {row['sector']:25} Short: {row['trend_short']:7} Med: {row['trend_medium']:7} Long: {row['trend_long']}")
    
    print()
    print("🔴 BEARISH ACROSS ALL TIMEFRAMES") 
    print("-" * 35)
    for row in trend_summary['BEARISH_ALL']:
        rsi = row['rsi']
        rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        print(f"  • {row['sector']:25} RSI: {rsi:.1f} ({rsi_status})")
    
    print()
    print("🔄 SECTOR ROTATION ANALYSIS")
    print("=" * 30)
    print()
    
    # Economic cycle analysis (only use sectors with valid 3M data)
    defensive_sectors = ['XLU', 'XLP', 'XLV', 'XLRE']
    cyclical_sectors = ['XLK', 'XLY', 'XLI', 'XLB', 'XLE']
    financial_sectors = ['XLF']
    
    # Calculate averages only for non-NaN values
    defensive_perf = df[df['symbol'].isin(defensive_sectors) & ~pd.isna(df['perf_3m'])]['perf_3m'].mean()
    cyclical_perf = df[df['symbol'].isin(cyclical_sectors) & ~pd.isna(df['perf_3m'])]['perf_3m'].mean()
    financial_perf = df[df['symbol'].isin(financial_sectors) & ~pd.isna(df['perf_3m'])]['perf_3m'].mean()
    
    print("📈 ECONOMIC CYCLE POSITIONING")
    print("-" * 32)
    print()
    print(f"  Cyclical Sectors (3M Avg):   {format_performance(cyclical_perf, False)}")
    print(f"  Financial Sector (3M):       {format_performance(financial_perf, False)}") 
    print(f"  Defensive Sectors (3M Avg):  {format_performance(defensive_perf, False)}")
    print()
    
    # Handle NaN values in cycle analysis
    if pd.isna(cyclical_perf) or pd.isna(defensive_perf):
        cycle_phase = "⚠️ INSUFFICIENT DATA"
        cycle_desc = "Unable to determine cycle phase due to missing data"
    elif cyclical_perf > defensive_perf + 2:
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
    
    # Identify trading opportunities (handle NaN values)
    opportunities = []
    
    for idx, row in df.iterrows():
        signals = []
        
        # Oversold opportunities
        if row['rsi'] < 30 and not pd.isna(row['perf_5d']) and row['perf_5d'] < -2:
            signals.append("OVERSOLD BOUNCE")
        
        # Momentum breakouts
        if row['momentum_score'] >= 5 and not pd.isna(row['perf_5d']) and row['perf_5d'] > 1:
            signals.append("MOMENTUM BREAKOUT")
        
        # Relative strength (use valid 1M data)
        valid_1m_data = df[~pd.isna(df['perf_1m'])]['perf_1m']
        if not valid_1m_data.empty and not pd.isna(row['perf_1m']):
            if row['perf_1m'] > valid_1m_data.quantile(0.75):
                signals.append("RELATIVE STRENGTH")
        
        # Mean reversion
        if row['rsi'] > 70 and not pd.isna(row['perf_5d']) and row['perf_5d'] > 3:
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
            print(f"    RSI: {opp['rsi']:.1f} | 5D: {format_performance(opp['perf_5d'], False)}")
            print()
    
    print("🔴 BEARISH OPPORTUNITIES")
    print("-" * 23)
    for opp in opportunities:
        bearish_signals = [s for s in opp['signals'] if 'FADE' in s]
        if bearish_signals:
            print(f"  • {opp['sector']:25} ({opp['symbol']})")
            for signal in bearish_signals:
                print(f"    └─ {signal}")
            print(f"    RSI: {opp['rsi']:.1f} | 5D: {format_performance(opp['perf_5d'], False)}")
            print()
    
    print("📊 DATA QUALITY SUMMARY")
    print("=" * 25)
    print()
    
    # Show data availability
    data_quality = {}
    for period in ['5d', '1m', '3m', '6m', '1y']:
        col = f'perf_{period}'
        valid_count = df[~pd.isna(df[col])].shape[0]
        total_count = df.shape[0]
        data_quality[period] = f"{valid_count}/{total_count}"
    
    print("📈 DATA AVAILABILITY:")
    for period, availability in data_quality.items():
        print(f"  {period.upper():3}: {availability} sectors have valid data")
    
    print()
    print("=" * 85)
    print("✅ COMPREHENSIVE SECTOR ANALYSIS COMPLETE (NaN VALUES HANDLED)")
    print("⚠️  Monitor daily for rotation signals and momentum shifts")
    print("📊 Use sector ETFs for efficient diversified exposure")
    print("🎯 Combine with individual stock picking within strong sectors")

if __name__ == "__main__":
    asyncio.run(analyze_sector_rotation())