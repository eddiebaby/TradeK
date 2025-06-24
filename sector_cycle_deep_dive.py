#!/usr/bin/env python3
"""
📊 SECTOR CYCLE DEEP DIVE ANALYSIS
Advanced economic cycle positioning and sector rotation insights
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

async def sector_cycle_analysis():
    """Deep dive into sector cycles and rotation patterns"""
    
    print("📊 SECTOR CYCLE DEEP DIVE ANALYSIS")
    print("=" * 60)
    print("🔄 Economic cycle positioning and rotation insights")
    print("-" * 60)
    print()
    
    # Enhanced sector classification with cycle sensitivity
    SECTOR_CYCLE_MAP = {
        # EARLY CYCLE (Economic Recovery)
        'early_cycle': {
            'XLF': 'Financials',
            'XLI': 'Industrials', 
            'XLB': 'Materials',
            'XLE': 'Energy'
        },
        
        # MID CYCLE (Economic Expansion) 
        'mid_cycle': {
            'XLK': 'Technology',
            'XLY': 'Consumer Discretionary',
            'XLC': 'Communication Services'
        },
        
        # LATE CYCLE (Economic Peak)
        'late_cycle': {
            'XLE': 'Energy',  # Also early cycle
            'XLB': 'Materials',  # Also early cycle
            'XLRE': 'Real Estate'
        },
        
        # RECESSION/DEFENSIVE (Economic Contraction)
        'defensive': {
            'XLU': 'Utilities',
            'XLP': 'Consumer Staples', 
            'XLV': 'Healthcare'
        }
    }
    
    # Get SPY for market comparison
    spy = yf.Ticker("SPY")
    spy_hist = spy.history(period="1y")
    spy_current = spy_hist['Close'].iloc[-1]
    spy_3m_start = spy_hist['Close'].iloc[-60]
    spy_3m_perf = ((spy_current - spy_3m_start) / spy_3m_start) * 100
    
    print(f"📈 MARKET CONTEXT")
    print("-" * 16)
    print(f"SPY 3-Month Performance: {spy_3m_perf:+.1f}%")
    print(f"Current Level: {spy_current:.2f}")
    print()
    
    # Analyze each cycle group
    cycle_performance = {}
    
    for cycle_phase, sectors in SECTOR_CYCLE_MAP.items():
        print(f"🔄 {cycle_phase.upper().replace('_', ' ')} SECTOR ANALYSIS")
        print("-" * 50)
        print()
        
        phase_data = []
        
        for symbol, sector_name in sectors.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                
                if hist.empty:
                    continue
                    
                current_price = hist['Close'].iloc[-1]
                
                # Multiple timeframe analysis
                perf_1w = ((current_price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]) * 100
                perf_1m = ((current_price - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20]) * 100
                perf_3m = ((current_price - hist['Close'].iloc[-60]) / hist['Close'].iloc[-60]) * 100
                
                # Relative performance vs SPY
                rel_perf_3m = perf_3m - spy_3m_perf
                
                # RSI
                close_prices = hist['Close']
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                # Moving average analysis
                sma_50 = close_prices.rolling(window=50).mean()
                sma_200 = close_prices.rolling(window=200).mean()
                
                ma_trend = "BULLISH" if current_price > sma_50.iloc[-1] > sma_200.iloc[-1] else "BEARISH"
                
                sector_data = {
                    'symbol': symbol,
                    'sector': sector_name,
                    'perf_1w': perf_1w,
                    'perf_1m': perf_1m,
                    'perf_3m': perf_3m,
                    'rel_perf_3m': rel_perf_3m,
                    'rsi': current_rsi,
                    'ma_trend': ma_trend,
                    'current_price': current_price
                }
                
                phase_data.append(sector_data)
                
                # Format output
                rel_symbol = "📈" if rel_perf_3m > 2 else "📉" if rel_perf_3m < -2 else "➡️"
                trend_symbol = "🟢" if ma_trend == "BULLISH" else "🔴"
                
                print(f"  {rel_symbol} {sector_name:25} ({symbol})")
                print(f"     3M Performance: {perf_3m:+.1f}% | Relative to SPY: {rel_perf_3m:+.1f}%")
                print(f"     RSI: {current_rsi:.1f} | Trend: {trend_symbol} {ma_trend}")
                print()
                
            except Exception as e:
                print(f"     ❌ Error analyzing {symbol}: {e}")
                continue
        
        # Calculate phase average performance
        if phase_data:
            avg_perf_3m = np.mean([d['perf_3m'] for d in phase_data])
            avg_rel_perf = np.mean([d['rel_perf_3m'] for d in phase_data])
            cycle_performance[cycle_phase] = {
                'avg_perf_3m': avg_perf_3m,
                'avg_rel_perf': avg_rel_perf,
                'sectors': phase_data
            }
            
            print(f"  📊 {cycle_phase.upper().replace('_', ' ')} AVERAGE:")
            print(f"     3M Performance: {avg_perf_3m:+.1f}%")
            print(f"     Relative Performance: {avg_rel_perf:+.1f}%")
            print()
        
        print()
    
    print("🎯 ECONOMIC CYCLE INTERPRETATION")
    print("=" * 35)
    print()
    
    # Rank cycle phases by performance
    phase_rankings = []
    for phase, data in cycle_performance.items():
        phase_rankings.append({
            'phase': phase,
            'avg_perf': data['avg_perf_3m'],
            'avg_rel_perf': data['avg_rel_perf']
        })
    
    phase_rankings.sort(key=lambda x: x['avg_rel_perf'], reverse=True)
    
    print("📈 CYCLE PHASE RANKINGS (3M Relative Performance)")
    print("-" * 50)
    for i, phase_data in enumerate(phase_rankings, 1):
        phase = phase_data['phase'].replace('_', ' ').title()
        rel_perf = phase_data['avg_rel_perf']
        
        if i == 1:
            status = "🥇 LEADING"
        elif i == 2:
            status = "🥈 STRONG"
        elif i == len(phase_rankings) - 1:
            status = "🥉 LAGGING"
        else:
            status = "📊 MIXED"
            
        print(f"  {i}. {status:12} {phase:20} ({rel_perf:+.1f}% vs SPY)")
    
    print()
    
    # Economic cycle inference
    best_phase = phase_rankings[0]['phase']
    worst_phase = phase_rankings[-1]['phase']
    
    print("🔍 ECONOMIC CYCLE DIAGNOSIS")
    print("-" * 30)
    print()
    
    if best_phase == 'early_cycle':
        cycle_stage = "🌅 EARLY CYCLE - Economic Recovery"
        interpretation = """
        The economy is emerging from recession/slowdown. Key characteristics:
        • Interest rates likely at or near bottom
        • Corporate earnings beginning to recover
        • Credit conditions improving
        • Financials and Industrials leading (credit expansion, capex)
        """
    elif best_phase == 'mid_cycle':
        cycle_stage = "☀️ MID CYCLE - Economic Expansion"
        interpretation = """
        The economy is in full expansion mode. Key characteristics:
        • GDP growth accelerating
        • Consumer confidence high
        • Technology and Consumer Discretionary leading
        • Risk appetite elevated
        """
    elif best_phase == 'late_cycle':
        cycle_stage = "🌅 LATE CYCLE - Economic Peak"
        interpretation = """
        The economy is approaching peak growth. Key characteristics:
        • Inflation pressures building
        • Interest rates rising
        • Materials and Energy benefiting from resource constraints
        • Early signs of economic stress emerging
        """
    else:  # defensive
        cycle_stage = "🌧️ DEFENSIVE CYCLE - Economic Contraction"
        interpretation = """
        The economy is slowing or in recession. Key characteristics:
        • Flight to quality underway
        • Defensive sectors outperforming
        • Central bank likely cutting rates
        • Risk assets under pressure
        """
    
    print(f"Current Cycle Stage: {cycle_stage}")
    print(interpretation)
    print()
    
    print("💡 INVESTMENT IMPLICATIONS")
    print("-" * 25)
    print()
    
    # Investment strategy based on cycle
    if best_phase == 'early_cycle':
        print("✅ RECOMMENDED STRATEGY:")
        print("  • Overweight Financials (XLF) - credit expansion")
        print("  • Overweight Industrials (XLI) - capex recovery") 
        print("  • Add Materials (XLB) - infrastructure demand")
        print("  • Underweight Defensive sectors")
        print()
        print("🎯 KEY CATALYSTS TO WATCH:")
        print("  • Fed policy pivots and rate cuts")
        print("  • Credit spread tightening")
        print("  • Manufacturing PMI improvement")
        
    elif best_phase == 'mid_cycle':
        print("✅ RECOMMENDED STRATEGY:")
        print("  • Overweight Technology (XLK) - growth acceleration")
        print("  • Overweight Consumer Discretionary (XLY) - spending boom")
        print("  • Add Communication Services (XLC)")
        print("  • Reduce defensive exposure")
        print()
        print("🎯 KEY CATALYSTS TO WATCH:")
        print("  • Earnings growth acceleration")
        print("  • Consumer spending data")
        print("  • Technology innovation cycles")
        
    elif best_phase == 'late_cycle':
        print("✅ RECOMMENDED STRATEGY:")
        print("  • Overweight Energy (XLE) - resource scarcity")
        print("  • Overweight Materials (XLB) - commodity pricing")
        print("  • Consider Real Estate (XLRE) - inflation hedge")
        print("  • Begin defensive positioning")
        print()
        print("🎯 KEY CATALYSTS TO WATCH:")
        print("  • Inflation data and Fed hawkishness")
        print("  • Commodity price trends")
        print("  • Yield curve dynamics")
        
    else:  # defensive
        print("✅ RECOMMENDED STRATEGY:")
        print("  • Overweight Utilities (XLU) - stable dividends")
        print("  • Overweight Healthcare (XLV) - defensive growth")
        print("  • Overweight Consumer Staples (XLP) - necessity demand")
        print("  • Underweight cyclical sectors")
        print()
        print("🎯 KEY CATALYSTS TO WATCH:")
        print("  • Recession depth and duration")
        print("  • Policy stimulus measures")
        print("  • Early recovery indicators")
    
    print()
    print("📊 SECTOR ROTATION SIGNALS")
    print("=" * 28)
    print()
    
    # Identify rotation signals
    rotation_signals = []
    
    for phase, data in cycle_performance.items():
        rel_perf = data['avg_rel_perf']
        if rel_perf > 3:
            rotation_signals.append(f"🟢 {phase.replace('_', ' ').title()} sectors showing strong momentum")
        elif rel_perf < -3:
            rotation_signals.append(f"🔴 {phase.replace('_', ' ').title()} sectors underperforming significantly")
    
    if rotation_signals:
        print("⚡ ACTIVE ROTATION SIGNALS:")
        for signal in rotation_signals:
            print(f"  {signal}")
    else:
        print("⚖️ No strong rotation signals - market in transition")
    
    print()
    print("🎯 TACTICAL TRADING OPPORTUNITIES")
    print("=" * 35)
    print()
    
    # Find best sector opportunities within winning cycle
    winning_cycle_data = cycle_performance.get(best_phase, {})
    if winning_cycle_data:
        best_sectors = sorted(winning_cycle_data['sectors'], 
                            key=lambda x: x['rel_perf_3m'], reverse=True)
        
        print(f"📈 TOP OPPORTUNITIES IN {best_phase.replace('_', ' ').upper()} CYCLE:")
        print("-" * 55)
        
        for sector in best_sectors[:3]:
            rel_perf = sector['rel_perf_3m']
            rsi = sector['rsi']
            
            if rsi < 40:
                entry_signal = "🟢 OVERSOLD - Good Entry"
            elif rsi > 70:
                entry_signal = "🔴 OVERBOUGHT - Wait for Pullback"
            else:
                entry_signal = "🟡 NEUTRAL - Monitor"
                
            print(f"  • {sector['sector']:25} ({sector['symbol']})")
            print(f"    Relative Performance: {rel_perf:+.1f}%")
            print(f"    Entry Signal: {entry_signal} (RSI: {rsi:.1f})")
            print()
    
    print("=" * 60)
    print("✅ SECTOR CYCLE ANALYSIS COMPLETE")
    print("🔄 Economic cycle phase determines optimal sector allocation")
    print("📊 Monitor relative performance for early rotation signals")
    print("⚡ Combine cycle analysis with technical timing for entries")

if __name__ == "__main__":
    asyncio.run(sector_cycle_analysis())