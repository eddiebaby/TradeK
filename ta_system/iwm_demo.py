#!/usr/bin/env python3
"""
Simple IWM Analysis Demo

This demonstrates running technical analysis on IWM using our TA system.
"""

import sys
import os
from datetime import datetime, timezone
from decimal import Decimal

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import OHLCV
from src.indicators import (
    IndicatorCalculator,
    RSICalculator,
    SMApCalculator,
    EMACalculator,
    MACDCalculator,
)

def create_iwm_sample_data():
    """Create realistic IWM sample data for demonstration."""
    # IWM typical price range: $180-220
    base_price = 195.0
    
    sample_data = []
    for i in range(30):  # 30 days of data
        # Simulate realistic price movement
        price_change = (i % 5 - 2) * 0.8  # Small daily changes
        price = base_price + price_change + (i * 0.1)  # Slight upward trend
        
        ohlcv = OHLCV(
            symbol="IWM",
            timestamp=datetime.now(timezone.utc),
            open=Decimal(str(round(price - 0.5, 2))),
            high=Decimal(str(round(price + 1.2, 2))),
            low=Decimal(str(round(price - 1.1, 2))),
            close=Decimal(str(round(price, 2))),
            volume=45000000 + (i * 100000)  # Typical IWM volume
        )
        sample_data.append(ohlcv)
    
    return sample_data

def analyze_iwm():
    """Run IWM technical analysis."""
    print("🚀 IWM (iShares Russell 2000 ETF) Technical Analysis")
    print("=" * 55)
    
    # Setup indicator calculator
    calculator = IndicatorCalculator()
    calculator.register("RSI_14", RSICalculator(period=14))
    calculator.register("SMA_20", SMApCalculator(period=20))
    calculator.register("EMA_12", EMACalculator(period=12))
    calculator.register("MACD_12_26_9", MACDCalculator(fast=12, slow=26, signal=9))
    
    print(f"📊 Registered {len(calculator.indicators)} technical indicators")
    
    # Get sample data
    iwm_data = create_iwm_sample_data()
    print(f"📈 Processing {len(iwm_data)} IWM data points")
    
    # Calculate indicators
    print("\n⚡ Calculating indicators...")
    results = []
    
    for i, ohlcv in enumerate(iwm_data):
        indicator_results = calculator.calculate_all(ohlcv)
        
        if indicator_results:  # Only store when we have results
            results.append({
                'day': i + 1,
                'price': float(ohlcv.close),
                'volume': ohlcv.volume,
                'indicators': {
                    name: {
                        'value': float(result.value),
                        'components': {k: float(v) for k, v in result.components.items()} if result.components else None
                    }
                    for name, result in indicator_results.items()
                }
            })
    
    # Display latest results
    if results:
        latest = results[-1]
        print(f"\n📊 IWM Analysis Results (Day {latest['day']})")
        print("-" * 40)
        print(f"💰 Current Price: ${latest['price']:.2f}")
        print(f"📊 Volume: {latest['volume']:,}")
        
        print("\n🔍 Technical Indicators:")
        for name, data in latest['indicators'].items():
            print(f"  • {name}: {data['value']:.4f}")
            
            if data['components']:
                for comp_name, comp_value in data['components'].items():
                    print(f"    └─ {comp_name}: {comp_value:.4f}")
        
        # Generate signals
        print("\n🎯 Trading Signals:")
        signals = []
        
        # RSI signals
        if 'RSI_14' in latest['indicators']:
            rsi = latest['indicators']['RSI_14']['value']
            if rsi < 30:
                signals.append(f"📈 BUY Signal: RSI Oversold ({rsi:.1f})")
            elif rsi > 70:
                signals.append(f"📉 SELL Signal: RSI Overbought ({rsi:.1f})")
            else:
                signals.append(f"🔄 NEUTRAL: RSI in normal range ({rsi:.1f})")
        
        # MACD signals
        if 'MACD_12_26_9' in latest['indicators']:
            macd_data = latest['indicators']['MACD_12_26_9']
            if macd_data['components']:
                histogram = macd_data['components']['histogram']
                if histogram > 0:
                    signals.append(f"📈 BULLISH: MACD Histogram positive ({histogram:.4f})")
                else:
                    signals.append(f"📉 BEARISH: MACD Histogram negative ({histogram:.4f})")
        
        # Price vs SMA
        if 'SMA_20' in latest['indicators']:
            sma_20 = latest['indicators']['SMA_20']['value']
            if latest['price'] > sma_20:
                signals.append(f"📈 BULLISH: Price above SMA-20 (${sma_20:.2f})")
            else:
                signals.append(f"📉 BEARISH: Price below SMA-20 (${sma_20:.2f})")
        
        for signal in signals:
            print(f"  {signal}")
        
        # Market analysis
        print(f"\n📈 Market Analysis:")
        print(f"  • IWM represents Russell 2000 small-cap companies")
        print(f"  • Current analysis based on {len(iwm_data)} data points")
        print(f"  • Indicators suggest: {'Mixed signals' if len(signals) > 2 else 'Clear trend'}")
        
        # Show trend over last few days
        if len(results) >= 5:
            print(f"\n📊 5-Day Price Trend:")
            for result in results[-5:]:
                print(f"  Day {result['day']:2d}: ${result['price']:6.2f}")
        
    else:
        print("❌ No indicator results generated. Need more data points.")
    
    print(f"\n✅ IWM Technical Analysis Complete!")
    print(f"   Analyzed with {len(calculator.indicators)} indicators")
    print(f"   Generated {len(results)} result sets")

if __name__ == "__main__":
    try:
        analyze_iwm()
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()