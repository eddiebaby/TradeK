#!/usr/bin/env python3
"""
IWM Options Trading Analysis Generator
Specialized analysis for entry/exit signals on IWM options positions
"""

import asyncio
from src.comprehensive_analyzer import ComprehensiveStockAnalyzer
from datetime import datetime, date

async def main():
    analyzer = ComprehensiveStockAnalyzer()
    analysis = await analyzer.analyze_stock('IWM')
    
    # Extract key options trading data
    current_price = float(analysis.market_data.current_price)
    ytd_return = float(analysis.market_data.ytd_return)
    rsi = float(analysis.technical_analysis.rsi_14) if analysis.technical_analysis.rsi_14 else 50.0
    volatility = float(analysis.technical_analysis.volatility)
    atr = float(analysis.technical_analysis.atr_14) if analysis.technical_analysis.atr_14 else current_price * 0.015
    
    # Support/Resistance levels for options strikes
    support_1 = current_price * 0.95  # 5% below
    support_2 = current_price * 0.90  # 10% below
    resistance_1 = current_price * 1.05  # 5% above
    resistance_2 = current_price * 1.10  # 10% above
    
    # Generate options-focused analysis
    options_report = f"""# 📈 IWM Options Trading Analysis - Entry/Exit Framework
================================================================================

## 🎯 Current Market Setup (As of {datetime.now().strftime('%B %d, %Y')})

**Underlying: iShares Russell 2000 ETF (IWM)**
- **Current Price**: ${current_price:.2f}
- **YTD Performance**: {ytd_return:.1f}%
- **Implied Volatility Environment**: {volatility:.1f}% annualized
- **Average True Range (14)**: ${atr:.2f}
- **RSI (14)**: {rsi:.1f} {'🔴 OVERBOUGHT' if rsi > 70 else '🟢 OVERSOLD' if rsi < 30 else '🟡 NEUTRAL'}

## ⚡ Key Options Strike Levels

### 📉 **PUT OPTION TARGETS** (Bearish/Protective)
```
SUPPORT LEVEL 1: ${support_1:.2f} (-5.0%)  ← **Primary Put Strike**
SUPPORT LEVEL 2: ${support_2:.2f} (-10.0%) ← **Deep Put Strike**
```

### 📈 **CALL OPTION TARGETS** (Bullish)
```
RESISTANCE 1: ${resistance_1:.2f} (+5.0%)  ← **Primary Call Strike**
RESISTANCE 2: ${resistance_2:.2f} (+10.0%) ← **Extended Call Strike**
```

## 🚨 ENTRY SIGNALS - When to Open Positions

### 🟢 **BULLISH CALL ENTRY CONDITIONS**
1. **RSI Oversold**: RSI < 30 (Current: {rsi:.1f})
2. **Price at Support**: IWM trading near ${support_1:.2f} level
3. **Volume Confirmation**: Above-average volume on bounce
4. **Market Environment**: Risk-on sentiment, small-cap rotation
5. **VIX Consideration**: Elevated fear levels presenting opportunity

**Recommended Call Strategy**:
- **Strike**: {(current_price + 5):.0f} (slightly OTM)
- **Expiration**: 30-45 DTE (Days to Expiration)
- **Entry Price**: When IV rank < 50th percentile

### 🔴 **BEARISH PUT ENTRY CONDITIONS**
1. **RSI Overbought**: RSI > 70 (Current: {rsi:.1f})
2. **Price at Resistance**: IWM trading near ${resistance_1:.2f} level  
3. **Economic Headwinds**: Rate hikes, credit tightening
4. **Large-cap Outperformance**: SPY/QQQ strength vs IWM weakness
5. **Yield Curve Inversion**: Recession warning signals

**Recommended Put Strategy**:
- **Strike**: {(current_price - 5):.0f} (slightly OTM)
- **Expiration**: 30-45 DTE
- **Entry Price**: When IV rank > 50th percentile

## 🎯 EXIT RULES - When to Close Positions

### ✅ **PROFIT-TAKING EXITS**
- **Target 1**: 25% profit (quick scalp on momentum)
- **Target 2**: 50% profit (standard swing trade)
- **Target 3**: 75% profit (home run trade - rare)

### 🛑 **STOP-LOSS EXITS**
- **Time Decay**: Close at 7 DTE if position unprofitable
- **Loss Limit**: -50% of premium paid (hard stop)
- **Technical Break**: Price moves beyond ATR range

### ⚖️ **RISK MANAGEMENT RULES**

1. **Position Sizing**: Never risk more than 2% of portfolio per trade
2. **Diversification**: Maximum 3 IWM option positions simultaneously  
3. **Delta Management**: Maintain delta exposure appropriate for outlook
4. **Theta Decay**: Avoid holding through earnings or major events

## 📊 CURRENT TECHNICAL SETUP

### Small-Cap Market Drivers
- **Federal Reserve Policy**: {'Hawkish environment pressuring small-caps' if ytd_return < 0 else 'Accommodative stance supporting growth'}
- **Economic Cycle**: {'Late cycle concerns' if rsi > 60 else 'Early cycle opportunity' if rsi < 40 else 'Mid-cycle positioning'}
- **Dollar Strength**: Monitor DXY for small-cap headwinds
- **Credit Spreads**: Watch HYG for small-cap funding environment

### Options Market Intelligence
- **Put/Call Ratio**: Monitor for sentiment extremes
- **Options Volume**: Look for unusual activity in key strikes
- **IV Percentile**: Current volatility vs historical range
- **Gamma Levels**: Major dealer hedging at round numbers

## 🔮 FORWARD-LOOKING CATALYSTS

### 📈 **BULLISH CATALYSTS** (Call Option Opportunities)
1. **Rate Cut Cycle**: Fed pivot supporting small-cap growth
2. **Economic Acceleration**: GDP growth favoring domestic companies
3. **M&A Activity**: Increased small-cap acquisition activity
4. **Russell Rebalancing**: Institutional flow supporting IWM

### 📉 **BEARISH CATALYSTS** (Put Option Opportunities)
1. **Credit Crunch**: Tightening conditions hurting small-caps
2. **Dollar Strength**: International headwinds
3. **Recession Signals**: Yield curve, leading indicators
4. **Large-cap Flight**: Quality rotation away from small-caps

## ⚡ ACTIONABLE TRADING PLAN

### Current Environment Assessment: {'OVERSOLD BOUNCE SETUP' if rsi < 35 else 'OVERBOUGHT FADE SETUP' if rsi > 65 else 'RANGE-BOUND NEUTRAL'}

**Immediate Strategy**:
```
{'🟢 LOOK FOR CALL ENTRIES on weakness near $' + f'{support_1:.2f}' if rsi < 50 else '🔴 LOOK FOR PUT ENTRIES on strength near $' + f'{resistance_1:.2f}'}
{'🎯 Target strikes: ' + f'{(current_price + 3):.0f}C' + ' (moderate upside)' if rsi < 50 else '🎯 Target strikes: ' + f'{(current_price - 3):.0f}P' + ' (moderate downside)'}
⏰ Optimal timing: 30-45 DTE for theta efficiency
```

### **SPECIFIC STRIKE RECOMMENDATIONS**

#### For Call Options (Bullish):
- **Conservative**: {int(current_price) + 2}C (${int(current_price) + 2})
- **Moderate**: {int(current_price) + 5}C (${int(current_price) + 5})
- **Aggressive**: {int(current_price) + 10}C (${int(current_price) + 10})

#### For Put Options (Bearish/Hedge):
- **Conservative**: {int(current_price) - 2}P (${int(current_price) - 2})
- **Moderate**: {int(current_price) - 5}P (${int(current_price) - 5})
- **Aggressive**: {int(current_price) - 10}P (${int(current_price) - 10})

## 🚨 KEY MONITORING POINTS

1. **Daily**: IWM vs SPY relative performance
2. **Weekly**: Options flow and unusual activity  
3. **Monthly**: Economic data (jobs, GDP, Fed policy)
4. **Quarterly**: Russell rebalancing and earnings impact

## 📋 OPTIONS TRADING CHECKLIST

### Before Entry:
- [ ] Check IV percentile (aim for <30% for calls, >70% for puts)
- [ ] Confirm RSI divergence or extreme levels
- [ ] Verify volume confirmation on price moves
- [ ] Assess overall market risk-on/risk-off sentiment
- [ ] Review economic calendar for upcoming events

### During Position:
- [ ] Monitor delta exposure daily
- [ ] Track time decay (theta burn)
- [ ] Watch for technical level breaks
- [ ] Maintain position sizing discipline
- [ ] Prepare exit strategy based on price action

### Exit Execution:
- [ ] Take profits at predetermined levels
- [ ] Cut losses at 50% of premium
- [ ] Close positions with 7 DTE if unprofitable
- [ ] Reassess market conditions for new entries

---

## 💡 ADDITIONAL STRATEGY NOTES

### **Earnings Considerations**
IWM doesn't have traditional earnings, but Russell rebalancing (typically in June) creates significant volatility and options opportunities.

### **Volatility Trading**
- **High IV Environment**: Sell premium (spreads, covered calls)
- **Low IV Environment**: Buy premium (long calls/puts)
- **IV Crush**: Avoid holding through major announcements

### **Small-Cap Seasonal Patterns**
- **January Effect**: Small-caps often outperform in January
- **Russell Rebalancing**: June volatility from index changes
- **Year-end Tax Selling**: December weakness, January recovery

---
**⚠️ DISCLAIMER**: This analysis is for educational purposes. Options trading involves substantial risk of loss. Always use proper position sizing and risk management. Past performance does not guarantee future results.

**📊 Analysis Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EST
**🔄 Next Update**: Monitor for RSI regime changes and technical level breaks
================================================================================"""
    
    # Save the options-focused report
    filename = f'IWM_OPTIONS_TRADING_ANALYSIS_{date.today().strftime("%Y%m%d")}.md'
    with open(filename, 'w') as f:
        f.write(options_report)
    
    print(f'✅ Generated options trading analysis: {filename}')
    print(f'📊 Current IWM: ${current_price:.2f} | RSI: {rsi:.1f} | Volatility: {volatility:.1f}%')
    
    # Current market assessment
    if rsi < 35:
        print("🟢 BULLISH SETUP: Oversold conditions favor call options")
        print(f"🎯 Target Call Strike: {int(current_price) + 3}C")
    elif rsi > 65:
        print("🔴 BEARISH SETUP: Overbought conditions favor put options")
        print(f"🎯 Target Put Strike: {int(current_price) - 3}P")
    else:
        print("🟡 NEUTRAL SETUP: Range-bound, consider spreads or short premium")
    
    return filename

if __name__ == "__main__":
    asyncio.run(main())