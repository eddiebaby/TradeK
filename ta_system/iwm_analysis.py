#!/usr/bin/env python3
"""
IWM (iShares Russell 2000 ETF) Technical Analysis Demo

This script demonstrates how to run comprehensive technical analysis on IWM using:
1. Real market data acquisition (yfinance)
2. Our TA system's indicator calculations
3. Professional analysis and visualization
4. Trading signal generation
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

import yfinance as yf
import pandas as pd
import numpy as np
from decimal import Decimal

# Import our TA system components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import OHLCV, Timeframe
from src.indicators import (
    IndicatorCalculator,
    RSICalculator,
    MACDCalculator,
    BollingerBandsCalculator,
    SMApCalculator,
    EMACalculator,
    ATRCalculator,
)

# Try to import rich for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    from rich.progress import Progress, SpinnerColumn, TextColumn
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None
    rprint = print


class IWMAnalyzer:
    """Professional IWM Technical Analysis System."""
    
    def __init__(self):
        """Initialize the IWM analyzer with our TA system."""
        self.symbol = "IWM"
        self.calculator = IndicatorCalculator()
        self._setup_indicators()
    
    def _setup_indicators(self):
        """Setup comprehensive indicator suite for IWM analysis."""
        # Trend Indicators
        self.calculator.register("SMA_20", SMApCalculator(period=20))
        self.calculator.register("SMA_50", SMApCalculator(period=50))
        self.calculator.register("SMA_200", SMApCalculator(period=200))
        self.calculator.register("EMA_12", EMACalculator(period=12))
        self.calculator.register("EMA_26", EMACalculator(period=26))
        
        # Momentum Indicators
        self.calculator.register("RSI_14", RSICalculator(period=14))
        self.calculator.register("RSI_21", RSICalculator(period=21))
        
        # Volatility Indicators
        self.calculator.register("BB_20_2", BollingerBandsCalculator(period=20, std_dev=2))
        self.calculator.register("ATR_14", ATRCalculator(period=14))
        
        # Convergence/Divergence
        self.calculator.register("MACD_12_26_9", MACDCalculator(fast=12, slow=26, signal=9))
    
    def fetch_iwm_data(self, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch real IWM market data from Yahoo Finance."""
        if HAS_RICH:
            console.print(f"[cyan]Fetching IWM data for {period} with {interval} intervals...[/cyan]")
        else:
            print(f"Fetching IWM data for {period} with {interval} intervals...")
        
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError("No data retrieved for IWM")
            
            if HAS_RICH:
                console.print(f"✅ [green]Retrieved {len(data)} data points[/green]")
            else:
                print(f"✅ Retrieved {len(data)} data points")
            
            return data
            
        except Exception as e:
            if HAS_RICH:
                console.print(f"❌ [red]Error fetching IWM data: {e}[/red]")
            else:
                print(f"❌ Error fetching IWM data: {e}")
            raise
    
    def convert_to_ohlcv(self, df: pd.DataFrame) -> List[OHLCV]:
        """Convert pandas DataFrame to our OHLCV model format."""
        ohlcv_data = []
        
        for timestamp, row in df.iterrows():
            try:
                ohlcv = OHLCV(
                    symbol=self.symbol,
                    timestamp=timestamp.to_pydatetime().replace(tzinfo=timezone.utc),
                    open=Decimal(str(round(row['Open'], 4))),
                    high=Decimal(str(round(row['High'], 4))),
                    low=Decimal(str(round(row['Low'], 4))),
                    close=Decimal(str(round(row['Close'], 4))),
                    volume=int(row['Volume'])
                )
                ohlcv_data.append(ohlcv)
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[yellow]Warning: Skipping invalid data point: {e}[/yellow]")
                else:
                    print(f"Warning: Skipping invalid data point: {e}")
                continue
        
        return ohlcv_data
    
    def calculate_indicators(self, ohlcv_data: List[OHLCV]) -> Dict[str, List]:
        """Calculate all technical indicators for IWM data."""
        if HAS_RICH:
            console.print("\n[cyan]Calculating technical indicators...[/cyan]")
        else:
            print("\nCalculating technical indicators...")
        
        # Reset calculators for fresh analysis
        self.calculator.reset_all()
        
        # Store results by indicator
        results = {name: [] for name in self.calculator.indicators.keys()}
        
        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing data points...", total=len(ohlcv_data))
                
                for ohlcv in ohlcv_data:
                    indicator_results = self.calculator.calculate_all(ohlcv)
                    
                    for name, result in indicator_results.items():
                        results[name].append({
                            'timestamp': result.timestamp,
                            'value': float(result.value),
                            'components': {k: float(v) for k, v in result.components.items()} if result.components else None
                        })
                    
                    progress.advance(task)
        else:
            for i, ohlcv in enumerate(ohlcv_data):
                if i % 50 == 0:
                    print(f"Processing {i}/{len(ohlcv_data)} data points...")
                
                indicator_results = self.calculator.calculate_all(ohlcv)
                
                for name, result in indicator_results.items():
                    results[name].append({
                        'timestamp': result.timestamp,
                        'value': float(result.value),
                        'components': {k: float(v) for k, v in result.components.items()} if result.components else None
                    })
        
        return results
    
    def analyze_current_state(self, ohlcv_data: List[OHLCV], indicators: Dict[str, List]) -> Dict:
        """Analyze current IWM technical state and generate insights."""
        if not ohlcv_data or not indicators:
            return {"error": "No data available for analysis"}
        
        latest_price = float(ohlcv_data[-1].close)
        latest_volume = ohlcv_data[-1].volume
        
        # Get latest indicator values
        latest_indicators = {}
        for name, values in indicators.items():
            if values:
                latest_indicators[name] = values[-1]
        
        analysis = {
            "timestamp": ohlcv_data[-1].timestamp.isoformat(),
            "current_price": latest_price,
            "volume": latest_volume,
            "indicators": latest_indicators,
            "signals": [],
            "trend_analysis": {},
            "momentum_analysis": {},
            "volatility_analysis": {}
        }
        
        # Trend Analysis
        if "SMA_20" in latest_indicators and "SMA_50" in latest_indicators and "SMA_200" in latest_indicators:
            sma_20 = latest_indicators["SMA_20"]["value"]
            sma_50 = latest_indicators["SMA_50"]["value"]
            sma_200 = latest_indicators["SMA_200"]["value"]
            
            analysis["trend_analysis"] = {
                "short_term": "Bullish" if latest_price > sma_20 else "Bearish",
                "medium_term": "Bullish" if latest_price > sma_50 else "Bearish",
                "long_term": "Bullish" if latest_price > sma_200 else "Bearish",
                "golden_cross": sma_50 > sma_200,
                "death_cross": sma_50 < sma_200
            }
        
        # Momentum Analysis
        if "RSI_14" in latest_indicators:
            rsi = latest_indicators["RSI_14"]["value"]
            analysis["momentum_analysis"]["rsi_14"] = {
                "value": rsi,
                "condition": "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            }
        
        if "MACD_12_26_9" in latest_indicators:
            macd = latest_indicators["MACD_12_26_9"]
            if macd["components"]:
                macd_line = macd["components"]["macd"]
                signal_line = macd["components"]["signal"]
                histogram = macd["components"]["histogram"]
                
                analysis["momentum_analysis"]["macd"] = {
                    "macd_line": macd_line,
                    "signal_line": signal_line,
                    "histogram": histogram,
                    "bullish": macd_line > signal_line,
                    "momentum": "Increasing" if histogram > 0 else "Decreasing"
                }
        
        # Volatility Analysis
        if "BB_20_2" in latest_indicators:
            bb = latest_indicators["BB_20_2"]
            if bb["components"]:
                position = bb["value"]  # Position within bands (0-1)
                analysis["volatility_analysis"]["bollinger_bands"] = {
                    "position": position,
                    "upper": bb["components"]["upper"],
                    "middle": bb["components"]["middle"],
                    "lower": bb["components"]["lower"],
                    "squeeze": (bb["components"]["upper"] - bb["components"]["lower"]) / bb["components"]["middle"] < 0.1
                }
        
        if "ATR_14" in latest_indicators:
            atr = latest_indicators["ATR_14"]["value"]
            analysis["volatility_analysis"]["atr"] = {
                "value": atr,
                "volatility": "High" if atr > latest_price * 0.03 else "Low"
            }
        
        # Generate Trading Signals
        signals = []
        
        # RSI Signals
        if "RSI_14" in latest_indicators:
            rsi = latest_indicators["RSI_14"]["value"]
            if rsi < 30:
                signals.append({
                    "type": "BUY",
                    "reason": "RSI Oversold",
                    "confidence": 0.7,
                    "indicator": "RSI_14",
                    "value": rsi
                })
            elif rsi > 70:
                signals.append({
                    "type": "SELL",
                    "reason": "RSI Overbought", 
                    "confidence": 0.7,
                    "indicator": "RSI_14",
                    "value": rsi
                })
        
        # MACD Signals
        if "MACD_12_26_9" in latest_indicators and len(indicators["MACD_12_26_9"]) > 1:
            current_macd = latest_indicators["MACD_12_26_9"]
            previous_macd = indicators["MACD_12_26_9"][-2]
            
            if (current_macd["components"] and previous_macd["components"]):
                current_hist = current_macd["components"]["histogram"]
                previous_hist = previous_macd["components"]["histogram"]
                
                if previous_hist < 0 and current_hist > 0:
                    signals.append({
                        "type": "BUY",
                        "reason": "MACD Bullish Crossover",
                        "confidence": 0.8,
                        "indicator": "MACD",
                        "value": current_hist
                    })
                elif previous_hist > 0 and current_hist < 0:
                    signals.append({
                        "type": "SELL", 
                        "reason": "MACD Bearish Crossover",
                        "confidence": 0.8,
                        "indicator": "MACD",
                        "value": current_hist
                    })
        
        analysis["signals"] = signals
        return analysis
    
    def display_analysis(self, analysis: Dict):
        """Display comprehensive IWM analysis results."""
        if "error" in analysis:
            if HAS_RICH:
                console.print(f"[red]❌ {analysis['error']}[/red]")
            else:
                print(f"❌ {analysis['error']}")
            return
        
        if HAS_RICH:
            # Create comprehensive display
            console.print(f"\n[bold blue]🔍 IWM Technical Analysis Report[/bold blue]")
            console.print(f"[dim]Generated at: {analysis['timestamp']}[/dim]")
            
            # Current State Panel
            current_info = f"""
💰 Current Price: ${analysis['current_price']:.2f}
📊 Volume: {analysis['volume']:,}
📅 Analysis Date: {analysis['timestamp'][:10]}
            """
            console.print(Panel(current_info.strip(), title="📈 Current Market State", border_style="green"))
            
            # Trend Analysis
            if analysis["trend_analysis"]:
                trend = analysis["trend_analysis"]
                trend_table = Table(title="📈 Trend Analysis")
                trend_table.add_column("Timeframe", style="cyan")
                trend_table.add_column("Direction", style="green")
                trend_table.add_column("Status", style="yellow")
                
                trend_table.add_row("Short Term (20 SMA)", trend["short_term"], "✅" if trend["short_term"] == "Bullish" else "❌")
                trend_table.add_row("Medium Term (50 SMA)", trend["medium_term"], "✅" if trend["medium_term"] == "Bullish" else "❌")
                trend_table.add_row("Long Term (200 SMA)", trend["long_term"], "✅" if trend["long_term"] == "Bullish" else "❌")
                
                if trend["golden_cross"]:
                    trend_table.add_row("Special Signal", "Golden Cross", "🚀 BULLISH")
                elif trend["death_cross"]:
                    trend_table.add_row("Special Signal", "Death Cross", "📉 BEARISH")
                
                console.print(trend_table)
            
            # Momentum Analysis
            if analysis["momentum_analysis"]:
                momentum = analysis["momentum_analysis"]
                momentum_table = Table(title="⚡ Momentum Analysis")
                momentum_table.add_column("Indicator", style="cyan")
                momentum_table.add_column("Value", style="green")
                momentum_table.add_column("Signal", style="yellow")
                
                if "rsi_14" in momentum:
                    rsi = momentum["rsi_14"]
                    momentum_table.add_row("RSI (14)", f"{rsi['value']:.2f}", rsi["condition"])
                
                if "macd" in momentum:
                    macd = momentum["macd"]
                    momentum_table.add_row("MACD Line", f"{macd['macd_line']:.4f}", "")
                    momentum_table.add_row("Signal Line", f"{macd['signal_line']:.4f}", "")
                    momentum_table.add_row("Histogram", f"{macd['histogram']:.4f}", macd["momentum"])
                
                console.print(momentum_table)
            
            # Trading Signals
            if analysis["signals"]:
                signals_table = Table(title="🎯 Trading Signals")
                signals_table.add_column("Signal", style="cyan")
                signals_table.add_column("Reason", style="green")
                signals_table.add_column("Confidence", style="yellow")
                signals_table.add_column("Indicator", style="magenta")
                
                for signal in analysis["signals"]:
                    confidence_str = f"{signal['confidence']*100:.0f}%"
                    signals_table.add_row(
                        signal["type"],
                        signal["reason"],
                        confidence_str,
                        signal["indicator"]
                    )
                
                console.print(signals_table)
            else:
                console.print(Panel("No strong trading signals detected", title="🎯 Trading Signals", border_style="yellow"))
        
        else:
            # Plain text output
            print(f"\n🔍 IWM Technical Analysis Report")
            print(f"Generated at: {analysis['timestamp']}")
            print(f"Current Price: ${analysis['current_price']:.2f}")
            print(f"Volume: {analysis['volume']:,}")
            
            if analysis["trend_analysis"]:
                print("\n📈 Trend Analysis:")
                trend = analysis["trend_analysis"]
                print(f"  Short Term: {trend['short_term']}")
                print(f"  Medium Term: {trend['medium_term']}")
                print(f"  Long Term: {trend['long_term']}")
            
            if analysis["momentum_analysis"]:
                print("\n⚡ Momentum Analysis:")
                momentum = analysis["momentum_analysis"]
                if "rsi_14" in momentum:
                    rsi = momentum["rsi_14"]
                    print(f"  RSI (14): {rsi['value']:.2f} - {rsi['condition']}")
            
            if analysis["signals"]:
                print("\n🎯 Trading Signals:")
                for signal in analysis["signals"]:
                    print(f"  {signal['type']}: {signal['reason']} (Confidence: {signal['confidence']*100:.0f}%)")
            else:
                print("\n🎯 No strong trading signals detected")

    async def run_full_analysis(self, period: str = "6mo", interval: str = "1d"):
        """Run complete IWM technical analysis."""
        try:
            if HAS_RICH:
                console.print("[bold yellow]🚀 Starting IWM Technical Analysis[/bold yellow]\n")
            else:
                print("🚀 Starting IWM Technical Analysis\n")
            
            # Step 1: Fetch data
            df = self.fetch_iwm_data(period=period, interval=interval)
            
            # Step 2: Convert to our format
            ohlcv_data = self.convert_to_ohlcv(df)
            
            if HAS_RICH:
                console.print(f"[green]✅ Converted {len(ohlcv_data)} data points to OHLCV format[/green]")
            else:
                print(f"✅ Converted {len(ohlcv_data)} data points to OHLCV format")
            
            # Step 3: Calculate indicators
            indicators = self.calculate_indicators(ohlcv_data)
            
            # Step 4: Analyze current state
            analysis = self.analyze_current_state(ohlcv_data, indicators)
            
            # Step 5: Display results
            self.display_analysis(analysis)
            
            if HAS_RICH:
                console.print(f"\n[bold green]✅ IWM Analysis Complete![/bold green]")
                console.print(f"[dim]Analyzed {len(ohlcv_data)} data points with {len(self.calculator.indicators)} indicators[/dim]")
            else:
                print(f"\n✅ IWM Analysis Complete!")
                print(f"Analyzed {len(ohlcv_data)} data points with {len(self.calculator.indicators)} indicators")
            
            return analysis
            
        except Exception as e:
            if HAS_RICH:
                console.print(f"[red]❌ Analysis failed: {e}[/red]")
            else:
                print(f"❌ Analysis failed: {e}")
            raise


async def main():
    """Main function to run IWM analysis."""
    analyzer = IWMAnalyzer()
    
    # Run analysis with different timeframes
    timeframes = [
        ("1mo", "1d", "Short-term (1 month daily)"),
        ("6mo", "1d", "Medium-term (6 months daily)"),
        ("1y", "1wk", "Long-term (1 year weekly)")
    ]
    
    for period, interval, description in timeframes:
        if HAS_RICH:
            console.print(f"\n[bold cyan]📊 {description} Analysis[/bold cyan]")
            console.print("=" * 60)
        else:
            print(f"\n📊 {description} Analysis")
            print("=" * 60)
        
        try:
            await analyzer.run_full_analysis(period=period, interval=interval)
            
            if HAS_RICH:
                console.print("\n" + "─" * 60)
            else:
                print("\n" + "─" * 60)
                
        except Exception as e:
            if HAS_RICH:
                console.print(f"[red]Failed to analyze {description}: {e}[/red]")
            else:
                print(f"Failed to analyze {description}: {e}")
            continue


if __name__ == "__main__":
    try:
        # Install required dependencies if needed
        try:
            import yfinance
        except ImportError:
            print("Installing yfinance...")
            os.system("pip install yfinance")
            import yfinance as yf
        
        asyncio.run(main())
        
    except KeyboardInterrupt:
        if HAS_RICH:
            console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        else:
            print("\nAnalysis interrupted by user")
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]Analysis failed: {e}[/red]")
        else:
            print(f"\nAnalysis failed: {e}")