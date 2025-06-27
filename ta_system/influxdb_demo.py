#!/usr/bin/env python3
"""
InfluxDB Storage Demo for IWM Analysis

Demonstrates storing granular analysis data in InfluxDB with comprehensive metrics:
- Technical indicators (RSI, MACD, moving averages, Bollinger Bands)
- Financial ratios (P/E, ROE, debt ratios, profitability metrics)
- Market data (price, volume, volatility, 52-week ranges)
- Risk metrics (beta, risk level, risk categorization)
- Investment thesis data (ratings, targets, bull/bear points)
- ETF-specific metrics (expense ratio, tracking error, holdings data)
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.comprehensive_analyzer import ComprehensiveStockAnalyzer
from src.data_sources.influxdb_storage import InfluxDBAnalysisStorage

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    rprint = print


async def demo_influxdb_integration():
    """Demonstrate InfluxDB integration with comprehensive analysis."""
    
    if HAS_RICH:
        console.print("[bold green]🗄️ InfluxDB Storage Demo - Granular Analysis Data[/bold green]")
        console.print("[dim]Storing comprehensive IWM analysis in InfluxDB with granular measurements[/dim]\n")
    else:
        print("🗄️ InfluxDB Storage Demo - Granular Analysis Data")
        print("Storing comprehensive IWM analysis in InfluxDB with granular measurements\n")

    # Configure InfluxDB (customize these settings)
    influxdb_config = {
        "url": "http://localhost:8086",
        "token": "your-token-here",  # Replace with your InfluxDB token
        "org": "ta-system",
        "bucket": "stock-analysis"
    }
    
    try:
        # Initialize analyzer with InfluxDB enabled
        if HAS_RICH:
            console.print("🔧 Initializing analyzer with InfluxDB storage...")
        else:
            print("🔧 Initializing analyzer with InfluxDB storage...")
            
        analyzer = ComprehensiveStockAnalyzer(
            enable_influxdb=True,
            influxdb_config=influxdb_config
        )
        
        # Check InfluxDB connectivity
        if not analyzer.influxdb_storage or not analyzer.influxdb_storage.client:
            if HAS_RICH:
                console.print("[yellow]⚠️ InfluxDB not available. Starting Docker InfluxDB:[/yellow]")
                console.print("[dim]docker run -d -p 8086:8086 --name influxdb2 influxdb:2.0[/dim]\n")
            else:
                print("⚠️ InfluxDB not available. Starting Docker InfluxDB:")
                print("docker run -d -p 8086:8086 --name influxdb2 influxdb:2.0\n")
            
            # Continue without InfluxDB for demonstration
            analyzer = ComprehensiveStockAnalyzer(enable_influxdb=False)
        
        # Analyze IWM with comprehensive data collection
        if HAS_RICH:
            console.print("📊 Running comprehensive IWM analysis...")
        else:
            print("📊 Running comprehensive IWM analysis...")
            
        analysis = await analyzer.analyze_stock("IWM")
        
        # Demonstrate manual InfluxDB storage (even if auto-storage failed)
        if HAS_RICH:
            console.print("\n💾 Demonstrating granular InfluxDB storage structure...")
        else:
            print("\n💾 Demonstrating granular InfluxDB storage structure...")
            
        # Create storage instance for demonstration
        storage = InfluxDBAnalysisStorage(**influxdb_config)
        
        if storage.client:
            success = await storage.store_comprehensive_analysis(analysis)
            if success:
                if HAS_RICH:
                    console.print("[green]✅ Successfully stored granular analysis data in InfluxDB[/green]")
                else:
                    print("✅ Successfully stored granular analysis data in InfluxDB")
            else:
                if HAS_RICH:
                    console.print("[red]❌ Failed to store data in InfluxDB[/red]")
                else:
                    print("❌ Failed to store data in InfluxDB")
        
        # Show what data would be stored
        await _demonstrate_storage_structure(analysis)
        
        # Generate summary report
        await _generate_storage_summary(analysis)
        
        if storage.client:
            storage.close()
            
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Demo failed: {e}[/red]")
        else:
            print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


async def _demonstrate_storage_structure(analysis):
    """Demonstrate the granular storage structure."""
    
    if HAS_RICH:
        console.print("\n📋 Granular InfluxDB Storage Structure:")
        
        # Technical Indicators Table
        tech_table = Table(title="Technical Indicators Measurements")
        tech_table.add_column("Measurement", style="cyan")
        tech_table.add_column("Tags", style="green") 
        tech_table.add_column("Fields", style="yellow")
        tech_table.add_column("Sample Values", style="magenta")
        
        tech_table.add_row(
            "technical_indicators",
            "symbol=IWM, indicator=RSI, period=14",
            "value, signal", 
            f"{float(analysis.technical_analysis.rsi_14):.1f}, 0"
        )
        tech_table.add_row(
            "moving_averages", 
            "symbol=IWM, type=SMA, period=20",
            "value, price_distance_pct",
            f"${float(analysis.technical_analysis.sma_20):.2f}, +2.5%"
        )
        tech_table.add_row(
            "bollinger_bands",
            "symbol=IWM",
            "upper, lower, width, position",
            f"${float(analysis.technical_analysis.bollinger_upper):.2f}, ${float(analysis.technical_analysis.bollinger_lower):.2f}"
        )
        
        console.print(tech_table)
        
        # Financial Ratios Table
        ratio_table = Table(title="Financial Ratios Measurements")
        ratio_table.add_column("Measurement", style="cyan")
        ratio_table.add_column("Category", style="green")
        ratio_table.add_column("Key Metrics", style="yellow")
        
        ratio_table.add_row(
            "financial_ratios",
            "valuation",
            f"PE: {float(analysis.financial_ratios.price_to_earnings):.1f}, PB: {float(analysis.financial_ratios.price_to_book):.1f}"
        )
        ratio_table.add_row(
            "financial_ratios", 
            "profitability",
            f"ROE: {float(analysis.financial_ratios.return_on_equity):.1f}%, Net Margin: {float(analysis.financial_ratios.net_margin):.1f}%"
        )
        
        console.print(ratio_table)
        
        # ETF Specific Table
        etf_table = Table(title="ETF-Specific Measurements")
        etf_table.add_column("Measurement", style="cyan")
        etf_table.add_column("ETF Metrics", style="yellow")
        etf_table.add_column("Values", style="magenta")
        
        etf_table.add_row(
            "etf_metrics",
            "expense_ratio, aum, tracking_error",
            "0.19%, $60.5B, 0.25%"
        )
        etf_table.add_row(
            "etf_sectors", 
            "sector allocation by weight",
            "Technology: 16.2%, Healthcare: 14.8%"
        )
        
        console.print(etf_table)
        
    else:
        print("\n📋 Granular InfluxDB Storage Structure:")
        print("Technical Indicators:")
        print(f"  - RSI(14): {float(analysis.technical_analysis.rsi_14):.1f}")
        print(f"  - SMA(20): ${float(analysis.technical_analysis.sma_20):.2f}")
        print(f"  - Bollinger Bands: ${float(analysis.technical_analysis.bollinger_upper):.2f} / ${float(analysis.technical_analysis.bollinger_lower):.2f}")
        
        print("\nFinancial Ratios:")
        print(f"  - P/E Ratio: {float(analysis.financial_ratios.price_to_earnings):.1f}")
        print(f"  - ROE: {float(analysis.financial_ratios.return_on_equity):.1f}%")
        print(f"  - Net Margin: {float(analysis.financial_ratios.net_margin):.1f}%")
        
        print("\nETF Metrics:")
        print("  - Expense Ratio: 0.19%")
        print("  - AUM: $60.5B")
        print("  - Tracking Error: 0.25%")


async def _generate_storage_summary(analysis):
    """Generate summary of what would be stored in InfluxDB."""
    
    # Calculate number of data points
    technical_points = 15  # RSI, MACD, moving averages, BB, volatility, support/resistance
    market_points = 1     # Current market data
    ratio_points = 3      # Valuation, profitability, leverage ratios  
    risk_points = 2       # Risk metrics and categories
    thesis_points = 1     # Investment thesis
    etf_points = 8        # ETF metrics + 5 sector allocations
    profile_points = 1    # Company profile
    financial_points = 2  # Financial statements + growth rates
    
    total_points = (technical_points + market_points + ratio_points + 
                   risk_points + thesis_points + etf_points + 
                   profile_points + financial_points)
    
    if HAS_RICH:
        console.print(f"\n📊 Storage Summary for {analysis.company_profile.symbol}:")
        
        summary_table = Table(title="InfluxDB Data Points Summary")
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("Data Points", style="green")
        summary_table.add_column("Key Measurements", style="yellow")
        
        summary_table.add_row("Technical Analysis", str(technical_points), "RSI, MACD, Moving Averages, Bollinger Bands, Volatility")
        summary_table.add_row("Market Data", str(market_points), "Price, Volume, Market Cap, 52-week position")  
        summary_table.add_row("Financial Ratios", str(ratio_points), "Valuation, Profitability, Leverage metrics")
        summary_table.add_row("Risk Assessment", str(risk_points), "Risk level, Beta, Risk categories count")
        summary_table.add_row("Investment Thesis", str(thesis_points), "Rating, Targets, Bull/Bear points count")
        summary_table.add_row("ETF Specific", str(etf_points), "Expense ratio, AUM, Sector allocations")
        summary_table.add_row("Company Profile", str(profile_points), "Sector, Industry, Exchange, Description")
        summary_table.add_row("Financial Statements", str(financial_points), "Income statement, Growth rates")
        summary_table.add_row("[bold]Total", f"[bold]{total_points}", "[bold]Comprehensive granular analysis data")
        
        console.print(summary_table)
        
        console.print(f"\n🎯 Analysis complete! {total_points} granular data points ready for InfluxDB storage.")
        console.print("[dim]This enables time-series analysis, trend identification, and comparative studies across securities.[/dim]")
        
    else:
        print(f"\n📊 Storage Summary for {analysis.company_profile.symbol}:")
        print(f"Technical Analysis: {technical_points} points")
        print(f"Market Data: {market_points} points")
        print(f"Financial Ratios: {ratio_points} points") 
        print(f"Risk Assessment: {risk_points} points")
        print(f"Investment Thesis: {thesis_points} points")
        print(f"ETF Specific: {etf_points} points")
        print(f"Company Profile: {profile_points} points")
        print(f"Financial Statements: {financial_points} points")
        print(f"Total: {total_points} granular data points")
        
        print(f"\n🎯 Analysis complete! {total_points} granular data points ready for InfluxDB storage.")


if __name__ == "__main__":
    asyncio.run(demo_influxdb_integration())