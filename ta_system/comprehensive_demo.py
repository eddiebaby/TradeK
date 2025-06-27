#!/usr/bin/env python3
"""
Comprehensive Stock Analysis Demo

This script demonstrates the institutional-grade comprehensive analysis system
that combines technical analysis, fundamental analysis, and professional reporting.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.comprehensive_analyzer import ComprehensiveStockAnalyzer

# Try to import rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich import print as rprint
    from rich.progress import Progress, SpinnerColumn, TextColumn
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None
    rprint = print


async def demonstrate_comprehensive_analysis(symbol: str = "GOOGL"):
    """Demonstrate comprehensive stock analysis."""
    
    if HAS_RICH:
        console.print(f"\n[bold cyan]🏛️ Institutional-Grade Stock Analysis System[/bold cyan]")
        console.print(f"[bold yellow]Analyzing {symbol.upper()} with Professional Standards[/bold yellow]")
        console.print("=" * 70)
    else:
        print(f"\n🏛️ Institutional-Grade Stock Analysis System")
        print(f"Analyzing {symbol.upper()} with Professional Standards")
        print("=" * 70)
    
    try:
        # Initialize comprehensive analyzer
        analyzer = ComprehensiveStockAnalyzer()
        
        if HAS_RICH:
            console.print(f"\n[green]✅ Initialized comprehensive analyzer with:[/green]")
            console.print(f"  • Fundamental analysis engine with financial ratios")
            console.print(f"  • Technical analysis with 9 indicators")
            console.print(f"  • Professional report generation system")
            console.print(f"  • Risk assessment framework")
            console.print(f"  • Investment thesis generation")
        else:
            print(f"\n✅ Initialized comprehensive analyzer")
            print(f"  • Fundamental analysis engine")
            print(f"  • Technical analysis system")
            print(f"  • Report generation")
        
        # Perform comprehensive analysis
        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Performing comprehensive analysis...", total=None)
                
                report = await analyzer.analyze_and_report(symbol)
                
                progress.remove_task(task)
        else:
            print(f"\n🔄 Performing comprehensive analysis...")
            report = await analyzer.analyze_and_report(symbol)
        
        # Display the report
        if HAS_RICH:
            console.print(f"\n[bold green]📊 Comprehensive Analysis Report Generated![/bold green]")
            console.print(f"[dim]Report length: {len(report):,} characters[/dim]")
            
            # Show a preview of the report
            preview_lines = report.split('\n')[:50]  # First 50 lines
            preview = '\n'.join(preview_lines)
            
            console.print(Panel(
                preview,
                title=f"📈 {symbol.upper()} Analysis Report Preview",
                border_style="green"
            ))
            
            console.print(f"\n[yellow]💾 Saving full report to file...[/yellow]")
        else:
            print(f"\n📊 Comprehensive Analysis Report Generated!")
            print(f"Report length: {len(report):,} characters")
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol.upper()}_comprehensive_analysis_{timestamp}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        if HAS_RICH:
            console.print(f"[green]✅ Full report saved to: {filename}[/green]")
            
            # Show key highlights
            console.print(f"\n[bold cyan]🎯 Analysis Highlights:[/bold cyan]")
            
            # Extract key metrics from report
            lines = report.split('\n')
            for line in lines:
                if 'Investment Rating:' in line:
                    console.print(f"  📈 {line.strip()}")
                elif 'Current Price:' in line and '$' in line:
                    console.print(f"  💰 {line.strip()}")
                elif 'Market Cap:' in line:
                    console.print(f"  🏢 {line.strip()}")
                elif 'P/E Ratio:' in line:
                    console.print(f"  📊 {line.strip()}")
                elif 'Risk Level:' in line:
                    console.print(f"  ⚠️ {line.strip()}")
        else:
            print(f"✅ Full report saved to: {filename}")
        
        # Provide next steps
        if HAS_RICH:
            console.print(f"\n[bold yellow]🚀 Next Steps:[/bold yellow]")
            console.print(f"  1. Review the complete analysis in {filename}")
            console.print(f"  2. Customize analysis parameters in comprehensive_analyzer.py")
            console.print(f"  3. Try analyzing other stocks: python3 comprehensive_demo.py AAPL")
            console.print(f"  4. Integrate with trading systems or portfolio management")
            
            console.print(f"\n[bold green]🎉 Institutional-grade analysis complete![/bold green]")
        else:
            print(f"\n🚀 Next Steps:")
            print(f"  1. Review the complete analysis in {filename}")
            print(f"  2. Try other stocks: python3 comprehensive_demo.py AAPL")
            print(f"\n🎉 Analysis complete!")
        
        return filename
        
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Analysis failed: {e}[/red]")
        else:
            print(f"❌ Analysis failed: {e}")
        raise


async def main():
    """Main demonstration function."""
    
    # Get symbol from command line or use default
    symbol = sys.argv[1] if len(sys.argv) > 1 else "GOOGL"
    
    if HAS_RICH:
        console.print(f"[dim]Starting comprehensive analysis for {symbol.upper()}...[/dim]")
        console.print(f"[dim]This demonstrates institutional-grade analysis capabilities[/dim]")
    else:
        print(f"Starting comprehensive analysis for {symbol.upper()}...")
    
    try:
        filename = await demonstrate_comprehensive_analysis(symbol)
        
        if HAS_RICH:
            console.print(f"\n[bold green]🎯 Demo completed successfully![/bold green]")
            console.print(f"[dim]Report saved as: {filename}[/dim]")
        else:
            print(f"\n🎯 Demo completed successfully!")
            print(f"Report saved as: {filename}")
        
    except KeyboardInterrupt:
        if HAS_RICH:
            console.print(f"\n[yellow]Demo interrupted by user[/yellow]")
        else:
            print(f"\nDemo interrupted by user")
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]Demo failed: {e}[/red]")
        else:
            print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Install required dependencies if needed
    try:
        import yfinance
    except ImportError:
        print("Installing yfinance...")
        os.system("pip install yfinance")
    
    try:
        import rich
    except ImportError:
        print("Installing rich for better output...")
        os.system("pip install rich")
    
    asyncio.run(main())