#!/usr/bin/env python3
"""
YTD Validation Test - Random Stock Verification

Test the refactored YTD calculation system with random stocks to ensure accuracy.
This validates that our London TDD fix works across different asset types.
"""

import asyncio
import sys
import os
from datetime import datetime
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.comprehensive_analyzer import ComprehensiveStockAnalyzer
from src.data_sources.ytd_calculator import YTDCalculator

try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    rprint = print


async def test_random_stocks_ytd():
    """Test YTD calculation accuracy with random stocks."""
    
    # Random sample of stocks and ETFs
    test_symbols = [
        # ETFs
        'IWM', 'SPY', 'QQQ', 'VTI', 'GLD', 'TLT',
        # Large cap stocks  
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        # Mid/small cap stocks
        'AMD', 'NVDA', 'CRM', 'SHOP', 'ROKU'
    ]
    
    # Randomly select 5 symbols to test
    random_symbols = random.sample(test_symbols, 5)
    
    if HAS_RICH:
        console.print(f"[bold green]🎲 Testing YTD accuracy with random sample:[/bold green]")
        console.print(f"[dim]Selected symbols: {', '.join(random_symbols)}[/dim]\n")
    else:
        print(f"🎲 Testing YTD accuracy with random sample:")
        print(f"Selected symbols: {', '.join(random_symbols)}\n")
    
    # Initialize systems
    analyzer = ComprehensiveStockAnalyzer()
    ytd_calculator = YTDCalculator()
    
    results = []
    
    for symbol in random_symbols:
        try:
            if HAS_RICH:
                console.print(f"[cyan]Testing {symbol}...[/cyan]")
            else:
                print(f"Testing {symbol}...")
            
            # Method 1: Direct YTD calculator
            direct_ytd = ytd_calculator.calculate_ytd_return(symbol)
            
            # Method 2: Through comprehensive analyzer
            analysis = await analyzer.analyze_stock(symbol)
            analyzer_ytd = float(analysis.market_data.ytd_return)
            
            # Validate consistency
            if direct_ytd:
                direct_value = float(direct_ytd)
                consistency_check = abs(direct_value - analyzer_ytd) < 0.1  # Within 0.1%
            else:
                direct_value = None
                consistency_check = False
            
            # Validate reasonable bounds
            reasonable_bounds = -100 <= analyzer_ytd <= 100  # ±100% is reasonable for most stocks
            
            # Get metadata
            metadata = ytd_calculator.get_calculation_metadata(symbol)
            
            results.append({
                'symbol': symbol,
                'direct_ytd': direct_value,
                'analyzer_ytd': analyzer_ytd,
                'consistent': consistency_check,
                'reasonable': reasonable_bounds,
                'asset_type': analyzer.detect_asset_type(symbol),
                'calculation_date': metadata['calculation_date'][:10]  # Just date part
            })
            
        except Exception as e:
            if HAS_RICH:
                console.print(f"[red]❌ Error testing {symbol}: {e}[/red]")
            else:
                print(f"❌ Error testing {symbol}: {e}")
            
            results.append({
                'symbol': symbol,
                'direct_ytd': None,
                'analyzer_ytd': None,
                'consistent': False,
                'reasonable': False,
                'asset_type': 'UNKNOWN',
                'calculation_date': datetime.now().strftime('%Y-%m-%d')
            })
    
    # Display results
    await _display_validation_results(results)
    
    # Summary statistics
    await _display_validation_summary(results)
    
    return results


async def _display_validation_results(results):
    """Display detailed validation results."""
    
    if HAS_RICH:
        table = Table(title="🔍 YTD Validation Results")
        table.add_column("Symbol", style="cyan")
        table.add_column("Asset Type", style="blue")
        table.add_column("Direct YTD", style="green")
        table.add_column("Analyzer YTD", style="green")
        table.add_column("Consistent", style="yellow")
        table.add_column("Reasonable", style="magenta")
        
        for result in results:
            direct_str = f"{result['direct_ytd']:.1f}%" if result['direct_ytd'] else "N/A"
            analyzer_str = f"{result['analyzer_ytd']:.1f}%" if result['analyzer_ytd'] else "N/A"
            consistent_str = "✅" if result['consistent'] else "❌"
            reasonable_str = "✅" if result['reasonable'] else "❌"
            
            table.add_row(
                result['symbol'],
                result['asset_type'],
                direct_str,
                analyzer_str,
                consistent_str,
                reasonable_str
            )
        
        console.print(table)
        
    else:
        print("\n🔍 YTD Validation Results:")
        print("=" * 80)
        print(f"{'Symbol':<8} {'Type':<6} {'Direct':<10} {'Analyzer':<10} {'Consistent':<10} {'Reasonable'}")
        print("-" * 80)
        
        for result in results:
            direct_str = f"{result['direct_ytd']:.1f}%" if result['direct_ytd'] else "N/A"
            analyzer_str = f"{result['analyzer_ytd']:.1f}%" if result['analyzer_ytd'] else "N/A"
            consistent_str = "✅" if result['consistent'] else "❌"
            reasonable_str = "✅" if result['reasonable'] else "❌"
            
            print(f"{result['symbol']:<8} {result['asset_type']:<6} {direct_str:<10} {analyzer_str:<10} {consistent_str:<10} {reasonable_str}")


async def _display_validation_summary(results):
    """Display validation summary statistics."""
    
    total_tests = len(results)
    consistent_count = sum(1 for r in results if r['consistent'])
    reasonable_count = sum(1 for r in results if r['reasonable'])
    successful_calcs = sum(1 for r in results if r['analyzer_ytd'] is not None)
    
    consistency_rate = (consistent_count / total_tests) * 100
    reasonableness_rate = (reasonable_count / total_tests) * 100
    success_rate = (successful_calcs / total_tests) * 100
    
    if HAS_RICH:
        console.print(f"\n[bold]📊 Validation Summary:[/bold]")
        console.print(f"[green]Total Tests: {total_tests}[/green]")
        console.print(f"[green]Successful Calculations: {successful_calcs}/{total_tests} ({success_rate:.1f}%)[/green]")
        console.print(f"[yellow]Consistency Rate: {consistent_count}/{total_tests} ({consistency_rate:.1f}%)[/yellow]")
        console.print(f"[magenta]Reasonableness Rate: {reasonable_count}/{total_tests} ({reasonableness_rate:.1f}%)[/magenta]")
        
        if consistency_rate >= 80 and reasonableness_rate >= 80:
            console.print(f"[bold green]✅ YTD calculation system validation PASSED[/bold green]")
        else:
            console.print(f"[bold red]❌ YTD calculation system validation FAILED[/bold red]")
            
    else:
        print(f"\n📊 Validation Summary:")
        print(f"Total Tests: {total_tests}")
        print(f"Successful Calculations: {successful_calcs}/{total_tests} ({success_rate:.1f}%)")
        print(f"Consistency Rate: {consistent_count}/{total_tests} ({consistency_rate:.1f}%)")
        print(f"Reasonableness Rate: {reasonable_count}/{total_tests} ({reasonableness_rate:.1f}%)")
        
        if consistency_rate >= 80 and reasonableness_rate >= 80:
            print(f"✅ YTD calculation system validation PASSED")
        else:
            print(f"❌ YTD calculation system validation FAILED")


async def test_edge_cases():
    """Test edge cases for YTD calculation."""
    
    if HAS_RICH:
        console.print(f"\n[bold blue]🧪 Testing Edge Cases:[/bold blue]")
    else:
        print(f"\n🧪 Testing Edge Cases:")
    
    edge_cases = [
        ('INVALID_SYMBOL', 'Non-existent symbol'),
        ('BRK.A', 'High-priced stock'),
        ('TQQQ', 'Leveraged ETF'),
    ]
    
    ytd_calculator = YTDCalculator()
    
    for symbol, description in edge_cases:
        try:
            ytd = ytd_calculator.calculate_ytd_return(symbol)
            
            if HAS_RICH:
                if ytd:
                    console.print(f"[green]{symbol} ({description}): {ytd:.1f}%[/green]")
                else:
                    console.print(f"[yellow]{symbol} ({description}): Calculation failed (expected)[/yellow]")
            else:
                if ytd:
                    print(f"{symbol} ({description}): {ytd:.1f}%")
                else:
                    print(f"{symbol} ({description}): Calculation failed (expected)")
                    
        except Exception as e:
            if HAS_RICH:
                console.print(f"[red]{symbol} ({description}): Error - {e}[/red]")
            else:
                print(f"{symbol} ({description}): Error - {e}")


async def main():
    """Run comprehensive YTD validation."""
    
    if HAS_RICH:
        console.print("[bold green]🔬 YTD Calculation System Validation[/bold green]")
        console.print("[dim]Testing refactored YTD calculator with random samples[/dim]\n")
    else:
        print("🔬 YTD Calculation System Validation")
        print("Testing refactored YTD calculator with random samples\n")
    
    # Test random stocks
    results = await test_random_stocks_ytd()
    
    # Test edge cases
    await test_edge_cases()
    
    if HAS_RICH:
        console.print("\n[bold]🎯 Validation Complete![/bold]")
    else:
        print("\n🎯 Validation Complete!")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())