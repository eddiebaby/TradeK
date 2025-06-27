#!/usr/bin/env python3
"""
Technical Analysis System Demo

This script demonstrates the TA system capabilities by:
1. Starting the FastAPI server
2. Making sample API calls
3. Showing indicator calculations
4. Displaying results in a formatted way
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import List, Dict

import httpx
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Try to import rich for beautiful output, fallback to basic print
try:
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    rprint = print

console = Console() if HAS_RICH else None


def create_sample_data() -> List[Dict]:
    """Create sample OHLCV data for demonstration."""
    data = []
    base_price = 150.0
    
    # Create 50 data points with realistic price movement
    for i in range(50):
        # Simulate price trends and volatility
        trend = 0.1 * i  # Slight upward trend
        volatility = 2.0 * (0.5 - abs((i % 10) - 5) / 10)  # Cyclical volatility
        
        price = base_price + trend + volatility
        
        data.append({
            "symbol": "DEMO",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "open": round(price - 0.5, 2),
            "high": round(price + 1.0, 2),
            "low": round(price - 1.0, 2),
            "close": round(price, 2),
            "volume": 1000000 + i * 10000
        })
    
    return data


async def test_api_endpoints(base_url: str = "http://localhost:8000"):
    """Test various API endpoints and display results."""
    
    if HAS_RICH:
        console.print("\n🚀 [bold blue]Technical Analysis System Demo[/bold blue]")
        console.print("=" * 50)
    else:
        print("\n🚀 Technical Analysis System Demo")
        print("=" * 50)

    async with httpx.AsyncClient() as client:
        
        # Test 1: Health Check
        if HAS_RICH:
            console.print("\n[bold green]1. Health Check[/bold green]")
        else:
            print("\n1. Health Check")
            
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                if HAS_RICH:
                    console.print("✅ System is healthy")
                else:
                    print("✅ System is healthy")
            else:
                if HAS_RICH:
                    console.print(f"❌ Health check failed: {response.status_code}")
                else:
                    print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            if HAS_RICH:
                console.print(f"❌ Connection failed: {e}")
            else:
                print(f"❌ Connection failed: {e}")
            return

        # Test 2: System Status
        if HAS_RICH:
            console.print("\n[bold green]2. System Status[/bold green]")
        else:
            print("\n2. System Status")
            
        response = await client.get(f"{base_url}/status")
        if response.status_code == 200:
            status = response.json()
            if HAS_RICH:
                table = Table(title="System Information")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Status", status["status"])
                table.add_row("Version", status["version"])
                table.add_row("Available Indicators", str(len(status["available_indicators"])))
                
                console.print(table)
            else:
                print(f"Status: {status['status']}")
                print(f"Version: {status['version']}")
                print(f"Available Indicators: {len(status['available_indicators'])}")

        # Test 3: Available Indicators
        if HAS_RICH:
            console.print("\n[bold green]3. Available Indicators[/bold green]")
        else:
            print("\n3. Available Indicators")
            
        response = await client.get(f"{base_url}/indicators/available")
        if response.status_code == 200:
            indicators = response.json()
            if HAS_RICH:
                panel_content = "\n".join([f"• {indicator}" for indicator in indicators])
                console.print(Panel(panel_content, title="Technical Indicators"))
            else:
                for indicator in indicators:
                    print(f"  • {indicator}")

        # Test 4: Calculate Indicators
        if HAS_RICH:
            console.print("\n[bold green]4. Calculating Technical Indicators[/bold green]")
        else:
            print("\n4. Calculating Technical Indicators")

        sample_data = create_sample_data()
        request_payload = {
            "ohlcv_data": sample_data,
            "indicators": ["RSI_14", "SMA_20", "EMA_10", "MACD_12_26_9", "BB_20_2"]
        }

        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing indicators...", total=None)
                
                response = await client.post(
                    f"{base_url}/indicators/calculate",
                    json=request_payload,
                    timeout=30.0
                )
                
                progress.remove_task(task)
        else:
            print("Processing indicators...")
            response = await client.post(
                f"{base_url}/indicators/calculate",
                json=request_payload,
                timeout=30.0
            )

        if response.status_code == 200:
            results = response.json()
            
            # Group results by indicator
            indicator_results = {}
            for result in results:
                indicator = result["indicator"]
                if indicator not in indicator_results:
                    indicator_results[indicator] = []
                indicator_results[indicator].append(result)

            if HAS_RICH:
                console.print(f"\n✅ [green]Calculated {len(results)} indicator values[/green]")
                
                # Display latest values for each indicator
                table = Table(title="Latest Indicator Values")
                table.add_column("Indicator", style="cyan")
                table.add_column("Value", style="green")
                table.add_column("Components", style="yellow")
                
                for indicator, values in indicator_results.items():
                    if values:
                        latest = values[-1]
                        value_str = f"{latest['value']:.4f}"
                        
                        components_str = ""
                        if latest.get("components"):
                            components_str = ", ".join([
                                f"{k}: {v:.4f}" for k, v in latest["components"].items()
                            ])
                        
                        table.add_row(indicator, value_str, components_str or "N/A")
                
                console.print(table)
                
            else:
                print(f"\n✅ Calculated {len(results)} indicator values")
                print("\nLatest Indicator Values:")
                for indicator, values in indicator_results.items():
                    if values:
                        latest = values[-1]
                        print(f"  {indicator}: {latest['value']:.4f}")
                        if latest.get("components"):
                            for k, v in latest["components"].items():
                                print(f"    {k}: {v:.4f}")

        else:
            if HAS_RICH:
                console.print(f"❌ [red]Failed to calculate indicators: {response.status_code}[/red]")
                console.print(response.text)
            else:
                print(f"❌ Failed to calculate indicators: {response.status_code}")
                print(response.text)

        # Test 5: Individual Indicator Info
        if HAS_RICH:
            console.print("\n[bold green]5. Indicator Information[/bold green]")
        else:
            print("\n5. Indicator Information")
            
        response = await client.get(f"{base_url}/indicators/RSI_14/info")
        if response.status_code == 200:
            info = response.json()
            if HAS_RICH:
                console.print(f"[cyan]{info['name']}[/cyan]: {info['type']}")
            else:
                print(f"{info['name']}: {info['type']}")


def start_server():
    """Start the FastAPI server in a separate process."""
    import multiprocessing
    import sys
    import os
    
    def run_server():
        # Add src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from src.api import app
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()
    
    # Wait for server to start
    time.sleep(3)
    
    return server_process


async def main():
    """Main demo function."""
    
    if HAS_RICH:
        console.print("[bold yellow]Starting Technical Analysis System Demo...[/bold yellow]")
        console.print("This demo will show the system's capabilities.\n")
    else:
        print("Starting Technical Analysis System Demo...")
        print("This demo will show the system's capabilities.\n")

    # Check if server is already running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=2.0)
            server_running = response.status_code == 200
    except:
        server_running = False

    if not server_running:
        if HAS_RICH:
            console.print("Starting API server...")
        else:
            print("Starting API server...")
        
        server_process = start_server()
        
        try:
            await test_api_endpoints()
        finally:
            if HAS_RICH:
                console.print("\n[yellow]Stopping server...[/yellow]")
            else:
                print("\nStopping server...")
            server_process.terminate()
            server_process.join()
    else:
        if HAS_RICH:
            console.print("Using existing API server...")
        else:
            print("Using existing API server...")
        await test_api_endpoints()

    if HAS_RICH:
        console.print("\n🎉 [bold green]Demo completed successfully![/bold green]")
        console.print("\n[dim]To explore more:")
        console.print("• Visit http://localhost:8000/docs for interactive API documentation")
        console.print("• Try different indicator combinations")
        console.print("• Test with real market data")
    else:
        print("\n🎉 Demo completed successfully!")
        print("\nTo explore more:")
        print("• Visit http://localhost:8000/docs for interactive API documentation") 
        print("• Try different indicator combinations")
        print("• Test with real market data")


if __name__ == "__main__":
    try:
        if HAS_RICH:
            # Install rich for better output
            console.print("[dim]For best experience, install rich: pip install rich[/dim]\n")
        
        asyncio.run(main())
    except KeyboardInterrupt:
        if HAS_RICH:
            console.print("\n[yellow]Demo interrupted by user[/yellow]")
        else:
            print("\nDemo interrupted by user")
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[red]Demo failed: {e}[/red]")
        else:
            print(f"\nDemo failed: {e}")