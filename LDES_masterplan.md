# LDES API Connection and Data Storage Guide

## Introduction

This guide walks you through connecting to each data source for the LDES project, storing real-time data, and retrieving historical information. Think of this as your cookbook - each section provides the exact recipe you need to get data flowing from a specific source into your InfluxDB database.

## Prerequisites and Environment Setup

Before we dive into specific APIs, let's establish a solid foundation. First, you'll need to set up your development environment with the necessary tools and create a secure configuration system.

### Setting Up Your Python Environment

Create a new virtual environment specifically for this project. This isolation prevents conflicts with other Python projects on your system:

```bash
# Create a new virtual environment
python -m venv ldes-env

# Activate it (Windows)
ldes-env\Scripts\activate

# Install core dependencies
pip install python-dotenv influxdb-client pandas numpy asyncio aiohttp
```

### Creating Your Configuration File

Security is paramount when dealing with financial APIs. Never hardcode credentials in your source code. Instead, create a `.env` file in your project root:

```bash
# .env file - NEVER commit this to version control
# Alpaca Configuration (Paper Trading)
ALPACA_API_KEY=PKN1X38722O4JSU5M0IN
ALPACA_SECRET_KEY=hap5VDqFlxlJZwyq6oLvf4knADcii9mGz1yqluN5
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2

# InfluxDB Configuration
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_influxdb_token
INFLUXDB_ORG=ldes
INFLUXDB_BUCKET=market_data

# Schwab Configuration (when available)
SCHWAB_APP_KEY=your_app_key
SCHWAB_SECRET=your_secret
SCHWAB_ACCOUNT_ID=scott.schwiezer@gmail.com

# Binance Configuration (optional for authenticated endpoints)
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
```

### Setting Up InfluxDB

InfluxDB serves as our time-series database, perfectly suited for financial data. Here's how to get it running:

```bash
# Using Docker (recommended for consistency)
docker run -d \
  --name influxdb \
  -p 8086:8086 \
  -v C:\Users\scott\.influxdb:/var/lib/influxdb2 \
  -e INFLUXDB_DB=ldes \
  -e INFLUXDB_ADMIN_USER=admin \
  -e INFLUXDB_ADMIN_PASSWORD=your_secure_password \
  influxdb:2.7

# After container starts, access the UI at http://localhost:8086
# Create a new bucket called "market_data" with 30-day retention
```

## Alpaca Markets API Connection

Alpaca provides excellent real-time market data through their WebSocket API. Their paper trading environment is perfect for development - it provides real market data without any risk.

### Initial Setup and Authentication

First, let's create a reusable Alpaca client class that handles authentication and connection management:

```python
# alpaca_client.py
import os
from dotenv import load_dotenv
from alpaca.data import StockDataStream
from alpaca.data.models import Trade, Quote
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.historical import StockHistoricalDataClient

load_dotenv()

class AlpacaDataCollector:
    def __init__(self):
        # Load credentials from environment variables
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        # Initialize the WebSocket client for real-time data
        self.stream = StockDataStream(self.api_key, self.secret_key)
        
        # Initialize the REST client for historical data
        self.historical_client = StockHistoricalDataClient(
            self.api_key, 
            self.secret_key
        )
        
        # Set up InfluxDB connection
        self.influx_writer = InfluxWriter()  # We'll define this later
        
    async def handle_trade(self, trade: Trade):
        """Process incoming trade data"""
        # This method is called for every trade that occurs
        point = {
            "measurement": "market_data",
            "tags": {
                "symbol": trade.symbol,
                "source": "alpaca",
                "asset_type": "stock"
            },
            "fields": {
                "last_price": float(trade.price),
                "last_size": trade.size,
                "volume": trade.size  # We'll accumulate this
            },
            "time": trade.timestamp
        }
        await self.influx_writer.write_point(point)
        
    async def handle_quote(self, quote: Quote):
        """Process incoming quote (bid/ask) data"""
        point = {
            "measurement": "market_data",
            "tags": {
                "symbol": quote.symbol,
                "source": "alpaca",
                "asset_type": "stock"
            },
            "fields": {
                "bid_price": float(quote.bid_price),
                "ask_price": float(quote.ask_price),
                "bid_size": quote.bid_size,
                "ask_size": quote.ask_size,
                "spread_bps": ((quote.ask_price - quote.bid_price) / quote.bid_price) * 10000
            },
            "time": quote.timestamp
        }
        await self.influx_writer.write_point(point)
```

### Connecting to Real-Time Data

Now let's establish the WebSocket connection and start receiving data:

```python
# Starting real-time data collection
async def start_alpaca_stream(symbols=['SPY', 'QQQ', 'AAPL']):
    collector = AlpacaDataCollector()
    
    # Subscribe to both trades and quotes for our symbols
    for symbol in symbols:
        collector.stream.subscribe_trades(symbol, collector.handle_trade)
        collector.stream.subscribe_quotes(symbol, collector.handle_quote)
    
    # Start the WebSocket connection
    print(f"Starting Alpaca stream for symbols: {symbols}")
    await collector.stream.run()
```

### Retrieving Historical Data

Historical data is crucial for backtesting and analysis. Alpaca makes this straightforward:

```python
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

def fetch_historical_bars(symbol, days_back=30):
    """Fetch historical OHLCV data for analysis"""
    collector = AlpacaDataCollector()
    
    # Define our request parameters
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,  # 1-minute bars
        start=datetime.now() - timedelta(days=days_back),
        end=datetime.now()
    )
    
    # Fetch the data
    bars = collector.historical_client.get_stock_bars(request)
    
    # Convert to DataFrame for easier manipulation
    df = bars.df
    
    # Write to InfluxDB
    for index, row in df.iterrows():
        point = {
            "measurement": "market_data_historical",
            "tags": {
                "symbol": symbol,
                "source": "alpaca",
                "timeframe": "1min"
            },
            "fields": {
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row['volume'])
            },
            "time": index  # Timestamp is in the index
        }
        collector.influx_writer.write_point_sync(point)
    
    return df
```

## Yahoo Finance (yfinance) Integration

Yahoo Finance provides free access to a vast array of market data. While it doesn't offer real-time WebSocket connections, we can poll it efficiently for near-real-time data and excellent historical coverage.

### Setting Up yfinance

The beauty of yfinance is its simplicity - no API keys required:

```python
# yfinance_client.py
import yfinance as yf
import asyncio
from datetime import datetime, timedelta
import pandas as pd

class YFinanceCollector:
    def __init__(self, symbols, interval='1m'):
        self.symbols = symbols
        self.interval = interval
        self.influx_writer = InfluxWriter()
        
    async def collect_current_data(self):
        """Fetch current market data for all symbols"""
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get current quote data
                info = ticker.info
                
                # Get the latest price data
                history = ticker.history(period='1d', interval='1m')
                
                if not history.empty:
                    latest = history.iloc[-1]
                    
                    point = {
                        "measurement": "market_data",
                        "tags": {
                            "symbol": symbol,
                            "source": "yfinance",
                            "asset_type": "stock"
                        },
                        "fields": {
                            "last_price": float(latest['Close']),
                            "volume": int(latest['Volume']),
                            "high": float(latest['High']),
                            "low": float(latest['Low'])
                        },
                        "time": datetime.now()
                    }
                    
                    await self.influx_writer.write_point(point)
                    
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                
    async def run_polling_loop(self, interval_seconds=30):
        """Continuously poll for new data"""
        while True:
            await self.collect_current_data()
            await asyncio.sleep(interval_seconds)
```

### Fetching Historical Data with yfinance

One of yfinance's strengths is its historical data access:

```python
def fetch_yfinance_historical(symbol, period='1mo', interval='1h'):
    """
    Fetch historical data from Yahoo Finance
    
    Periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    Intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    """
    ticker = yf.Ticker(symbol)
    
    # Fetch the historical data
    history = ticker.history(period=period, interval=interval)
    
    # Write to InfluxDB
    influx_writer = InfluxWriter()
    
    for index, row in history.iterrows():
        point = {
            "measurement": "market_data_historical",
            "tags": {
                "symbol": symbol,
                "source": "yfinance",
                "interval": interval
            },
            "fields": {
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            },
            "time": index
        }
        influx_writer.write_point_sync(point)
    
    return history
```

## Schwab Developer API

The Schwab API requires a more complex authentication flow using OAuth2. Let's walk through the setup process:

### Initial Setup and App Registration

Before you can use the Schwab API, you need to register your application:

1. Visit https://developer.schwab.com
2. Create a new app and note your App Key and Secret
3. Set up your redirect URI (use http://localhost:8000/callback for development)

### Implementing OAuth2 Authentication

Schwab uses OAuth2 for authentication, which requires a bit more setup:

```python
# schwab_client.py
import requests
from urllib.parse import urlencode
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

class SchwabAuthHandler(BaseHTTPRequestHandler):
    """Handle OAuth callback"""
    def do_GET(self):
        # Extract authorization code from callback
        query = self.path.split('?')[1]
        params = dict(param.split('=') for param in query.split('&'))
        self.server.auth_code = params.get('code')
        
        # Send response to browser
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<html><body><h1>Authentication successful! You can close this window.</h1></body></html>')

class SchwabClient:
    def __init__(self):
        self.app_key = os.getenv('SCHWAB_APP_KEY')
        self.app_secret = os.getenv('SCHWAB_SECRET')
        self.redirect_uri = 'http://localhost:8000/callback'
        self.token = None
        
    def authenticate(self):
        """Perform OAuth2 authentication flow"""
        # Step 1: Get authorization code
        auth_url = f"https://api.schwabapi.com/v1/oauth/authorize"
        params = {
            'response_type': 'code',
            'client_id': self.app_key,
            'redirect_uri': self.redirect_uri,
            'scope': 'read'
        }
        
        # Open browser for user authorization
        full_url = f"{auth_url}?{urlencode(params)}"
        webbrowser.open(full_url)
        
        # Start local server to receive callback
        server = HTTPServer(('localhost', 8000), SchwabAuthHandler)
        server.handle_request()
        
        # Step 2: Exchange code for token
        token_url = "https://api.schwabapi.com/v1/oauth/token"
        data = {
            'grant_type': 'authorization_code',
            'code': server.auth_code,
            'client_id': self.app_key,
            'client_secret': self.app_secret,
            'redirect_uri': self.redirect_uri
        }
        
        response = requests.post(token_url, data=data)
        self.token = response.json()['access_token']
        
        return self.token
```

### Fetching Market Data from Schwab

Once authenticated, you can fetch market data:

```python
def fetch_schwab_quotes(self, symbols):
    """Fetch current quotes from Schwab"""
    headers = {
        'Authorization': f'Bearer {self.token}',
        'Accept': 'application/json'
    }
    
    # Schwab uses symbol lookup
    for symbol in symbols:
        url = f"https://api.schwabapi.com/marketdata/v1/quotes/{symbol}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            point = {
                "measurement": "market_data",
                "tags": {
                    "symbol": symbol,
                    "source": "schwab",
                    "asset_type": "stock"
                },
                "fields": {
                    "last_price": float(data['lastPrice']),
                    "bid_price": float(data['bidPrice']),
                    "ask_price": float(data['askPrice']),
                    "volume": int(data['totalVolume'])
                },
                "time": datetime.now()
            }
            
            self.influx_writer.write_point_sync(point)
```

## Binance API for Cryptocurrency

Binance offers excellent WebSocket APIs for real-time crypto data. The best part? Public market data doesn't require authentication.

### Setting Up Binance WebSocket

```python
# binance_client.py
import websocket
import json
import threading

class BinanceWebSocketClient:
    def __init__(self, symbols=['BTCUSDT', 'ETHUSDT']):
        self.symbols = [s.lower() for s in symbols]
        self.influx_writer = InfluxWriter()
        self.ws = None
        
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        data = json.loads(message)
        
        # Handle different message types
        if 'e' in data:
            if data['e'] == 'trade':
                self.process_trade(data)
            elif data['e'] == 'depthUpdate':
                self.process_depth(data)
                
    def process_trade(self, data):
        """Process trade data"""
        point = {
            "measurement": "market_data",
            "tags": {
                "symbol": data['s'],
                "source": "binance",
                "asset_type": "crypto"
            },
            "fields": {
                "last_price": float(data['p']),
                "last_size": float(data['q']),
                "trade_id": data['t']
            },
            "time": datetime.fromtimestamp(data['T'] / 1000)  # Convert ms to seconds
        }
        self.influx_writer.write_point_sync(point)
        
    def start_stream(self):
        """Connect to Binance WebSocket"""
        # Create subscription string for multiple symbols
        streams = []
        for symbol in self.symbols:
            streams.append(f"{symbol}@trade")
            streams.append(f"{symbol}@depth20@100ms")
            
        stream_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        
        self.ws = websocket.WebSocketApp(
            stream_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Run in a separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
```

### Fetching Historical Crypto Data

```python
def fetch_binance_historical(symbol='BTCUSDT', interval='1h', limit=1000):
    """
    Fetch historical kline/candlestick data
    
    Intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit  # Max 1000
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    influx_writer = InfluxWriter()
    
    for candle in data:
        # Binance returns array: [timestamp, open, high, low, close, volume, ...]
        point = {
            "measurement": "market_data_historical",
            "tags": {
                "symbol": symbol,
                "source": "binance",
                "interval": interval
            },
            "fields": {
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5])
            },
            "time": datetime.fromtimestamp(candle[0] / 1000)
        }
        influx_writer.write_point_sync(point)
    
    return data
```

## InfluxDB Writer Implementation

Now let's implement the InfluxDB writer that all our collectors use:

```python
# influx_writer.py
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
import asyncio
from collections import deque
import threading

class InfluxWriter:
    def __init__(self):
        self.client = InfluxDBClient(
            url=os.getenv('INFLUXDB_URL'),
            token=os.getenv('INFLUXDB_TOKEN'),
            org=os.getenv('INFLUXDB_ORG')
        )
        self.bucket = os.getenv('INFLUXDB_BUCKET')
        
        # Async write API for real-time data
        self.write_api_async = self.client.write_api(write_options=ASYNCHRONOUS)
        
        # Sync write API for historical data
        self.write_api_sync = self.client.write_api(write_options=SYNCHRONOUS)
        
        # Buffer for batching
        self.buffer = deque(maxlen=10000)
        self.buffer_lock = threading.Lock()
        
        # Start batch writer thread
        self.start_batch_writer()
        
    async def write_point(self, point_dict):
        """Write a point asynchronously"""
        point = Point(point_dict['measurement'])
        
        for key, value in point_dict['tags'].items():
            point = point.tag(key, value)
            
        for key, value in point_dict['fields'].items():
            point = point.field(key, value)
            
        point = point.time(point_dict['time'])
        
        # Add to buffer
        with self.buffer_lock:
            self.buffer.append(point)
            
    def write_point_sync(self, point_dict):
        """Write a point synchronously (for historical data)"""
        point = Point(point_dict['measurement'])
        
        for key, value in point_dict['tags'].items():
            point = point.tag(key, value)
            
        for key, value in point_dict['fields'].items():
            point = point.field(key, value)
            
        point = point.time(point_dict['time'])
        
        self.write_api_sync.write(self.bucket, self.org, point)
        
    def start_batch_writer(self):
        """Background thread for batch writing"""
        def writer_loop():
            while True:
                if len(self.buffer) >= 1000:
                    # Write batch
                    with self.buffer_lock:
                        batch = list(self.buffer)
                        self.buffer.clear()
                        
                    self.write_api_async.write(self.bucket, self.org, batch)
                    
                time.sleep(1)  # Check every second
                
        thread = threading.Thread(target=writer_loop)
        thread.daemon = True
        thread.start()
```

## Putting It All Together

Here's the main orchestrator that runs all collectors:

```python
# main.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def main():
    # Initialize all collectors
    alpaca_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
    yfinance_symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'MSFT']
    crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    # Start Alpaca WebSocket
    alpaca_task = asyncio.create_task(start_alpaca_stream(alpaca_symbols))
    
    # Start yfinance polling
    yf_collector = YFinanceCollector(yfinance_symbols)
    yf_task = asyncio.create_task(yf_collector.run_polling_loop())
    
    # Start Binance WebSocket
    binance_client = BinanceWebSocketClient(crypto_symbols)
    binance_client.start_stream()
    
    # Start Schwab polling (if authenticated)
    # schwab_client = SchwabClient()
    # schwab_client.authenticate()
    # schwab_task = asyncio.create_task(schwab_polling_loop(schwab_client))
    
    # Keep the program running
    await asyncio.gather(alpaca_task, yf_task)

if __name__ == "__main__":
    asyncio.run(main())
```

## Verifying Data Collection

Once everything is running, you'll want to verify that data is flowing correctly into InfluxDB. Here are some essential queries to run in the InfluxDB UI:

### Check Data Freshness
```sql
-- See the latest data points for each symbol
SELECT last("last_price") 
FROM "market_data" 
WHERE time > now() - 5m 
GROUP BY "symbol", "source"
```

### Monitor Ingestion Rate
```sql
-- Count data points per minute by source
SELECT count("last_price") 
FROM "market_data" 
WHERE time > now() - 10m 
GROUP BY time(1m), "source"
```

### Verify Symbol Coverage
```sql
-- See which symbols are being collected
SELECT distinct("symbol") 
FROM "market_data" 
WHERE time > now() - 1h 
GROUP BY "source"
```

## Troubleshooting Common Issues

When setting up data collection, you might encounter these common problems:

### Alpaca Connection Issues

If your Alpaca WebSocket disconnects frequently, implement exponential backoff:

```python
async def start_alpaca_with_retry(symbols, max_retries=5):
    retry_count = 0
    backoff = 1
    
    while retry_count < max_retries:
        try:
            await start_alpaca_stream(symbols)
        except Exception as e:
            print(f"Alpaca connection failed: {e}")
            retry_count += 1
            wait_time = backoff * (2 ** retry_count)
            print(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
    
    raise Exception("Max retries exceeded for Alpaca connection")
```

### yfinance Rate Limiting

If you hit rate limits with yfinance, implement request throttling:

```python
class RateLimiter:
    def __init__(self, max_requests=2000, window=3600):
        self.max_requests = max_requests
        self.window = window
        self.requests = deque()
        
    async def acquire(self):
        now = time.time()
        # Remove old requests outside the window
        while self.requests and self.requests[0] < now - self.window:
            self.requests.popleft()
            
        if len(self.requests) >= self.max_requests:
            # Wait until we can make another request
            sleep_time = self.window - (now - self.requests[0])
            await asyncio.sleep(sleep_time)
            
        self.requests.append(now)
```

### InfluxDB Performance Tuning

If you're experiencing slow writes, optimize your InfluxDB configuration:

```toml
# influxdb.conf
[data]
  # Increase cache size for better write performance
  cache-max-memory-size = "2g"
  
  # Increase WAL size
  wal-max-write-delay = "10m"
  
  # Optimize compaction
  compact-throughput = "50m"
  compact-throughput-burst = "100m"
```

## Historical Data Backfill Strategy

To build a comprehensive dataset, you'll want to backfill historical data efficiently:

```python
async def backfill_all_sources(symbols, days_back=30):
    """Orchestrate historical data backfill from all sources"""
    
    print(f"Starting backfill for {len(symbols)} symbols, {days_back} days back")
    
    # Use ThreadPoolExecutor for parallel backfill
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Alpaca historical
        alpaca_futures = [
            executor.submit(fetch_historical_bars, symbol, days_back)
            for symbol in symbols
        ]
        
        # yfinance historical
        yf_futures = [
            executor.submit(fetch_yfinance_historical, symbol, f"{days_back}d", "1h")
            for symbol in symbols
        ]
        
        # Wait for all to complete
        for future in alpaca_futures + yf_futures:
            future.result()
            
    print("Backfill complete!")
```

## Production Deployment Checklist

Before deploying to production, ensure you've completed these steps:

1. **Security Hardening**
   - All API keys in environment variables
   - .env file added to .gitignore
   - Implement API key rotation schedule
   - Use secrets manager for production

2. **Monitoring Setup**
   - Configure Grafana dashboards for real-time monitoring
   - Set up alerts for data gaps
   - Implement health check endpoints
   - Monitor system resource usage

3. **Data Validation**
   - Implement price sanity checks
   - Validate volume data
   - Check for timestamp consistency
   - Monitor for duplicate data

4. **Backup and Recovery**
   - Set up InfluxDB backup schedule
   - Test recovery procedures
   - Document rollback process
   - Maintain configuration backups

5. **Performance Optimization**
   - Profile code for bottlenecks
   - Optimize database queries
   - Implement connection pooling
   - Monitor memory usage

## Next Steps

With data collection operational, you can now focus on:

1. **Building the Liquidation Detection Algorithm** - Use the streaming data to identify volume spikes and price movements
2. **Implementing Risk Management** - Add position sizing and portfolio management
3. **Creating a Backtesting Framework** - Use historical data to validate strategies
4. **Developing the Execution System** - Connect order placement when signals are detected

Remember, the quality of your data directly impacts the performance of your trading system. Take time to ensure your data collection is robust, accurate, and reliable before moving on to strategy development.