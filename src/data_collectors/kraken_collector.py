#!/usr/bin/env python3
"""
Kraken Pro API Data Collector

Comprehensive data collection system for Kraken cryptocurrency exchange:
- Historical data backfilling from 2024 to current
- Real-time WebSocket data streaming
- OHLCV candle data collection
- Automatic gap detection and filling
- InfluxDB integration with batching
- Error handling and recovery

Usage:
    python src/data_collectors/kraken_collector.py [--backfill] [--live] [--symbols BTC,ETH,LTC]
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
import urllib.parse
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import websockets
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KrakenAPI:
    """Kraken API client with authentication and rate limiting."""

    def __init__(
        self, api_key: str, private_key: str, base_url: str = "https://api.kraken.com"
    ):
        """
        Initialize Kraken API client.

        Args:
            api_key: Kraken API key
            private_key: Kraken private key
            base_url: Kraken API base URL
        """
        self.api_key = api_key
        self.private_key = private_key
        self.base_url = base_url

        # Rate limiting - Kraken allows 2 requests per second for public API
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 1 request per 2 seconds to be safe

        # Session management
        self._session: aiohttp.ClientSession | None = None

        logger.info("Kraken API client initialized")

    async def _ensure_session(self) -> None:
        """Ensure HTTP session is available."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _get_signature(self, url_path: str, data: dict[str, Any], nonce: str) -> str:
        """Generate API signature for authenticated requests."""
        postdata = urllib.parse.urlencode(data)
        encoded = (nonce + postdata).encode()
        message = url_path.encode() + hashlib.sha256(encoded).digest()

        mac = hmac.new(base64.b64decode(self.private_key), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            await asyncio.sleep(wait_time)

        self.last_request_time = time.time()

    async def public_request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make public API request."""
        await self._ensure_session()
        await self._rate_limit()

        url = f"{self.base_url}/0/public/{endpoint}"

        try:
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(
                        f"API error {response.status}: {await response.text()}"
                    )

                data = await response.json()

                if data.get("error"):
                    raise Exception(f"Kraken API error: {data['error']}")

                return data.get("result", {})

        except Exception as e:
            logger.error(f"Public request failed: {e}")
            raise

    async def private_request(
        self, endpoint: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make private API request with authentication."""
        await self._ensure_session()
        await self._rate_limit()

        if data is None:
            data = {}

        # Add nonce
        nonce = str(int(1000 * time.time()))
        data["nonce"] = nonce

        url_path = f"/0/private/{endpoint}"
        url = f"{self.base_url}{url_path}"

        # Generate signature
        signature = self._get_signature(url_path, data, nonce)

        headers = {
            "API-Key": self.api_key,
            "API-Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        try:
            async with self._session.post(url, data=data, headers=headers) as response:
                if response.status != 200:
                    raise Exception(
                        f"API error {response.status}: {await response.text()}"
                    )

                response_data = await response.json()

                if response_data.get("error"):
                    raise Exception(f"Kraken API error: {response_data['error']}")

                return response_data.get("result", {})

        except Exception as e:
            logger.error(f"Private request failed: {e}")
            raise


class KrakenDataCollector:
    """
    Comprehensive Kraken data collector with backfilling and live streaming.

    Features:
    - Historical data backfilling from specified start date
    - Real-time WebSocket data streaming
    - Gap detection and automatic filling
    - InfluxDB integration with batching
    - Error recovery and reconnection
    """

    # Supported trading pairs - Updated with correct Kraken symbols
    SUPPORTED_PAIRS = {
        "BTC": "XXBTZUSD",  # Bitcoin (base asset: XXBT)
        "ETH": "XETHZUSD",  # Ethereum (base asset: XETH)
        "LTC": "XLTCZUSD",  # Litecoin (base asset: XLTC)
        "ADA": "ADAUSD",  # Cardano (base asset: ADA)
        "DOT": "DOTUSD",  # Polkadot (base asset: DOT)
        "LINK": "LINKUSD",  # Chainlink (base asset: LINK)
        "SOL": "SOLUSD",  # Solana (base asset: SOL)
        "MATIC": "POLUSD",  # Polygon (rebranded from MATIC to POL)
    }

    # Timeframes (Kraken intervals)
    TIMEFRAMES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
        "1w": 10080,
    }

    def __init__(
        self,
        api_key: str,
        private_key: str,
        influx_url: str,
        influx_token: str,
        influx_org: str,
        influx_bucket: str,
        websocket_url: str = "wss://ws.kraken.com/v2",
    ):
        """
        Initialize Kraken data collector.

        Args:
            api_key: Kraken API key
            private_key: Kraken private key
            influx_url: InfluxDB URL
            influx_token: InfluxDB token
            influx_org: InfluxDB organization
            influx_bucket: InfluxDB bucket
            websocket_url: Kraken WebSocket URL
        """
        # API client
        self.api = KrakenAPI(api_key, private_key)

        # InfluxDB client
        self.influx_client = InfluxDBClient(
            url=influx_url, token=influx_token, org=influx_org
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.influx_client.query_api()
        self.bucket = influx_bucket
        self.org = influx_org

        # WebSocket
        self.websocket_url = websocket_url
        self.websocket = None

        # Data management
        self.data_buffer = []
        self.buffer_size = 1000
        self.subscribed_pairs: set[str] = set()

        # Statistics
        self.stats = {
            "backfill_records": 0,
            "live_records": 0,
            "errors": 0,
            "last_update": None,
        }

        logger.info("Kraken data collector initialized")

    async def get_available_pairs(self) -> dict[str, str]:
        """Get available trading pairs from Kraken."""
        try:
            data = await self.api.public_request("AssetPairs")

            # Create mapping from common symbols to Kraken pair IDs
            pairs = {}
            symbol_mapping = {}

            # Build reverse mapping from Kraken symbols to common symbols
            for common_symbol, kraken_pair in self.SUPPORTED_PAIRS.items():
                symbol_mapping[kraken_pair] = common_symbol

            # Verify all our supported pairs are available
            for pair_id, pair_info in data.items():
                if (
                    pair_info.get("quote") == "ZUSD"
                    and pair_info.get("status") == "online"
                ):
                    if pair_id in symbol_mapping:
                        common_symbol = symbol_mapping[pair_id]
                        pairs[common_symbol] = pair_id
                        logger.debug(f"Mapped {common_symbol} -> {pair_id}")

            # Add any missing pairs from our supported list
            for common_symbol, kraken_pair in self.SUPPORTED_PAIRS.items():
                if common_symbol not in pairs:
                    logger.warning(
                        f"Supported pair {common_symbol} ({kraken_pair}) not found in available pairs"
                    )
                    pairs[common_symbol] = kraken_pair  # Use it anyway

            logger.info(f"Mapped {len(pairs)} supported USD pairs")
            return pairs

        except Exception as e:
            logger.error(f"Failed to get available pairs: {e}")
            return self.SUPPORTED_PAIRS

    async def get_historical_data(
        self,
        pair: str,
        timeframe: str = "1h",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get historical OHLCV data for a trading pair.

        Args:
            pair: Trading pair (e.g., 'XXBTZUSD')
            timeframe: Timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')
            start_time: Start time (if None, gets last 720 candles)
            end_time: End time (if None, uses current time)

        Returns:
            List of OHLCV data dictionaries
        """
        try:
            params = {"pair": pair, "interval": self.TIMEFRAMES.get(timeframe, 60)}

            # Add time parameters if specified
            if start_time:
                params["since"] = int(start_time.timestamp())

            data = await self.api.public_request("OHLC", params)

            # Extract OHLC data
            pair_data = data.get(pair, [])
            if not pair_data:
                logger.warning(f"No data returned for {pair}")
                return []

            # Convert to standard format
            ohlcv_data = []
            for candle in pair_data:
                ohlcv_data.append(
                    {
                        "timestamp": datetime.fromtimestamp(int(candle[0])),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "vwap": float(candle[5]),
                        "volume": float(candle[6]),
                        "count": int(candle[7]),
                        "pair": pair,
                        "timeframe": timeframe,
                    }
                )

            logger.info(f"Retrieved {len(ohlcv_data)} {timeframe} candles for {pair}")
            return ohlcv_data

        except Exception as e:
            logger.error(f"Failed to get historical data for {pair}: {e}")
            return []

    async def backfill_historical_data(
        self,
        symbols: list[str] = None,
        timeframes: list[str] = None,
        start_date: datetime = None,
        batch_size: int = 500,
    ) -> None:
        """
        Backfill historical data for specified symbols and timeframes.

        Args:
            symbols: List of symbols to backfill (default: BTC, ETH, LTC)
            timeframes: List of timeframes (default: 15m, 1h, 1d)
            start_date: Start date for backfill (default: 2024-01-01)
            batch_size: Batch size for InfluxDB writes
        """
        if symbols is None:
            symbols = ["BTC", "ETH", "LTC"]
        if timeframes is None:
            timeframes = ["15m", "1h", "1d"]
        if start_date is None:
            start_date = datetime(2024, 1, 1)

        logger.info(f"Starting backfill for {symbols} from {start_date}")

        # Get available pairs
        available_pairs = await self.get_available_pairs()

        for symbol in symbols:
            if symbol not in available_pairs:
                logger.warning(f"Symbol {symbol} not available, skipping")
                continue

            pair = available_pairs[symbol]

            for timeframe in timeframes:
                await self._backfill_pair_timeframe(
                    pair, symbol, timeframe, start_date, batch_size
                )

        logger.info("Backfill completed")

    async def _backfill_pair_timeframe(
        self,
        pair: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        batch_size: int,
    ) -> None:
        """Backfill data for a specific pair and timeframe."""
        logger.info(f"Backfilling {symbol} {timeframe} from {start_date}")

        current_time = start_date
        end_time = datetime.now()

        # Calculate chunk size based on timeframe
        if timeframe in ["1m", "5m"]:
            chunk_hours = 24  # 1 day chunks for minute data
        elif timeframe in ["15m", "30m", "1h"]:
            chunk_hours = 168  # 1 week chunks
        else:
            chunk_hours = 720  # 30 day chunks for daily data

        while current_time < end_time:
            chunk_end = min(current_time + timedelta(hours=chunk_hours), end_time)

            try:
                # Get historical data
                data = await self.get_historical_data(
                    pair=pair,
                    timeframe=timeframe,
                    start_time=current_time,
                    end_time=chunk_end,
                )

                if data:
                    # Convert to InfluxDB points
                    points = []
                    for candle in data:
                        point = (
                            Point("kraken_ohlcv")
                            .tag("symbol", symbol)
                            .tag("timeframe", timeframe)
                            .tag("pair", pair)
                            .field("open", candle["open"])
                            .field("high", candle["high"])
                            .field("low", candle["low"])
                            .field("close", candle["close"])
                            .field("volume", candle["volume"])
                            .field("vwap", candle["vwap"])
                            .field("count", candle["count"])
                            .time(candle["timestamp"], WritePrecision.S)
                        )
                        points.append(point)

                    # Write in batches
                    for i in range(0, len(points), batch_size):
                        batch = points[i : i + batch_size]
                        self.write_api.write(bucket=self.bucket, record=batch)
                        self.stats["backfill_records"] += len(batch)

                    logger.info(
                        f"  Backfilled {len(data)} records for {symbol} {timeframe} ({current_time.date()})"
                    )

                current_time = chunk_end
                await asyncio.sleep(
                    2
                )  # Rate limiting - additional delay between chunks

            except Exception as e:
                logger.error(
                    f"Backfill error for {symbol} {timeframe} at {current_time}: {e}"
                )
                current_time += timedelta(hours=chunk_hours)
                await asyncio.sleep(5)  # Wait before retry

    async def start_live_collection(self, symbols: list[str] = None) -> None:
        """Start live data collection via WebSocket."""
        if symbols is None:
            symbols = ["BTC", "ETH", "LTC"]

        logger.info(f"Starting live data collection for {symbols}")

        # Get available pairs
        available_pairs = await self.get_available_pairs()

        # Subscribe to OHLC data for each symbol
        subscriptions = []
        for symbol in symbols:
            if symbol in available_pairs:
                pair = available_pairs[symbol]
                subscriptions.append(
                    {
                        "method": "subscribe",
                        "params": {
                            "channel": "ohlc",
                            "symbol": [pair],
                            "interval": [60],  # 1-hour candles
                        },
                    }
                )
                self.subscribed_pairs.add(pair)

        # Start WebSocket connection
        await self._websocket_loop(subscriptions)

    async def _websocket_loop(self, subscriptions: list[dict[str, Any]]) -> None:
        """Main WebSocket connection loop with auto-reconnection."""
        while True:
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    self.websocket = websocket
                    logger.info("WebSocket connected")

                    # Send subscriptions
                    for subscription in subscriptions:
                        await websocket.send(json.dumps(subscription))
                        logger.info(
                            f"Subscribed to {subscription['params']['channel']}"
                        )

                    # Listen for messages
                    async for message in websocket:
                        try:
                            await self._handle_websocket_message(message)
                        except Exception as e:
                            logger.error(f"Error handling WebSocket message: {e}")

            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                logger.info("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def _handle_websocket_message(self, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            # Handle OHLC data
            if data.get("channel") == "ohlc" and "data" in data:
                for ohlc_data in data["data"]:
                    await self._process_ohlc_data(ohlc_data)

            # Handle heartbeat
            elif data.get("method") == "heartbeat":
                await self.websocket.send(json.dumps({"method": "heartbeat"}))

        except Exception as e:
            logger.error(f"Failed to handle WebSocket message: {e}")

    async def _process_ohlc_data(self, ohlc_data: dict[str, Any]) -> None:
        """Process OHLC data from WebSocket."""
        try:
            symbol = ohlc_data.get("symbol", "")

            # Extract symbol name from pair
            symbol_name = symbol.replace("ZUSD", "").replace("USD", "")
            if symbol_name.startswith("X"):
                symbol_name = symbol_name[1:]

            # Create InfluxDB point
            point = (
                Point("kraken_live_ohlcv")
                .tag("symbol", symbol_name)
                .tag("timeframe", "1h")
                .tag("pair", symbol)
                .field("open", float(ohlc_data["open"]))
                .field("high", float(ohlc_data["high"]))
                .field("low", float(ohlc_data["low"]))
                .field("close", float(ohlc_data["close"]))
                .field("volume", float(ohlc_data["volume"]))
                .field("vwap", float(ohlc_data["vwap"]))
                .field("count", int(ohlc_data["count"]))
                .time(datetime.now(), WritePrecision.S)
            )

            # Add to buffer
            self.data_buffer.append(point)

            # Flush buffer if full
            if len(self.data_buffer) >= self.buffer_size:
                await self._flush_buffer()

            self.stats["live_records"] += 1
            self.stats["last_update"] = datetime.now()

            logger.debug(f"Processed live OHLC for {symbol_name}: {ohlc_data['close']}")

        except Exception as e:
            logger.error(f"Failed to process OHLC data: {e}")
            self.stats["errors"] += 1

    async def _flush_buffer(self) -> None:
        """Flush data buffer to InfluxDB."""
        if not self.data_buffer:
            return

        try:
            self.write_api.write(bucket=self.bucket, record=self.data_buffer)
            buffer_size = len(self.data_buffer)
            self.data_buffer.clear()
            logger.debug(f"Flushed {buffer_size} records to InfluxDB")

        except Exception as e:
            logger.error(f"Failed to flush buffer: {e}")
            self.stats["errors"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        return {
            **self.stats,
            "buffer_size": len(self.data_buffer),
            "subscribed_pairs": list(self.subscribed_pairs),
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Flush remaining buffer
        await self._flush_buffer()

        # Close connections
        if self.websocket:
            await self.websocket.close()

        await self.api.close()
        self.influx_client.close()

        logger.info("Kraken data collector cleaned up")


async def main():
    """Main function for running the data collector."""
    import argparse

    parser = argparse.ArgumentParser(description="Kraken data collector")
    parser.add_argument(
        "--backfill", action="store_true", help="Run historical data backfill"
    )
    parser.add_argument(
        "--live", action="store_true", help="Start live data collection"
    )
    parser.add_argument(
        "--symbols", default="BTC,ETH,LTC", help="Comma-separated symbols"
    )
    parser.add_argument(
        "--timeframes", default="15m,1h,1d", help="Comma-separated timeframes"
    )
    parser.add_argument("--start-date", help="Start date for backfill (YYYY-MM-DD)")
    args = parser.parse_args()

    # Get configuration from environment
    api_key = os.getenv("KRAKEN_API_KEY")
    private_key = os.getenv("KRAKEN_PRIVATE_KEY")
    influx_url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
    influx_token = os.getenv("INFLUXDB_TOKEN")
    influx_org = os.getenv("INFLUXDB_ORG", "TradeKnowledge")
    influx_bucket = os.getenv("INFLUXDB_BUCKET", "data")

    if not all([api_key, private_key, influx_token]):
        logger.error("Missing required environment variables")
        return

    # Parse arguments
    symbols = [s.strip() for s in args.symbols.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")]

    start_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

    # Initialize collector
    collector = KrakenDataCollector(
        api_key=api_key,
        private_key=private_key,
        influx_url=influx_url,
        influx_token=influx_token,
        influx_org=influx_org,
        influx_bucket=influx_bucket,
    )

    try:
        if args.backfill:
            await collector.backfill_historical_data(
                symbols=symbols, timeframes=timeframes, start_date=start_date
            )

        if args.live:
            await collector.start_live_collection(symbols=symbols)

        if not args.backfill and not args.live:
            logger.info("No action specified. Use --backfill or --live")

    finally:
        await collector.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
