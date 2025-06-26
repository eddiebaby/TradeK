"""
Comprehensive Market Data Service for Trade Knowledge
Maximum granular data collection across multiple timeframes
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class MarketDataPoint:
    """Single market data point with comprehensive information"""

    symbol: str
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float | None = None
    vwap: float | None = None
    trades_count: int | None = None
    session_type: str = "regular"  # pre_market, regular, after_hours

    # Technical indicators
    rsi: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None

    # Data quality metrics
    data_quality_score: float = 1.0
    source: str = "yfinance"


class MarketDataService:
    """Comprehensive market data service with maximum granular collection"""

    def __init__(self, db_service=None):
        self.db_service = db_service
        self.executor = ThreadPoolExecutor(max_workers=10)

        # Maximum data collection parameters
        self.timeframe_limits = {
            "1m": {"period": "7d", "description": "7 days of 1-minute bars"},
            "2m": {"period": "60d", "description": "60 days of 2-minute bars"},
            "5m": {"period": "60d", "description": "60 days of 5-minute bars"},
            "15m": {"period": "60d", "description": "60 days of 15-minute bars"},
            "30m": {"period": "60d", "description": "60 days of 30-minute bars"},
            "1h": {"period": "730d", "description": "2 years of hourly bars"},
            "1d": {"period": "max", "description": "Maximum daily history"},
            "1wk": {"period": "max", "description": "Maximum weekly history"},
            "1mo": {"period": "max", "description": "Maximum monthly history"},
        }

        self.supported_symbols = [
            "SPY",
            "QQQ",
            "IWM",
            "DIA",  # Major ETFs
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",  # Tech giants
            "NVDA",
            "META",
            "NFLX",
            "CRM",
            "ORCL",  # More tech
            "VIX",
            "TLT",
            "GLD",
            "USO",  # Volatility, bonds, commodities
        ]

    async def fetch_comprehensive_data(
        self, symbols: list[str]
    ) -> dict[str, dict[str, list[MarketDataPoint]]]:
        """
        Fetch maximum granular data for symbols across all timeframes

        Returns:
            Dict[symbol][timeframe] = List[MarketDataPoint]
        """
        logger.info(f"Starting comprehensive data fetch for {len(symbols)} symbols")

        all_data = {}

        # Process symbols concurrently
        tasks = []
        for symbol in symbols:
            task = self._fetch_symbol_all_timeframes(symbol)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, result in zip(symbols, results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Error fetching data for {symbol}: {result}")
                all_data[symbol] = {}
            else:
                all_data[symbol] = result

        logger.info(
            f"Completed comprehensive data fetch. Total symbols processed: {len(all_data)}"
        )
        return all_data

    async def _fetch_symbol_all_timeframes(
        self, symbol: str
    ) -> dict[str, list[MarketDataPoint]]:
        """Fetch data for a single symbol across all timeframes"""
        symbol_data = {}

        for timeframe, config in self.timeframe_limits.items():
            try:
                logger.info(
                    f"Fetching {symbol} {timeframe} data - {config['description']}"
                )

                # Run yfinance in thread executor to avoid blocking
                loop = asyncio.get_event_loop()
                raw_data = await loop.run_in_executor(
                    self.executor,
                    self._fetch_yfinance_data,
                    symbol,
                    timeframe,
                    config["period"],
                )

                if raw_data is not None and not raw_data.empty:
                    # Process and enhance the data
                    processed_data = self._process_raw_data(raw_data, symbol, timeframe)
                    symbol_data[timeframe] = processed_data

                    logger.info(
                        f"Successfully fetched {len(processed_data)} {timeframe} bars for {symbol}"
                    )
                else:
                    logger.warning(f"No data returned for {symbol} {timeframe}")
                    symbol_data[timeframe] = []

            except Exception as e:
                logger.error(f"Error fetching {symbol} {timeframe}: {e}")
                symbol_data[timeframe] = []

        return symbol_data

    def _fetch_yfinance_data(
        self, symbol: str, interval: str, period: str
    ) -> pd.DataFrame:
        """Fetch data from yfinance (runs in thread executor)"""
        try:
            ticker = yf.Ticker(symbol)

            # Handle different period specifications
            if period == "max":
                # For daily/weekly/monthly, get maximum history
                if interval in ["1d", "1wk", "1mo"]:
                    data = ticker.history(period="max", interval=interval)
                else:
                    # For intraday, use 2 years as practical maximum
                    data = ticker.history(period="2y", interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)

            if data.empty:
                logger.warning(f"No data returned for {symbol} {interval} {period}")
                return None

            return data

        except Exception as e:
            logger.error(f"yfinance error for {symbol} {interval}: {e}")
            return None

    def _process_raw_data(
        self, raw_data: pd.DataFrame, symbol: str, timeframe: str
    ) -> list[MarketDataPoint]:
        """Process raw yfinance data into MarketDataPoint objects with technical indicators"""

        # Calculate technical indicators
        enhanced_data = self._calculate_technical_indicators(raw_data.copy())

        data_points = []

        for timestamp, row in enhanced_data.iterrows():
            try:
                # Calculate VWAP if volume data is available
                vwap = None
                if "Volume" in row and row["Volume"] > 0:
                    typical_price = (row["High"] + row["Low"] + row["Close"]) / 3
                    vwap = typical_price  # Simplified VWAP calculation

                # Determine session type based on timestamp
                session_type = self._determine_session_type(timestamp, timeframe)

                # Calculate data quality score
                quality_score = self._calculate_data_quality_score(row)

                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=(
                        timestamp.to_pydatetime()
                        if hasattr(timestamp, "to_pydatetime")
                        else timestamp
                    ),
                    timeframe=timeframe,
                    open=float(row["Open"]) if pd.notna(row["Open"]) else 0.0,
                    high=float(row["High"]) if pd.notna(row["High"]) else 0.0,
                    low=float(row["Low"]) if pd.notna(row["Low"]) else 0.0,
                    close=float(row["Close"]) if pd.notna(row["Close"]) else 0.0,
                    volume=int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
                    adj_close=(
                        float(row.get("Adj Close", row["Close"]))
                        if pd.notna(row.get("Adj Close", row["Close"]))
                        else None
                    ),
                    vwap=vwap,
                    session_type=session_type,
                    # Technical indicators
                    rsi=(
                        float(row.get("RSI_14", np.nan))
                        if pd.notna(row.get("RSI_14"))
                        else None
                    ),
                    macd_line=(
                        float(row.get("MACD_12_26_9", np.nan))
                        if pd.notna(row.get("MACD_12_26_9"))
                        else None
                    ),
                    macd_signal=(
                        float(row.get("MACDs_12_26_9", np.nan))
                        if pd.notna(row.get("MACDs_12_26_9"))
                        else None
                    ),
                    macd_histogram=(
                        float(row.get("MACDh_12_26_9", np.nan))
                        if pd.notna(row.get("MACDh_12_26_9"))
                        else None
                    ),
                    bb_upper=(
                        float(row.get("BBU_20_2.0", np.nan))
                        if pd.notna(row.get("BBU_20_2.0"))
                        else None
                    ),
                    bb_middle=(
                        float(row.get("BBM_20_2.0", np.nan))
                        if pd.notna(row.get("BBM_20_2.0"))
                        else None
                    ),
                    bb_lower=(
                        float(row.get("BBL_20_2.0", np.nan))
                        if pd.notna(row.get("BBL_20_2.0"))
                        else None
                    ),
                    sma_20=(
                        float(row.get("SMA_20", np.nan))
                        if pd.notna(row.get("SMA_20"))
                        else None
                    ),
                    sma_50=(
                        float(row.get("SMA_50", np.nan))
                        if pd.notna(row.get("SMA_50"))
                        else None
                    ),
                    sma_200=(
                        float(row.get("SMA_200", np.nan))
                        if pd.notna(row.get("SMA_200"))
                        else None
                    ),
                    data_quality_score=quality_score,
                    source="yfinance",
                )

                data_points.append(data_point)

            except Exception as e:
                logger.error(
                    f"Error processing data point for {symbol} at {timestamp}: {e}"
                )
                continue

        return data_points

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        try:
            # RSI
            df.ta.rsi(length=14, append=True)

            # MACD
            df.ta.macd(fast=12, slow=26, signal=9, append=True)

            # Bollinger Bands
            df.ta.bbands(length=20, std=2.0, append=True)

            # Moving Averages
            df.ta.sma(length=20, append=True)
            df.ta.sma(length=50, append=True)
            df.ta.sma(length=200, append=True)

            # Volume indicators
            if "Volume" in df.columns and df["Volume"].sum() > 0:
                df.ta.vwap(append=True)
                df.ta.ad(append=True)  # Accumulation/Distribution

            # Momentum indicators
            df.ta.stoch(append=True)  # Stochastic
            df.ta.cci(append=True)  # Commodity Channel Index

            # Volatility indicators
            df.ta.atr(append=True)  # Average True Range

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")

        return df

    def _determine_session_type(self, timestamp: pd.Timestamp, timeframe: str) -> str:
        """Determine market session type based on timestamp"""
        if timeframe in ["1d", "1wk", "1mo"]:
            return "regular"

        # For intraday data, check time
        time_hour = timestamp.hour

        if 4 <= time_hour < 9.5:  # 4:00 AM - 9:30 AM ET
            return "pre_market"
        elif 9.5 <= time_hour < 16:  # 9:30 AM - 4:00 PM ET
            return "regular"
        else:  # 4:00 PM - 8:00 PM ET
            return "after_hours"

    def _calculate_data_quality_score(self, row: pd.Series) -> float:
        """Calculate data quality score based on completeness and validity"""
        score = 1.0

        # Check for missing values
        required_fields = ["Open", "High", "Low", "Close", "Volume"]
        missing_fields = sum(1 for field in required_fields if pd.isna(row.get(field)))
        score -= (missing_fields / len(required_fields)) * 0.3

        # Check for invalid OHLC relationships
        try:
            if not pd.isna(row["High"]) and not pd.isna(row["Low"]):
                if row["High"] < row["Low"]:
                    score -= 0.2

            if (
                not pd.isna(row["Open"])
                and not pd.isna(row["High"])
                and not pd.isna(row["Low"])
            ):
                if not (row["Low"] <= row["Open"] <= row["High"]):
                    score -= 0.1

            if (
                not pd.isna(row["Close"])
                and not pd.isna(row["High"])
                and not pd.isna(row["Low"])
            ):
                if not (row["Low"] <= row["Close"] <= row["High"]):
                    score -= 0.1
        except:
            score -= 0.1

        # Check for zero volume (suspicious for active symbols)
        if pd.isna(row.get("Volume")) or row.get("Volume", 0) == 0:
            score -= 0.1

        return max(0.0, score)

    async def store_market_data(
        self, symbol_data: dict[str, dict[str, list[MarketDataPoint]]]
    ) -> dict[str, Any]:
        """Store comprehensive market data in databases"""
        if not self.db_service:
            logger.warning("No database service available for storage")
            return {"status": "skipped", "reason": "no_database_service"}

        storage_stats = {
            "symbols_processed": 0,
            "total_data_points": 0,
            "influxdb_points": 0,
            "postgresql_records": 0,
            "errors": [],
        }

        try:
            for symbol, timeframe_data in symbol_data.items():
                storage_stats["symbols_processed"] += 1

                for timeframe, data_points in timeframe_data.items():
                    if not data_points:
                        continue

                    # Store in InfluxDB for time-series analysis
                    await self._store_influxdb_data(
                        symbol, timeframe, data_points, storage_stats
                    )

                    # Store summary in PostgreSQL for analysis tracking
                    await self._store_postgresql_summary(
                        symbol, timeframe, data_points, storage_stats
                    )

                    storage_stats["total_data_points"] += len(data_points)

        except Exception as e:
            error_msg = f"Error storing market data: {e}"
            logger.error(error_msg)
            storage_stats["errors"].append(error_msg)

        return storage_stats

    async def _store_influxdb_data(
        self,
        symbol: str,
        timeframe: str,
        data_points: list[MarketDataPoint],
        stats: dict[str, Any],
    ) -> None:
        """Store data points in InfluxDB"""
        try:
            for point in data_points:
                # Market data measurement
                await self.db_service.influx.write_market_data(
                    symbol,
                    {
                        "price": point.close,
                        "volume": point.volume,
                        "open": point.open,
                        "high": point.high,
                        "low": point.low,
                        "adj_close": point.adj_close,
                        "vwap": point.vwap,
                        "timestamp": point.timestamp,
                        "exchange": "US",
                        "data_source": point.source,
                        "timeframe": timeframe,
                        "session_type": point.session_type,
                        "data_quality_score": point.data_quality_score,
                    },
                )

                # Technical indicators measurement (if available)
                if any([point.rsi, point.macd_line, point.bb_upper, point.sma_20]):
                    await self._store_technical_indicators(symbol, point, timeframe)

                stats["influxdb_points"] += 1

        except Exception as e:
            error_msg = f"Error storing InfluxDB data for {symbol} {timeframe}: {e}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)

    async def _store_technical_indicators(
        self, symbol: str, point: MarketDataPoint, timeframe: str
    ) -> None:
        """Store technical indicators in InfluxDB"""
        try:
            # This would be implemented with actual InfluxDB technical indicators measurement
            # For now, using the market data storage as a placeholder
            pass
        except Exception as e:
            logger.error(f"Error storing technical indicators for {symbol}: {e}")

    async def _store_postgresql_summary(
        self,
        symbol: str,
        timeframe: str,
        data_points: list[MarketDataPoint],
        stats: dict[str, Any],
    ) -> None:
        """Store market data summary in PostgreSQL for analysis tracking"""
        try:
            if not data_points:
                return

            # Calculate summary statistics
            prices = [p.close for p in data_points if p.close > 0]
            volumes = [p.volume for p in data_points if p.volume > 0]

            if not prices:
                return

            summary = {
                "symbol": symbol,
                "timeframe": timeframe,
                "data_points_count": len(data_points),
                "start_date": min(p.timestamp for p in data_points),
                "end_date": max(p.timestamp for p in data_points),
                "avg_price": np.mean(prices),
                "avg_volume": np.mean(volumes) if volumes else 0,
                "avg_quality_score": np.mean(
                    [p.data_quality_score for p in data_points]
                ),
                "source": "yfinance",
            }

            # This would be stored in a market_data_summaries table
            # For now, just log the summary
            logger.info(
                f"Market data summary for {symbol} {timeframe}: {len(data_points)} points, "
                f"avg_price={summary['avg_price']:.2f}, avg_quality={summary['avg_quality_score']:.2f}"
            )

            stats["postgresql_records"] += 1

        except Exception as e:
            error_msg = (
                f"Error storing PostgreSQL summary for {symbol} {timeframe}: {e}"
            )
            logger.error(error_msg)
            stats["errors"].append(error_msg)

    async def get_latest_data(
        self, symbol: str, timeframe: str = "1d", limit: int = 100
    ) -> list[MarketDataPoint]:
        """Get latest market data for a symbol"""
        try:
            # Fetch fresh data
            symbol_data = await self._fetch_symbol_all_timeframes(symbol)

            if timeframe in symbol_data and symbol_data[timeframe]:
                # Return most recent data points
                data_points = symbol_data[timeframe]
                return sorted(data_points, key=lambda x: x.timestamp, reverse=True)[
                    :limit
                ]

            return []

        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {e}")
            return []

    async def analyze_data_quality(
        self, symbol_data: dict[str, dict[str, list[MarketDataPoint]]]
    ) -> dict[str, Any]:
        """Analyze the quality of collected market data"""
        analysis = {
            "total_symbols": len(symbol_data),
            "total_data_points": 0,
            "timeframe_coverage": {},
            "quality_metrics": {},
            "data_gaps": [],
            "overall_quality_score": 0.0,
        }

        all_quality_scores = []

        for symbol, timeframe_data in symbol_data.items():
            for timeframe, data_points in timeframe_data.items():
                if not data_points:
                    continue

                analysis["total_data_points"] += len(data_points)

                # Timeframe coverage
                if timeframe not in analysis["timeframe_coverage"]:
                    analysis["timeframe_coverage"][timeframe] = {
                        "symbols": 0,
                        "total_points": 0,
                    }

                analysis["timeframe_coverage"][timeframe]["symbols"] += 1
                analysis["timeframe_coverage"][timeframe]["total_points"] += len(
                    data_points
                )

                # Quality metrics per symbol/timeframe
                quality_scores = [p.data_quality_score for p in data_points]
                avg_quality = np.mean(quality_scores) if quality_scores else 0.0

                key = f"{symbol}_{timeframe}"
                analysis["quality_metrics"][key] = {
                    "avg_quality": avg_quality,
                    "data_points": len(data_points),
                    "min_quality": min(quality_scores) if quality_scores else 0.0,
                    "max_quality": max(quality_scores) if quality_scores else 0.0,
                }

                all_quality_scores.extend(quality_scores)

                # Detect data gaps
                if len(data_points) > 1:
                    timestamps = sorted([p.timestamp for p in data_points])
                    gaps = self._detect_data_gaps(timestamps, timeframe)
                    for gap in gaps:
                        analysis["data_gaps"].append(
                            {
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "gap_start": gap["start"],
                                "gap_end": gap["end"],
                                "gap_duration": gap["duration"],
                            }
                        )

        # Overall quality score
        analysis["overall_quality_score"] = (
            np.mean(all_quality_scores) if all_quality_scores else 0.0
        )

        return analysis

    def _detect_data_gaps(
        self, timestamps: list[datetime], timeframe: str
    ) -> list[dict[str, Any]]:
        """Detect gaps in time series data"""
        gaps = []

        # Expected intervals for different timeframes
        interval_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
            "1wk": timedelta(weeks=1),
            "1mo": timedelta(days=30),
        }

        expected_interval = interval_map.get(timeframe, timedelta(days=1))
        max_gap = expected_interval * 3  # Allow up to 3x the expected interval

        for i in range(1, len(timestamps)):
            gap_duration = timestamps[i] - timestamps[i - 1]

            if gap_duration > max_gap:
                gaps.append(
                    {
                        "start": timestamps[i - 1],
                        "end": timestamps[i],
                        "duration": gap_duration,
                    }
                )

        return gaps
