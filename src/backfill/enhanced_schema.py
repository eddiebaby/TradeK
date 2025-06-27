"""
Enhanced InfluxDB Schema for ML-Ready Multi-Asset Data

Optimized schema design for storing multi-asset market data with ML features
supporting arbitrage strategies, HFT algorithms, and cross-asset analysis.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from influxdb_client import Point
import logging

logger = logging.getLogger(__name__)


@dataclass
class AssetDataPoint:
    """Standardized data point for any asset class"""
    symbol: str
    timestamp: datetime
    asset_class: str  # crypto, equity, futures, options
    source: str
    granularity: str  # tick, 1min, 5min, 1hr, daily
    
    # Core OHLCV
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Optional market microstructure
    vwap: Optional[float] = None
    trades: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    
    # ML-derived features
    volatility: Optional[float] = None
    return_1min: Optional[float] = None
    rsi: Optional[float] = None
    volume_profile: Optional[str] = None
    
    # Asset-specific fields
    extra_fields: Optional[Dict[str, Any]] = None


class MLOptimizedSchema:
    """Enhanced InfluxDB schema optimized for ML workloads and multi-asset strategies"""
    
    def __init__(self):
        self.measurement_configs = self._define_measurement_schema()
        
    def _define_measurement_schema(self) -> Dict[str, Dict]:
        """Define measurement schemas for different asset classes and granularities"""
        return {
            # Crypto measurements
            "crypto_prices_1min": {
                "tags": ["symbol", "source", "exchange", "pair_type"],
                "fields": ["open", "high", "low", "close", "volume", "trades", "bid_ask_spread", "volatility", "return_1min"],
                "retention_policy": "5_years",
                "description": "1-minute crypto OHLCV with microstructure"
            },
            "crypto_prices_tick": {
                "tags": ["symbol", "source", "exchange", "trade_type"],
                "fields": ["price", "size", "side", "timestamp_ns"],
                "retention_policy": "1_year",
                "description": "Tick-level crypto trades for HFT strategies"
            },
            
            # Equity measurements
            "equity_prices_1min": {
                "tags": ["symbol", "source", "market", "sector", "market_cap"],
                "fields": ["open", "high", "low", "close", "volume", "vwap", "transactions", "volatility", "return_1min"],
                "retention_policy": "10_years",
                "description": "1-minute equity OHLCV with enhanced metadata"
            },
            "equity_prices_daily": {
                "tags": ["symbol", "source", "sector", "index_member"],
                "fields": ["open", "high", "low", "close", "volume", "adj_close", "dividend", "split_factor"],
                "retention_policy": "permanent",
                "description": "Daily equity data with corporate actions"
            },
            
            # Futures measurements
            "futures_prices_1min": {
                "tags": ["symbol", "source", "contract_month", "underlying"],
                "fields": ["open", "high", "low", "close", "volume", "open_interest", "settlement_price"],
                "retention_policy": "7_years",
                "description": "1-minute futures OHLCV with open interest"
            },
            
            # Options measurements
            "options_prices_1min": {
                "tags": ["symbol", "underlying", "expiry", "strike", "option_type"],
                "fields": ["bid", "ask", "last", "volume", "open_interest", "implied_vol", "delta", "gamma", "theta", "vega"],
                "retention_policy": "3_years",
                "description": "Options pricing with Greeks for volatility strategies"
            },
            
            # Cross-asset correlation measurements
            "asset_correlations": {
                "tags": ["asset1", "asset2", "timeframe", "correlation_type"],
                "fields": ["correlation", "rolling_correlation", "correlation_stability"],
                "retention_policy": "5_years",
                "description": "Cross-asset correlations for pairs trading"
            },
            
            # ML feature measurements
            "ml_features": {
                "tags": ["symbol", "feature_type", "model_version"],
                "fields": ["feature_value", "feature_confidence", "feature_importance"],
                "retention_policy": "3_years",
                "description": "Pre-computed ML features for strategy execution"
            },
            
            # Strategy performance measurements
            "strategy_performance": {
                "tags": ["strategy_name", "symbol", "strategy_type", "timeframe"],
                "fields": ["pnl", "sharpe", "max_drawdown", "win_rate", "position_size"],
                "retention_policy": "permanent",
                "description": "Strategy performance metrics for optimization"
            }
        }
    
    def create_data_point(self, data: AssetDataPoint) -> Point:
        """Create optimized InfluxDB point from standardized data"""
        measurement = self._get_measurement_name(data.asset_class, data.granularity)
        
        point = Point(measurement)
        
        # Add standard tags
        point.tag("symbol", data.symbol)
        point.tag("asset_class", data.asset_class)
        point.tag("source", data.source)
        point.tag("granularity", data.granularity)
        
        # Add asset-specific tags
        if data.asset_class == "crypto":
            point.tag("exchange", self._extract_exchange(data.source))
            point.tag("pair_type", self._get_crypto_pair_type(data.symbol))
        elif data.asset_class == "equity":
            point.tag("market", "us_equity")
            point.tag("sector", self._get_sector(data.symbol))
            point.tag("market_cap", self._get_market_cap_category(data.symbol))
        elif data.asset_class == "futures":
            point.tag("contract_month", self._extract_contract_month(data.symbol))
            point.tag("underlying", self._get_futures_underlying(data.symbol))
        
        # Add core OHLCV fields
        point.field("open", float(data.open))
        point.field("high", float(data.high))
        point.field("low", float(data.low))
        point.field("close", float(data.close))
        point.field("volume", int(data.volume))
        
        # Add optional market microstructure
        if data.vwap is not None:
            point.field("vwap", float(data.vwap))
        if data.trades is not None:
            point.field("trades", int(data.trades))
        if data.bid is not None and data.ask is not None:
            point.field("bid", float(data.bid))
            point.field("ask", float(data.ask))
            point.field("bid_ask_spread", float(data.ask - data.bid))
        
        # Add ML-derived features
        if data.volatility is not None:
            point.field("volatility", float(data.volatility))
        if data.return_1min is not None:
            point.field("return_1min", float(data.return_1min))
        if data.rsi is not None:
            point.field("rsi", float(data.rsi))
        
        # Add extra fields
        if data.extra_fields:
            for key, value in data.extra_fields.items():
                if isinstance(value, (int, float)):
                    point.field(key, value)
                elif isinstance(value, str):
                    point.tag(key, value)
        
        # Set timestamp
        point.time(data.timestamp)
        
        return point
    
    def _get_measurement_name(self, asset_class: str, granularity: str) -> str:
        """Get appropriate measurement name based on asset class and granularity"""
        base_measurement = f"{asset_class}_prices_{granularity}"
        
        # Ensure measurement exists in schema
        if base_measurement not in self.measurement_configs:
            # Fallback to generic measurement
            return f"{asset_class}_prices_1min"
        
        return base_measurement
    
    def _extract_exchange(self, source: str) -> str:
        """Extract exchange from source string"""
        exchange_mapping = {
            "kraken": "kraken",
            "coinbase": "coinbase_pro",
            "polygon": "consolidated",
            "alpaca": "alpaca"
        }
        return exchange_mapping.get(source.lower(), source)
    
    def _get_crypto_pair_type(self, symbol: str) -> str:
        """Determine crypto pair type"""
        if "/USD" in symbol:
            return "fiat_pair"
        elif "/BTC" in symbol:
            return "btc_pair"
        elif "/ETH" in symbol:
            return "eth_pair"
        else:
            return "other_pair"
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for equity symbol"""
        sector_mapping = {
            "XLF": "financials",
            "XLK": "technology", 
            "XLE": "energy",
            "XLI": "industrials",
            "XLV": "healthcare",
            "XLP": "consumer_staples",
            "SPY": "broad_market",
            "QQQ": "tech_heavy",
            "IWM": "small_cap",
            "DIA": "blue_chip"
        }
        return sector_mapping.get(symbol, "unknown")
    
    def _get_market_cap_category(self, symbol: str) -> str:
        """Get market cap category"""
        if symbol in ["SPY", "QQQ", "DIA"]:
            return "large_cap"
        elif symbol == "IWM":
            return "small_cap"
        else:
            return "mixed"
    
    def _extract_contract_month(self, symbol: str) -> str:
        """Extract contract month from futures symbol"""
        # Simplified extraction - would need more sophisticated parsing
        if symbol.startswith("/"):
            return "continuous"
        return "unknown"
    
    def _get_futures_underlying(self, symbol: str) -> str:
        """Get underlying asset for futures"""
        underlying_mapping = {
            "/ES": "SP500",
            "/NQ": "NASDAQ",
            "/YM": "DOW",
            "/RTY": "RUSSELL2000",
            "/GC": "GOLD",
            "/CL": "CRUDE_OIL"
        }
        return underlying_mapping.get(symbol, "unknown")
    
    def create_correlation_point(
        self, 
        asset1: str, 
        asset2: str, 
        correlation: float,
        timeframe: str,
        timestamp: datetime
    ) -> Point:
        """Create correlation measurement point"""
        point = (
            Point("asset_correlations")
            .tag("asset1", asset1)
            .tag("asset2", asset2)
            .tag("timeframe", timeframe)
            .tag("correlation_type", "pearson")
            .field("correlation", float(correlation))
            .field("abs_correlation", abs(float(correlation)))
            .time(timestamp)
        )
        return point
    
    def create_ml_feature_point(
        self,
        symbol: str,
        feature_name: str,
        feature_value: float,
        model_version: str,
        timestamp: datetime,
        confidence: Optional[float] = None
    ) -> Point:
        """Create ML feature measurement point"""
        point = (
            Point("ml_features")
            .tag("symbol", symbol)
            .tag("feature_type", feature_name)
            .tag("model_version", model_version)
            .field("feature_value", float(feature_value))
            .time(timestamp)
        )
        
        if confidence is not None:
            point.field("feature_confidence", float(confidence))
            
        return point
    
    def create_strategy_performance_point(
        self,
        strategy_name: str,
        symbol: str,
        pnl: float,
        timestamp: datetime,
        **metrics
    ) -> Point:
        """Create strategy performance measurement point"""
        point = (
            Point("strategy_performance")
            .tag("strategy_name", strategy_name)
            .tag("symbol", symbol)
            .tag("strategy_type", self._infer_strategy_type(strategy_name))
            .field("pnl", float(pnl))
            .time(timestamp)
        )
        
        # Add additional performance metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                point.field(metric_name, float(metric_value))
        
        return point
    
    def _infer_strategy_type(self, strategy_name: str) -> str:
        """Infer strategy type from strategy name"""
        name_lower = strategy_name.lower()
        if "arbitrage" in name_lower:
            return "arbitrage"
        elif "hft" in name_lower or "market_making" in name_lower:
            return "hft"
        elif "pairs" in name_lower:
            return "pairs_trading"
        elif "mean_reversion" in name_lower:
            return "mean_reversion"
        else:
            return "other"
    
    def get_query_templates(self) -> Dict[str, str]:
        """Get optimized query templates for common ML use cases"""
        return {
            "recent_prices": '''
                from(bucket: "{bucket}")
                |> range(start: -{timeframe})
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r._field == "close")
                |> aggregateWindow(every: {window}, fn: last)
            ''',
            
            "cross_correlations": '''
                asset1 = from(bucket: "{bucket}")
                |> range(start: -{timeframe})
                |> filter(fn: (r) => r._measurement == "asset_correlations")
                |> filter(fn: (r) => r.asset1 == "{asset1}")
                |> filter(fn: (r) => r.asset2 == "{asset2}")
                |> filter(fn: (r) => r._field == "correlation")
                |> last()
            ''',
            
            "volatility_surface": '''
                from(bucket: "{bucket}")
                |> range(start: -{timeframe})
                |> filter(fn: (r) => r._measurement == "options_prices_1min")
                |> filter(fn: (r) => r.underlying == "{underlying}")
                |> filter(fn: (r) => r._field == "implied_vol")
                |> group(columns: ["strike", "expiry"])
                |> last()
            ''',
            
            "ml_features": '''
                from(bucket: "{bucket}")
                |> range(start: -{timeframe})
                |> filter(fn: (r) => r._measurement == "ml_features")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.feature_type == "{feature_type}")
                |> last()
            ''',
            
            "strategy_performance": '''
                from(bucket: "{bucket}")
                |> range(start: -{timeframe})
                |> filter(fn: (r) => r._measurement == "strategy_performance")
                |> filter(fn: (r) => r.strategy_name == "{strategy_name}")
                |> filter(fn: (r) => r._field == "pnl")
                |> sum()
            '''
        }
    
    def get_schema_documentation(self) -> Dict[str, Any]:
        """Get comprehensive schema documentation for development teams"""
        return {
            "version": "1.0.0",
            "description": "ML-optimized multi-asset schema for algorithmic trading",
            "measurements": self.measurement_configs,
            "supported_strategies": [
                "statistical_arbitrage",
                "pairs_trading", 
                "mean_reversion",
                "volatility_arbitrage",
                "hft_market_making",
                "cross_exchange_arbitrage"
            ],
            "performance_targets": {
                "query_latency": "<50ms for feature extraction",
                "throughput": "10K+ points/second ingestion",
                "retention": "Up to 10 years for core data"
            },
            "query_templates": self.get_query_templates(),
            "best_practices": {
                "indexing": "Use symbol and timestamp tags for optimal performance",
                "batching": "Write in batches of 5K-10K points",
                "retention": "Configure appropriate retention policies by data type",
                "monitoring": "Track schema evolution and query performance"
            }
        }


# Export the schema for use by other modules
ml_schema = MLOptimizedSchema()

# Example usage
if __name__ == "__main__":
    # Example data point creation
    sample_data = AssetDataPoint(
        symbol="BTC/USD",
        timestamp=datetime.utcnow(),
        asset_class="crypto",
        source="kraken",
        granularity="1min",
        open=50000.0,
        high=50100.0,
        low=49900.0,
        close=50050.0,
        volume=100,
        vwap=50025.0,
        trades=250,
        volatility=0.02,
        return_1min=0.001
    )
    
    # Create InfluxDB point
    point = ml_schema.create_data_point(sample_data)
    print(f"Created point: {point}")
    
    # Get schema documentation
    schema_docs = ml_schema.get_schema_documentation()
    print(f"Schema supports {len(schema_docs['measurements'])} measurement types")