"""
LDES Configuration Management

Extends the TradeKnowledge configuration system with LDES-specific settings.
Follows the same patterns as the existing config.py for consistency.
"""

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MarketDataConfig(BaseModel):
    """Market data source configuration."""

    # Alpaca configuration
    alpaca_api_key: str | None = Field(
        default_factory=lambda: os.getenv("ALPACA_API_KEY")
    )
    alpaca_secret_key: str | None = Field(
        default_factory=lambda: os.getenv("ALPACA_SECRET_KEY")
    )
    alpaca_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2"
        )
    )
    alpaca_data_url: str = Field(
        default_factory=lambda: os.getenv(
            "ALPACA_DATA_URL", "https://data.alpaca.markets/v2"
        )
    )
    alpaca_stream_url: str = Field(
        default_factory=lambda: os.getenv(
            "ALPACA_STREAM_URL", "wss://stream.data.alpaca.markets/v2"
        )
    )
    alpaca_symbols: list[str] = Field(default=["SPY", "QQQ", "IWM", "TLT", "GLD"])

    # Binance configuration
    binance_api_key: str | None = Field(
        default_factory=lambda: os.getenv("BINANCE_API_KEY")
    )
    binance_secret_key: str | None = Field(
        default_factory=lambda: os.getenv("BINANCE_SECRET_KEY")
    )
    binance_base_url: str = Field(default="https://api.binance.com")
    binance_stream_url: str = Field(default="wss://stream.binance.com:9443")
    binance_symbols: list[str] = Field(default=["BTCUSDT", "ETHUSDT", "BNBUSDT"])

    # Yahoo Finance configuration
    yfinance_symbols: list[str] = Field(default=["^VIX", "^TNX", "DXY"])
    yfinance_poll_interval_seconds: int = Field(default=30)

    # Schwab configuration
    schwab_app_key: str | None = Field(
        default_factory=lambda: os.getenv("SCHWAB_APP_KEY")
    )
    schwab_secret: str | None = Field(
        default_factory=lambda: os.getenv("SCHWAB_SECRET")
    )
    schwab_account_id: str | None = Field(
        default_factory=lambda: os.getenv("SCHWAB_ACCOUNT_ID")
    )
    schwab_redirect_uri: str = Field(default="http://localhost:8000/callback")

    @field_validator("alpaca_symbols", "binance_symbols", "yfinance_symbols")
    @classmethod
    def validate_symbols(cls, v):
        """Ensure symbol lists are not empty and contain valid symbols."""
        if not v:
            return v
        # Basic validation - symbols should be uppercase strings
        return [symbol.upper().strip() for symbol in v if symbol.strip()]


class InfluxDBConfig(BaseModel):
    """InfluxDB time-series database configuration."""

    url: str = Field(
        default_factory=lambda: os.getenv("INFLUXDB_URL", "http://localhost:8086")
    )
    token: str | None = Field(default_factory=lambda: os.getenv("INFLUXDB_TOKEN"))
    org: str = Field(default_factory=lambda: os.getenv("INFLUXDB_ORG", "ldes"))
    bucket: str = Field(
        default_factory=lambda: os.getenv("INFLUXDB_BUCKET", "market_data")
    )
    timeout_ms: int = Field(default=30000)
    batch_size: int = Field(default=1000)
    flush_interval_ms: int = Field(default=1000)
    retention_days: int = Field(default=90)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        """Ensure URL is properly formatted."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("InfluxDB URL must start with http:// or https://")
        return v


class DetectionConfig(BaseModel):
    """Liquidity detection algorithm configuration."""

    # Performance requirements
    latency_target_ms: int = Field(
        default=100, description="Target detection latency in milliseconds"
    )
    throughput_target_per_second: int = Field(
        default=10000, description="Target throughput per second"
    )

    # Signal thresholds
    min_signal_strength: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum signal strength threshold"
    )
    min_confidence: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )

    # Time windows
    lookback_period_minutes: int = Field(
        default=5, description="Historical data lookback period"
    )
    volume_window_minutes: int = Field(
        default=1, description="Volume spike detection window"
    )
    price_velocity_window_seconds: int = Field(
        default=30, description="Price velocity calculation window"
    )

    # Detection parameters
    volume_spike_threshold: float = Field(
        default=3.0, description="Volume spike threshold multiplier"
    )
    price_velocity_threshold_bps: float = Field(
        default=50.0, description="Price velocity threshold in bps"
    )
    spread_expansion_threshold: float = Field(
        default=2.0, description="Spread expansion threshold multiplier"
    )
    order_book_imbalance_threshold: float = Field(
        default=0.7, description="Order book imbalance threshold"
    )

    # Machine learning
    ml_model_enabled: bool = Field(
        default=True, description="Enable ML model predictions"
    )
    ml_model_path: str = Field(
        default="models/liquidation_detector.pkl", description="ML model file path"
    )
    ml_feature_window_minutes: int = Field(
        default=10, description="ML feature calculation window"
    )
    ml_prediction_threshold: float = Field(
        default=0.5, description="ML prediction threshold"
    )

    # Signal types to detect
    detect_forced_liquidations: bool = Field(default=True)
    detect_margin_calls: bool = Field(default=True)
    detect_portfolio_rebalancing: bool = Field(default=True)
    detect_volume_spikes: bool = Field(default=True)
    detect_spread_expansion: bool = Field(default=True)
    detect_order_book_imbalance: bool = Field(default=True)


class RiskConfig(BaseModel):
    """Risk management configuration."""

    # Position limits
    max_position_size_pct: float = Field(
        default=5.0, description="Maximum position size as % of portfolio"
    )
    max_sector_exposure_pct: float = Field(
        default=20.0, description="Maximum sector exposure as % of portfolio"
    )
    max_daily_loss_pct: float = Field(
        default=2.0, description="Maximum daily loss as % of portfolio"
    )
    max_drawdown_pct: float = Field(
        default=10.0, description="Maximum drawdown threshold"
    )

    # Stop losses and targets
    stop_loss_pct: float = Field(
        default=5.0, description="Default stop loss as % of position"
    )
    profit_target_pct: float = Field(
        default=10.0, description="Default profit target as % of position"
    )
    trailing_stop_enabled: bool = Field(
        default=True, description="Enable trailing stops"
    )
    trailing_stop_pct: float = Field(
        default=2.0, description="Trailing stop distance as % of position"
    )

    # Kelly criterion
    kelly_criterion_enabled: bool = Field(
        default=True, description="Use Kelly criterion for position sizing"
    )
    kelly_safety_factor: float = Field(
        default=0.25, description="Kelly safety factor (quarter-Kelly)"
    )
    kelly_max_allocation_pct: float = Field(
        default=20.0, description="Maximum Kelly allocation per position"
    )

    # Portfolio limits
    max_positions: int = Field(
        default=20, description="Maximum number of open positions"
    )
    max_correlation_exposure: float = Field(
        default=0.8, description="Maximum correlation exposure threshold"
    )

    # Risk metrics
    var_confidence_level: float = Field(
        default=0.95, description="VaR confidence level"
    )
    var_time_horizon_days: int = Field(
        default=1, description="VaR time horizon in days"
    )
    stress_test_scenarios: list[str] = Field(
        default=["market_crash", "liquidity_crisis", "volatility_spike"]
    )

    @field_validator(
        "max_position_size_pct", "max_sector_exposure_pct", "max_daily_loss_pct"
    )
    @classmethod
    def validate_percentages(cls, v):
        """Ensure percentage values are reasonable."""
        if not 0.1 <= v <= 50.0:  # Between 0.1% and 50%
            raise ValueError("Percentage values must be between 0.1 and 50")
        return v


class ExecutionConfig(BaseModel):
    """Trade execution configuration."""

    # Order routing
    default_order_type: str = Field(default="limit", description="Default order type")
    aggressive_execution_enabled: bool = Field(
        default=True, description="Enable aggressive execution during signals"
    )
    passive_execution_enabled: bool = Field(
        default=True, description="Enable passive execution during normal conditions"
    )

    # Execution algorithms
    twap_enabled: bool = Field(default=True, description="Enable TWAP execution")
    vwap_enabled: bool = Field(default=True, description="Enable VWAP execution")
    iceberg_orders_enabled: bool = Field(
        default=False, description="Enable iceberg orders"
    )

    # Slippage and timing
    max_slippage_bps: float = Field(
        default=10.0, description="Maximum acceptable slippage in bps"
    )
    order_timeout_seconds: int = Field(
        default=30, description="Order timeout in seconds"
    )
    partial_fill_threshold_pct: float = Field(
        default=10.0, description="Minimum fill % before considering partial"
    )

    # Market making
    market_making_enabled: bool = Field(
        default=False, description="Enable market making strategies"
    )
    quote_spread_bps: float = Field(
        default=5.0, description="Market making quote spread in bps"
    )
    quote_size_shares: int = Field(default=100, description="Market making quote size")

    # Paper trading
    paper_trading: bool = Field(default=True, description="Enable paper trading mode")
    simulate_slippage: bool = Field(
        default=True, description="Simulate realistic slippage"
    )
    simulate_latency: bool = Field(
        default=True, description="Simulate execution latency"
    )


class BacktestConfig(BaseModel):
    """Backtesting configuration."""

    # Capital and costs
    initial_capital: float = Field(
        default=100000.0, description="Initial capital for backtesting"
    )
    commission_per_trade: float = Field(
        default=1.0, description="Commission cost per trade"
    )
    slippage_bps: float = Field(
        default=2.0, description="Assumed slippage in basis points"
    )
    interest_rate: float = Field(
        default=0.02, description="Risk-free interest rate for Sharpe calculation"
    )

    # Data and timing
    start_date: str | None = Field(
        None, description="Backtest start date (YYYY-MM-DD)"
    )
    end_date: str | None = Field(None, description="Backtest end date (YYYY-MM-DD)")
    warmup_period_days: int = Field(
        default=30, description="Warmup period for indicators"
    )

    # Performance metrics
    benchmark_symbol: str = Field(
        default="SPY", description="Benchmark for performance comparison"
    )
    calculate_risk_metrics: bool = Field(
        default=True, description="Calculate VaR, CVaR, etc."
    )
    save_trade_details: bool = Field(
        default=True, description="Save individual trade details"
    )
    save_daily_metrics: bool = Field(
        default=True, description="Save daily performance metrics"
    )

    # Output
    output_directory: str = Field(
        default="backtests", description="Output directory for results"
    )
    generate_plots: bool = Field(default=True, description="Generate performance plots")
    export_to_csv: bool = Field(default=True, description="Export results to CSV")


class LDESConfig(BaseModel):
    """Main LDES system configuration."""

    # System settings
    enable_ldes: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_LDES", "false").lower() == "true"
    )
    log_level: str = Field(default_factory=lambda: os.getenv("LDES_LOG_LEVEL", "INFO"))
    environment: str = Field(
        default_factory=lambda: os.getenv("LDES_ENV", "development")
    )

    # Component configurations
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    influxdb: InfluxDBConfig = Field(default_factory=InfluxDBConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    # Advanced settings
    performance_monitoring_enabled: bool = Field(
        default=True, description="Enable performance monitoring"
    )
    metrics_collection_interval_seconds: int = Field(
        default=1, description="Metrics collection interval"
    )
    health_check_interval_seconds: int = Field(
        default=30, description="Health check interval"
    )

    # Storage settings
    data_retention_days: int = Field(
        default=90, description="Data retention period in days"
    )
    backup_enabled: bool = Field(default=True, description="Enable automatic backups")
    backup_interval_hours: int = Field(
        default=24, description="Backup interval in hours"
    )

    # Integration settings
    tradeknowledge_integration_enabled: bool = Field(
        default=True, description="Enable TradeKnowledge integration"
    )
    agent_integration_enabled: bool = Field(
        default=True, description="Enable agent system integration"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Ensure log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Ensure environment is valid."""
        valid_envs = ["development", "testing", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()

    def get_symbols(self) -> list[str]:
        """Get all configured symbols across all data sources."""
        symbols = set()
        symbols.update(self.market_data.alpaca_symbols)
        symbols.update(self.market_data.binance_symbols)
        symbols.update(self.market_data.yfinance_symbols)
        return sorted(list(symbols))

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def is_paper_trading(self) -> bool:
        """Check if paper trading is enabled."""
        return self.execution.paper_trading or not self.is_production()

    def get_config_summary(self) -> dict[str, Any]:
        """Get a summary of key configuration settings."""
        return {
            "environment": self.environment,
            "paper_trading": self.is_paper_trading(),
            "symbols_count": len(self.get_symbols()),
            "detection_enabled": self.enable_ldes,
            "ml_enabled": self.detection.ml_model_enabled,
            "risk_limits": {
                "max_position_pct": self.risk.max_position_size_pct,
                "max_daily_loss_pct": self.risk.max_daily_loss_pct,
                "kelly_enabled": self.risk.kelly_criterion_enabled,
            },
            "performance_targets": {
                "latency_ms": self.detection.latency_target_ms,
                "throughput_per_sec": self.detection.throughput_target_per_second,
            },
        }
