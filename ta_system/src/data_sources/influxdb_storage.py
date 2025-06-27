#!/usr/bin/env python3
"""
InfluxDB Storage for Granular Analysis Data

Stores comprehensive analysis data in InfluxDB with granular measurements:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Financial ratios (P/E, ROE, debt ratios, etc.) 
- Market data (price, volume, volatility)
- Risk metrics (beta, VaR, drawdown)
- Investment thesis data points
- ETF-specific metrics (expense ratio, tracking error, holdings)
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    logging.warning("InfluxDB client not available. Install with: pip install influxdb-client")

from ..fundamental.models import ComprehensiveAnalysis


class InfluxDBAnalysisStorage:
    """Stores comprehensive analysis data in InfluxDB with granular measurements."""
    
    def __init__(
        self, 
        url: str = "http://localhost:8086",
        token: str = "your-token",
        org: str = "ta-system",
        bucket: str = "stock-analysis"
    ):
        """Initialize InfluxDB storage."""
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.client = None
        self.write_api = None
        self.logger = logging.getLogger(__name__)
        
        if not INFLUXDB_AVAILABLE:
            self.logger.error("InfluxDB client not available")
            return
            
        try:
            self.client = InfluxDBClient(url=url, token=token, org=org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.logger.info(f"InfluxDB client initialized: {url}")
        except Exception as e:
            self.logger.error(f"Failed to initialize InfluxDB client: {e}")

    async def store_comprehensive_analysis(
        self, 
        analysis: ComprehensiveAnalysis,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Store comprehensive analysis data with granular measurements."""
        
        if not self.client:
            self.logger.error("InfluxDB client not available")
            return False
            
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        try:
            points = []
            symbol = analysis.company_profile.symbol.upper()
            
            # Technical Analysis Points
            points.extend(self._create_technical_points(analysis, symbol, timestamp))
            
            # Market Data Points  
            points.extend(self._create_market_data_points(analysis, symbol, timestamp))
            
            # Financial Ratios Points
            points.extend(self._create_financial_ratio_points(analysis, symbol, timestamp))
            
            # Risk Assessment Points
            points.extend(self._create_risk_assessment_points(analysis, symbol, timestamp))
            
            # Investment Thesis Points
            points.extend(self._create_investment_thesis_points(analysis, symbol, timestamp))
            
            # ETF-specific Points (if applicable)
            if self._is_etf(symbol):
                points.extend(self._create_etf_specific_points(analysis, symbol, timestamp))
            
            # Company Profile Points
            points.extend(self._create_company_profile_points(analysis, symbol, timestamp))
            
            # Financial Statements Points (if available)
            if analysis.income_statements:
                points.extend(self._create_financial_statements_points(analysis, symbol, timestamp))
            
            # Write all points to InfluxDB
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
            
            self.logger.info(f"Stored {len(points)} data points for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store analysis data: {e}")
            return False

    def _create_technical_points(
        self, 
        analysis: ComprehensiveAnalysis, 
        symbol: str, 
        timestamp: datetime
    ) -> List[Point]:
        """Create technical analysis data points."""
        points = []
        ta = analysis.technical_analysis
        
        if ta.rsi_14:
            points.append(
                Point("technical_indicators")
                .tag("symbol", symbol)
                .tag("indicator", "RSI")
                .tag("period", "14")
                .field("value", float(ta.rsi_14))
                .field("signal", self._get_rsi_signal_numeric(ta.rsi_14))
                .time(timestamp, WritePrecision.NS)
            )
            
        if ta.macd:
            points.append(
                Point("technical_indicators")
                .tag("symbol", symbol)
                .tag("indicator", "MACD")
                .field("macd", float(ta.macd))
                .field("signal", float(ta.macd_signal) if ta.macd_signal else 0.0)
                .field("histogram", float(ta.macd_histogram) if ta.macd_histogram else 0.0)
                .time(timestamp, WritePrecision.NS)
            )
            
        # Moving Averages
        for period, value in [("20", ta.sma_20), ("50", ta.sma_50), ("200", ta.sma_200)]:
            if value:
                points.append(
                    Point("moving_averages")
                    .tag("symbol", symbol)
                    .tag("type", "SMA")
                    .tag("period", period)
                    .field("value", float(value))
                    .field("price_distance_pct", self._calc_price_distance_pct(ta.current_price, value))
                    .time(timestamp, WritePrecision.NS)
                )
                
        # Bollinger Bands
        if ta.bollinger_upper and ta.bollinger_lower:
            points.append(
                Point("bollinger_bands")
                .tag("symbol", symbol)
                .field("upper", float(ta.bollinger_upper))
                .field("lower", float(ta.bollinger_lower))
                .field("width", float(ta.bollinger_upper - ta.bollinger_lower))
                .field("position", self._calc_bb_position(ta.current_price, ta.bollinger_upper, ta.bollinger_lower))
                .time(timestamp, WritePrecision.NS)
            )
            
        # Volatility & ATR
        if ta.volatility:
            points.append(
                Point("volatility_metrics")
                .tag("symbol", symbol)
                .field("volatility_annual", float(ta.volatility))
                .field("atr_14", float(ta.atr_14) if ta.atr_14 else 0.0)
                .field("volatility_percentile", self._calc_volatility_percentile(float(ta.volatility)))
                .time(timestamp, WritePrecision.NS)
            )
            
        # Support/Resistance Levels
        for i, level in enumerate(ta.support_levels[:3], 1):
            if level:
                points.append(
                    Point("support_resistance")
                    .tag("symbol", symbol)
                    .tag("type", "support")
                    .tag("level", f"S{i}")
                    .field("price", float(level))
                    .field("distance_pct", self._calc_price_distance_pct(ta.current_price, level))
                    .time(timestamp, WritePrecision.NS)
                )
                
        for i, level in enumerate(ta.resistance_levels[:3], 1):
            if level:
                points.append(
                    Point("support_resistance")
                    .tag("symbol", symbol)
                    .tag("type", "resistance")
                    .tag("level", f"R{i}")
                    .field("price", float(level))
                    .field("distance_pct", self._calc_price_distance_pct(ta.current_price, level))
                    .time(timestamp, WritePrecision.NS)
                )
                
        return points

    def _create_market_data_points(
        self, 
        analysis: ComprehensiveAnalysis, 
        symbol: str, 
        timestamp: datetime
    ) -> List[Point]:
        """Create market data points."""
        points = []
        md = analysis.market_data
        
        # Current market data
        point = (
            Point("market_data")
            .tag("symbol", symbol)
            .field("current_price", float(md.current_price))
            .field("previous_close", float(md.previous_close))
            .field("day_change", float(md.current_price - md.previous_close))
            .field("day_change_pct", float((md.current_price - md.previous_close) / md.previous_close * 100))
            .time(timestamp, WritePrecision.NS)
        )
        
        if md.volume:
            point.field("volume", float(md.volume))
            
        if md.market_cap:
            point.field("market_cap", float(md.market_cap))
            
        if md.ytd_return:
            point.field("ytd_return", float(md.ytd_return))
            
        if md.week_52_high and md.week_52_low:
            point.field("week_52_high", float(md.week_52_high))
            point.field("week_52_low", float(md.week_52_low))
            point.field("week_52_position", float((md.current_price - md.week_52_low) / (md.week_52_high - md.week_52_low) * 100))
            
        points.append(point)
        return points

    def _create_financial_ratio_points(
        self, 
        analysis: ComprehensiveAnalysis, 
        symbol: str, 
        timestamp: datetime
    ) -> List[Point]:
        """Create financial ratio data points."""
        points = []
        ratios = analysis.financial_ratios
        
        # Valuation ratios
        valuation_point = Point("financial_ratios").tag("symbol", symbol).tag("category", "valuation")
        
        if ratios.price_to_earnings:
            valuation_point.field("pe_ratio", float(ratios.price_to_earnings))
            valuation_point.field("pe_percentile", self._calc_pe_percentile(float(ratios.price_to_earnings)))
            
        if ratios.price_to_book:
            valuation_point.field("pb_ratio", float(ratios.price_to_book))
            
        if ratios.price_to_sales:
            valuation_point.field("ps_ratio", float(ratios.price_to_sales))
            
        if ratios.enterprise_value_to_revenue:
            valuation_point.field("ev_revenue", float(ratios.enterprise_value_to_revenue))
            
        if ratios.peg_ratio:
            valuation_point.field("peg_ratio", float(ratios.peg_ratio))
            
        valuation_point.time(timestamp, WritePrecision.NS)
        points.append(valuation_point)
        
        # Profitability ratios
        profitability_point = Point("financial_ratios").tag("symbol", symbol).tag("category", "profitability")
        
        if ratios.return_on_equity:
            profitability_point.field("roe", float(ratios.return_on_equity))
            
        if ratios.return_on_assets:
            profitability_point.field("roa", float(ratios.return_on_assets))
            
        if ratios.net_margin:
            profitability_point.field("net_margin", float(ratios.net_margin))
            
        if ratios.operating_margin:
            profitability_point.field("operating_margin", float(ratios.operating_margin))
            
        if ratios.gross_margin:
            profitability_point.field("gross_margin", float(ratios.gross_margin))
            
        profitability_point.time(timestamp, WritePrecision.NS)
        points.append(profitability_point)
        
        # Leverage ratios
        if ratios.debt_to_equity or ratios.debt_to_assets or ratios.interest_coverage:
            leverage_point = Point("financial_ratios").tag("symbol", symbol).tag("category", "leverage")
            
            if ratios.debt_to_equity:
                leverage_point.field("debt_to_equity", float(ratios.debt_to_equity))
                
            if ratios.debt_to_assets:
                leverage_point.field("debt_to_assets", float(ratios.debt_to_assets))
                
            if ratios.interest_coverage:
                leverage_point.field("interest_coverage", float(ratios.interest_coverage))
                
            leverage_point.time(timestamp, WritePrecision.NS)
            points.append(leverage_point)
            
        return points

    def _create_risk_assessment_points(
        self, 
        analysis: ComprehensiveAnalysis, 
        symbol: str, 
        timestamp: datetime
    ) -> List[Point]:
        """Create risk assessment data points."""
        points = []
        risk = analysis.risk_assessment
        
        # Overall risk metrics
        risk_point = (
            Point("risk_metrics")
            .tag("symbol", symbol)
            .field("risk_level_numeric", self._risk_level_to_numeric(risk.overall_risk_level))
            .field("risk_level", risk.overall_risk_level.value)
            .time(timestamp, WritePrecision.NS)
        )
        
        if risk.beta:
            risk_point.field("beta", float(risk.beta))
            risk_point.field("beta_category", self._categorize_beta(float(risk.beta)))
            
        points.append(risk_point)
        
        # Risk categories count
        points.append(
            Point("risk_categories")
            .tag("symbol", symbol)
            .field("regulatory_risks_count", len(risk.regulatory_risks))
            .field("business_risks_count", len(risk.business_risks))
            .field("growth_catalysts_count", len(risk.growth_catalysts))
            .time(timestamp, WritePrecision.NS)
        )
        
        return points

    def _create_investment_thesis_points(
        self, 
        analysis: ComprehensiveAnalysis, 
        symbol: str, 
        timestamp: datetime
    ) -> List[Point]:
        """Create investment thesis data points."""
        points = []
        thesis = analysis.investment_thesis
        
        # Investment rating and targets
        points.append(
            Point("investment_thesis")
            .tag("symbol", symbol)
            .field("rating_numeric", self._rating_to_numeric(thesis.rating))
            .field("rating", thesis.rating.value)
            .field("price_target", float(thesis.price_target))
            .field("bull_case_target", float(thesis.bull_case_target))
            .field("bear_case_target", float(thesis.bear_case_target))
            .field("upside_potential", float((thesis.price_target / analysis.market_data.current_price - 1) * 100))
            .field("bull_upside", float((thesis.bull_case_target / analysis.market_data.current_price - 1) * 100))
            .field("bear_downside", float((thesis.bear_case_target / analysis.market_data.current_price - 1) * 100))
            .field("bull_points_count", len(thesis.bull_case_points))
            .field("bear_points_count", len(thesis.bear_case_points))
            .field("monitoring_points_count", len(thesis.monitoring_points))
            .time(timestamp, WritePrecision.NS)
        )
        
        return points

    def _create_etf_specific_points(
        self, 
        analysis: ComprehensiveAnalysis, 
        symbol: str, 
        timestamp: datetime
    ) -> List[Point]:
        """Create ETF-specific data points."""
        points = []
        
        # ETF metrics (using typical values for demonstration)
        points.append(
            Point("etf_metrics")
            .tag("symbol", symbol)
            .field("expense_ratio", 0.19)  # IWM typical
            .field("aum", float(analysis.market_data.market_cap))
            .field("tracking_error", 0.25)  # Typical small-cap ETF
            .field("dividend_yield", 1.7)  # IWM typical
            .field("holdings_count", 2000)  # Russell 2000
            .field("top_10_concentration", 11.5)  # Typical
            .time(timestamp, WritePrecision.NS)
        )
        
        # Sector allocation (placeholder - would be real data in production)
        sectors = [
            ("Technology", 16.2),
            ("Healthcare", 14.8), 
            ("Financial Services", 13.5),
            ("Industrials", 12.1),
            ("Consumer Discretionary", 11.8)
        ]
        
        for sector, weight in sectors:
            points.append(
                Point("etf_sectors")
                .tag("symbol", symbol)
                .tag("sector", sector)
                .field("weight", weight)
                .time(timestamp, WritePrecision.NS)
            )
            
        return points

    def _create_company_profile_points(
        self, 
        analysis: ComprehensiveAnalysis, 
        symbol: str, 
        timestamp: datetime
    ) -> List[Point]:
        """Create company profile data points."""
        points = []
        profile = analysis.company_profile
        
        points.append(
            Point("company_profile")
            .tag("symbol", symbol)
            .tag("sector", profile.sector)
            .tag("industry", profile.industry)
            .tag("exchange", profile.exchange)
            .field("market_cap", float(profile.market_cap) if profile.market_cap else 0.0)
            .field("description_length", len(profile.description))
            .time(timestamp, WritePrecision.NS)
        )
        
        return points

    def _create_financial_statements_points(
        self, 
        analysis: ComprehensiveAnalysis, 
        symbol: str, 
        timestamp: datetime
    ) -> List[Point]:
        """Create financial statements data points."""
        points = []
        
        # Latest income statement
        if analysis.income_statements:
            latest = analysis.income_statements[-1]
            
            point = Point("financial_statements").tag("symbol", symbol).tag("statement", "income")
            
            if latest.revenue:
                point.field("revenue", float(latest.revenue))
                
            if latest.operating_income:
                point.field("operating_income", float(latest.operating_income))
                
            if latest.net_income:
                point.field("net_income", float(latest.net_income))
                
            if latest.eps_diluted:
                point.field("eps_diluted", float(latest.eps_diluted))
                
            point.field("fiscal_year", latest.fiscal_year)
            if latest.fiscal_quarter:
                point.field("fiscal_quarter", latest.fiscal_quarter)
                
            point.time(timestamp, WritePrecision.NS)
            points.append(point)
            
        # Growth rates (if multiple periods available)
        if len(analysis.income_statements) >= 2:
            current = analysis.income_statements[-1]
            previous = analysis.income_statements[-2]
            
            if current.revenue and previous.revenue and previous.revenue != 0:
                revenue_growth = float((current.revenue - previous.revenue) / previous.revenue * 100)
                points.append(
                    Point("growth_rates")
                    .tag("symbol", symbol)
                    .tag("metric", "revenue")
                    .field("yoy_growth", revenue_growth)
                    .time(timestamp, WritePrecision.NS)
                )
                
        return points

    # Helper methods
    def _is_etf(self, symbol: str) -> bool:
        """Check if symbol is an ETF."""
        etf_symbols = {
            "IWM", "SPY", "QQQ", "VTI", "EFA", "GLD", "TLT", "EEM", "VEA", "IEFA",
            "AGG", "BND", "VWO", "IEMG", "IJH", "IJR", "MDY", "SLY", "VB", "VBR"
        }
        return symbol.upper() in etf_symbols

    def _get_rsi_signal_numeric(self, rsi: Decimal) -> int:
        """Convert RSI to numeric signal."""
        if rsi > 70:
            return 1  # Overbought
        if rsi < 30:
            return -1  # Oversold
        return 0  # Neutral

    def _calc_price_distance_pct(self, current_price: Decimal, target_price: Decimal) -> float:
        """Calculate percentage distance between prices."""
        if target_price == 0:
            return 0.0
        return float((current_price - target_price) / target_price * 100)

    def _calc_bb_position(self, price: Decimal, upper: Decimal, lower: Decimal) -> float:
        """Calculate position within Bollinger Bands (0-100%)."""
        if upper == lower:
            return 50.0
        return float((price - lower) / (upper - lower) * 100)

    def _calc_volatility_percentile(self, volatility: float) -> int:
        """Calculate volatility percentile (rough estimate)."""
        if volatility < 15:
            return 25
        if volatility < 25:
            return 50
        if volatility < 35:
            return 75
        return 90

    def _calc_pe_percentile(self, pe: float) -> int:
        """Calculate P/E ratio percentile (rough estimate)."""
        if pe < 15:
            return 25
        if pe < 25:
            return 50
        if pe < 35:
            return 75
        return 90

    def _risk_level_to_numeric(self, risk_level) -> int:
        """Convert risk level to numeric value."""
        mapping = {
            "LOW": 1,
            "MODERATE": 2, 
            "MODERATE_HIGH": 3,
            "HIGH": 4,
            "VERY_HIGH": 5
        }
        return mapping.get(risk_level.value, 3)

    def _rating_to_numeric(self, rating) -> int:
        """Convert investment rating to numeric value."""
        mapping = {
            "STRONG_SELL": 1,
            "SELL": 2,
            "HOLD": 3,
            "BUY": 4,
            "STRONG_BUY": 5
        }
        return mapping.get(rating.value, 3)

    def _categorize_beta(self, beta: float) -> str:
        """Categorize beta value."""
        if beta < 0.8:
            return "Low Volatility"
        if beta < 1.2:
            return "Market Volatility"
        return "High Volatility"

    def close(self):
        """Close InfluxDB client connection."""
        if self.client:
            self.client.close()


# Example usage function
async def demo_influxdb_storage():
    """Demonstrate InfluxDB storage functionality."""
    from ..comprehensive_analyzer import ComprehensiveStockAnalyzer
    
    # Initialize storage
    storage = InfluxDBAnalysisStorage()
    
    if not storage.client:
        print("InfluxDB not available - run with Docker: docker run -p 8086:8086 influxdb:2.0")
        return
    
    # Analyze stock
    analyzer = ComprehensiveStockAnalyzer()
    analysis = await analyzer.analyze_stock("IWM")
    
    # Store in InfluxDB
    success = await storage.store_comprehensive_analysis(analysis)
    
    if success:
        print(f"✅ Successfully stored {analysis.company_profile.symbol} analysis in InfluxDB")
    else:
        print(f"❌ Failed to store {analysis.company_profile.symbol} analysis")
    
    storage.close()


if __name__ == "__main__":
    asyncio.run(demo_influxdb_storage())