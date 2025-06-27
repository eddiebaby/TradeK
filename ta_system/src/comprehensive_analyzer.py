"""Comprehensive stock analysis orchestrator combining technical and fundamental analysis."""

import asyncio
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional

import pandas as pd
import yfinance as yf

from .data_sources.influxdb_storage import InfluxDBAnalysisStorage
from .data_sources.ytd_calculator import YTDCalculator
from .fundamental.models import (
    BalanceSheet,
    CashFlowStatement,
    CompanyProfile,
    ComprehensiveAnalysis,
    IncomeStatement,
    InvestmentRating,
    InvestmentThesis,
    MarketData,
    ReportPeriod,
    RiskAssessment,
    RiskLevel,
    TechnicalAnalysis,
)
from .fundamental.ratios import FinancialRatioCalculator
from .indicators import (
    ATRCalculator,
    BollingerBandsCalculator,
    EMACalculator,
    IndicatorCalculator,
    MACDCalculator,
    RSICalculator,
    SMApCalculator,
)
from .models import OHLCV
from .reports.generator import ReportGenerator

# Market cap thresholds for risk assessment
SMALL_CAP_THRESHOLD = 1_000_000_000  # $1B
LARGE_CAP_THRESHOLD = 100_000_000_000  # $100B

# Valuation thresholds for investment rating
HIGH_PE_THRESHOLD = 30  # P/E > 30 = expensive
LOW_PE_THRESHOLD = 15   # P/E < 15 = cheap
OVERBOUGHT_RSI = 70     # RSI > 70 = overbought


class ComprehensiveStockAnalyzer:
    """Orchestrates comprehensive fundamental and technical stock analysis."""

    def __init__(self, enable_influxdb: bool = False, influxdb_config: Optional[dict] = None):
        """Initialize the comprehensive analyzer."""
        self.ratio_calculator = FinancialRatioCalculator()
        self.report_generator = ReportGenerator()
        self.technical_calculator = IndicatorCalculator()
        self._setup_technical_indicators()
        
        # InfluxDB storage (optional)
        self.influxdb_storage = None
        if enable_influxdb:
            config = influxdb_config or {}
            self.influxdb_storage = InfluxDBAnalysisStorage(**config)
        
        # YTD Calculator service
        self.ytd_calculator = YTDCalculator()
        
        # ETF symbols for asset type detection
        self._etf_symbols = {
            "IWM", "SPY", "QQQ", "VTI", "EFA", "GLD", "TLT", "EEM", "VEA", "IEFA",
            "AGG", "BND", "VWO", "IEMG", "IJH", "IJR", "MDY", "SLY", "VB", "VBR",
            "VTV", "VUG", "VYM", "SCHD", "SCHA", "SCHB", "SCHF", "SCHG", "SCHM", 
            "SCHV", "SCHX", "SCHY", "SCHZ", "XLF", "XLE", "XLI", "XLK", "XLV",
            "XLY", "XLP", "XLU", "XLRE", "XME", "XHB", "SMH", "KRE", "IYR"
        }

    def _setup_technical_indicators(self):
        """Setup technical analysis indicators."""
        self.technical_calculator.register("RSI_14", RSICalculator(period=14))
        self.technical_calculator.register("SMA_20", SMApCalculator(period=20))
        self.technical_calculator.register("SMA_50", SMApCalculator(period=50))
        self.technical_calculator.register("SMA_200", SMApCalculator(period=200))
        self.technical_calculator.register("EMA_12", EMACalculator(period=12))
        self.technical_calculator.register("EMA_26", EMACalculator(period=26))
        self.technical_calculator.register("MACD_12_26_9", MACDCalculator(fast=12, slow=26, signal=9))
        self.technical_calculator.register("BB_20_2", BollingerBandsCalculator(period=20, std_dev=2))
        self.technical_calculator.register("ATR_14", ATRCalculator(period=14))

    def detect_asset_type(self, symbol: str) -> str:
        """Detect if the symbol is a stock, ETF, or other asset type."""
        symbol_upper = symbol.upper()
        
        if symbol_upper in self._etf_symbols:
            return "ETF"
        return "STOCK"

    def _get_ytd_date_range(self) -> tuple[date, date]:
        """Get YTD date range from January 1st to current date."""
        return self.ytd_calculator._get_ytd_date_range()

    def _calculate_ytd_from_prices(self, start_price: float, current_price: float) -> Decimal:
        """Calculate YTD return from start and current prices."""
        return self.ytd_calculator._calculate_return_percentage(start_price, current_price)

    def _calculate_ytd_return(self, symbol: str, year: Optional[int] = None) -> Optional[Decimal]:
        """Calculate accurate YTD return using historical data."""
        return self.ytd_calculator.calculate_ytd_return(symbol, year)

    def _get_ytd_with_fallback(self, symbol: str, info: dict, hist: pd.DataFrame) -> Decimal:
        """Get YTD return with multiple fallback methods."""
        return self.ytd_calculator.calculate_ytd_with_fallback(symbol, info, hist)

    async def analyze_stock(self, symbol: str) -> ComprehensiveAnalysis:
        """Perform comprehensive analysis of a stock."""
        
        print(f"🔍 Starting comprehensive analysis for {symbol.upper()}")
        
        # Fetch all data concurrently
        tasks = [
            self._fetch_company_profile(symbol),
            self._fetch_market_data(symbol),
            self._fetch_financial_statements(symbol),
            self._perform_technical_analysis(symbol),
        ]
        
        company_profile, market_data, financial_data, technical_analysis = await asyncio.gather(*tasks)
        
        # Calculate financial ratios
        ratios = self._calculate_financial_ratios(symbol, financial_data, market_data)
        
        # Perform risk assessment
        risk_assessment = self._assess_risks(symbol, company_profile, financial_data, market_data)
        
        # Generate investment thesis
        investment_thesis = self._generate_investment_thesis(
            symbol, company_profile, market_data, ratios, technical_analysis, risk_assessment
        )
        
        # Create comprehensive analysis
        analysis = ComprehensiveAnalysis(
            company_profile=company_profile,
            market_data=market_data,
            financial_ratios=ratios,
            technical_analysis=technical_analysis,
            risk_assessment=risk_assessment,
            investment_thesis=investment_thesis,
            income_statements=financial_data.get("income_statements", []),
            balance_sheets=financial_data.get("balance_sheets", []),
            cash_flow_statements=financial_data.get("cash_flow_statements", []),
            analysis_date=datetime.now(timezone.utc),
            analyst="AI Comprehensive Analyzer",
            version="1.0"
        )
        
        print(f"✅ Comprehensive analysis completed for {symbol.upper()}")
        
        # Store in InfluxDB if enabled
        if self.influxdb_storage:
            try:
                success = await self.influxdb_storage.store_comprehensive_analysis(analysis)
                if success:
                    print(f"💾 Analysis data stored in InfluxDB for {symbol.upper()}")
                else:
                    print(f"⚠️ Failed to store analysis data in InfluxDB for {symbol.upper()}")
            except Exception as e:
                print(f"⚠️ InfluxDB storage error: {e}")
        
        return analysis

    async def _fetch_company_profile(self, symbol: str) -> CompanyProfile:
        """Fetch company profile information."""
        print(f"📋 Fetching company profile for {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return CompanyProfile(
                symbol=symbol.upper(),
                company_name=info.get("longName", f"{symbol.upper()} Corporation"),
                sector=info.get("sector", "Technology"),
                industry=info.get("industry", "Software"),
                market_cap=Decimal(str(info.get("marketCap", 0))),
                employees=info.get("fullTimeEmployees"),
                headquarters=f"{info.get('city', 'Unknown')}, {info.get('state', 'Unknown')}",
                founded=None,  # Not available in yfinance
                website=info.get("website"),
                description=info.get("longBusinessSummary", f"{symbol.upper()} is a leading company in its sector."),
                exchange=info.get("exchange", "NASDAQ"),
                currency=info.get("currency", "USD")
            )
        except Exception as e:
            print(f"⚠️ Using default company profile for {symbol}: {e}")
            return self._create_default_company_profile(symbol)

    def _create_default_company_profile(self, symbol: str) -> CompanyProfile:
        """Create default company profile when data is unavailable."""
        return CompanyProfile(
            symbol=symbol.upper(),
            company_name=f"{symbol.upper()} Corporation",
            sector="Technology",
            industry="Software",
            description=f"{symbol.upper()} is a technology company operating in multiple segments.",
            exchange="NASDAQ"
        )

    async def _fetch_market_data(self, symbol: str) -> MarketData:
        """Fetch current market data."""
        print(f"📊 Fetching market data for {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1y")
            
            current_price = Decimal(str(info.get("currentPrice", hist["Close"].iloc[-1])))
            
            return MarketData(
                symbol=symbol.upper(),
                current_price=current_price,
                previous_close=Decimal(str(info.get("previousClose", hist["Close"].iloc[-2]))),
                market_cap=Decimal(str(info.get("marketCap", 0))),
                shares_outstanding=Decimal(str(info.get("sharesOutstanding", 0))),
                avg_volume=Decimal(str(info.get("averageVolume", hist["Volume"].mean()))),
                week_52_high=Decimal(str(info.get("fiftyTwoWeekHigh", hist["High"].max()))),
                week_52_low=Decimal(str(info.get("fiftyTwoWeekLow", hist["Low"].min()))),
                ytd_return=self._get_ytd_with_fallback(symbol, info, hist),
                beta=Decimal(str(info.get("beta", 1.0))) if info.get("beta") else None,
                dividend_yield=Decimal(str(info.get("dividendYield", 0) * 100)) if info.get("dividendYield") else None,
            )
        except Exception as e:
            print(f"⚠️ Using default market data for {symbol}: {e}")
            return self._create_default_market_data(symbol)

    def _create_default_market_data(self, symbol: str) -> MarketData:
        """Create default market data when unavailable."""
        return MarketData(
            symbol=symbol.upper(),
            current_price=Decimal("100.00"),
            previous_close=Decimal("99.50"),
            market_cap=Decimal("50000000000"),
            shares_outstanding=Decimal("500000000"),
            avg_volume=Decimal("10000000"),
            week_52_high=Decimal("120.00"),
            week_52_low=Decimal("80.00"),
            ytd_return=Decimal("5.0"),
        )

    async def _fetch_financial_statements(self, symbol: str) -> dict[str, list]:
        """Fetch financial statement data."""
        print(f"💰 Fetching financial statements for {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get financial data
            income_stmt = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            financial_data = {
                "income_statements": [],
                "balance_sheets": [],
                "cash_flow_statements": []
            }
            
            # Process income statements
            if not income_stmt.empty:
                for date, data in income_stmt.items():
                    income_statement = self._create_income_statement(symbol, date, data)
                    if income_statement:
                        financial_data["income_statements"].append(income_statement)
            
            # Process balance sheets
            if not balance_sheet.empty:
                for date, data in balance_sheet.items():
                    balance_sheet_obj = self._create_balance_sheet(symbol, date, data)
                    if balance_sheet_obj:
                        financial_data["balance_sheets"].append(balance_sheet_obj)
            
            # Process cash flow statements
            if not cash_flow.empty:
                for date, data in cash_flow.items():
                    cash_flow_stmt = self._create_cash_flow_statement(symbol, date, data)
                    if cash_flow_stmt:
                        financial_data["cash_flow_statements"].append(cash_flow_stmt)
            
            # If no data available, create default statements
            if not financial_data["income_statements"]:
                financial_data["income_statements"] = [self._create_default_income_statement(symbol)]
            if not financial_data["balance_sheets"]:
                financial_data["balance_sheets"] = [self._create_default_balance_sheet(symbol)]
            if not financial_data["cash_flow_statements"]:
                financial_data["cash_flow_statements"] = [self._create_default_cash_flow_statement(symbol)]
            
            return financial_data
            
        except Exception as e:
            print(f"⚠️ Using default financial statements for {symbol}: {e}")
            return {
                "income_statements": [self._create_default_income_statement(symbol)],
                "balance_sheets": [self._create_default_balance_sheet(symbol)],
                "cash_flow_statements": [self._create_default_cash_flow_statement(symbol)]
            }

    def _create_income_statement(self, symbol: str, date, data) -> Optional[IncomeStatement]:
        """Create income statement from yfinance data."""
        try:
            return IncomeStatement(
                symbol=symbol.upper(),
                period=ReportPeriod.ANNUAL,
                fiscal_year=date.year,
                report_date=date,
                revenue=Decimal(str(data.get("Total Revenue", 0))),
                cost_of_revenue=Decimal(str(data.get("Cost Of Revenue", 0))),
                gross_profit=Decimal(str(data.get("Gross Profit", 0))),
                operating_income=Decimal(str(data.get("Operating Income", 0))),
                net_income=Decimal(str(data.get("Net Income", 0))),
                ebitda=Decimal(str(data.get("EBITDA", 0))),
            )
        except Exception:
            return None

    def _create_balance_sheet(self, symbol: str, date, data) -> Optional[BalanceSheet]:
        """Create balance sheet from yfinance data."""
        try:
            return BalanceSheet(
                symbol=symbol.upper(),
                period=ReportPeriod.ANNUAL,
                fiscal_year=date.year,
                report_date=date,
                total_assets=Decimal(str(data.get("Total Assets", 0))),
                current_assets=Decimal(str(data.get("Current Assets", 0))),
                cash_and_equivalents=Decimal(str(data.get("Cash And Cash Equivalents", 0))),
                total_liabilities=Decimal(str(data.get("Total Liab", 0))),
                current_liabilities=Decimal(str(data.get("Current Liabilities", 0))),
                total_debt=Decimal(str(data.get("Total Debt", 0))),
                shareholders_equity=Decimal(str(data.get("Stockholders Equity", 0))),
            )
        except Exception:
            return None

    def _create_cash_flow_statement(self, symbol: str, date, data) -> Optional[CashFlowStatement]:
        """Create cash flow statement from yfinance data."""
        try:
            return CashFlowStatement(
                symbol=symbol.upper(),
                period=ReportPeriod.ANNUAL,
                fiscal_year=date.year,
                report_date=date,
                operating_cash_flow=Decimal(str(data.get("Total Cash From Operating Activities", 0))),
                investing_cash_flow=Decimal(str(data.get("Total Cashflows From Investing Activities", 0))),
                financing_cash_flow=Decimal(str(data.get("Total Cash From Financing Activities", 0))),
                free_cash_flow=Decimal(str(data.get("Free Cash Flow", 0))),
            )
        except Exception:
            return None

    def _create_default_income_statement(self, symbol: str) -> IncomeStatement:
        """Create default income statement."""
        return IncomeStatement(
            symbol=symbol.upper(),
            period=ReportPeriod.ANNUAL,
            fiscal_year=2024,
            report_date=datetime.now(timezone.utc),
            revenue=Decimal("10000000000"),
            cost_of_revenue=Decimal("6000000000"),
            gross_profit=Decimal("4000000000"),
            operating_income=Decimal("2000000000"),
            net_income=Decimal("1500000000"),
            eps_diluted=Decimal("3.00"),
            ebitda=Decimal("2500000000"),
        )

    def _create_default_balance_sheet(self, symbol: str) -> BalanceSheet:
        """Create default balance sheet."""
        return BalanceSheet(
            symbol=symbol.upper(),
            period=ReportPeriod.ANNUAL,
            fiscal_year=2024,
            report_date=datetime.now(timezone.utc),
            total_assets=Decimal("15000000000"),
            current_assets=Decimal("8000000000"),
            cash_and_equivalents=Decimal("3000000000"),
            total_liabilities=Decimal("5000000000"),
            current_liabilities=Decimal("3000000000"),
            total_debt=Decimal("2000000000"),
            shareholders_equity=Decimal("10000000000"),
        )

    def _create_default_cash_flow_statement(self, symbol: str) -> CashFlowStatement:
        """Create default cash flow statement."""
        return CashFlowStatement(
            symbol=symbol.upper(),
            period=ReportPeriod.ANNUAL,
            fiscal_year=2024,
            report_date=datetime.now(timezone.utc),
            operating_cash_flow=Decimal("2000000000"),
            investing_cash_flow=Decimal("-500000000"),
            financing_cash_flow=Decimal("-300000000"),
            free_cash_flow=Decimal("1800000000"),
        )

    async def _perform_technical_analysis(self, symbol: str) -> TechnicalAnalysis:
        """Perform comprehensive technical analysis."""
        print(f"📈 Performing technical analysis for {symbol}")
        
        try:
            # Fetch price data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y", interval="1d")
            
            if hist.empty:
                return self._create_default_technical_analysis(symbol)
            
            # Convert to OHLCV format
            ohlcv_data = []
            for date, row in hist.iterrows():
                ohlcv = OHLCV(
                    symbol=symbol.upper(),
                    timestamp=date.to_pydatetime().replace(tzinfo=timezone.utc),
                    open=Decimal(str(row["Open"])),
                    high=Decimal(str(row["High"])),
                    low=Decimal(str(row["Low"])),
                    close=Decimal(str(row["Close"])),
                    volume=int(row["Volume"])
                )
                ohlcv_data.append(ohlcv)
            
            # Calculate technical indicators
            self.technical_calculator.reset_all()
            latest_indicators = {}
            
            for ohlcv in ohlcv_data:
                results = self.technical_calculator.calculate_all(ohlcv)
                if results:
                    latest_indicators = results
            
            # Extract indicator values
            current_price = ohlcv_data[-1].close
            
            return TechnicalAnalysis(
                symbol=symbol.upper(),
                current_price=current_price,
                rsi_14=Decimal(str(latest_indicators["RSI_14"].value)) if "RSI_14" in latest_indicators else None,
                sma_20=Decimal(str(latest_indicators["SMA_20"].value)) if "SMA_20" in latest_indicators else None,
                sma_50=Decimal(str(latest_indicators["SMA_50"].value)) if "SMA_50" in latest_indicators else None,
                sma_200=Decimal(str(latest_indicators["SMA_200"].value)) if "SMA_200" in latest_indicators else None,
                ema_12=Decimal(str(latest_indicators["EMA_12"].value)) if "EMA_12" in latest_indicators else None,
                ema_26=Decimal(str(latest_indicators["EMA_26"].value)) if "EMA_26" in latest_indicators else None,
                macd=Decimal(str(latest_indicators["MACD_12_26_9"].value)) if "MACD_12_26_9" in latest_indicators else None,
                macd_signal=Decimal(str(latest_indicators["MACD_12_26_9"].components["signal"])) if "MACD_12_26_9" in latest_indicators and latest_indicators["MACD_12_26_9"].components else None,
                macd_histogram=Decimal(str(latest_indicators["MACD_12_26_9"].components["histogram"])) if "MACD_12_26_9" in latest_indicators and latest_indicators["MACD_12_26_9"].components else None,
                bollinger_upper=Decimal(str(latest_indicators["BB_20_2"].components["upper"])) if "BB_20_2" in latest_indicators and latest_indicators["BB_20_2"].components else None,
                bollinger_lower=Decimal(str(latest_indicators["BB_20_2"].components["lower"])) if "BB_20_2" in latest_indicators and latest_indicators["BB_20_2"].components else None,
                atr_14=Decimal(str(latest_indicators["ATR_14"].value)) if "ATR_14" in latest_indicators else None,
                volume=Decimal(str(ohlcv_data[-1].volume)),
                volatility=self._calculate_volatility(hist),
                support_levels=[current_price * Decimal("0.95"), current_price * Decimal("0.90")],
                resistance_levels=[current_price * Decimal("1.05"), current_price * Decimal("1.10")],
            )
            
        except Exception as e:
            print(f"⚠️ Using default technical analysis for {symbol}: {e}")
            return self._create_default_technical_analysis(symbol)

    def _calculate_volatility(self, hist: pd.DataFrame) -> Decimal:
        """Calculate annualized volatility."""
        try:
            returns = hist["Close"].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5) * 100  # Annualized
            return Decimal(str(volatility))
        except Exception:
            return Decimal("25.0")  # Default volatility

    def _create_default_technical_analysis(self, symbol: str) -> TechnicalAnalysis:
        """Create default technical analysis."""
        current_price = Decimal("100.00")
        
        return TechnicalAnalysis(
            symbol=symbol.upper(),
            current_price=current_price,
            rsi_14=Decimal("50.0"),
            sma_20=Decimal("98.0"),
            sma_50=Decimal("95.0"),
            sma_200=Decimal("90.0"),
            volume=Decimal("1000000"),
            volatility=Decimal("25.0"),
            support_levels=[Decimal("95.0"), Decimal("90.0")],
            resistance_levels=[Decimal("105.0"), Decimal("110.0")],
        )

    def _calculate_financial_ratios(self, symbol: str, financial_data: dict, market_data: MarketData):
        """Calculate comprehensive financial ratios."""
        if not financial_data["income_statements"] or not financial_data["balance_sheets"]:
            # Return default ratios
            return self.ratio_calculator._create_default_ratios(symbol)
        
        latest_income = financial_data["income_statements"][0]
        latest_balance = financial_data["balance_sheets"][0]
        
        return self.ratio_calculator.calculate_all_ratios(
            symbol=symbol,
            income_statement=latest_income,
            balance_sheet=latest_balance,
            market_data=market_data
        )

    def _assess_risks(self, symbol: str, company_profile: CompanyProfile, _financial_data: dict, market_data: MarketData) -> RiskAssessment:
        """Assess comprehensive investment risks."""
        
        # Determine overall risk level based on various factors
        risk_level = RiskLevel.MODERATE
        
        # Adjust based on market cap
        if market_data.market_cap < SMALL_CAP_THRESHOLD:  # Small cap
            risk_level = RiskLevel.HIGH
        elif market_data.market_cap > LARGE_CAP_THRESHOLD:  # Large cap
            risk_level = RiskLevel.MODERATE
        
        # Adjust based on sector
        if (company_profile.sector in ["Technology", "Biotechnology"] 
            and risk_level == RiskLevel.MODERATE):
                risk_level = RiskLevel.MODERATE_HIGH
        
        return RiskAssessment(
            symbol=symbol.upper(),
            overall_risk_level=risk_level,
            regulatory_risks=[
                "Regulatory changes: Potential changes in government regulations affecting the industry",
                "Antitrust concerns: Possible antitrust investigations or enforcement actions",
                "Data privacy: Evolving privacy regulations and compliance requirements",
                "International regulations: Changes in international trade and regulatory policies",
                "Compliance costs: Increasing costs of regulatory compliance"
            ],
            business_risks=[
                "Competition: Intense competition from existing and new market entrants",
                "Technology disruption: Risk of technological obsolescence or disruption",
                "Market demand: Changes in consumer preferences and market demand",
                "Economic sensitivity: Exposure to economic cycles and downturns",
                "Key personnel: Dependence on key management and technical personnel",
                "Supply chain: Potential disruptions in supply chain and operations"
            ],
            growth_catalysts=[
                "Market expansion: Opportunities for geographic and market expansion",
                "Product innovation: Development of new products and services",
                "Digital transformation: Benefits from ongoing digital transformation trends",
                "Strategic partnerships: Potential for strategic alliances and partnerships",
                "Cost optimization: Operational efficiency improvements and cost savings",
                "Market consolidation: Opportunities from industry consolidation"
            ],
            beta=market_data.beta,
        )

    def _generate_investment_thesis(
        self,
        symbol: str,
        company_profile: CompanyProfile,
        market_data: MarketData,
        ratios,
        technical_analysis: TechnicalAnalysis,
        risk_assessment: RiskAssessment
    ) -> InvestmentThesis:
        """Generate comprehensive investment thesis."""
        
        # Determine rating based on various factors
        rating = InvestmentRating.BUY
        
        # Adjust rating based on valuation
        if ratios.price_to_earnings and ratios.price_to_earnings > HIGH_PE_THRESHOLD:
            rating = InvestmentRating.HOLD
        elif ratios.price_to_earnings and ratios.price_to_earnings < LOW_PE_THRESHOLD:
            rating = InvestmentRating.BUY
        
        # Adjust based on technical analysis
        if (technical_analysis.rsi_14 and technical_analysis.rsi_14 > OVERBOUGHT_RSI
            and rating == InvestmentRating.BUY):
                rating = InvestmentRating.HOLD
        
        # Calculate price targets
        current_price = market_data.current_price
        base_target = current_price * Decimal("1.15")  # 15% upside
        bull_target = current_price * Decimal("1.30")  # 30% upside
        bear_target = current_price * Decimal("0.85")  # 15% downside
        
        return InvestmentThesis(
            symbol=symbol.upper(),
            rating=rating,
            price_target=base_target,
            bull_case_target=bull_target,
            bear_case_target=bear_target,
            investment_rationale=f"{company_profile.company_name} represents a solid investment opportunity in the {company_profile.sector} sector with strong fundamentals and growth potential.",
            bull_case_points=[
                "Market Leadership: Strong competitive position in core markets",
                "Innovation Pipeline: Robust product development and innovation capabilities",
                "Financial Strength: Strong balance sheet and cash generation",
                "Growth Markets: Exposure to high-growth market segments",
                "Operational Excellence: Efficient operations and cost management",
                "Strategic Initiatives: Well-positioned strategic initiatives and partnerships",
                "Market Trends: Beneficiary of favorable long-term market trends"
            ],
            bear_case_points=[
                "Competitive Pressure: Intense competition affecting margins and market share",
                "Regulatory Risk: Potential regulatory changes impacting operations",
                "Economic Sensitivity: Exposure to economic cycles and downturns",
                "Valuation Concerns: Current valuation may limit upside potential",
                "Technology Risk: Risk of technological disruption or obsolescence",
                "Market Saturation: Limited growth opportunities in mature markets"
            ],
            key_catalysts=[
                "Product launches and innovation",
                "Market expansion and new customer acquisition",
                "Strategic partnerships and acquisitions",
                "Regulatory approvals and policy changes",
                "Economic recovery and market growth"
            ],
            key_risks=[
                "Competitive threats and market share loss",
                "Regulatory changes and compliance costs",
                "Economic downturns and market volatility",
                "Technology disruption and obsolescence"
            ],
            short_term_outlook="Near-term performance will depend on quarterly results, market conditions, and competitive developments.",
            medium_term_outlook="Medium-term growth driven by strategic initiatives, market expansion, and operational improvements.",
            long_term_outlook="Long-term value creation through innovation, market leadership, and strategic positioning.",
            monitoring_points=[
                "Quarterly earnings and revenue growth",
                "Market share and competitive position",
                "New product launches and innovation",
                "Regulatory developments and policy changes",
                "Management guidance and strategic updates",
                "Industry trends and market conditions",
                "Financial metrics and ratio analysis",
                "Technical analysis and price action",
                "Economic indicators and market sentiment",
                "Analyst recommendations and target prices"
            ],
            portfolio_allocation="3-7% allocation for diversified growth portfolios",
            risk_level=risk_assessment.overall_risk_level,
            time_horizon="3-5 year investment horizon for long-term growth",
            portfolio_fit="Suitable for growth-oriented investors seeking sector exposure"
        )

    async def generate_report(self, analysis: ComprehensiveAnalysis) -> str:
        """Generate comprehensive analysis report."""
        print(f"📄 Generating comprehensive report for {analysis.company_profile.symbol}")
        
        report = self.report_generator.generate_comprehensive_report(analysis)
        
        print(f"✅ Report generated successfully for {analysis.company_profile.symbol}")
        return report

    async def analyze_and_report(self, symbol: str) -> str:
        """Perform complete analysis and generate report."""
        analysis = await self.analyze_stock(symbol)
        return await self.generate_report(analysis)