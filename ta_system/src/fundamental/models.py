"""Fundamental analysis data models."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ReportPeriod(str, Enum):
    """Financial reporting periods."""
    
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    TTM = "ttm"  # Trailing Twelve Months


class InvestmentRating(str, Enum):
    """Investment recommendation ratings."""
    
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    
    LOW = "Low"
    MODERATE = "Moderate"
    MODERATE_HIGH = "Moderate-High"
    HIGH = "High"
    VERY_HIGH = "Very High"


class CompanyProfile(BaseModel):
    """Company overview and profile information."""
    
    symbol: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    sector: str = Field(..., description="Business sector")
    industry: str = Field(..., description="Industry classification")
    market_cap: Optional[Decimal] = Field(None, description="Market capitalization")
    employees: Optional[int] = Field(None, description="Number of employees")
    headquarters: Optional[str] = Field(None, description="Headquarters location")
    founded: Optional[int] = Field(None, description="Year founded")
    website: Optional[str] = Field(None, description="Company website")
    description: str = Field(..., description="Business description")
    exchange: str = Field(..., description="Stock exchange")
    currency: str = Field(default="USD", description="Reporting currency")


class FinancialStatement(BaseModel):
    """Base class for financial statements."""
    
    symbol: str = Field(..., description="Stock ticker symbol")
    period: ReportPeriod = Field(..., description="Reporting period")
    fiscal_year: int = Field(..., description="Fiscal year")
    fiscal_quarter: Optional[int] = Field(None, description="Fiscal quarter (if quarterly)")
    report_date: datetime = Field(..., description="Report date")
    currency: str = Field(default="USD", description="Currency")


class IncomeStatement(FinancialStatement):
    """Income statement data."""
    
    revenue: Decimal = Field(..., description="Total revenue")
    cost_of_revenue: Optional[Decimal] = Field(None, description="Cost of goods sold")
    gross_profit: Optional[Decimal] = Field(None, description="Gross profit")
    operating_expenses: Optional[Decimal] = Field(None, description="Operating expenses")
    operating_income: Optional[Decimal] = Field(None, description="Operating income")
    interest_expense: Optional[Decimal] = Field(None, description="Interest expense")
    pretax_income: Optional[Decimal] = Field(None, description="Income before tax")
    tax_expense: Optional[Decimal] = Field(None, description="Tax expense")
    net_income: Decimal = Field(..., description="Net income")
    eps_basic: Optional[Decimal] = Field(None, description="Basic EPS")
    eps_diluted: Optional[Decimal] = Field(None, description="Diluted EPS")
    shares_outstanding: Optional[Decimal] = Field(None, description="Shares outstanding")
    ebitda: Optional[Decimal] = Field(None, description="EBITDA")


class BalanceSheet(FinancialStatement):
    """Balance sheet data."""
    
    total_assets: Decimal = Field(..., description="Total assets")
    current_assets: Optional[Decimal] = Field(None, description="Current assets")
    cash_and_equivalents: Optional[Decimal] = Field(None, description="Cash and cash equivalents")
    accounts_receivable: Optional[Decimal] = Field(None, description="Accounts receivable")
    inventory: Optional[Decimal] = Field(None, description="Inventory")
    total_liabilities: Decimal = Field(..., description="Total liabilities")
    current_liabilities: Optional[Decimal] = Field(None, description="Current liabilities")
    long_term_debt: Optional[Decimal] = Field(None, description="Long-term debt")
    total_debt: Optional[Decimal] = Field(None, description="Total debt")
    shareholders_equity: Decimal = Field(..., description="Shareholders' equity")
    retained_earnings: Optional[Decimal] = Field(None, description="Retained earnings")


class CashFlowStatement(FinancialStatement):
    """Cash flow statement data."""
    
    operating_cash_flow: Decimal = Field(..., description="Operating cash flow")
    investing_cash_flow: Optional[Decimal] = Field(None, description="Investing cash flow")
    financing_cash_flow: Optional[Decimal] = Field(None, description="Financing cash flow")
    free_cash_flow: Optional[Decimal] = Field(None, description="Free cash flow")
    capital_expenditures: Optional[Decimal] = Field(None, description="Capital expenditures")
    dividend_payments: Optional[Decimal] = Field(None, description="Dividend payments")
    share_buybacks: Optional[Decimal] = Field(None, description="Share buybacks")


class FinancialRatios(BaseModel):
    """Calculated financial ratios."""
    
    symbol: str = Field(..., description="Stock ticker symbol")
    as_of_date: datetime = Field(..., description="Calculation date")
    
    # Profitability Ratios
    gross_margin: Optional[Decimal] = Field(None, description="Gross profit margin")
    operating_margin: Optional[Decimal] = Field(None, description="Operating margin")
    net_margin: Optional[Decimal] = Field(None, description="Net profit margin")
    return_on_assets: Optional[Decimal] = Field(None, description="ROA")
    return_on_equity: Optional[Decimal] = Field(None, description="ROE")
    return_on_invested_capital: Optional[Decimal] = Field(None, description="ROIC")
    
    # Liquidity Ratios
    current_ratio: Optional[Decimal] = Field(None, description="Current ratio")
    quick_ratio: Optional[Decimal] = Field(None, description="Quick ratio")
    cash_ratio: Optional[Decimal] = Field(None, description="Cash ratio")
    
    # Leverage Ratios
    debt_to_equity: Optional[Decimal] = Field(None, description="Debt-to-equity ratio")
    debt_to_assets: Optional[Decimal] = Field(None, description="Debt-to-assets ratio")
    interest_coverage: Optional[Decimal] = Field(None, description="Interest coverage ratio")
    
    # Efficiency Ratios
    asset_turnover: Optional[Decimal] = Field(None, description="Asset turnover")
    inventory_turnover: Optional[Decimal] = Field(None, description="Inventory turnover")
    receivables_turnover: Optional[Decimal] = Field(None, description="Receivables turnover")
    
    # Valuation Ratios
    price_to_earnings: Optional[Decimal] = Field(None, description="P/E ratio")
    price_to_book: Optional[Decimal] = Field(None, description="P/B ratio")
    price_to_sales: Optional[Decimal] = Field(None, description="P/S ratio")
    enterprise_value_to_revenue: Optional[Decimal] = Field(None, description="EV/Revenue")
    enterprise_value_to_ebitda: Optional[Decimal] = Field(None, description="EV/EBITDA")
    peg_ratio: Optional[Decimal] = Field(None, description="PEG ratio")


class MarketData(BaseModel):
    """Current market data and trading information."""
    
    symbol: str = Field(..., description="Stock ticker symbol")
    current_price: Decimal = Field(..., description="Current stock price")
    previous_close: Decimal = Field(..., description="Previous closing price")
    market_cap: Decimal = Field(..., description="Market capitalization")
    shares_outstanding: Decimal = Field(..., description="Shares outstanding")
    avg_volume: Decimal = Field(..., description="Average trading volume")
    week_52_high: Decimal = Field(..., description="52-week high")
    week_52_low: Decimal = Field(..., description="52-week low")
    ytd_return: Decimal = Field(..., description="Year-to-date return")
    beta: Optional[Decimal] = Field(None, description="Beta coefficient")
    dividend_yield: Optional[Decimal] = Field(None, description="Dividend yield")
    ex_dividend_date: Optional[datetime] = Field(None, description="Ex-dividend date")
    earnings_date: Optional[datetime] = Field(None, description="Next earnings date")


class TechnicalAnalysis(BaseModel):
    """Technical analysis summary."""
    
    symbol: str = Field(..., description="Stock ticker symbol")
    current_price: Decimal = Field(..., description="Current price")
    rsi_14: Optional[Decimal] = Field(None, description="14-day RSI")
    sma_20: Optional[Decimal] = Field(None, description="20-day SMA")
    sma_50: Optional[Decimal] = Field(None, description="50-day SMA") 
    sma_200: Optional[Decimal] = Field(None, description="200-day SMA")
    ema_12: Optional[Decimal] = Field(None, description="12-day EMA")
    ema_26: Optional[Decimal] = Field(None, description="26-day EMA")
    macd: Optional[Decimal] = Field(None, description="MACD line")
    macd_signal: Optional[Decimal] = Field(None, description="MACD signal line")
    macd_histogram: Optional[Decimal] = Field(None, description="MACD histogram")
    bollinger_upper: Optional[Decimal] = Field(None, description="Bollinger upper band")
    bollinger_lower: Optional[Decimal] = Field(None, description="Bollinger lower band")
    atr_14: Optional[Decimal] = Field(None, description="14-day ATR")
    volume: Optional[Decimal] = Field(None, description="Current volume")
    volatility: Optional[Decimal] = Field(None, description="Annualized volatility")
    support_levels: List[Decimal] = Field(default_factory=list, description="Support levels")
    resistance_levels: List[Decimal] = Field(default_factory=list, description="Resistance levels")


class RiskAssessment(BaseModel):
    """Risk analysis and assessment."""
    
    symbol: str = Field(..., description="Stock ticker symbol")
    overall_risk_level: RiskLevel = Field(..., description="Overall risk assessment")
    
    # Risk Categories
    regulatory_risks: List[str] = Field(default_factory=list, description="Regulatory risks")
    business_risks: List[str] = Field(default_factory=list, description="Business risks")
    competitive_risks: List[str] = Field(default_factory=list, description="Competitive risks")
    financial_risks: List[str] = Field(default_factory=list, description="Financial risks")
    operational_risks: List[str] = Field(default_factory=list, description="Operational risks")
    
    # Growth Catalysts
    growth_catalysts: List[str] = Field(default_factory=list, description="Growth catalysts")
    
    # Risk Metrics
    value_at_risk: Optional[Decimal] = Field(None, description="Value at Risk (95%)")
    beta: Optional[Decimal] = Field(None, description="Market beta")
    correlation_spy: Optional[Decimal] = Field(None, description="Correlation with SPY")
    max_drawdown: Optional[Decimal] = Field(None, description="Maximum drawdown")


class InvestmentThesis(BaseModel):
    """Investment thesis and recommendation."""
    
    symbol: str = Field(..., description="Stock ticker symbol")
    rating: InvestmentRating = Field(..., description="Investment rating")
    price_target: Decimal = Field(..., description="12-month price target")
    bull_case_target: Decimal = Field(..., description="Bull case price target")
    bear_case_target: Decimal = Field(..., description="Bear case price target")
    
    # Thesis Components
    investment_rationale: str = Field(..., description="Investment rationale")
    bull_case_points: List[str] = Field(default_factory=list, description="Bull case arguments")
    bear_case_points: List[str] = Field(default_factory=list, description="Bear case arguments")
    key_catalysts: List[str] = Field(default_factory=list, description="Key catalysts")
    key_risks: List[str] = Field(default_factory=list, description="Key risks")
    
    # Time Horizons
    short_term_outlook: str = Field(..., description="3-6 month outlook")
    medium_term_outlook: str = Field(..., description="6-18 month outlook")
    long_term_outlook: str = Field(..., description="2-5 year outlook")
    
    # Monitoring Points
    monitoring_points: List[str] = Field(default_factory=list, description="Key monitoring points")
    
    # Portfolio Recommendations
    portfolio_allocation: str = Field(..., description="Recommended allocation")
    risk_level: RiskLevel = Field(..., description="Risk level")
    time_horizon: str = Field(..., description="Investment time horizon")
    portfolio_fit: str = Field(..., description="Portfolio fit description")


class ComprehensiveAnalysis(BaseModel):
    """Complete comprehensive stock analysis."""
    
    # Core Components
    company_profile: CompanyProfile
    market_data: MarketData
    financial_ratios: FinancialRatios
    technical_analysis: TechnicalAnalysis
    risk_assessment: RiskAssessment
    investment_thesis: InvestmentThesis
    
    # Additional Data
    income_statements: List[IncomeStatement] = Field(default_factory=list)
    balance_sheets: List[BalanceSheet] = Field(default_factory=list)
    cash_flow_statements: List[CashFlowStatement] = Field(default_factory=list)
    
    # Analysis Metadata
    analysis_date: datetime = Field(..., description="Analysis generation date")
    analyst: Optional[str] = Field(None, description="Analyst name")
    version: str = Field(default="1.0", description="Analysis version")
    
    class Config:
        """Pydantic configuration."""
        
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }