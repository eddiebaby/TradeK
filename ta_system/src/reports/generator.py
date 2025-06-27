"""Professional report generation system for comprehensive stock analysis."""

import os
import re
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional

from ..fundamental.models import (
    ComprehensiveAnalysis,
    InvestmentRating,
    RiskLevel,
)


class ReportGenerator:
    """Generates professional stock analysis reports from templates."""

    def __init__(self, template_dir: Optional[str] = None):
        """Initialize report generator with template directory."""
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), "../../templates")
        self.template_dir = Path(template_dir)
        self.template_cache: Dict[str, str] = {}

    def load_template(self, template_name: str) -> str:
        """Load and cache report template."""
        if template_name not in self.template_cache:
            template_path = self.template_dir / template_name
            if not template_path.exists():
                raise FileNotFoundError(f"Template not found: {template_path}")
            
            with open(template_path, encoding="utf-8") as f:
                self.template_cache[template_name] = f.read()
        
        return self.template_cache[template_name]

    def select_template(self, analysis: ComprehensiveAnalysis) -> str:
        """Select appropriate template based on asset type."""
        symbol = analysis.company_profile.symbol.upper()
        
        # ETF symbols detection
        etf_symbols = {
            "IWM", "SPY", "QQQ", "VTI", "EFA", "GLD", "TLT", "EEM", "VEA", "IEFA",
            "AGG", "BND", "VWO", "IEMG", "IJH", "IJR", "MDY", "SLY", "VB", "VBR",
            "VTV", "VUG", "VYM", "SCHD", "SCHA", "SCHB", "SCHF", "SCHG", "SCHM", 
            "SCHV", "SCHX", "SCHY", "SCHZ", "XLF", "XLE", "XLI", "XLK", "XLV",
            "XLY", "XLP", "XLU", "XLRE", "XME", "XHB", "SMH", "KRE", "IYR"
        }
        
        if symbol in etf_symbols:
            return "etf_analysis_template.md"
        return "comprehensive_stock_analysis_template.md"

    def generate_comprehensive_report(
        self,
        analysis: ComprehensiveAnalysis,
        template_name: Optional[str] = None
    ) -> str:
        """Generate comprehensive stock analysis report."""
        
        # Auto-select template if not provided
        if template_name is None:
            template_name = self.select_template(analysis)
        
        template = self.load_template(template_name)
        
        # Prepare template variables based on asset type
        if "etf" in template_name.lower():
            variables = self._prepare_etf_variables(analysis)
        else:
            variables = self._prepare_template_variables(analysis)
        
        # Replace all template placeholders
        report = self._replace_template_variables(template, variables)
        
        return report

    def _prepare_template_variables(self, analysis: ComprehensiveAnalysis) -> Dict[str, str]:
        """Prepare all template variables from analysis data."""
        
        # Helper function to format currency
        def format_currency(amount: Optional[Decimal], show_billions: bool = True) -> str:
            if amount is None:
                return "N/A"
            
            amount_float = float(amount)
            if show_billions and abs(amount_float) >= 1_000_000_000:
                return f"${amount_float / 1_000_000_000:.1f}B"
            if abs(amount_float) >= 1_000_000:
                return f"${amount_float / 1_000_000:.1f}M"
            if abs(amount_float) >= 1_000:
                return f"${amount_float / 1_000:.1f}K"
            return f"${amount_float:.2f}"

        # Helper function to format percentage
        def format_percentage(value: Optional[Decimal], decimals: int = 1) -> str:
            if value is None:
                return "N/A"
            return f"{float(value):.{decimals}f}%"

        # Helper function to format number
        def format_number(value: Optional[Decimal], decimals: int = 2) -> str:
            if value is None:
                return "N/A"
            return f"{float(value):.{decimals}f}"

        # Get the latest financial data
        latest_income = analysis.income_statements[-1] if analysis.income_statements else None
        latest_balance = analysis.balance_sheets[-1] if analysis.balance_sheets else None
        latest_cash_flow = analysis.cash_flow_statements[-1] if analysis.cash_flow_statements else None

        variables = {
            # Company Information
            "COMPANY_NAME": analysis.company_profile.company_name,
            "SYMBOL": analysis.company_profile.symbol.upper(),
            "COMPANY_DESCRIPTION": analysis.company_profile.description,
            
            # Financial Performance
            "LATEST_QUARTER": self._get_latest_quarter(latest_income),
            "REVENUE": format_currency(latest_income.revenue if latest_income else None),
            "REVENUE_GROWTH": format_percentage(self._calculate_revenue_growth(analysis)),
            "OPERATING_INCOME": format_currency(latest_income.operating_income if latest_income else None),
            "OPERATING_INCOME_GROWTH": format_percentage(self._calculate_operating_income_growth(analysis)),
            "NET_INCOME": format_currency(latest_income.net_income if latest_income else None),
            "EPS": format_number(latest_income.eps_diluted if latest_income else None),
            "FREE_CASH_FLOW": format_currency(latest_cash_flow.free_cash_flow if latest_cash_flow else None),
            
            # Market Data
            "MARKET_CAP": format_currency(analysis.market_data.market_cap, show_billions=True),
            "CURRENT_PRICE": format_currency(analysis.market_data.current_price, show_billions=False),
            "YTD_RETURN": format_percentage(analysis.market_data.ytd_return),
            "WEEK_52_LOW": format_currency(analysis.market_data.week_52_low, show_billions=False),
            "WEEK_52_HIGH": format_currency(analysis.market_data.week_52_high, show_billions=False),
            
            # Financial Ratios
            "PE_RATIO": format_number(analysis.financial_ratios.price_to_earnings),
            "FORWARD_PE": format_number(analysis.financial_ratios.price_to_earnings),  # Placeholder
            "PEG_RATIO": format_number(analysis.financial_ratios.peg_ratio),
            "PRICE_SALES": format_number(analysis.financial_ratios.price_to_sales),
            "PRICE_BOOK": format_number(analysis.financial_ratios.price_to_book),
            "EV_REVENUE": format_number(analysis.financial_ratios.enterprise_value_to_revenue),
            "ROE": format_percentage(analysis.financial_ratios.return_on_equity),
            "PROFIT_MARGIN": format_percentage(analysis.financial_ratios.net_margin),
            "OPERATING_MARGIN": format_percentage(analysis.financial_ratios.operating_margin),
            
            # Technical Analysis
            "RSI_14": format_number(analysis.technical_analysis.rsi_14),
            "RSI_SIGNAL": self._get_rsi_signal(analysis.technical_analysis.rsi_14),
            "RSI_CONDITION": self._get_rsi_condition(analysis.technical_analysis.rsi_14),
            "SMA_20": format_currency(analysis.technical_analysis.sma_20, show_billions=False),
            "SMA_50": format_currency(analysis.technical_analysis.sma_50, show_billions=False),
            "SMA_200": format_currency(analysis.technical_analysis.sma_200, show_billions=False),
            "PRICE_VS_SMA_20": self._calculate_price_vs_sma(analysis.technical_analysis.current_price, analysis.technical_analysis.sma_20),
            "PRICE_VS_SMA_200": self._calculate_price_vs_sma(analysis.technical_analysis.current_price, analysis.technical_analysis.sma_200),
            "VOLATILITY": format_percentage(analysis.technical_analysis.volatility),
            "ATR_14": format_currency(analysis.technical_analysis.atr_14, show_billions=False),
            "VOLUME": f"{float(analysis.technical_analysis.volume or 0):,.0f}",
            "RESISTANCE_1": format_currency(analysis.technical_analysis.resistance_levels[0] if analysis.technical_analysis.resistance_levels else None, show_billions=False),
            "RESISTANCE_2": format_currency(analysis.technical_analysis.resistance_levels[1] if len(analysis.technical_analysis.resistance_levels) > 1 else None, show_billions=False),
            "SUPPORT_1": format_currency(analysis.technical_analysis.support_levels[0] if analysis.technical_analysis.support_levels else None, show_billions=False),
            "SUPPORT_2": format_currency(analysis.technical_analysis.support_levels[1] if len(analysis.technical_analysis.support_levels) > 1 else None, show_billions=False),
            "BB_UPPER": format_currency(analysis.technical_analysis.bollinger_upper, show_billions=False),
            "BB_LOWER": format_currency(analysis.technical_analysis.bollinger_lower, show_billions=False),
            "MACD_VALUE": format_number(analysis.technical_analysis.macd),
            "MACD_SIGNAL": self._get_macd_signal(analysis.technical_analysis.macd, analysis.technical_analysis.macd_signal),
            
            # Investment Thesis
            "INVESTMENT_RATING": analysis.investment_thesis.rating.value,
            "RATING_EMOJI": self._get_rating_emoji(analysis.investment_thesis.rating),
            "INVESTMENT_RATIONALE": analysis.investment_thesis.investment_rationale,
            
            # Price Targets
            "BULL_TARGET_LOW": format_currency(analysis.investment_thesis.bull_case_target * Decimal("0.95"), show_billions=False),
            "BULL_TARGET_HIGH": format_currency(analysis.investment_thesis.bull_case_target, show_billions=False),
            "BASE_TARGET_LOW": format_currency(analysis.investment_thesis.price_target * Decimal("0.9"), show_billions=False),
            "BASE_TARGET_HIGH": format_currency(analysis.investment_thesis.price_target, show_billions=False),
            "BEAR_TARGET_LOW": format_currency(analysis.investment_thesis.bear_case_target, show_billions=False),
            "BEAR_TARGET_HIGH": format_currency(analysis.investment_thesis.bear_case_target * Decimal("1.05"), show_billions=False),
            
            # Risk Assessment
            "RISK_LEVEL": analysis.risk_assessment.overall_risk_level.value,
            "RISK_DESCRIPTION": self._get_risk_description(analysis.risk_assessment.overall_risk_level),
            
            # Time Horizons
            "SHORT_TERM_PERIOD": "3-6",
            "MEDIUM_TERM_PERIOD": "6-18",
            "SHORT_TERM_FOCUS": self._extract_first_sentence(analysis.investment_thesis.short_term_outlook),
            "MEDIUM_TERM_DRIVERS": self._extract_first_sentence(analysis.investment_thesis.medium_term_outlook),
            
            # Additional fields
            "TIME_HORIZON": analysis.investment_thesis.time_horizon,
            "PORTFOLIO_FIT": analysis.investment_thesis.portfolio_fit,
        }
        
        # Add strategic developments and other list-based content
        variables.update(self._prepare_list_variables(analysis))
        
        return variables

    def _prepare_etf_variables(self, analysis: ComprehensiveAnalysis) -> Dict[str, str]:
        """Prepare ETF-specific template variables from analysis data."""
        
        # Helper function to format currency
        def format_currency(amount: Optional[Decimal], show_billions: bool = True) -> str:
            if amount is None:
                return "N/A"
            
            amount_float = float(amount)
            if show_billions and abs(amount_float) >= 1_000_000_000:
                return f"${amount_float / 1_000_000_000:.1f}B"
            if abs(amount_float) >= 1_000_000:
                return f"${amount_float / 1_000_000:.1f}M"
            if abs(amount_float) >= 1_000:
                return f"${amount_float / 1_000:.1f}K"
            return f"${amount_float:.2f}"

        # Helper function to format percentage
        def format_percentage(value: Optional[Decimal], decimals: int = 1) -> str:
            if value is None:
                return "N/A"
            return f"{float(value):.{decimals}f}%"

        # Helper function to format number
        def format_number(value: Optional[Decimal], decimals: int = 2) -> str:
            if value is None:
                return "N/A"
            return f"{float(value):.{decimals}f}"

        # ETF-specific variables
        variables = {
            # Company Information
            "COMPANY_NAME": analysis.company_profile.company_name,
            "SYMBOL": analysis.company_profile.symbol.upper(),
            "COMPANY_DESCRIPTION": analysis.company_profile.description,
            
            # ETF Specific Metrics
            "AUM": format_currency(analysis.market_data.market_cap, show_billions=True),
            "EXPENSE_RATIO": "0.19",  # IWM typical expense ratio
            "DIVIDEND_YIELD": "1.7",  # IWM typical dividend yield  
            "TRACKING_ERROR": "0.25",  # Typical small-cap ETF tracking error
            "HOLDINGS_COUNT": "~2000",  # Russell 2000 holdings count
            "TOP_10_WEIGHT": "11.5",  # Typical top 10 concentration
            "DISTRIBUTION_FREQUENCY": "Quarterly",
            "INVESTMENT_CATEGORY": "small-cap",
            
            # Market Data
            "MARKET_CAP": format_currency(analysis.market_data.market_cap, show_billions=True),
            "CURRENT_PRICE": format_currency(analysis.market_data.current_price, show_billions=False),
            "YTD_RETURN": format_percentage(analysis.market_data.ytd_return),
            "WEEK_52_LOW": format_currency(analysis.market_data.week_52_low, show_billions=False),
            "WEEK_52_HIGH": format_currency(analysis.market_data.week_52_high, show_billions=False),
            
            # Financial Ratios
            "PE_RATIO": format_number(analysis.financial_ratios.price_to_earnings),
            "PRICE_SALES": format_number(analysis.financial_ratios.price_to_sales),
            "PRICE_BOOK": format_number(analysis.financial_ratios.price_to_book),
            
            # Technical Analysis
            "RSI_14": format_number(analysis.technical_analysis.rsi_14),
            "RSI_SIGNAL": self._get_rsi_signal(analysis.technical_analysis.rsi_14),
            "RSI_CONDITION": self._get_rsi_condition(analysis.technical_analysis.rsi_14),
            "SMA_20": format_currency(analysis.technical_analysis.sma_20, show_billions=False),
            "SMA_50": format_currency(analysis.technical_analysis.sma_50, show_billions=False),
            "SMA_200": format_currency(analysis.technical_analysis.sma_200, show_billions=False),
            "PRICE_VS_SMA_20": self._calculate_price_vs_sma(analysis.technical_analysis.current_price, analysis.technical_analysis.sma_20),
            "PRICE_VS_SMA_200": self._calculate_price_vs_sma(analysis.technical_analysis.current_price, analysis.technical_analysis.sma_200),
            "VOLATILITY": format_percentage(analysis.technical_analysis.volatility),
            "ATR_14": format_currency(analysis.technical_analysis.atr_14, show_billions=False),
            "VOLUME": f"{float(analysis.technical_analysis.volume or 0):,.0f}",
            "RESISTANCE_1": format_currency(analysis.technical_analysis.resistance_levels[0] if analysis.technical_analysis.resistance_levels else None, show_billions=False),
            "RESISTANCE_2": format_currency(analysis.technical_analysis.resistance_levels[1] if len(analysis.technical_analysis.resistance_levels) > 1 else None, show_billions=False),
            "SUPPORT_1": format_currency(analysis.technical_analysis.support_levels[0] if analysis.technical_analysis.support_levels else None, show_billions=False),
            "SUPPORT_2": format_currency(analysis.technical_analysis.support_levels[1] if len(analysis.technical_analysis.support_levels) > 1 else None, show_billions=False),
            "BB_UPPER": format_currency(analysis.technical_analysis.bollinger_upper, show_billions=False),
            "BB_LOWER": format_currency(analysis.technical_analysis.bollinger_lower, show_billions=False),
            "MACD_VALUE": format_number(analysis.technical_analysis.macd),
            "MACD_SIGNAL": self._get_macd_signal(analysis.technical_analysis.macd, analysis.technical_analysis.macd_signal),
            
            # Investment Thesis
            "INVESTMENT_RATING": analysis.investment_thesis.rating.value,
            "RATING_EMOJI": self._get_rating_emoji(analysis.investment_thesis.rating),
            "INVESTMENT_RATIONALE": analysis.investment_thesis.investment_rationale,
            
            # Price Targets
            "BULL_TARGET_LOW": format_currency(analysis.investment_thesis.bull_case_target * Decimal("0.95"), show_billions=False),
            "BULL_TARGET_HIGH": format_currency(analysis.investment_thesis.bull_case_target, show_billions=False),
            "BASE_TARGET_LOW": format_currency(analysis.investment_thesis.price_target * Decimal("0.9"), show_billions=False),
            "BASE_TARGET_HIGH": format_currency(analysis.investment_thesis.price_target, show_billions=False),
            "BEAR_TARGET_LOW": format_currency(analysis.investment_thesis.bear_case_target, show_billions=False),
            "BEAR_TARGET_HIGH": format_currency(analysis.investment_thesis.bear_case_target * Decimal("1.05"), show_billions=False),
            
            # Risk Assessment
            "RISK_LEVEL": analysis.risk_assessment.overall_risk_level.value,
            "RISK_DESCRIPTION": self._get_risk_description(analysis.risk_assessment.overall_risk_level),
            
            # Time Horizons
            "SHORT_TERM_PERIOD": "3-6",
            "MEDIUM_TERM_PERIOD": "6-18",
            "SHORT_TERM_FOCUS": self._extract_first_sentence(analysis.investment_thesis.short_term_outlook),
            "MEDIUM_TERM_DRIVERS": self._extract_first_sentence(analysis.investment_thesis.medium_term_outlook),
            
            # Additional fields
            "TIME_HORIZON": analysis.investment_thesis.time_horizon,
            "PORTFOLIO_FIT": analysis.investment_thesis.portfolio_fit,
            
            # ETF Holdings placeholders (would be enhanced with real data)
            "TOP_HOLDING_1": "Apple Inc",
            "TOP_HOLDING_1_WEIGHT": "2.1",
            "TOP_HOLDING_2": "Tesla Inc", 
            "TOP_HOLDING_2_WEIGHT": "1.8",
            "TOP_HOLDING_3": "Nvidia Corp",
            "TOP_HOLDING_3_WEIGHT": "1.5",
            "TOP_HOLDING_4": "Microsoft Corp",
            "TOP_HOLDING_4_WEIGHT": "1.4",
            "TOP_HOLDING_5": "Amazon.com Inc",
            "TOP_HOLDING_5_WEIGHT": "1.3",
            
            # Sector allocation placeholders
            "SECTOR_1": "Technology",
            "SECTOR_1_WEIGHT": "16.2",
            "SECTOR_2": "Healthcare",
            "SECTOR_2_WEIGHT": "14.8",
            "SECTOR_3": "Financial Services",
            "SECTOR_3_WEIGHT": "13.5",
            "SECTOR_4": "Industrials",
            "SECTOR_4_WEIGHT": "12.1",
            "SECTOR_5": "Consumer Discretionary",
            "SECTOR_5_WEIGHT": "11.8",
        }
        
        # Add ETF-specific list variables
        variables.update(self._prepare_etf_list_variables(analysis))
        
        return variables

    def _prepare_etf_list_variables(self, analysis: ComprehensiveAnalysis) -> Dict[str, str]:
        """Prepare ETF-specific list variables."""
        variables = {}
        
        # Bull case points
        for i, point in enumerate(analysis.investment_thesis.bull_case_points[:7], 1):
            variables[f"BULL_POINT_{i}"] = self._extract_title(point)
            variables[f"BULL_DESC_{i}"] = self._extract_description(point)
        
        # Bear case points
        for i, point in enumerate(analysis.investment_thesis.bear_case_points[:6], 1):
            variables[f"BEAR_POINT_{i}"] = self._extract_title(point)
            variables[f"BEAR_DESC_{i}"] = self._extract_description(point)
        
        # Risks
        for i, risk in enumerate(analysis.risk_assessment.regulatory_risks[:5], 1):
            variables[f"RISK_{i}"] = self._extract_title(risk)
            variables[f"RISK_{i}_DESC"] = self._extract_description(risk)
        
        for i, risk in enumerate(analysis.risk_assessment.business_risks[:6], 1):
            variables[f"BUSINESS_RISK_{i}"] = self._extract_title(risk)
            variables[f"BUSINESS_RISK_{i}_DESC"] = self._extract_description(risk)
        
        # Growth catalysts
        for i, catalyst in enumerate(analysis.risk_assessment.growth_catalysts[:6], 1):
            variables[f"CATALYST_{i}"] = self._extract_title(catalyst)
            variables[f"CATALYST_{i}_DESC"] = self._extract_description(catalyst)
        
        # Monitoring points
        for i, point in enumerate(analysis.investment_thesis.monitoring_points[:10], 1):
            variables[f"MONITOR_POINT_{i}"] = point
        
        return variables

    def _prepare_list_variables(self, analysis: ComprehensiveAnalysis) -> Dict[str, str]:
        """Prepare list-based template variables."""
        variables = {}
        
        # Bull case points
        for i, point in enumerate(analysis.investment_thesis.bull_case_points[:7], 1):
            variables[f"BULL_POINT_{i}"] = self._extract_title(point)
            variables[f"BULL_DESC_{i}"] = self._extract_description(point)
        
        # Bear case points
        for i, point in enumerate(analysis.investment_thesis.bear_case_points[:6], 1):
            variables[f"BEAR_POINT_{i}"] = self._extract_title(point)
            variables[f"BEAR_DESC_{i}"] = self._extract_description(point)
        
        # Risk categories
        variables["RISK_CATEGORY_1"] = "REGULATORY & ANTITRUST RISKS"
        variables["RISK_CATEGORY_2"] = "BUSINESS & COMPETITIVE RISKS"
        
        # Risks
        for i, risk in enumerate(analysis.risk_assessment.regulatory_risks[:5], 1):
            variables[f"RISK_{i}"] = self._extract_title(risk)
            variables[f"RISK_{i}_DESC"] = self._extract_description(risk)
        
        for i, risk in enumerate(analysis.risk_assessment.business_risks[:6], 1):
            variables[f"BUSINESS_RISK_{i}"] = self._extract_title(risk)
            variables[f"BUSINESS_RISK_{i}_DESC"] = self._extract_description(risk)
        
        # Growth catalysts
        for i, catalyst in enumerate(analysis.risk_assessment.growth_catalysts[:6], 1):
            variables[f"CATALYST_{i}"] = self._extract_title(catalyst)
            variables[f"CATALYST_{i}_DESC"] = self._extract_description(catalyst)
        
        # Monitoring points
        for i, point in enumerate(analysis.investment_thesis.monitoring_points[:10], 1):
            variables[f"MONITOR_POINT_{i}"] = point
        
        return variables

    def _replace_template_variables(self, template: str, variables: Dict[str, str]) -> str:
        """Replace template placeholders with actual values."""
        result = template
        
        # Replace all {VARIABLE_NAME} placeholders
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, str(value))
        
        # Clean up any remaining unreplaced placeholders with intelligent handling
        remaining_placeholders = re.findall(r"\{[A-Z_0-9]+\}", result)
        if remaining_placeholders:
            # Log the missing placeholders for debugging
            print(f"Warning: Missing template variables: {remaining_placeholders}")
            # Replace with empty strings or context-appropriate values instead of N/A
            result = re.sub(r"\{[A-Z_0-9]+\}", "", result)
        
        return result

    def _get_latest_quarter(self, income_statement) -> str:
        """Get formatted latest quarter string."""
        if not income_statement:
            return "Latest Quarter"
        
        if income_statement.fiscal_quarter:
            return f"Q{income_statement.fiscal_quarter} {income_statement.fiscal_year}"
        return f"FY {income_statement.fiscal_year}"

    def _calculate_revenue_growth(self, analysis: ComprehensiveAnalysis) -> Optional[Decimal]:
        """Calculate revenue growth rate."""
        if len(analysis.income_statements) < 2:
            return None
        
        current = analysis.income_statements[-1]
        previous = analysis.income_statements[-2]
        
        if previous.revenue == 0:
            return None
        
        return ((current.revenue - previous.revenue) / previous.revenue) * 100

    def _calculate_operating_income_growth(self, analysis: ComprehensiveAnalysis) -> Optional[Decimal]:
        """Calculate operating income growth rate."""
        if len(analysis.income_statements) < 2:
            return None
        
        current = analysis.income_statements[-1]
        previous = analysis.income_statements[-2]
        
        if not current.operating_income or not previous.operating_income or previous.operating_income == 0:
            return None
        
        return ((current.operating_income - previous.operating_income) / previous.operating_income) * 100

    def _get_rsi_signal(self, rsi: Optional[Decimal]) -> str:
        """Get RSI signal emoji."""
        if not rsi:
            return "🟡"
        
        if rsi > 70:
            return "🔴"
        if rsi < 30:
            return "🟢"
        return "🟡"

    def _get_rsi_condition(self, rsi: Optional[Decimal]) -> str:
        """Get RSI condition text."""
        if not rsi:
            return "NEUTRAL"
        
        if rsi > 70:
            return "OVERBOUGHT"
        if rsi < 30:
            return "OVERSOLD"
        return "NEUTRAL"

    def _calculate_price_vs_sma(self, current_price: Decimal, sma: Optional[Decimal]) -> str:
        """Calculate price vs SMA percentage."""
        if not sma or sma == 0:
            return "N/A"
        
        percentage = ((current_price - sma) / sma) * 100
        return f"{float(percentage):+.1f}"

    def _get_macd_signal(self, macd: Optional[Decimal], signal: Optional[Decimal]) -> str:
        """Get MACD signal interpretation."""
        if not macd or not signal:
            return "NEUTRAL"
        
        if macd > signal:
            return "BULLISH"
        return "BEARISH"

    def _get_rating_emoji(self, rating: InvestmentRating) -> str:
        """Get emoji for investment rating."""
        emoji_map = {
            InvestmentRating.STRONG_BUY: "🚀",
            InvestmentRating.BUY: "📈",
            InvestmentRating.HOLD: "🔄",
            InvestmentRating.SELL: "📉",
            InvestmentRating.STRONG_SELL: "🚨",
        }
        return emoji_map.get(rating, "🔄")

    def _get_risk_description(self, risk_level: RiskLevel) -> str:
        """Get risk level description."""
        descriptions = {
            RiskLevel.LOW: "large cap dividend stock with stable fundamentals",
            RiskLevel.MODERATE: "established company with predictable business model",
            RiskLevel.MODERATE_HIGH: "large cap growth with regulatory overhang",
            RiskLevel.HIGH: "high growth company with competitive risks",
            RiskLevel.VERY_HIGH: "speculative investment with significant volatility",
        }
        return descriptions.get(risk_level, "diversified investment")

    def _extract_title(self, text: str) -> str:
        """Extract title from text (first part before colon)."""
        return text.split(":")[0].strip() if ":" in text else text[:50] + "..."

    def _extract_description(self, text: str) -> str:
        """Extract description from text (part after colon)."""
        return text.split(":", 1)[1].strip() if ":" in text else text

    def _extract_first_sentence(self, text: str) -> str:
        """Extract first sentence from text."""
        sentences = text.split(".")
        return sentences[0].strip() + "." if sentences else text

    def generate_executive_summary(self, analysis: ComprehensiveAnalysis) -> str:
        """Generate executive summary section."""
        summary_parts = [
            f"{analysis.company_profile.company_name} ({analysis.company_profile.symbol}) - {analysis.investment_thesis.rating.value}",
            f"Current Price: ${analysis.market_data.current_price:.2f}",
            f"12M Target: ${analysis.investment_thesis.price_target:.2f}",
            f"Risk Level: {analysis.risk_assessment.overall_risk_level.value}",
        ]
        
        return " | ".join(summary_parts)

    def export_to_html(self, markdown_content: str) -> str:
        """Convert markdown report to HTML format."""
        try:
            import markdown
            html = markdown.markdown(markdown_content, extensions=["tables", "fenced_code"])
            return self._wrap_html(html)
        except ImportError:
            # Fallback: simple markdown-to-HTML conversion
            return self._simple_markdown_to_html(markdown_content)

    def _wrap_html(self, content: str) -> str:
        """Wrap HTML content with proper document structure."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Stock Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .emoji {{ font-size: 1.2em; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 4px; }}
    </style>
</head>
<body>
{content}
</body>
</html>
        """

    def _simple_markdown_to_html(self, markdown_content: str) -> str:
        """Simple markdown to HTML conversion."""
        html = markdown_content
        
        # Convert headers
        html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        
        # Convert line breaks
        html = html.replace("\n\n", "</p><p>")
        html = f"<p>{html}</p>"
        
        return self._wrap_html(html)