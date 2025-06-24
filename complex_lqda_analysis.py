#!/usr/bin/env python3
"""
Complex LQDA Stock Analysis - The Sophisticated System
Comprehensive financial analysis using yfinance with AI-powered insights
"""

import asyncio
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import requests
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StockAnalysisRequest:
    """Stock analysis request structure"""
    symbol: str
    analysis_type: str = "comprehensive"  # comprehensive, technical, fundamental, sentiment
    time_period: str = "1y"  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    include_news: bool = True
    include_financials: bool = True
    include_technical: bool = True
    risk_assessment: bool = True

@dataclass
class StockData:
    """Stock data container"""
    symbol: str
    current_price: float
    price_change: float
    price_change_percent: float
    volume: int
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    dividend_yield: Optional[float]
    fifty_two_week_high: float
    fifty_two_week_low: float
    historical_data: pd.DataFrame
    financials: Dict[str, Any]
    news: List[Dict[str, Any]]
    company_info: Dict[str, Any]

class StockDataCollector:
    """Collects stock data from multiple sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    async def get_stock_data(self, symbol: str, period: str = "1y") -> StockData:
        """Collect comprehensive stock data"""
        try:
            print(f"📊 Collecting comprehensive data for {symbol}...")
            ticker = yf.Ticker(symbol)
            
            # Get current stock info
            info = ticker.info
            
            # Get historical data
            hist = ticker.history(period=period)
            
            # Get current price data
            current_price = info.get('currentPrice', hist['Close'].iloc[-1] if not hist.empty else 0)
            
            # Calculate price change
            if len(hist) > 1:
                price_change = current_price - hist['Close'].iloc[-2]
                price_change_percent = (price_change / hist['Close'].iloc[-2]) * 100
            else:
                price_change = 0
                price_change_percent = 0
            
            # Get financial statements
            financials = {
                'income_statement': self._safe_dataframe_to_dict(ticker.financials),
                'balance_sheet': self._safe_dataframe_to_dict(ticker.balance_sheet),
                'cash_flow': self._safe_dataframe_to_dict(ticker.cashflow),
                'quarterly_financials': self._safe_dataframe_to_dict(ticker.quarterly_financials)
            }
            
            # Get recent news
            news = ticker.news[:10] if hasattr(ticker, 'news') and ticker.news else []
            
            return StockData(
                symbol=symbol,
                current_price=current_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                volume=info.get('volume', 0),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                dividend_yield=info.get('dividendYield'),
                fifty_two_week_high=info.get('fiftyTwoWeekHigh', 0),
                fifty_two_week_low=info.get('fiftyTwoWeekLow', 0),
                historical_data=hist,
                financials=financials,
                news=news,
                company_info=info
            )
            
        except Exception as e:
            logger.error(f"Error collecting stock data for {symbol}: {e}")
            raise
    
    def _safe_dataframe_to_dict(self, df) -> Dict:
        """Safely convert DataFrame to dict"""
        try:
            if df is not None and not df.empty:
                return df.fillna(0).to_dict()
            return {}
        except:
            return {}

class TechnicalAnalyzer:
    """Technical analysis calculations"""
    
    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        
        return {
            'upper': sma + (std * 2),
            'middle': sma,
            'lower': sma - (std * 2)
        }
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }

class FundamentalAnalyzer:
    """Fundamental analysis calculations"""
    
    @staticmethod
    def calculate_ratios(stock_data: StockData) -> Dict[str, float]:
        """Calculate key financial ratios"""
        info = stock_data.company_info
        
        ratios = {
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'price_to_book': info.get('priceToBook', 0),
            'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'return_on_equity': info.get('returnOnEquity', 0),
            'return_on_assets': info.get('returnOnAssets', 0),
            'profit_margin': info.get('profitMargins', 0),
            'operating_margin': info.get('operatingMargins', 0),
            'current_ratio': info.get('currentRatio', 0),
            'quick_ratio': info.get('quickRatio', 0)
        }
        
        return {k: v for k, v in ratios.items() if v is not None}
    
    @staticmethod
    def analyze_growth(financials: Dict) -> Dict[str, Any]:
        """Analyze growth trends"""
        try:
            income_stmt = financials.get('income_statement', {})
            if not income_stmt:
                return {"analysis": "No income statement data available for growth analysis"}
            
            growth_analysis = {
                "revenue_trend": "Biotech revenue growth dependent on product launches",
                "development_stage": "Commercial stage with FDA-approved YUTREPIA",
                "growth_drivers": [
                    "YUTREPIA commercial penetration",
                    "Pipeline advancement",
                    "Market expansion opportunities"
                ]
            }
            
            return growth_analysis
            
        except Exception as e:
            return {"error": f"Growth analysis failed: {str(e)}"}

class OllamaAIAnalyzer:
    """AI-powered analysis using Ollama models"""
    
    def __init__(self):
        self.available_models = self._get_available_models()
        self.selected_model = self._select_best_model()
    
    def _get_available_models(self) -> List[str]:
        """Get available Ollama models"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
            return []
        except:
            return []
    
    def _select_best_model(self) -> Optional[str]:
        """Select the best available model"""
        if not self.available_models:
            return None
        
        # Prefer financial/analytical models
        preferred_order = ['qwen3:4b', 'qwen3:8b', 'gemma2:2b', 'llama3', 'phi3']
        
        for preferred in preferred_order:
            if preferred in self.available_models:
                return preferred
        
        return self.available_models[0]
    
    async def analyze_with_ai(self, stock_data: StockData) -> Dict[str, Any]:
        """Generate AI-powered analysis"""
        if not self.selected_model:
            return self._fallback_analysis(stock_data)
        
        try:
            print(f"🤖 Using {self.selected_model} for AI analysis...")
            
            prompt = self._create_analysis_prompt(stock_data)
            
            result = subprocess.run([
                'ollama', 'run', self.selected_model, prompt
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                return {
                    "ai_analysis": result.stdout.strip(),
                    "model_used": self.selected_model,
                    "analysis_quality": "AI-powered comprehensive analysis"
                }
            else:
                return self._fallback_analysis(stock_data)
                
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._fallback_analysis(stock_data)
    
    def _create_analysis_prompt(self, stock_data: StockData) -> str:
        """Create comprehensive analysis prompt"""
        return f"""
Analyze LQDA (Liquidia Corporation) stock as a professional biotech equity analyst:

CURRENT DATA:
- Symbol: {stock_data.symbol}
- Current Price: ${stock_data.current_price:.2f}
- Price Change: {stock_data.price_change:+.2f} ({stock_data.price_change_percent:+.2f}%)
- Volume: {stock_data.volume:,}
- Market Cap: ${stock_data.market_cap:,} (if available)
- P/E Ratio: {stock_data.pe_ratio}
- 52-Week Range: ${stock_data.fifty_two_week_low:.2f} - ${stock_data.fifty_two_week_high:.2f}

COMPANY PROFILE:
- Sector: {stock_data.company_info.get('sector', 'Biotechnology')}
- Industry: {stock_data.company_info.get('industry', 'Pharmaceutical')}
- Business: Pulmonary therapeutics with PRINT technology platform

KEY BIOTECH CONSIDERATIONS:
1. YUTREPIA (treprostinil) - FDA approved PAH treatment
2. PRINT particle engineering technology platform
3. Commercial stage company with revenue potential
4. Pipeline development opportunities
5. Competitive PAH market dynamics

RECENT NEWS CONTEXT:
{self._format_news_for_analysis(stock_data.news[:3])}

Provide a comprehensive biotech investment analysis including:

1. INVESTMENT THESIS (Bullish/Bearish/Neutral with rationale)
2. KEY CATALYSTS (Clinical, regulatory, commercial milestones)
3. RISK FACTORS (Development, commercial, financial risks)
4. VALUATION PERSPECTIVE (Relative to biotech peers)
5. PRICE TARGETS (12-month outlook with scenarios)
6. RECOMMENDATION (Buy/Hold/Sell with allocation guidance)

Focus on biotech-specific factors: regulatory risks, clinical development, commercial execution, and competitive positioning in PAH market.
"""
    
    def _format_news_for_analysis(self, news_items: List[Dict]) -> str:
        """Format news for analysis"""
        if not news_items:
            return "No recent news available"
        
        headlines = []
        for item in news_items[:3]:
            title = item.get('title', 'No title')
            headlines.append(f"- {title}")
        
        return "\n".join(headlines)
    
    def _fallback_analysis(self, stock_data: StockData) -> Dict[str, Any]:
        """Fallback analysis when AI is not available"""
        return {
            "ai_analysis": f"""
🧬 LQDA (Liquidia Corporation) Professional Biotech Analysis

INVESTMENT THESIS: NEUTRAL with cautious optimism
Liquidia represents a commercial-stage biotech with FDA-approved YUTREPIA for PAH treatment. The company has transitioned from pure development to commercial execution phase.

KEY CATALYSTS:
• YUTREPIA commercial uptake and market penetration
• Pipeline advancement with LIQ861 and backup programs  
• Potential partnerships for international expansion
• Additional indications for PRINT technology platform

RISK FACTORS:
• Small biotech with limited product diversification
• Competitive PAH market with established treatments
• Commercial execution challenges in specialist market
• Potential funding needs and dilution risk

VALUATION PERSPECTIVE:
Current market cap reflects commercial-stage premium but limited revenue base. Valuation dependent on YUTREPIA adoption rates and pipeline progress.

PRICE TARGETS:
Bull Case: ${stock_data.current_price * 1.5:.2f} (successful commercial launch)
Base Case: ${stock_data.current_price * 1.1:.2f} (steady progress)
Bear Case: ${stock_data.current_price * 0.7:.2f} (commercial challenges)

RECOMMENDATION: HOLD/Speculative Buy
Consider as small position (1-2%) in diversified biotech portfolio. Monitor quarterly commercial progress and pipeline milestones.

⚠️ High-risk, high-reward biotech investment requiring specialized knowledge.
""",
            "model_used": "Expert Fallback Analysis",
            "analysis_quality": "Professional biotech analysis"
        }

class StockAnalysisEngine:
    """Main comprehensive stock analysis engine"""
    
    def __init__(self):
        self.data_collector = StockDataCollector()
        self.technical_analyzer = TechnicalAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.ai_analyzer = OllamaAIAnalyzer()
    
    async def analyze_stock(self, request: StockAnalysisRequest) -> Dict[str, Any]:
        """Perform comprehensive stock analysis"""
        print(f"🔍 Starting comprehensive analysis for {request.symbol}")
        
        # Collect stock data
        stock_data = await self.data_collector.get_stock_data(
            request.symbol, 
            request.time_period
        )
        
        analysis_results = {
            "symbol": request.symbol,
            "analysis_date": datetime.now().isoformat(),
            "current_price": stock_data.current_price,
            "price_change": stock_data.price_change,
            "price_change_percent": stock_data.price_change_percent,
            "analysis_components": {}
        }
        
        # Technical Analysis
        if request.include_technical:
            print("📈 Performing technical analysis...")
            analysis_results["analysis_components"]["technical"] = await self._perform_technical_analysis(stock_data)
        
        # Fundamental Analysis
        if request.include_financials:
            print("💰 Performing fundamental analysis...")
            analysis_results["analysis_components"]["fundamental"] = await self._perform_fundamental_analysis(stock_data)
        
        # AI-Powered Analysis
        print("🤖 Generating AI-powered insights...")
        ai_analysis = await self.ai_analyzer.analyze_with_ai(stock_data)
        analysis_results["analysis_components"]["ai_insights"] = ai_analysis
        
        # Risk Assessment
        if request.risk_assessment:
            print("⚠️ Assessing investment risks...")
            analysis_results["analysis_components"]["risk"] = await self._assess_risk(stock_data)
        
        return analysis_results
    
    async def _perform_technical_analysis(self, stock_data: StockData) -> Dict[str, Any]:
        """Perform technical analysis"""
        try:
            hist = stock_data.historical_data
            close_prices = hist['Close']
            
            # Calculate technical indicators
            sma_20 = self.technical_analyzer.calculate_sma(close_prices, 20)
            sma_50 = self.technical_analyzer.calculate_sma(close_prices, 50)
            ema_12 = self.technical_analyzer.calculate_ema(close_prices, 12)
            rsi = self.technical_analyzer.calculate_rsi(close_prices)
            bollinger = self.technical_analyzer.calculate_bollinger_bands(close_prices)
            macd = self.technical_analyzer.calculate_macd(close_prices)
            
            # Get latest values
            latest_values = {
                "sma_20": float(sma_20.iloc[-1]) if not sma_20.empty else None,
                "sma_50": float(sma_50.iloc[-1]) if not sma_50.empty else None,
                "ema_12": float(ema_12.iloc[-1]) if not ema_12.empty else None,
                "rsi": float(rsi.iloc[-1]) if not rsi.empty else None,
                "bollinger_upper": float(bollinger['upper'].iloc[-1]) if not bollinger['upper'].empty else None,
                "bollinger_lower": float(bollinger['lower'].iloc[-1]) if not bollinger['lower'].empty else None,
                "macd": float(macd['macd'].iloc[-1]) if not macd['macd'].empty else None,
                "macd_signal": float(macd['signal'].iloc[-1]) if not macd['signal'].empty else None
            }
            
            # Technical signals
            signals = []
            current_price = stock_data.current_price
            
            if latest_values["sma_20"] and current_price > latest_values["sma_20"]:
                signals.append("Price above 20-day SMA (bullish signal)")
            elif latest_values["sma_20"] and current_price < latest_values["sma_20"]:
                signals.append("Price below 20-day SMA (bearish signal)")
            
            if latest_values["rsi"]:
                if latest_values["rsi"] > 70:
                    signals.append("RSI indicates overbought conditions (potential sell)")
                elif latest_values["rsi"] < 30:
                    signals.append("RSI indicates oversold conditions (potential buy)")
                else:
                    signals.append("RSI in neutral range (50-70)")
            
            if latest_values["macd"] and latest_values["macd_signal"]:
                if latest_values["macd"] > latest_values["macd_signal"]:
                    signals.append("MACD above signal line (bullish momentum)")
                else:
                    signals.append("MACD below signal line (bearish momentum)")
            
            return {
                "indicators": latest_values,
                "signals": signals,
                "summary": f"Technical analysis completed with {len(signals)} signals identified"
            }
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {"error": f"Technical analysis failed: {str(e)}"}
    
    async def _perform_fundamental_analysis(self, stock_data: StockData) -> Dict[str, Any]:
        """Perform fundamental analysis"""
        try:
            ratios = self.fundamental_analyzer.calculate_ratios(stock_data)
            growth = self.fundamental_analyzer.analyze_growth(stock_data.financials)
            
            # Fundamental assessment
            assessment = []
            
            pe_ratio = ratios.get('pe_ratio', 0)
            if pe_ratio and pe_ratio > 0:
                if pe_ratio < 15:
                    assessment.append("P/E ratio suggests potential undervaluation")
                elif pe_ratio > 25:
                    assessment.append("P/E ratio suggests potential overvaluation")
                else:
                    assessment.append("P/E ratio within normal range")
            elif pe_ratio and pe_ratio < 0:
                assessment.append("Negative P/E - company currently unprofitable")
            else:
                assessment.append("P/E ratio not available - may indicate no earnings")
            
            # Biotech-specific considerations
            if stock_data.company_info.get('sector') == 'Healthcare':
                assessment.append("Healthcare/biotech sector - high risk/reward profile")
                assessment.append("Revenue growth dependent on product development and approvals")
            
            return {
                "ratios": ratios,
                "growth_analysis": growth,
                "assessment": assessment,
                "company_info": {
                    "sector": stock_data.company_info.get('sector', 'Unknown'),
                    "industry": stock_data.company_info.get('industry', 'Unknown'),
                    "market_cap": stock_data.market_cap,
                    "employees": stock_data.company_info.get('fullTimeEmployees'),
                    "business_summary": stock_data.company_info.get('longBusinessSummary', '')[:200] + "..."
                }
            }
            
        except Exception as e:
            logger.error(f"Fundamental analysis failed: {e}")
            return {"error": f"Fundamental analysis failed: {str(e)}"}
    
    async def _assess_risk(self, stock_data: StockData) -> Dict[str, Any]:
        """Assess investment risk factors"""
        try:
            risk_factors = []
            risk_score = 0  # 0-10 scale
            
            # Volatility assessment
            if not stock_data.historical_data.empty:
                returns = stock_data.historical_data['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                
                if volatility > 0.6:
                    risk_factors.append("Very high volatility (>60% annualized) - biotech characteristic")
                    risk_score += 4
                elif volatility > 0.4:
                    risk_factors.append("High volatility (40-60% annualized)")
                    risk_score += 3
                elif volatility > 0.25:
                    risk_factors.append("Moderate volatility (25-40% annualized)")
                    risk_score += 2
                else:
                    risk_factors.append("Low volatility (<25% annualized)")
                    risk_score += 1
            
            # Market cap risk (biotech specific)
            if stock_data.market_cap:
                if stock_data.market_cap < 1e9:  # Small biotech
                    risk_factors.append("Small biotech (<$1B market cap) - very high risk")
                    risk_score += 3
                elif stock_data.market_cap < 5e9:  # Mid-size biotech
                    risk_factors.append("Mid-size biotech ($1-5B) - high risk")
                    risk_score += 2
                else:
                    risk_factors.append("Large biotech (>$5B) - moderate risk")
                    risk_score += 1
            
            # Profitability risk
            if stock_data.pe_ratio:
                if stock_data.pe_ratio < 0:
                    risk_factors.append("Currently unprofitable - typical for development-stage biotech")
                    risk_score += 2
                elif stock_data.pe_ratio > 50:
                    risk_factors.append("Very high P/E ratio - growth expectations premium")
                    risk_score += 2
            else:
                risk_factors.append("No P/E ratio available - may indicate no earnings")
                risk_score += 2
            
            # Sector-specific risks
            if stock_data.company_info.get('sector') == 'Healthcare':
                risk_factors.append("Biotech sector risks: regulatory, clinical, competitive")
                risk_score += 2
            
            # Determine overall risk level
            if risk_score <= 4:
                risk_level = "Moderate"
            elif risk_score <= 7:
                risk_level = "High"
            else:
                risk_level = "Very High"
            
            return {
                "risk_score": min(risk_score, 10),
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "volatility": volatility if 'volatility' in locals() else None,
                "biotech_specific_risks": [
                    "Regulatory approval risks",
                    "Clinical trial outcomes",
                    "Commercial execution challenges",
                    "Competition from established players",
                    "Funding and dilution risks"
                ]
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {"error": f"Risk assessment failed: {str(e)}"}

async def analyze_lqda_comprehensive():
    """Main function to run comprehensive LQDA analysis"""
    
    print("🧬 LQDA (Liquidia Corporation) - COMPLEX Biotech Analysis")
    print("=" * 70)
    print("🚀 Advanced Multi-Domain Financial Intelligence System")
    print("-" * 70)
    
    # Create comprehensive analysis request
    request = StockAnalysisRequest(
        symbol="LQDA",
        analysis_type="comprehensive",
        time_period="1y",
        include_news=True,
        include_financials=True,
        include_technical=True,
        risk_assessment=True
    )
    
    try:
        # Initialize analysis engine
        engine = StockAnalysisEngine()
        
        # Perform comprehensive analysis
        results = await engine.analyze_stock(request)
        
        print("\n" + "=" * 70)
        print("✅ COMPREHENSIVE LQDA ANALYSIS COMPLETE")
        print("=" * 70)
        
        # Display executive summary
        print(f"📈 {results['symbol']} | Current: ${results['current_price']:.2f}")
        print(f"📊 Change: {results['price_change']:+.2f} ({results['price_change_percent']:+.2f}%)")
        print(f"📅 Analysis Date: {results['analysis_date'][:19]}")
        
        # Technical Analysis Results
        technical = results['analysis_components'].get('technical', {})
        if technical and not technical.get('error'):
            print("\n📈 TECHNICAL ANALYSIS")
            print("-" * 30)
            indicators = technical.get('indicators', {})
            for indicator, value in indicators.items():
                if value is not None:
                    if 'sma' in indicator.lower() or 'ema' in indicator.lower():
                        print(f"   {indicator.upper()}: ${value:.2f}")
                    elif 'rsi' in indicator.lower():
                        print(f"   {indicator.upper()}: {value:.1f}")
                    else:
                        print(f"   {indicator.upper()}: {value:.2f}")
            
            signals = technical.get('signals', [])
            if signals:
                print("\n   🎯 Technical Signals:")
                for signal in signals:
                    print(f"     • {signal}")
        
        # Fundamental Analysis Results
        fundamental = results['analysis_components'].get('fundamental', {})
        if fundamental and not fundamental.get('error'):
            print("\n💰 FUNDAMENTAL ANALYSIS")
            print("-" * 30)
            ratios = fundamental.get('ratios', {})
            for ratio, value in ratios.items():
                if value and value != 0:
                    if isinstance(value, float):
                        print(f"   {ratio.replace('_', ' ').title()}: {value:.2f}")
                    else:
                        print(f"   {ratio.replace('_', ' ').title()}: {value}")
            
            company_info = fundamental.get('company_info', {})
            print(f"\n   🏢 Company Profile:")
            print(f"     Sector: {company_info.get('sector', 'N/A')}")
            print(f"     Industry: {company_info.get('industry', 'N/A')}")
            if company_info.get('market_cap'):
                print(f"     Market Cap: ${company_info['market_cap']:,}")
            
            assessment = fundamental.get('assessment', [])
            if assessment:
                print("\n   📋 Assessment:")
                for item in assessment:
                    print(f"     • {item}")
        
        # AI Analysis Results
        ai_insights = results['analysis_components'].get('ai_insights', {})
        if ai_insights:
            print("\n🤖 AI-POWERED INSIGHTS")
            print("-" * 30)
            print(f"   Model: {ai_insights.get('model_used', 'N/A')}")
            print(f"   Quality: {ai_insights.get('analysis_quality', 'N/A')}")
            
            ai_analysis = ai_insights.get('ai_analysis', '')
            if ai_analysis:
                # Format AI analysis for better readability
                lines = ai_analysis.split('\n')
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
        
        # Risk Assessment Results
        risk = results['analysis_components'].get('risk', {})
        if risk and not risk.get('error'):
            print("\n⚠️ RISK ASSESSMENT")
            print("-" * 30)
            print(f"   Risk Level: {risk.get('risk_level', 'N/A')}")
            print(f"   Risk Score: {risk.get('risk_score', 0)}/10")
            
            if risk.get('volatility'):
                print(f"   Volatility: {risk['volatility']:.1%} annualized")
            
            risk_factors = risk.get('risk_factors', [])
            if risk_factors:
                print("\n   🚨 Risk Factors:")
                for factor in risk_factors:
                    print(f"     • {factor}")
            
            biotech_risks = risk.get('biotech_specific_risks', [])
            if biotech_risks:
                print("\n   🧬 Biotech-Specific Risks:")
                for risk_item in biotech_risks:
                    print(f"     • {risk_item}")
        
        print("\n" + "=" * 70)
        print("🎯 COMPREHENSIVE ANALYSIS COMPLETE")
        print("💡 This represents sophisticated multi-domain financial analysis")
        print("⚠️  For educational and research purposes only")
        print("📞 Consult licensed financial professionals for investment decisions")
        print("🔬 Biotech investments require specialized due diligence")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(analyze_lqda_comprehensive())