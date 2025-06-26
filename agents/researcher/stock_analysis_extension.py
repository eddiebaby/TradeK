"""
Stock Analysis Extension for RESEARCHER Agent

This module extends the RESEARCHER agent with comprehensive stock analysis capabilities
inspired by CrewAI's multi-agent stock analysis framework, optimized for Ollama local models.
"""

import asyncio
import time
import json
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from researcher_agent import ResearcherAgent, ResearchDomain, ResearchInsight
from core.agent_base import TaskContext
from blackboard import blackboard, write_task, update_status, log_performance


class StockAnalysisDomain(Enum):
    """Stock analysis specific research domains."""
    FINANCIAL_METRICS = "financial_analysis"
    MARKET_SENTIMENT = "market_intelligence"
    SEC_FILINGS = "regulatory_analysis"
    TECHNICAL_ANALYSIS = "price_patterns"
    COMPETITIVE_ANALYSIS = "industry_comparison"
    RISK_ASSESSMENT = "risk_evaluation"


@dataclass
class StockAnalysisRequest:
    """Structured stock analysis request."""
    request_id: str
    ticker_symbol: str
    analysis_domains: List[StockAnalysisDomain]
    analysis_depth: str  # "quick", "standard", "comprehensive"
    time_horizon: str  # "short_term", "medium_term", "long_term"
    context: Dict[str, Any]
    priority: int = 1
    timestamp: float = field(default_factory=time.time)


@dataclass
class FinancialMetrics:
    """Key financial metrics for analysis."""
    pe_ratio: Optional[float] = None
    eps_growth: Optional[float] = None
    revenue_growth: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    profit_margin: Optional[float] = None
    market_cap: Optional[float] = None
    book_value: Optional[float] = None


@dataclass
class StockAnalysisResult:
    """Comprehensive stock analysis results."""
    analysis_id: str
    ticker_symbol: str
    financial_metrics: FinancialMetrics
    market_sentiment: Dict[str, Any]
    sec_insights: List[Dict[str, Any]]
    technical_signals: Dict[str, Any]
    competitive_position: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    investment_recommendation: Dict[str, Any]
    confidence_score: float
    analysis_timestamp: float = field(default_factory=time.time)


class StockAnalysisTools:
    """Collection of stock analysis tools with local model optimization."""
    
    def __init__(self):
        self.sec_api_key = self._get_sec_api_key()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradeKnowledge Research Agent (contact@tradeknowledge.ai)'
        })
    
    def _get_sec_api_key(self) -> Optional[str]:
        """Get SEC API key from environment or config."""
        import os
        return os.getenv('SEC_API_KEY')
    
    async def get_financial_metrics(self, ticker: str) -> FinancialMetrics:
        """
        Extract financial metrics for the ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            FinancialMetrics object with key financial data
        """
        try:
            # This would integrate with actual financial data APIs
            # For now, simulate realistic financial metrics
            metrics = FinancialMetrics(
                pe_ratio=25.4,
                eps_growth=12.5,
                revenue_growth=18.3,
                debt_to_equity=0.45,
                roe=15.2,
                profit_margin=8.7,
                market_cap=1250000000000,  # $1.25T
                book_value=42.30
            )
            
            await write_task("RESEARCHER", "financial_metrics", {
                "ticker": ticker,
                "pe_ratio": metrics.pe_ratio
            })
            
            return metrics
            
        except Exception as e:
            logging.error(f"Failed to get financial metrics for {ticker}: {e}")
            return FinancialMetrics()
    
    async def analyze_sec_filings(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Analyze SEC 10-K and 10-Q filings for insights.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of insights from SEC filings
        """
        if not self.sec_api_key:
            return [{"error": "SEC API key not configured"}]
        
        try:
            insights = []
            
            # Analyze 10-K (Annual Report)
            tenk_insights = await self._analyze_10k_filing(ticker)
            insights.extend(tenk_insights)
            
            # Analyze 10-Q (Quarterly Report)
            tenq_insights = await self._analyze_10q_filing(ticker)
            insights.extend(tenq_insights)
            
            return insights
            
        except Exception as e:
            logging.error(f"SEC filing analysis failed for {ticker}: {e}")
            return [{"error": str(e)}]
    
    async def _analyze_10k_filing(self, ticker: str) -> List[Dict[str, Any]]:
        """Analyze 10-K filing for strategic insights."""
        
        # Simulated 10-K analysis results
        return [
            {
                "filing_type": "10-K",
                "insight_type": "business_strategy",
                "description": "Company focusing on AI and cloud infrastructure expansion",
                "risk_factors": ["Regulatory changes", "Competition", "Technology obsolescence"],
                "growth_initiatives": ["AI platform development", "International expansion"],
                "confidence": 0.92
            },
            {
                "filing_type": "10-K",
                "insight_type": "financial_health",
                "description": "Strong cash position with improving operational efficiency",
                "key_metrics": {"cash_flow": "positive", "debt_level": "manageable"},
                "confidence": 0.88
            }
        ]
    
    async def _analyze_10q_filing(self, ticker: str) -> List[Dict[str, Any]]:
        """Analyze 10-Q filing for quarterly insights."""
        
        # Simulated 10-Q analysis results
        return [
            {
                "filing_type": "10-Q",
                "insight_type": "quarterly_performance",
                "description": "Revenue growth accelerating with margin expansion",
                "quarterly_highlights": ["Revenue beat expectations", "Cost optimization"],
                "forward_guidance": "Raised full-year guidance",
                "confidence": 0.90
            }
        ]
    
    async def gather_market_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Gather market sentiment from news and analyst reports.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Market sentiment analysis
        """
        try:
            # Simulated market sentiment analysis
            sentiment = {
                "overall_sentiment": "positive",
                "sentiment_score": 0.72,  # -1 to 1 scale
                "news_summary": {
                    "positive_news": [
                        "Strong quarterly earnings beat expectations",
                        "New product launch gaining market traction",
                        "Strategic partnership announced"
                    ],
                    "negative_news": [
                        "Regulatory investigation ongoing",
                        "Competitive pressure in core market"
                    ],
                    "neutral_news": [
                        "Management reshuffling announced",
                        "Dividend policy review scheduled"
                    ]
                },
                "analyst_ratings": {
                    "buy": 12,
                    "hold": 8,
                    "sell": 2,
                    "average_target": 425.50,
                    "current_price": 380.25
                },
                "social_sentiment": {
                    "twitter_sentiment": "bullish",
                    "reddit_sentiment": "mixed",
                    "news_sentiment": "positive"
                },
                "confidence": 0.85
            }
            
            return sentiment
            
        except Exception as e:
            logging.error(f"Market sentiment analysis failed for {ticker}: {e}")
            return {"error": str(e)}
    
    async def perform_technical_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Perform technical analysis on price patterns.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Technical analysis results
        """
        try:
            # Simulated technical analysis
            technical_signals = {
                "trend_analysis": {
                    "short_term": "bullish",
                    "medium_term": "neutral",
                    "long_term": "bullish"
                },
                "support_resistance": {
                    "support_levels": [360.00, 340.00, 320.00],
                    "resistance_levels": [400.00, 420.00, 450.00]
                },
                "technical_indicators": {
                    "rsi": 58.3,  # Relative Strength Index
                    "macd": "bullish_crossover",
                    "moving_averages": {
                        "ma_50": 375.20,
                        "ma_200": 355.80,
                        "golden_cross": True
                    }
                },
                "volume_analysis": {
                    "average_volume": 2500000,
                    "recent_volume": 3200000,
                    "volume_trend": "increasing"
                },
                "confidence": 0.78
            }
            
            return technical_signals
            
        except Exception as e:
            logging.error(f"Technical analysis failed for {ticker}: {e}")
            return {"error": str(e)}
    
    async def analyze_competitive_position(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze competitive position within industry.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Competitive analysis results
        """
        try:
            competitive_analysis = {
                "industry_sector": "Technology",
                "market_position": "Market Leader",
                "competitive_advantages": [
                    "Strong brand recognition",
                    "Technological innovation",
                    "Scale economics",
                    "Distribution network"
                ],
                "competitive_threats": [
                    "Emerging competitors",
                    "Technology disruption",
                    "Regulatory challenges"
                ],
                "peer_comparison": {
                    "revenue_vs_peers": "above_average",
                    "profitability_vs_peers": "above_average",
                    "valuation_vs_peers": "premium",
                    "growth_vs_peers": "above_average"
                },
                "market_share": {
                    "current_share": 23.5,
                    "trend": "stable",
                    "key_competitors": ["MSFT", "GOOGL", "META"]
                },
                "confidence": 0.87
            }
            
            return competitive_analysis
            
        except Exception as e:
            logging.error(f"Competitive analysis failed for {ticker}: {e}")
            return {"error": str(e)}
    
    async def assess_investment_risks(self, ticker: str, metrics: FinancialMetrics) -> Dict[str, Any]:
        """
        Assess investment risks and calculate risk score.
        
        Args:
            ticker: Stock ticker symbol
            metrics: Financial metrics for risk assessment
            
        Returns:
            Risk assessment results
        """
        try:
            risk_assessment = {
                "overall_risk_score": 6.5,  # 1-10 scale (10 = highest risk)
                "risk_categories": {
                    "financial_risk": {
                        "score": 4.0,
                        "factors": ["Debt levels manageable", "Strong cash flow"],
                        "debt_to_equity": metrics.debt_to_equity or 0.45
                    },
                    "market_risk": {
                        "score": 7.0,
                        "factors": ["High market volatility", "Sector rotation risk"],
                        "beta": 1.25
                    },
                    "operational_risk": {
                        "score": 5.5,
                        "factors": ["Technology disruption", "Regulatory changes"],
                        "key_risks": ["AI regulation", "Data privacy laws"]
                    },
                    "valuation_risk": {
                        "score": 8.0,
                        "factors": ["High P/E ratio", "Growth expectations"],
                        "pe_ratio": metrics.pe_ratio or 25.4
                    }
                },
                "risk_mitigation": [
                    "Diversification across sectors",
                    "Position sizing control",
                    "Stop-loss implementation",
                    "Regular portfolio rebalancing"
                ],
                "confidence": 0.89
            }
            
            return risk_assessment
            
        except Exception as e:
            logging.error(f"Risk assessment failed for {ticker}: {e}")
            return {"error": str(e)}


class StockAnalysisResearcher(ResearcherAgent):
    """
    Extended RESEARCHER agent with comprehensive stock analysis capabilities.
    
    Integrates CrewAI-inspired multi-agent analysis patterns with local Ollama models
    for financial intelligence gathering and investment research.
    """
    
    def __init__(self):
        super().__init__()
        self.stock_tools = StockAnalysisTools()
        
        # Add stock analysis to research capabilities
        self.capabilities.extend([
            "stock_financial_analysis",
            "market_sentiment_research",
            "sec_filing_analysis",
            "technical_pattern_recognition",
            "competitive_intelligence",
            "investment_risk_assessment"
        ])
        
        # Add stock-specific research modes
        self.research_modes.update({
            "financial_deep_dive": "Comprehensive financial metrics and ratio analysis",
            "market_intelligence": "News, sentiment, and analyst opinion synthesis",
            "regulatory_analysis": "SEC filing analysis and compliance review",
            "technical_research": "Price pattern and technical indicator analysis",
            "competitive_research": "Industry positioning and peer comparison",
            "risk_intelligence": "Investment risk assessment and mitigation strategies"
        })
    
    async def analyze_stock(self, analysis_spec: Dict[str, Any]) -> StockAnalysisResult:
        """
        Conduct comprehensive stock analysis using multi-domain research.
        
        Args:
            analysis_spec: Stock analysis specification with ticker and requirements
            
        Returns:
            StockAnalysisResult: Comprehensive analysis with investment insights
        """
        analysis_start = time.time()
        analysis_id = f"stock_analysis_{int(time.time() * 1000)}"
        
        ticker = analysis_spec.get("ticker_symbol", "").upper()
        if not ticker:
            raise ValueError("Ticker symbol required for stock analysis")
        
        print(f"🔍 RESEARCHER conducting comprehensive stock analysis for {ticker}")
        
        # Create stock analysis request
        request = StockAnalysisRequest(
            request_id=analysis_id,
            ticker_symbol=ticker,
            analysis_domains=[StockAnalysisDomain(d) for d in analysis_spec.get("domains", 
                ["financial_metrics", "market_sentiment", "sec_filings"])],
            analysis_depth=analysis_spec.get("depth", "comprehensive"),
            time_horizon=analysis_spec.get("time_horizon", "medium_term"),
            context=analysis_spec.get("context", {}),
            priority=analysis_spec.get("priority", 1)
        )
        
        # Conduct parallel analysis across domains
        print("  📊 Analyzing financial metrics...")
        financial_metrics = await self.stock_tools.get_financial_metrics(ticker)
        
        print("  📰 Gathering market sentiment...")
        market_sentiment = await self.stock_tools.gather_market_sentiment(ticker)
        
        print("  📋 Analyzing SEC filings...")
        sec_insights = await self.stock_tools.analyze_sec_filings(ticker)
        
        print("  📈 Performing technical analysis...")
        technical_signals = await self.stock_tools.perform_technical_analysis(ticker)
        
        print("  🏢 Analyzing competitive position...")
        competitive_position = await self.stock_tools.analyze_competitive_position(ticker)
        
        print("  ⚠️ Assessing investment risks...")
        risk_assessment = await self.stock_tools.assess_investment_risks(ticker, financial_metrics)
        
        # Generate investment recommendation
        investment_recommendation = await self._generate_investment_recommendation(
            ticker, financial_metrics, market_sentiment, sec_insights,
            technical_signals, competitive_position, risk_assessment
        )
        
        # Calculate overall confidence score
        confidence_score = await self._calculate_analysis_confidence(
            market_sentiment, sec_insights, technical_signals, 
            competitive_position, risk_assessment
        )
        
        analysis_duration = time.time() - analysis_start
        
        # Log performance
        await log_performance("RESEARCHER", "stock_analysis", 
                             self._estimate_tokens(investment_recommendation), 
                             analysis_duration, True)
        
        return StockAnalysisResult(
            analysis_id=analysis_id,
            ticker_symbol=ticker,
            financial_metrics=financial_metrics,
            market_sentiment=market_sentiment,
            sec_insights=sec_insights,
            technical_signals=technical_signals,
            competitive_position=competitive_position,
            risk_assessment=risk_assessment,
            investment_recommendation=investment_recommendation,
            confidence_score=confidence_score
        )
    
    async def _generate_investment_recommendation(self, 
                                                ticker: str,
                                                metrics: FinancialMetrics,
                                                sentiment: Dict[str, Any],
                                                sec_insights: List[Dict[str, Any]],
                                                technical: Dict[str, Any],
                                                competitive: Dict[str, Any],
                                                risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive investment recommendation."""
        
        # Calculate recommendation score based on multiple factors
        financial_score = self._score_financial_metrics(metrics)
        sentiment_score = sentiment.get("sentiment_score", 0) * 10  # Convert to 1-10
        technical_score = self._score_technical_signals(technical)
        competitive_score = self._score_competitive_position(competitive)
        risk_score = 10 - risk.get("overall_risk_score", 5)  # Invert risk
        
        overall_score = (
            financial_score * 0.25 +
            sentiment_score * 0.20 +
            technical_score * 0.20 +
            competitive_score * 0.20 +
            risk_score * 0.15
        )
        
        # Determine recommendation
        if overall_score >= 8.0:
            recommendation = "Strong Buy"
            action = "Accumulate position with 3-5% portfolio allocation"
        elif overall_score >= 7.0:
            recommendation = "Buy"
            action = "Initiate position with 2-3% portfolio allocation"
        elif overall_score >= 6.0:
            recommendation = "Hold"
            action = "Maintain current position, monitor developments"
        elif overall_score >= 4.0:
            recommendation = "Weak Hold"
            action = "Consider reducing position, watch for exit signals"
        else:
            recommendation = "Sell"
            action = "Exit position, capital preservation priority"
        
        return {
            "recommendation": recommendation,
            "action": action,
            "overall_score": overall_score,
            "target_price": self._calculate_target_price(metrics, sentiment),
            "time_horizon": "6-12 months",
            "portfolio_allocation": self._suggest_portfolio_allocation(overall_score, risk),
            "key_catalysts": [
                "Quarterly earnings performance",
                "Market sentiment shifts",
                "Competitive developments",
                "Regulatory changes"
            ],
            "exit_conditions": [
                "Score drops below 5.0",
                "Risk score exceeds 8.0",
                "Technical breakdown below support",
                "Fundamental deterioration"
            ],
            "confidence": min(0.95, overall_score / 10)
        }
    
    def _score_financial_metrics(self, metrics: FinancialMetrics) -> float:
        """Score financial metrics on 1-10 scale."""
        
        score = 5.0  # Base score
        
        # P/E ratio scoring
        if metrics.pe_ratio:
            if metrics.pe_ratio < 15:
                score += 1.0
            elif metrics.pe_ratio < 25:
                score += 0.5
            elif metrics.pe_ratio > 35:
                score -= 1.0
        
        # Growth scoring
        if metrics.eps_growth and metrics.eps_growth > 15:
            score += 1.0
        elif metrics.eps_growth and metrics.eps_growth > 10:
            score += 0.5
        
        # Profitability scoring
        if metrics.roe and metrics.roe > 15:
            score += 1.0
        elif metrics.roe and metrics.roe > 10:
            score += 0.5
        
        # Debt scoring
        if metrics.debt_to_equity and metrics.debt_to_equity < 0.3:
            score += 1.0
        elif metrics.debt_to_equity and metrics.debt_to_equity > 0.8:
            score -= 1.0
        
        return max(1.0, min(10.0, score))
    
    def _score_technical_signals(self, technical: Dict[str, Any]) -> float:
        """Score technical signals on 1-10 scale."""
        
        score = 5.0  # Base score
        
        trend = technical.get("trend_analysis", {})
        if trend.get("short_term") == "bullish":
            score += 1.0
        if trend.get("medium_term") == "bullish":
            score += 1.5
        if trend.get("long_term") == "bullish":
            score += 1.0
        
        indicators = technical.get("technical_indicators", {})
        rsi = indicators.get("rsi", 50)
        if 40 <= rsi <= 60:  # Neutral RSI is good
            score += 0.5
        elif rsi > 70:  # Overbought
            score -= 1.0
        elif rsi < 30:  # Oversold, potential bounce
            score += 0.5
        
        if indicators.get("macd") == "bullish_crossover":
            score += 1.0
        
        return max(1.0, min(10.0, score))
    
    def _score_competitive_position(self, competitive: Dict[str, Any]) -> float:
        """Score competitive position on 1-10 scale."""
        
        position = competitive.get("market_position", "").lower()
        if "leader" in position:
            return 9.0
        elif "strong" in position:
            return 7.5
        elif "average" in position:
            return 5.0
        else:
            return 3.0
    
    def _calculate_target_price(self, metrics: FinancialMetrics, sentiment: Dict[str, Any]) -> float:
        """Calculate target price based on analysis."""
        
        # Simple target price calculation (would be more sophisticated in practice)
        current_price = sentiment.get("analyst_ratings", {}).get("current_price", 100)
        analyst_target = sentiment.get("analyst_ratings", {}).get("average_target", current_price * 1.1)
        
        # Adjust based on our analysis
        if metrics.eps_growth and metrics.eps_growth > 15:
            adjustment = 1.15  # 15% premium for high growth
        elif metrics.eps_growth and metrics.eps_growth < 5:
            adjustment = 0.95  # 5% discount for low growth
        else:
            adjustment = 1.05  # 5% premium baseline
        
        return round(analyst_target * adjustment, 2)
    
    def _suggest_portfolio_allocation(self, score: float, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest portfolio allocation based on score and risk."""
        
        risk_score = risk.get("overall_risk_score", 5)
        
        if score >= 8.0 and risk_score <= 6:
            allocation = "3-5%"
            risk_level = "Moderate"
        elif score >= 7.0 and risk_score <= 7:
            allocation = "2-3%"
            risk_level = "Moderate-High"
        elif score >= 6.0:
            allocation = "1-2%"
            risk_level = "High"
        else:
            allocation = "0%"
            risk_level = "Very High"
        
        return {
            "recommended_allocation": allocation,
            "risk_level": risk_level,
            "position_sizing": "Conservative approach recommended",
            "diversification": "Part of balanced portfolio only"
        }
    
    async def _calculate_analysis_confidence(self, 
                                           sentiment: Dict[str, Any],
                                           sec_insights: List[Dict[str, Any]],
                                           technical: Dict[str, Any],
                                           competitive: Dict[str, Any],
                                           risk: Dict[str, Any]) -> float:
        """Calculate overall analysis confidence score."""
        
        confidences = []
        
        if sentiment.get("confidence"):
            confidences.append(sentiment["confidence"])
        
        for insight in sec_insights:
            if insight.get("confidence"):
                confidences.append(insight["confidence"])
        
        if technical.get("confidence"):
            confidences.append(technical["confidence"])
        
        if competitive.get("confidence"):
            confidences.append(competitive["confidence"])
        
        if risk.get("confidence"):
            confidences.append(risk["confidence"])
        
        return sum(confidences) / len(confidences) if confidences else 0.75
    
    async def format_stock_analysis_for_strategy(self, analysis: StockAnalysisResult) -> Dict[str, Any]:
        """Format stock analysis results for MASTERMIND strategic planning."""
        
        return {
            "strategic_investment_insights": {
                "ticker": analysis.ticker_symbol,
                "investment_thesis": analysis.investment_recommendation,
                "risk_profile": analysis.risk_assessment,
                "competitive_moat": analysis.competitive_position,
                "growth_trajectory": analysis.financial_metrics.__dict__
            },
            "portfolio_strategy": {
                "allocation_guidance": analysis.investment_recommendation.get("portfolio_allocation", {}),
                "risk_management": analysis.risk_assessment.get("risk_mitigation", []),
                "diversification_impact": "Technology sector exposure",
                "rebalancing_triggers": analysis.investment_recommendation.get("exit_conditions", [])
            },
            "decision_support": {
                "confidence_level": analysis.confidence_score,
                "recommendation_strength": analysis.investment_recommendation.get("recommendation", "Hold"),
                "catalyst_timeline": "6-12 months",
                "monitoring_requirements": "Quarterly review recommended"
            }
        }
    
    async def format_stock_analysis_for_implementation(self, analysis: StockAnalysisResult) -> Dict[str, Any]:
        """Format stock analysis results for EXECUTOR implementation."""
        
        return {
            "implementation_guidance": {
                "execution_strategy": analysis.investment_recommendation.get("action", "Hold position"),
                "entry_criteria": "Technical confirmation required",
                "position_sizing": analysis.investment_recommendation.get("portfolio_allocation", {}),
                "risk_controls": analysis.risk_assessment.get("risk_mitigation", [])
            },
            "monitoring_specifications": {
                "key_metrics": list(analysis.financial_metrics.__dict__.keys()),
                "alert_conditions": analysis.investment_recommendation.get("exit_conditions", []),
                "review_frequency": "Monthly technical, quarterly fundamental",
                "data_sources": ["SEC filings", "Market data", "News sentiment"]
            },
            "automation_opportunities": {
                "automated_alerts": "Price/volume threshold monitoring",
                "rebalancing_triggers": "Risk score changes",
                "reporting_schedule": "Weekly portfolio impact assessment"
            }
        }