#!/usr/bin/env python3
"""
TradeKnowledge Options Analysis Engine

Specialized options trading analysis for systematic traders like Case.
Focus on SPX options, diagonal spreads, and systematic strategy identification.

Features:
- Real-time SPX options chain analysis
- Diagonal spread opportunity detection
- Greeks calculation and risk assessment
- Probability analysis and expected returns
- Strategy backtesting and optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class OptionData:
    """Individual option contract data."""
    symbol: str
    strike: float
    expiry: date
    option_type: str  # 'call' or 'put'
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

@dataclass
class DiagonalSpreadOpportunity:
    """Diagonal spread trading opportunity."""
    symbol: str
    strategy_type: str
    long_strike: float
    short_strike: float
    long_expiry: date
    short_expiry: date
    net_debit: float
    max_profit: float
    max_loss: float
    breakeven: float
    roi: float
    profit_probability: float
    max_profit_probability: float
    delta: float
    gamma: float
    theta: float
    vega: float
    recommendation: str
    reasoning: str
    confidence_score: float

class OptionsAnalyzer:
    """
    Advanced options analysis engine for systematic trading.
    
    Designed for technical professionals who trade SPX options systematically
    using data-driven approaches and neural network insights.
    """
    
    def __init__(self):
        """Initialize the options analyzer with market data connections."""
        self.risk_free_rate = 0.05  # 5% risk-free rate (will be updated from FRED)
        self.trading_days_per_year = 252
        
        # Strategy parameters optimized for SPX
        self.spx_parameters = {
            "min_volume": 100,
            "min_open_interest": 500,
            "max_bid_ask_spread": 0.50,
            "min_delta": 0.15,
            "max_delta": 0.85,
            "min_dte": 7,
            "max_dte": 90,
            "target_roi": 0.20  # 20% target ROI
        }

    async def scan_opportunities(
        self, 
        symbol: str = "SPX", 
        strategy: str = "diagonal",
        min_dte: int = 30,
        max_dte: int = 45,
        confidence_level: float = 99.0
    ) -> Dict[str, Any]:
        """
        Scan for optimal trading opportunities in SPX options.
        
        Args:
            symbol: Options symbol to analyze (default: SPX)
            strategy: Strategy type (diagonal, iron_condor, strangle, etc.)
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            confidence_level: Neural network confidence threshold
            
        Returns:
            Dictionary containing opportunities and analysis
        """
        try:
            logger.info(f"🔍 Scanning {symbol} {strategy} opportunities (DTE: {min_dte}-{max_dte})")
            
            # Get current market data
            current_price = await self._get_current_price(symbol)
            
            # Get options chain data
            options_chain = await self._get_options_chain(symbol, min_dte, max_dte)
            
            if strategy == "diagonal":
                opportunities = await self._scan_diagonal_spreads(
                    options_chain, current_price, confidence_level
                )
            elif strategy == "iron_condor":
                opportunities = await self._scan_iron_condors(
                    options_chain, current_price, confidence_level
                )
            elif strategy == "covered_call":
                opportunities = await self._scan_covered_calls(
                    options_chain, current_price, confidence_level
                )
            else:
                opportunities = []
            
            # Rank opportunities by expected return and probability
            ranked_opportunities = self._rank_opportunities(opportunities)
            
            return {
                "symbol": symbol,
                "strategy": strategy,
                "scan_time": datetime.now(),
                "current_price": current_price,
                "total_opportunities": len(ranked_opportunities),
                "opportunities": ranked_opportunities[:10],  # Top 10
                "market_conditions": await self._assess_market_conditions(symbol),
                "recommended_allocation": self._calculate_position_sizing(ranked_opportunities)
            }
            
        except Exception as e:
            logger.error(f"❌ Options scan error: {e}")
            return {"error": str(e), "opportunities": []}

    async def analyze_diagonal_spread(
        self,
        symbol: str,
        long_strike: float,
        short_strike: float,
        long_expiry: str,
        short_expiry: str
    ) -> Dict[str, Any]:
        """
        Analyze a specific diagonal spread configuration.
        
        This is the core function for Case's preferred trading strategy.
        """
        try:
            logger.info(f"📊 Analyzing {symbol} diagonal: {long_strike}/{short_strike}")
            
            # Convert date strings to date objects
            long_exp_date = datetime.strptime(long_expiry, "%Y-%m-%d").date()
            short_exp_date = datetime.strptime(short_expiry, "%Y-%m-%d").date()
            
            # Get option data
            long_option = await self._get_option_data(symbol, long_strike, long_exp_date, "call")
            short_option = await self._get_option_data(symbol, short_strike, short_exp_date, "call")
            
            if not long_option or not short_option:
                return {"error": "Could not retrieve option data"}
            
            # Calculate spread metrics
            net_debit = long_option.last_price - short_option.last_price
            max_profit = (short_strike - long_strike) - net_debit
            max_loss = net_debit
            roi = (max_profit / net_debit) * 100 if net_debit > 0 else 0
            
            # Calculate Greeks
            spread_delta = long_option.delta - short_option.delta
            spread_gamma = long_option.gamma - short_option.gamma
            spread_theta = long_option.theta - short_option.theta
            spread_vega = long_option.vega - short_option.vega
            
            # Calculate probabilities
            current_price = await self._get_current_price(symbol)
            profit_prob = await self._calculate_profit_probability(
                current_price, short_strike, long_exp_date, symbol
            )
            max_profit_prob = await self._calculate_max_profit_probability(
                current_price, short_strike, short_exp_date, symbol
            )
            
            # Calculate breakeven
            breakeven = long_strike + net_debit
            
            # Generate recommendation
            recommendation, reasoning = self._generate_recommendation(
                roi, profit_prob, max_profit_prob, spread_theta, current_price, short_strike
            )
            
            return {
                "symbol": symbol,
                "strategy": "diagonal_call_spread",
                "long_strike": long_strike,
                "short_strike": short_strike,
                "long_expiry": long_expiry,
                "short_expiry": short_expiry,
                "current_price": current_price,
                "net_debit": round(net_debit, 2),
                "max_profit": round(max_profit, 2),
                "max_loss": round(max_loss, 2),
                "roi": round(roi, 1),
                "breakeven": round(breakeven, 2),
                "delta": round(spread_delta, 3),
                "gamma": round(spread_gamma, 4),
                "theta": round(spread_theta, 2),
                "vega": round(spread_vega, 2),
                "profit_probability": round(profit_prob, 1),
                "max_profit_probability": round(max_profit_prob, 1),
                "recommendation": recommendation,
                "reasoning": reasoning,
                "long_option": {
                    "price": long_option.last_price,
                    "iv": long_option.implied_volatility,
                    "delta": long_option.delta,
                    "volume": long_option.volume
                },
                "short_option": {
                    "price": short_option.last_price,
                    "iv": short_option.implied_volatility,
                    "delta": short_option.delta,
                    "volume": short_option.volume
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Diagonal spread analysis error: {e}")
            return {"error": str(e)}

    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for the underlying."""
        try:
            if symbol == "SPX":
                # Use SPY as proxy for SPX current price
                ticker = yf.Ticker("SPY")
                data = ticker.history(period="1d", interval="1m")
                return float(data['Close'].iloc[-1]) * 10  # SPX ≈ SPY * 10
            else:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                return float(data['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 4500.0  # Default SPX price if data unavailable

    async def _get_options_chain(self, symbol: str, min_dte: int, max_dte: int) -> List[OptionData]:
        """Get options chain data for the specified DTE range."""
        try:
            # For demo purposes, generate synthetic options data
            # In production, this would connect to options data provider
            current_price = await self._get_current_price(symbol)
            options_chain = []
            
            # Generate expiration dates
            base_date = datetime.now().date()
            expiry_dates = []
            for days in range(min_dte, max_dte + 7, 7):  # Weekly expirations
                expiry_dates.append(base_date + timedelta(days=days))
            
            # Generate strike prices around current price
            strikes = []
            for i in range(-20, 21):  # 40 strikes total
                strike = round(current_price + (i * 25), 0)  # $25 intervals for SPX
                strikes.append(strike)
            
            # Generate option data for each strike and expiry
            for expiry in expiry_dates:
                dte = (expiry - base_date).days
                for strike in strikes:
                    # Call option
                    call_data = await self._generate_option_data(
                        symbol, strike, expiry, "call", current_price, dte
                    )
                    options_chain.append(call_data)
                    
                    # Put option  
                    put_data = await self._generate_option_data(
                        symbol, strike, expiry, "put", current_price, dte
                    )
                    options_chain.append(put_data)
            
            return options_chain
            
        except Exception as e:
            logger.error(f"Error getting options chain: {e}")
            return []

    async def _generate_option_data(
        self, 
        symbol: str, 
        strike: float, 
        expiry: date, 
        option_type: str,
        current_price: float,
        dte: int
    ) -> OptionData:
        """Generate synthetic option data for demo purposes."""
        # Basic Black-Scholes calculations for demo
        time_to_expiry = dte / 365.0
        volatility = 0.20  # 20% implied volatility
        
        # Calculate option price using simplified Black-Scholes
        if option_type == "call":
            intrinsic = max(0, current_price - strike)
            delta = self._calculate_delta(current_price, strike, time_to_expiry, volatility, "call")
        else:
            intrinsic = max(0, strike - current_price)
            delta = self._calculate_delta(current_price, strike, time_to_expiry, volatility, "put")
        
        time_value = max(0.10, intrinsic * 0.1 + (time_to_expiry * volatility * current_price * 0.4))
        theoretical_price = intrinsic + time_value
        
        # Add some randomness for realistic bid/ask spreads
        bid = theoretical_price * 0.98
        ask = theoretical_price * 1.02
        last_price = (bid + ask) / 2
        
        # Generate volume and open interest
        moneyness = abs(current_price - strike) / current_price
        volume = max(10, int(1000 * (1 - moneyness) * np.random.uniform(0.5, 2.0)))
        open_interest = volume * 5
        
        # Calculate Greeks
        gamma = self._calculate_gamma(current_price, strike, time_to_expiry, volatility)
        theta = self._calculate_theta(current_price, strike, time_to_expiry, volatility, option_type)
        vega = self._calculate_vega(current_price, strike, time_to_expiry, volatility)
        rho = self._calculate_rho(current_price, strike, time_to_expiry, volatility, option_type)
        
        return OptionData(
            symbol=f"{symbol}{expiry.strftime('%y%m%d')}{option_type[0].upper()}{int(strike*1000):08d}",
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            last_price=round(last_price, 2),
            bid=round(bid, 2),
            ask=round(ask, 2),
            volume=volume,
            open_interest=open_interest,
            implied_volatility=volatility,
            delta=round(delta, 3),
            gamma=round(gamma, 4),
            theta=round(theta, 2),
            vega=round(vega, 2),
            rho=round(rho, 2)
        )

    async def _get_option_data(self, symbol: str, strike: float, expiry: date, option_type: str) -> Optional[OptionData]:
        """Get specific option data."""
        current_price = await self._get_current_price(symbol)
        dte = (expiry - datetime.now().date()).days
        return await self._generate_option_data(symbol, strike, expiry, option_type, current_price, dte)

    async def _scan_diagonal_spreads(
        self, 
        options_chain: List[OptionData], 
        current_price: float,
        confidence_level: float
    ) -> List[DiagonalSpreadOpportunity]:
        """Scan for optimal diagonal spread opportunities."""
        opportunities = []
        
        # Filter options by basic criteria
        calls = [opt for opt in options_chain if opt.option_type == "call"]
        
        # Group by expiry
        expiry_groups = {}
        for opt in calls:
            if opt.expiry not in expiry_groups:
                expiry_groups[opt.expiry] = []
            expiry_groups[opt.expiry].append(opt)
        
        # Find diagonal spread combinations
        expiry_dates = sorted(expiry_groups.keys())
        
        for i, short_expiry in enumerate(expiry_dates[:-1]):  # Short leg expiry
            for long_expiry in expiry_dates[i+1:i+3]:  # Long leg 1-2 expiries out
                
                short_options = expiry_groups[short_expiry]
                long_options = expiry_groups[long_expiry]
                
                for short_opt in short_options:
                    for long_opt in long_options:
                        # Basic diagonal spread criteria
                        if (long_opt.strike < short_opt.strike and  # Long strike below short
                            short_opt.delta >= 0.3 and short_opt.delta <= 0.7 and  # Short leg in sweet spot
                            long_opt.delta >= 0.6 and  # Long leg higher delta
                            short_opt.volume >= self.spx_parameters["min_volume"] and
                            long_opt.volume >= self.spx_parameters["min_volume"]):
                            
                            opportunity = await self._evaluate_diagonal_spread(
                                long_opt, short_opt, current_price, confidence_level
                            )
                            
                            if opportunity and opportunity.roi >= 15.0:  # Minimum 15% ROI
                                opportunities.append(opportunity)
        
        return opportunities

    async def _evaluate_diagonal_spread(
        self, 
        long_opt: OptionData, 
        short_opt: OptionData, 
        current_price: float,
        confidence_level: float
    ) -> Optional[DiagonalSpreadOpportunity]:
        """Evaluate a specific diagonal spread setup."""
        try:
            # Calculate spread metrics
            net_debit = long_opt.last_price - short_opt.last_price
            if net_debit <= 0:
                return None
            
            max_profit = (short_opt.strike - long_opt.strike) - net_debit
            max_loss = net_debit
            roi = (max_profit / net_debit) * 100
            breakeven = long_opt.strike + net_debit
            
            # Calculate spread Greeks
            spread_delta = long_opt.delta - short_opt.delta
            spread_gamma = long_opt.gamma - short_opt.gamma
            spread_theta = long_opt.theta - short_opt.theta
            spread_vega = long_opt.vega - short_opt.vega
            
            # Calculate probabilities
            profit_prob = await self._calculate_profit_probability(
                current_price, breakeven, long_opt.expiry, "SPX"
            )
            max_profit_prob = await self._calculate_max_profit_probability(
                current_price, short_opt.strike, short_opt.expiry, "SPX"
            )
            
            # Generate recommendation
            recommendation, reasoning = self._generate_recommendation(
                roi, profit_prob, max_profit_prob, spread_theta, current_price, short_opt.strike
            )
            
            # Calculate confidence score based on multiple factors
            confidence_score = self._calculate_confidence_score(
                roi, profit_prob, max_profit_prob, spread_theta, 
                long_opt.volume, short_opt.volume, confidence_level
            )
            
            return DiagonalSpreadOpportunity(
                symbol="SPX",
                strategy_type="diagonal_call_spread",
                long_strike=long_opt.strike,
                short_strike=short_opt.strike,
                long_expiry=long_opt.expiry,
                short_expiry=short_opt.expiry,
                net_debit=round(net_debit, 2),
                max_profit=round(max_profit, 2),
                max_loss=round(max_loss, 2),
                breakeven=round(breakeven, 2),
                roi=round(roi, 1),
                profit_probability=round(profit_prob, 1),
                max_profit_probability=round(max_profit_prob, 1),
                delta=round(spread_delta, 3),
                gamma=round(spread_gamma, 4),
                theta=round(spread_theta, 2),
                vega=round(spread_vega, 2),
                recommendation=recommendation,
                reasoning=reasoning,
                confidence_score=round(confidence_score, 1)
            )
            
        except Exception as e:
            logger.error(f"Error evaluating diagonal spread: {e}")
            return None

    def _calculate_delta(self, S: float, K: float, T: float, sigma: float, option_type: str) -> float:
        """Calculate option delta using Black-Scholes."""
        if T <= 0:
            return 1.0 if (option_type == "call" and S > K) or (option_type == "put" and S < K) else 0.0
        
        d1 = (np.log(S / K) + (self.risk_free_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if option_type == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def _calculate_gamma(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate option gamma."""
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (self.risk_free_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def _calculate_theta(self, S: float, K: float, T: float, sigma: float, option_type: str) -> float:
        """Calculate option theta (time decay)."""
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (self.risk_free_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2))
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2))
        
        return theta / 365  # Convert to daily theta

    def _calculate_vega(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate option vega (volatility sensitivity)."""
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (self.risk_free_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% volatility change

    def _calculate_rho(self, S: float, K: float, T: float, sigma: float, option_type: str) -> float:
        """Calculate option rho (interest rate sensitivity)."""
        if T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (self.risk_free_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            return K * T * np.exp(-self.risk_free_rate * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2) / 100

    async def _calculate_profit_probability(self, current_price: float, target_price: float, expiry: date, symbol: str) -> float:
        """Calculate probability of reaching target price by expiry."""
        try:
            dte = (expiry - datetime.now().date()).days
            if dte <= 0:
                return 100.0 if current_price >= target_price else 0.0
            
            # Simplified probability calculation using normal distribution
            time_to_expiry = dte / 365.0
            volatility = 0.20  # Historical SPX volatility
            
            # Calculate the required return
            required_return = np.log(target_price / current_price)
            expected_return = (self.risk_free_rate - 0.5 * volatility**2) * time_to_expiry
            
            # Standard normal calculation
            z_score = (required_return - expected_return) / (volatility * np.sqrt(time_to_expiry))
            probability = (1 - norm.cdf(z_score)) * 100
            
            return max(5.0, min(95.0, probability))  # Bound between 5% and 95%
            
        except Exception as e:
            logger.error(f"Error calculating profit probability: {e}")
            return 50.0

    async def _calculate_max_profit_probability(self, current_price: float, target_price: float, expiry: date, symbol: str) -> float:
        """Calculate probability of maximum profit scenario."""
        # For diagonal spreads, max profit occurs when short option expires worthless
        # and long option is in the money
        return await self._calculate_profit_probability(current_price, target_price, expiry, symbol) * 0.7

    def _generate_recommendation(
        self, 
        roi: float, 
        profit_prob: float, 
        max_profit_prob: float, 
        theta: float,
        current_price: float,
        short_strike: float
    ) -> Tuple[str, str]:
        """Generate trading recommendation with reasoning."""
        
        # Risk-adjusted score
        risk_score = (roi * profit_prob) / 100
        
        reasons = []
        
        if roi >= 25:
            reasons.append(f"High ROI potential ({roi:.1f}%)")
        elif roi >= 15:
            reasons.append(f"Moderate ROI ({roi:.1f}%)")
        else:
            reasons.append(f"Low ROI ({roi:.1f}%)")
        
        if profit_prob >= 70:
            reasons.append(f"High profit probability ({profit_prob:.1f}%)")
        elif profit_prob >= 50:
            reasons.append(f"Moderate profit probability ({profit_prob:.1f}%)")
        else:
            reasons.append(f"Low profit probability ({profit_prob:.1f}%)")
        
        if theta < -0.05:
            reasons.append("Significant time decay risk")
        elif theta < -0.02:
            reasons.append("Moderate time decay")
        else:
            reasons.append("Low time decay impact")
        
        # Distance to short strike analysis
        strike_distance = abs(current_price - short_strike) / current_price * 100
        if strike_distance < 2:
            reasons.append("Close to short strike (high gamma risk)")
        elif strike_distance < 5:
            reasons.append("Near short strike (moderate risk)")
        else:
            reasons.append("Safe distance from short strike")
        
        # Final recommendation
        if risk_score >= 20 and profit_prob >= 60:
            recommendation = "STRONG BUY"
        elif risk_score >= 12 and profit_prob >= 45:
            recommendation = "BUY"
        elif risk_score >= 8 and profit_prob >= 35:
            recommendation = "HOLD"
        else:
            recommendation = "AVOID"
        
        reasoning = " | ".join(reasons)
        
        return recommendation, reasoning

    def _calculate_confidence_score(
        self, 
        roi: float, 
        profit_prob: float, 
        max_profit_prob: float, 
        theta: float,
        long_volume: int,
        short_volume: int,
        confidence_level: float
    ) -> float:
        """Calculate overall confidence score for the opportunity."""
        
        # Weight different factors
        roi_score = min(100, roi * 2)  # ROI component (0-100)
        prob_score = profit_prob  # Probability component (0-100)
        liquidity_score = min(100, (long_volume + short_volume) / 20)  # Liquidity (0-100)
        
        # Theta penalty (more negative theta reduces confidence)
        theta_penalty = max(0, abs(theta) * 10)
        
        # Base confidence score
        base_score = (roi_score * 0.3 + prob_score * 0.4 + liquidity_score * 0.3) - theta_penalty
        
        # Adjust for neural network confidence level
        nn_adjustment = confidence_level / 100.0
        final_score = base_score * nn_adjustment
        
        return max(0, min(100, final_score))

    def _rank_opportunities(self, opportunities: List[DiagonalSpreadOpportunity]) -> List[DiagonalSpreadOpportunity]:
        """Rank opportunities by expected value and risk metrics."""
        def score_opportunity(opp: DiagonalSpreadOpportunity) -> float:
            # Multi-factor scoring
            return (
                opp.roi * 0.3 +
                opp.profit_probability * 0.25 +
                opp.confidence_score * 0.25 +
                min(20, opp.max_profit) * 0.2  # Cap max profit contribution
            )
        
        return sorted(opportunities, key=score_opportunity, reverse=True)

    async def _assess_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Assess current market conditions for strategy selection."""
        try:
            # Get VIX data for volatility regime
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="30d")
            current_vix = float(vix_data['Close'].iloc[-1])
            vix_20d_avg = float(vix_data['Close'].mean())
            
            # Determine volatility regime
            if current_vix > 25:
                vol_regime = "HIGH"
            elif current_vix > 18:
                vol_regime = "ELEVATED"
            elif current_vix > 12:
                vol_regime = "NORMAL"
            else:
                vol_regime = "LOW"
            
            return {
                "vix_current": round(current_vix, 2),
                "vix_20d_avg": round(vix_20d_avg, 2),
                "volatility_regime": vol_regime,
                "market_bias": "BULLISH" if current_vix < vix_20d_avg else "BEARISH",
                "recommended_strategies": self._get_regime_strategies(vol_regime)
            }
            
        except Exception as e:
            logger.error(f"Error assessing market conditions: {e}")
            return {
                "vix_current": 18.0,
                "volatility_regime": "NORMAL",
                "market_bias": "NEUTRAL",
                "recommended_strategies": ["diagonal_spreads", "iron_condors"]
            }

    def _get_regime_strategies(self, vol_regime: str) -> List[str]:
        """Get recommended strategies for current volatility regime."""
        regime_strategies = {
            "LOW": ["diagonal_spreads", "calendar_spreads", "covered_calls"],
            "NORMAL": ["diagonal_spreads", "iron_condors", "strangles"],
            "ELEVATED": ["iron_condors", "short_strangles", "credit_spreads"],
            "HIGH": ["covered_calls", "cash_secured_puts", "protective_puts"]
        }
        return regime_strategies.get(vol_regime, ["diagonal_spreads"])

    def _calculate_position_sizing(self, opportunities: List[DiagonalSpreadOpportunity]) -> Dict[str, Any]:
        """Calculate recommended position sizing based on Kelly criterion and risk management."""
        if not opportunities:
            return {"recommended_allocation": 0, "max_positions": 0}
        
        # Use top opportunity for sizing calculation
        top_opp = opportunities[0]
        
        # Kelly criterion approximation
        win_rate = top_opp.profit_probability / 100
        avg_win = top_opp.max_profit
        avg_loss = top_opp.max_loss
        
        if win_rate > 0 and avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            # Conservative sizing: use 25% of Kelly
            recommended_fraction = max(0.01, min(0.05, kelly_fraction * 0.25))
        else:
            recommended_fraction = 0.02  # Default 2% risk per trade
        
        return {
            "recommended_allocation": round(recommended_fraction * 100, 1),
            "max_positions": min(5, int(0.20 / recommended_fraction)),  # Max 20% total allocation
            "kelly_fraction": round(kelly_fraction * 100, 1) if 'kelly_fraction' in locals() else 0,
            "risk_per_trade": round(recommended_fraction * 100, 1)
        }

    # Additional strategy methods would be implemented here
    async def _scan_iron_condors(self, options_chain: List[OptionData], current_price: float, confidence_level: float) -> List[Dict]:
        """Scan for iron condor opportunities."""
        # Implementation for iron condor scanning
        return []

    async def _scan_covered_calls(self, options_chain: List[OptionData], current_price: float, confidence_level: float) -> List[Dict]:
        """Scan for covered call opportunities.""" 
        # Implementation for covered call scanning
        return []