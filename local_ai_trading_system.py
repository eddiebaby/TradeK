#!/usr/bin/env python3
"""
Local AI Trading System - Zero Cloud Tokens
===========================================

Complete offline AI trading system using:
- Qwen2.5-Coder:7b (local Ollama)
- Processed trading books database
- LDES strategy framework
- No external API calls to Anthropic/OpenAI

This system operates entirely locally and can continue working
during API timeouts or rate limits.
"""

import asyncio
import json
import logging
import os
import requests
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenClient:
    """Local Qwen model client via Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "qwen2.5-coder:7b"
        self.available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if Qwen model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                for model in models:
                    if 'qwen2.5-coder' in model.get('name', ''):
                        logger.info(f"✅ Qwen model available: {model['name']}")
                        return True
            return False
        except Exception as e:
            logger.warning(f"⚠️ Ollama not accessible: {e}")
            return False
    
    def generate_strategy(self, prompt: str, context: str = "", max_tokens: int = 2000) -> Dict[str, Any]:
        """Generate trading strategy using local Qwen model"""
        if not self.available:
            return {
                "success": False,
                "error": "Qwen model not available",
                "fallback": True
            }
        
        # Enhanced prompt with context
        full_prompt = f"""
You are an expert algorithmic trading developer. Generate a complete, production-ready Python trading strategy.

CONTEXT FROM TRADING BOOKS:
{context}

REQUEST:
{prompt}

REQUIREMENTS:
1. Complete Python class inheriting from TradingStrategy
2. Implement all required methods: generate_signals, update_positions
3. Include proper risk management and position sizing
4. Add comprehensive backtesting capabilities
5. Use pandas, numpy, and standard trading libraries
6. Include detailed comments and docstrings
7. Make it production-ready with error handling

Generate the complete strategy implementation:
"""
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,  # Low temperature for consistent code
                "top_p": 0.9
            }
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            response_time = time.time() - start_time
            
            return {
                "success": True,
                "content": result.get("response", ""),
                "model": self.model,
                "response_time": response_time,
                "tokens_used": len(result.get("response", "").split()) * 1.3,
                "cost": 0.0,  # Local model is free
                "source": "local_qwen"
            }
            
        except Exception as e:
            logger.error(f"Qwen generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback": True
            }

class LocalBookSearch:
    """File-based book search without vector database"""
    
    def __init__(self):
        self.books_dir = Path("books and papers (pdf and epub)")
        self.processed_dir = Path("data/processed_books")
        self.knowledge_base = self._load_knowledge_base()
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load processed book content or create basic knowledge base"""
        knowledge_file = self.processed_dir / "knowledge_base.json"
        
        if knowledge_file.exists():
            try:
                with open(knowledge_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load knowledge base: {e}")
        
        # Create basic knowledge base from available books
        return self._create_basic_knowledge_base()
    
    def _create_basic_knowledge_base(self) -> Dict[str, Any]:
        """Create basic knowledge base from book titles and concepts"""
        logger.info("Creating basic knowledge base from available books...")
        
        knowledge = {
            "books": {},
            "concepts": {
                "momentum_strategies": {
                    "source": "Hilpisch - Python for Algorithmic Trading",
                    "concepts": [
                        "Moving average crossover strategies",
                        "Price momentum indicators",
                        "Trend following systems",
                        "Bollinger Bands trading",
                        "RSI-based momentum detection"
                    ],
                    "code_patterns": [
                        "pd.DataFrame.rolling().mean() for moving averages",
                        "numpy.where() for signal generation",
                        "Vectorized backtesting with pandas",
                        "Risk management with position sizing"
                    ]
                },
                "machine_learning_trading": {
                    "source": "Coqueret & Guida - ML for Factor Investing",
                    "concepts": [
                        "Factor construction and selection",
                        "ML model validation in finance",
                        "Portfolio optimization with ML",
                        "Risk attribution and decomposition"
                    ],
                    "code_patterns": [
                        "sklearn for ML model training",
                        "Factor analysis with PCA",
                        "Cross-validation for financial data",
                        "Portfolio weight optimization"
                    ]
                },
                "risk_management": {
                    "source": "Trading Systems and Methods",
                    "concepts": [
                        "Position sizing using Kelly criterion",
                        "Stop-loss and profit target setting",
                        "Portfolio diversification rules",
                        "Maximum drawdown controls"
                    ],
                    "code_patterns": [
                        "Kelly criterion calculation",
                        "Dynamic position sizing",
                        "Risk-adjusted returns measurement",
                        "Correlation-based diversification"
                    ]
                },
                "high_frequency_trading": {
                    "source": "High-Frequency Trading Guide",
                    "concepts": [
                        "Market microstructure analysis",
                        "Order book dynamics",
                        "Latency optimization",
                        "Market making strategies"
                    ],
                    "code_patterns": [
                        "Order book processing",
                        "Tick-by-tick data analysis",
                        "Execution cost analysis",
                        "Spread capture strategies"
                    ]
                }
            }
        }
        
        # Save for future use
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        with open(self.processed_dir / "knowledge_base.json", 'w') as f:
            json.dump(knowledge, f, indent=2)
        
        logger.info(f"✅ Created knowledge base with {len(knowledge['concepts'])} concept areas")
        return knowledge
    
    def search_relevant_context(self, query: str, limit: int = 3) -> str:
        """Search for relevant context based on query keywords"""
        query_lower = query.lower()
        relevant_concepts = []
        
        # Search for matching concepts
        for concept_name, concept_data in self.knowledge_base["concepts"].items():
            # Check if query matches concept name or content
            if (concept_name.replace("_", " ") in query_lower or
                any(keyword in query_lower for keyword in 
                    [c.lower() for c in concept_data["concepts"][:3]])):
                
                relevant_concepts.append((concept_name, concept_data))
        
        if not relevant_concepts:
            # Fallback to general trading concepts
            relevant_concepts = [
                ("momentum_strategies", self.knowledge_base["concepts"]["momentum_strategies"]),
                ("risk_management", self.knowledge_base["concepts"]["risk_management"])
            ]
        
        # Build context string
        context_parts = []
        for concept_name, concept_data in relevant_concepts[:limit]:
            context_parts.append(f"""
CONCEPT: {concept_name.replace('_', ' ').title()}
SOURCE: {concept_data['source']}

Key Concepts:
{chr(10).join(f"• {concept}" for concept in concept_data['concepts'])}

Implementation Patterns:
{chr(10).join(f"• {pattern}" for pattern in concept_data['code_patterns'])}
""")
        
        context = "\n" + "="*60 + "\n".join(context_parts)
        logger.info(f"🔍 Found context from {len(relevant_concepts)} concept areas")
        return context

class LocalTradingAI:
    """Main local AI trading system"""
    
    def __init__(self):
        self.qwen = QwenClient()
        self.book_search = LocalBookSearch()
        
        logger.info("🤖 Local AI Trading System initialized")
        logger.info(f"📚 Knowledge base: {len(self.book_search.knowledge_base['concepts'])} concept areas")
        logger.info(f"🧠 Qwen model: {'✅ Available' if self.qwen.available else '❌ Unavailable'}")
    
    def generate_strategy(self, request: str) -> Dict[str, Any]:
        """Generate complete trading strategy using local resources only"""
        logger.info(f"🎯 Generating strategy for: {request}")
        
        # Step 1: Search relevant context from books
        context = self.book_search.search_relevant_context(request)
        
        # Step 2: Generate strategy using Qwen + context
        if self.qwen.available:
            result = self.qwen.generate_strategy(request, context)
            
            if result["success"]:
                logger.info(f"✅ Strategy generated in {result['response_time']:.1f}s")
                return result
            else:
                logger.warning("Qwen generation failed, using fallback")
                return self._fallback_strategy(request, context)
        else:
            logger.info("Using fallback strategy generation")
            return self._fallback_strategy(request, context)
    
    def _fallback_strategy(self, request: str, context: str) -> Dict[str, Any]:
        """Fallback strategy when Qwen is unavailable"""
        
        # Generate basic strategy template based on request type
        if "momentum" in request.lower():
            strategy_code = self._generate_momentum_strategy_template()
        elif "machine learning" in request.lower() or "ml" in request.lower():
            strategy_code = self._generate_ml_strategy_template()
        elif "risk" in request.lower():
            strategy_code = self._generate_risk_management_template()
        else:
            strategy_code = self._generate_basic_strategy_template()
        
        return {
            "success": True,
            "content": strategy_code,
            "model": "fallback_template",
            "response_time": 0.1,
            "tokens_used": len(strategy_code.split()) * 1.3,
            "cost": 0.0,
            "source": "local_fallback",
            "context_used": context[:200] + "..." if len(context) > 200 else context
        }
    
    def _generate_momentum_strategy_template(self) -> str:
        """Generate basic momentum strategy template"""
        return '''
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class MomentumTradingStrategy:
    """
    Momentum Trading Strategy - Local Implementation
    
    This strategy uses moving average crossovers and RSI to identify
    momentum trends in financial instruments.
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50, rsi_period: int = 14):
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.positions = {}
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Moving averages
        data['SMA_short'] = data['close'].rolling(window=self.short_window).mean()
        data['SMA_long'] = data['close'].rolling(window=self.long_window).mean()
        
        # RSI calculation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        data = self.calculate_indicators(data)
        
        # Momentum signals
        data['signal'] = np.where(
            (data['SMA_short'] > data['SMA_long']) & (data['RSI'] > 50),
            1,  # Buy signal
            np.where(
                (data['SMA_short'] < data['SMA_long']) & (data['RSI'] < 50),
                -1,  # Sell signal
                0   # Hold
            )
        )
        
        return data
        
    def calculate_position_size(self, capital: float, price: float, risk_per_trade: float = 0.02) -> int:
        """Calculate position size using fixed risk per trade"""
        risk_amount = capital * risk_per_trade
        stop_loss_distance = price * 0.05  # 5% stop loss
        position_size = int(risk_amount / stop_loss_distance)
        return max(1, position_size)
        
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """Simple backtesting implementation"""
        data = self.generate_signals(data)
        
        capital = initial_capital
        position = 0
        trades = []
        
        for i in range(1, len(data)):
            current_price = data.iloc[i]['close']
            signal = data.iloc[i]['signal']
            
            if signal == 1 and position <= 0:  # Buy signal
                shares = self.calculate_position_size(capital, current_price)
                cost = shares * current_price
                if cost <= capital:
                    position += shares
                    capital -= cost
                    trades.append({'action': 'buy', 'price': current_price, 'shares': shares})
                    
            elif signal == -1 and position > 0:  # Sell signal
                proceeds = position * current_price
                capital += proceeds
                trades.append({'action': 'sell', 'price': current_price, 'shares': position})
                position = 0
        
        # Final portfolio value
        final_value = capital + (position * data.iloc[-1]['close'])
        total_return = (final_value - initial_capital) / initial_capital
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'num_trades': len(trades),
            'trades': trades
        }

# Example usage:
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Run strategy
    strategy = MomentumTradingStrategy()
    results = strategy.backtest(data)
    
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")
'''

    def _generate_ml_strategy_template(self) -> str:
        """Generate ML strategy template"""
        return '''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List

class MLTradingStrategy:
    """
    Machine Learning Trading Strategy - Local Implementation
    
    Uses Random Forest to predict price direction based on technical indicators.
    """
    
    def __init__(self, lookback_period: int = 20, prediction_horizon: int = 5):
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ML features from price data"""
        features = data.copy()
        
        # Technical indicators
        features['SMA_10'] = data['close'].rolling(10).mean()
        features['SMA_20'] = data['close'].rolling(20).mean()
        features['RSI'] = self.calculate_rsi(data['close'])
        features['BB_upper'], features['BB_lower'] = self.calculate_bollinger_bands(data['close'])
        features['volume_sma'] = data['volume'].rolling(10).mean()
        
        # Price-based features
        features['price_change'] = data['close'].pct_change()
        features['volatility'] = features['price_change'].rolling(10).std()
        features['momentum'] = data['close'] / data['close'].shift(10) - 1
        
        # Volume-based features
        features['volume_ratio'] = data['volume'] / features['volume_sma']
        
        return features
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
        
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create prediction labels (1: price up, 0: price down)"""
        future_returns = data['close'].shift(-self.prediction_horizon) / data['close'] - 1
        labels = (future_returns > 0).astype(int)
        return labels
        
    def prepare_training_data(self, data: pd.DataFrame):
        """Prepare features and labels for training"""
        features_df = self.create_features(data)
        labels = self.create_labels(data)
        
        # Select feature columns (exclude price columns to avoid lookahead bias)
        feature_columns = ['SMA_10', 'SMA_20', 'RSI', 'BB_upper', 'BB_lower', 
                          'volume_sma', 'price_change', 'volatility', 'momentum', 'volume_ratio']
        
        X = features_df[feature_columns].fillna(0)
        y = labels.fillna(0)
        
        # Remove rows with insufficient data
        valid_indices = X.index[self.lookback_period:-self.prediction_horizon]
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
        return X, y
        
    def train(self, data: pd.DataFrame):
        """Train the ML model"""
        X, y = self.prepare_training_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        self.is_trained = True
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        features_df = self.create_features(data)
        feature_columns = ['SMA_10', 'SMA_20', 'RSI', 'BB_upper', 'BB_lower', 
                          'volume_sma', 'price_change', 'volatility', 'momentum', 'volume_ratio']
        
        X = features_df[feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities

# Example usage:
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Train and evaluate strategy
    strategy = MLTradingStrategy()
    training_results = strategy.train(data)
    
    print(f"Training Accuracy: {training_results['train_accuracy']:.3f}")
    print(f"Test Accuracy: {training_results['test_accuracy']:.3f}")
    print("Feature Importance:")
    for feature, importance in training_results['feature_importance'].items():
        print(f"  {feature}: {importance:.3f}")
'''

    def _generate_risk_management_template(self) -> str:
        """Generate risk management template"""
        return '''
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class RiskManagementSystem:
    """
    Risk Management System - Local Implementation
    
    Implements comprehensive risk controls for trading strategies.
    """
    
    def __init__(self, max_risk_per_trade: float = 0.02, max_portfolio_risk: float = 0.10):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.positions = {}
        self.risk_metrics = {}
        
    def calculate_position_size(self, capital: float, entry_price: float, 
                               stop_loss_price: float) -> int:
        """Calculate position size using Kelly criterion"""
        risk_per_share = abs(entry_price - stop_loss_price)
        max_risk_amount = capital * self.max_risk_per_trade
        
        if risk_per_share > 0:
            position_size = int(max_risk_amount / risk_per_share)
        else:
            position_size = 0
            
        return max(0, position_size)
        
    def calculate_stop_loss(self, entry_price: float, volatility: float, 
                           multiplier: float = 2.0) -> float:
        """Calculate dynamic stop loss based on volatility"""
        stop_distance = volatility * multiplier
        stop_loss = entry_price * (1 - stop_distance)
        return stop_loss
        
    def calculate_portfolio_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk for portfolio"""
        if len(returns) < 30:
            return 0.0
            
        sorted_returns = returns.sort_values()
        var_index = int(len(sorted_returns) * confidence_level)
        var = abs(sorted_returns.iloc[var_index])
        
        return var
        
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0.0
            
        excess_returns = returns.mean() - risk_free_rate / 252  # Daily risk-free rate
        sharpe = excess_returns / returns.std() * np.sqrt(252)  # Annualized
        
        return sharpe
        
    def check_correlation_limits(self, new_position: Dict, existing_positions: List[Dict], 
                                max_correlation: float = 0.7) -> bool:
        """Check if new position violates correlation limits"""
        # Simplified correlation check - in practice, use historical price correlation
        sector_exposure = {}
        
        for position in existing_positions:
            sector = position.get('sector', 'unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + position['weight']
            
        new_sector = new_position.get('sector', 'unknown')
        new_exposure = sector_exposure.get(new_sector, 0) + new_position['weight']
        
        return new_exposure <= max_correlation
        
    def assess_market_regime(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Assess current market regime for risk adjustment"""
        recent_data = market_data.tail(30)  # Last 30 periods
        
        # Calculate regime indicators
        volatility = recent_data['close'].pct_change().std() * np.sqrt(252)
        trend_strength = abs(recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1)
        volume_trend = recent_data['volume'].tail(5).mean() / recent_data['volume'].head(5).mean()
        
        # Market stress indicators
        max_drawdown = self.calculate_max_drawdown(recent_data['close'])
        
        regime_score = {
            'volatility': min(volatility / 0.20, 2.0),  # Normalized to 20% baseline
            'trend_strength': min(trend_strength / 0.10, 2.0),  # Normalized to 10% baseline  
            'volume_surge': min(volume_trend, 3.0),
            'stress_level': min(max_drawdown / 0.15, 2.0)  # Normalized to 15% baseline
        }
        
        return regime_score
        
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        return max_drawdown
        
    def adjust_risk_for_regime(self, base_risk: float, regime_score: Dict[str, float]) -> float:
        """Adjust risk based on market regime"""
        # Reduce risk in high volatility/stress environments
        volatility_adjustment = 1 / (1 + regime_score['volatility'])
        stress_adjustment = 1 / (1 + regime_score['stress_level'])
        
        adjusted_risk = base_risk * volatility_adjustment * stress_adjustment
        
        return max(adjusted_risk, base_risk * 0.25)  # Never go below 25% of base risk

# Example usage:
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Test risk management
    risk_manager = RiskManagementSystem()
    
    # Example calculations
    capital = 100000
    entry_price = 150.0
    volatility = 0.02
    
    stop_loss = risk_manager.calculate_stop_loss(entry_price, volatility)
    position_size = risk_manager.calculate_position_size(capital, entry_price, stop_loss)
    
    print(f"Entry Price: ${entry_price}")
    print(f"Stop Loss: ${stop_loss:.2f}")
    print(f"Position Size: {position_size} shares")
    print(f"Risk Amount: ${(entry_price - stop_loss) * position_size:.2f}")
'''

    def _generate_basic_strategy_template(self) -> str:
        """Generate basic strategy template"""
        return '''
import pandas as pd
import numpy as np
from typing import Dict, List

class BasicTradingStrategy:
    """
    Basic Trading Strategy Template - Local Implementation
    
    A simple buy-and-hold strategy with basic risk management.
    """
    
    def __init__(self, rebalance_frequency: int = 30):
        self.rebalance_frequency = rebalance_frequency
        self.positions = {}
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic signals"""
        # Simple buy and hold with rebalancing
        data['signal'] = 1  # Always buy
        
        # Rebalance periodically
        rebalance_mask = np.arange(len(data)) % self.rebalance_frequency == 0
        data.loc[~rebalance_mask, 'signal'] = 0
        
        return data
        
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """Simple backtesting"""
        data = self.generate_signals(data)
        
        capital = initial_capital
        shares = 0
        
        for i, row in data.iterrows():
            if row['signal'] == 1:
                # Buy shares with available capital
                shares_to_buy = int(capital / row['close'])
                shares += shares_to_buy
                capital -= shares_to_buy * row['close']
        
        # Final value
        final_value = capital + shares * data.iloc[-1]['close']
        total_return = (final_value - initial_capital) / initial_capital
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return
        }

# Example usage
if __name__ == "__main__":
    strategy = BasicTradingStrategy()
    print("Basic Trading Strategy Template Generated")
'''

def main():
    """Main interface for local AI trading system"""
    print("🤖 Local AI Trading System - Zero Cloud Tokens")
    print("="*60)
    
    system = LocalTradingAI()
    
    if len(sys.argv) > 1:
        # Command line mode
        request = " ".join(sys.argv[1:])
        result = system.generate_strategy(request)
        
        print(f"\n📝 Generated Strategy:")
        print("="*60)
        print(result["content"])
        print("\n" + "="*60)
        print(f"⏱️  Generation time: {result.get('response_time', 0):.1f}s")
        print(f"🧠 Model used: {result.get('model', 'unknown')}")
        print(f"💰 Cost: ${result.get('cost', 0):.4f}")
        
    else:
        # Interactive mode
        print("\n🎯 Interactive Strategy Generation")
        print("Type strategy requests or 'quit' to exit")
        print("\nExample requests:")
        print("  • 'momentum trading strategy'")
        print("  • 'machine learning factor investing'") 
        print("  • 'risk management system'")
        print("  • 'high frequency trading strategy'")
        print()
        
        try:
            while True:
                request = input("🔥 Strategy request: ").strip()
                
                if not request:
                    continue
                    
                if request.lower() in ['quit', 'exit', 'q']:
                    break
                
                print(f"\n🚀 Generating strategy...")
                start_time = time.time()
                
                result = system.generate_strategy(request)
                
                print(f"\n📝 Generated Strategy:")
                print("="*60)
                print(result["content"])
                print("\n" + "="*60)
                print(f"⏱️  Generation time: {result.get('response_time', 0):.1f}s")
                print(f"🧠 Model used: {result.get('model', 'unknown')}")
                print(f"💰 Cost: ${result.get('cost', 0):.4f}")
                print("\n" + "="*80 + "\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Session ended")

if __name__ == "__main__":
    main()