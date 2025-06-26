"""
Test payloads and fixtures for search engine testing.

This module contains realistic search queries and expected results
for comprehensive testing of search functionality.
"""

from typing import List, Dict, Any
from datetime import datetime


# Sample search queries for different intents
RESEARCH_QUERIES = [
    "comprehensive analysis of momentum trading strategies and their effectiveness",
    "detailed study of market microstructure effects on algorithmic trading",
    "research on machine learning applications in quantitative finance",
    "investigate correlation between volatility clustering and returns",
    "academic literature review on portfolio optimization techniques"
]

QUICK_LOOKUP_QUERIES = [
    "RSI formula",
    "what is MACD",
    "VaR definition", 
    "Bollinger Bands calculation",
    "Sharpe ratio meaning",
    "Black-Scholes equation",
    "Greeks in options trading"
]

LEARNING_QUERIES = [
    "how to implement moving averages in Python",
    "learn about options pricing models",
    "tutorial on backtesting trading strategies",
    "beginner guide to portfolio optimization",
    "step by step LSTM implementation for time series",
    "understanding technical analysis indicators"
]

COMPARISON_QUERIES = [
    "SMA vs EMA effectiveness comparison",
    "compare momentum and mean reversion strategies",
    "LSTM vs ARIMA for time series prediction",
    "difference between value at risk and expected shortfall",
    "Random Forest vs XGBoost for trading signals",
    "compare different portfolio rebalancing methods"
]

EXPLORATION_QUERIES = [
    "explore alternative trading strategies",
    "discover new quantitative finance techniques",
    "what are emerging trends in algorithmic trading",
    "innovative approaches to risk management",
    "recent developments in market making algorithms"
]

# Sample malicious queries for security testing
MALICIOUS_QUERIES = [
    "'; DROP TABLE chunks; --",
    "' OR 1=1 UNION SELECT * FROM users --",
    "<script>alert('XSS')</script>",
    "../../../etc/passwd",
    "'; DELETE FROM books WHERE 1=1; --",
    "' OR '1'='1",
    "%27%20OR%201=1--",
    "admin'--",
    "' UNION SELECT password FROM users --",
    "<img src=x onerror=alert('XSS')>"
]

# Sample edge case queries
EDGE_CASE_QUERIES = [
    "",  # Empty query
    " ",  # Whitespace only
    "a",  # Single character
    "   trading   strategies   ",  # Multiple spaces
    "TRADING STRATEGIES",  # All caps
    "trading\nstrategies",  # With newlines
    "trading\tstrategies",  # With tabs
    "a" * 1000,  # Very long query
    "🚀📈💰",  # Emoji only
    "algorithmic-trading_strategies.pdf",  # Filename-like
]

# Sample search results for testing ranking
MOCK_SEARCH_RESULTS = [
    {
        "id": "chunk_001",
        "content": "Moving average crossover strategies are fundamental to algorithmic trading",
        "title": "Introduction to Moving Averages",
        "book_id": "trading_book_1",
        "book_title": "Algorithmic Trading Strategies",
        "author": "John Smith", 
        "score": 0.95,
        "semantic_score": 0.9,
        "exact_score": 0.8,
        "page_number": 42,
        "chapter": "Technical Indicators",
        "created_at": "2023-06-15T10:30:00Z",
        "metadata": {
            "difficulty": "beginner",
            "topics": ["moving_averages", "crossover", "signals"],
            "code_examples": True
        }
    },
    {
        "id": "chunk_002", 
        "content": "Risk management is crucial for any trading strategy implementation",
        "title": "Risk Management Fundamentals",
        "book_id": "risk_book_1",
        "book_title": "Trading Risk Management",
        "author": "Jane Doe",
        "score": 0.88,
        "semantic_score": 0.85,
        "exact_score": 0.7,
        "page_number": 15,
        "chapter": "Introduction to Risk",
        "created_at": "2023-07-20T14:15:00Z",
        "metadata": {
            "difficulty": "intermediate",
            "topics": ["risk_management", "position_sizing", "drawdown"],
            "code_examples": False
        }
    },
    {
        "id": "chunk_003",
        "content": "Machine learning models can enhance traditional trading strategies",
        "title": "ML in Trading",
        "book_id": "ml_book_1", 
        "book_title": "Machine Learning for Finance",
        "author": "Alex Johnson",
        "score": 0.82,
        "semantic_score": 0.8,
        "exact_score": 0.6,
        "page_number": 128,
        "chapter": "Predictive Models",
        "created_at": "2023-08-10T09:45:00Z",
        "metadata": {
            "difficulty": "advanced",
            "topics": ["machine_learning", "prediction", "features"],
            "code_examples": True
        }
    },
    {
        "id": "chunk_004",
        "content": "Backtesting allows traders to validate strategy performance on historical data",
        "title": "Strategy Backtesting",
        "book_id": "testing_book_1",
        "book_title": "Strategy Development and Testing", 
        "author": "Mike Wilson",
        "score": 0.79,
        "semantic_score": 0.75,
        "exact_score": 0.65,
        "page_number": 67,
        "chapter": "Performance Evaluation",
        "created_at": "2023-05-30T16:20:00Z",
        "metadata": {
            "difficulty": "intermediate",
            "topics": ["backtesting", "validation", "performance"],
            "code_examples": True
        }
    },
    {
        "id": "chunk_005",
        "content": "Portfolio optimization helps balance risk and return across multiple assets",
        "title": "Modern Portfolio Theory",
        "book_id": "portfolio_book_1",
        "book_title": "Portfolio Management Strategies",
        "author": "Sarah Chen",
        "score": 0.76,
        "semantic_score": 0.72,
        "exact_score": 0.58,
        "page_number": 89,
        "chapter": "Optimization Techniques",
        "created_at": "2023-09-05T11:10:00Z",
        "metadata": {
            "difficulty": "advanced",
            "topics": ["portfolio", "optimization", "diversification"],
            "code_examples": False
        }
    }
]

# Sample filter configurations
SAMPLE_FILTERS = {
    "book_filters": {
        "book_id": "trading_book_1",
        "author": "John Smith",
        "difficulty": "beginner"
    },
    "date_filters": {
        "date_from": "2023-06-01",
        "date_to": "2023-12-31"
    },
    "content_filters": {
        "has_code_examples": True,
        "topics": ["moving_averages", "risk_management"],
        "min_score": 0.7
    },
    "malicious_filters": {
        "book_id": "'; DROP TABLE books; --",
        "author": "<script>alert('xss')</script>",
        "topic": "../../../etc/passwd"
    }
}

# Sample pagination test cases
PAGINATION_TEST_CASES = [
    {"offset": 0, "limit": 10, "expected_results": 10},
    {"offset": 10, "limit": 10, "expected_results": 10},
    {"offset": 20, "limit": 10, "expected_results": 5},  # Partial page
    {"offset": 30, "limit": 10, "expected_results": 0},  # Beyond results
    {"offset": 0, "limit": 100, "expected_results": 25}, # Max results
]

# Sample user contexts for personalization testing
SAMPLE_USER_CONTEXTS = [
    {
        "user_id": "user_beginner",
        "skill_level": "beginner",
        "interests": ["basic_trading", "technical_analysis"],
        "search_history": ["RSI formula", "moving averages", "support resistance"],
        "preferred_complexity": "simple"
    },
    {
        "user_id": "user_intermediate", 
        "skill_level": "intermediate",
        "interests": ["algorithmic_trading", "backtesting", "risk_management"],
        "search_history": ["strategy optimization", "drawdown analysis", "Sharpe ratio"],
        "preferred_complexity": "moderate"
    },
    {
        "user_id": "user_advanced",
        "skill_level": "advanced", 
        "interests": ["machine_learning", "quantitative_analysis", "portfolio_optimization"],
        "search_history": ["LSTM models", "factor models", "Black-Litterman optimization"],
        "preferred_complexity": "complex"
    }
]

# Expected search response structures
EXPECTED_SEARCH_RESPONSE = {
    "query": "moving average strategies",
    "intent": "research",
    "results": [],
    "total_found": 0,
    "processing_time_ms": 0.0,
    "suggestions": [],
    "filters_applied": {},
    "pagination": None
}

# Performance benchmarks
PERFORMANCE_BENCHMARKS = {
    "max_response_time_ms": 500,  # 500ms max response time
    "max_results_per_query": 100,
    "cache_hit_ratio_target": 0.8,  # 80% cache hit ratio
    "concurrent_users_target": 50
}

# Test data for ranking algorithms
RANKING_TEST_DATA = [
    {
        "query": "moving average crossover",
        "results": [
            {"content": "Moving average crossover strategy implementation", "expected_rank": 1},
            {"content": "Technical analysis with moving averages", "expected_rank": 2}, 
            {"content": "Crossover signals in trading systems", "expected_rank": 3},
            {"content": "Portfolio optimization techniques", "expected_rank": 4}
        ]
    },
    {
        "query": "risk management",
        "results": [
            {"content": "Risk management in algorithmic trading", "expected_rank": 1},
            {"content": "Position sizing and risk control", "expected_rank": 2},
            {"content": "Portfolio risk assessment methods", "expected_rank": 3},
            {"content": "Machine learning model validation", "expected_rank": 4}
        ]
    }
]

def get_query_by_intent(intent: str) -> List[str]:
    """Get sample queries for a specific intent"""
    intent_mapping = {
        "research": RESEARCH_QUERIES,
        "quick_lookup": QUICK_LOOKUP_QUERIES,
        "learning": LEARNING_QUERIES,
        "comparison": COMPARISON_QUERIES,
        "exploration": EXPLORATION_QUERIES
    }
    return intent_mapping.get(intent, [])

def get_mock_results(count: int = 5) -> List[Dict[str, Any]]:
    """Get a subset of mock search results"""
    return MOCK_SEARCH_RESULTS[:count]

def create_test_user(skill_level: str = "intermediate") -> Dict[str, Any]:
    """Create a test user context"""
    for user in SAMPLE_USER_CONTEXTS:
        if user["skill_level"] == skill_level:
            return user.copy()
    return SAMPLE_USER_CONTEXTS[1].copy()  # Default to intermediate