"""
Advanced Search Result Ranking Engine

This module implements sophisticated ranking algorithms that combine
multiple signals to provide the most relevant results.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..core.models import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RankingSignal:
    """Individual ranking signal with weight"""

    name: str
    score: float
    weight: float
    explanation: str = ""


@dataclass
class RankingResult:
    """Result with ranking details"""

    result: SearchResult
    final_score: float
    signals: list[RankingSignal] = field(default_factory=list)
    ranking_explanation: str = ""


class HybridSearchRanker:
    """
    Advanced ranking engine that combines multiple signals for optimal relevance.

    Ranking Factors:
    - Semantic similarity score
    - Exact text match score
    - Content type relevance
    - Recency/freshness
    - Authority/source quality
    - User interaction history
    - Query-document length match
    - Structural importance (headers, summaries)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize ranking engine with configuration"""
        self.config = config or {}

        # Default ranking weights
        self.weights = {
            "semantic_similarity": self.config.get("semantic_weight", 0.4),
            "exact_match": self.config.get("exact_weight", 0.3),
            "content_type": self.config.get("content_type_weight", 0.1),
            "recency": self.config.get("recency_weight", 0.05),
            "authority": self.config.get("authority_weight", 0.1),
            "interaction": self.config.get("interaction_weight", 0.05),
            "length_match": self.config.get("length_match_weight", 0.05),
            "structural": self.config.get("structural_weight", 0.1),
        }

        # Content type preferences
        self.content_type_scores = {
            "strategy": 1.0,
            "summary": 0.9,
            "text": 0.8,
            "table": 0.7,
            "code": 0.6,
        }

        # Book authority scores (can be learned from user feedback)
        self.book_authority = defaultdict(lambda: 0.5)

        # User interaction tracking
        self.click_through_rates = defaultdict(lambda: 0.5)
        self.dwell_times = defaultdict(lambda: 30.0)  # seconds

        logger.info("Hybrid search ranker initialized")

    def calculate_semantic_similarity_score(
        self, result: SearchResult
    ) -> RankingSignal:
        """Calculate semantic similarity ranking signal"""
        # Use the existing similarity score from vector search
        raw_score = result.score

        # Normalize and enhance
        normalized_score = min(1.0, max(0.0, raw_score))

        # Apply sigmoid curve for better distribution
        enhanced_score = 1 / (1 + math.exp(-10 * (normalized_score - 0.5)))

        return RankingSignal(
            name="semantic_similarity",
            score=enhanced_score,
            weight=self.weights["semantic_similarity"],
            explanation=f"Vector similarity: {raw_score:.3f} → {enhanced_score:.3f}",
        )

    def calculate_exact_match_score(
        self, query: str, result: SearchResult
    ) -> RankingSignal:
        """Calculate exact text match ranking signal"""
        text = result.text.lower()
        query_lower = query.lower()
        query_terms = query_lower.split()

        if not query_terms:
            return RankingSignal("exact_match", 0.0, self.weights["exact_match"])

        # Term frequency scoring
        term_scores = []
        for term in query_terms:
            if len(term) < 2:  # Skip very short terms
                continue

            term_count = text.count(term)
            # TF-IDF style scoring (simplified)
            tf = term_count / len(text.split())
            score = min(1.0, tf * 100)  # Scale and cap
            term_scores.append(score)

        if not term_scores:
            exact_score = 0.0
        else:
            # Combine term scores
            exact_score = sum(term_scores) / len(term_scores)

        # Bonus for phrase matches
        if query_lower in text:
            exact_score = min(1.0, exact_score + 0.3)

        # Bonus for title/header matches
        if hasattr(result, "metadata") and result.metadata:
            header = result.metadata.get("header", "").lower()
            if query_lower in header:
                exact_score = min(1.0, exact_score + 0.2)

        return RankingSignal(
            name="exact_match",
            score=exact_score,
            weight=self.weights["exact_match"],
            explanation=f"Term matches and phrase detection: {exact_score:.3f}",
        )

    def calculate_content_type_score(
        self, query: str, result: SearchResult
    ) -> RankingSignal:
        """Calculate content type relevance score"""
        # Determine expected content type from query
        query_lower = query.lower()

        # Query intent detection
        if any(
            word in query_lower
            for word in ["strategy", "algorithm", "trading strategy"]
        ):
            preferred_type = "strategy"
        elif any(
            word in query_lower for word in ["summary", "overview", "introduction"]
        ):
            preferred_type = "summary"
        elif any(
            word in query_lower for word in ["code", "implementation", "function"]
        ):
            preferred_type = "code"
        elif any(word in query_lower for word in ["table", "data", "statistics"]):
            preferred_type = "table"
        else:
            preferred_type = "text"

        # Get result content type
        result_type = "text"  # default
        if hasattr(result, "chunk_type") and result.chunk_type:
            result_type = (
                result.chunk_type.value
                if hasattr(result.chunk_type, "value")
                else str(result.chunk_type)
            )
        elif hasattr(result, "metadata") and result.metadata:
            result_type = result.metadata.get("chunk_type", "text")

        # Calculate relevance score
        if result_type == preferred_type:
            content_score = 1.0
        else:
            content_score = self.content_type_scores.get(result_type, 0.5)

        return RankingSignal(
            name="content_type",
            score=content_score,
            weight=self.weights["content_type"],
            explanation=f"Query type '{preferred_type}' vs result type '{result_type}': {content_score:.3f}",
        )

    def calculate_recency_score(self, result: SearchResult) -> RankingSignal:
        """Calculate recency/freshness score"""
        # Get creation date from metadata
        created_date = None
        if hasattr(result, "metadata") and result.metadata:
            date_str = result.metadata.get("created_at") or result.metadata.get(
                "indexed_at"
            )
            if date_str:
                try:
                    created_date = datetime.fromisoformat(
                        date_str.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    pass

        if not created_date:
            # No date info, use neutral score
            return RankingSignal(
                "recency", 0.5, self.weights["recency"], "No date information"
            )

        # Calculate age in days
        age_days = (datetime.now(created_date.tzinfo) - created_date).days

        # Decay function: newer content scores higher
        # Score drops to 0.5 after 365 days, 0.1 after 1000 days
        recency_score = max(0.1, 1.0 - (age_days / 1000))

        return RankingSignal(
            name="recency",
            score=recency_score,
            weight=self.weights["recency"],
            explanation=f"Age: {age_days} days → {recency_score:.3f}",
        )

    def calculate_authority_score(self, result: SearchResult) -> RankingSignal:
        """Calculate source authority score"""
        # Get book/source identifier
        book_id = getattr(result, "book_id", "unknown")

        # Look up authority score
        authority_score = self.book_authority[book_id]

        # Boost for certain book types
        if hasattr(result, "metadata") and result.metadata:
            source_type = result.metadata.get("source_type", "")
            if "academic" in source_type.lower():
                authority_score = min(1.0, authority_score + 0.2)
            elif "professional" in source_type.lower():
                authority_score = min(1.0, authority_score + 0.1)

        return RankingSignal(
            name="authority",
            score=authority_score,
            weight=self.weights["authority"],
            explanation=f"Source authority for {book_id}: {authority_score:.3f}",
        )

    def calculate_interaction_score(self, result: SearchResult) -> RankingSignal:
        """Calculate user interaction score"""
        # Get result identifier
        result_id = getattr(result, "id", "unknown")

        # Get click-through rate and dwell time
        ctr = self.click_through_rates[result_id]
        dwell_time = self.dwell_times[result_id]

        # Combine CTR and dwell time into interaction score
        # High CTR (>0.3) and high dwell time (>60s) = good
        ctr_score = min(1.0, ctr / 0.3)  # Normalize to CTR of 30%
        dwell_score = min(1.0, dwell_time / 60.0)  # Normalize to 60 seconds

        interaction_score = (ctr_score + dwell_score) / 2

        return RankingSignal(
            name="interaction",
            score=interaction_score,
            weight=self.weights["interaction"],
            explanation=f"CTR: {ctr:.3f}, Dwell: {dwell_time:.1f}s → {interaction_score:.3f}",
        )

    def calculate_length_match_score(
        self, query: str, result: SearchResult
    ) -> RankingSignal:
        """Calculate query-document length matching score"""
        query_length = len(query.split())
        text_length = len(result.text.split())

        # Optimal chunk length depends on query length
        if query_length <= 3:
            # Short queries prefer concise answers
            optimal_length = 100
        elif query_length <= 8:
            # Medium queries prefer moderate chunks
            optimal_length = 200
        else:
            # Long queries may need detailed explanations
            optimal_length = 400

        # Calculate distance from optimal
        length_diff = abs(text_length - optimal_length)
        max_penalty = optimal_length  # Maximum penalty

        length_score = max(0.1, 1.0 - (length_diff / max_penalty))

        return RankingSignal(
            name="length_match",
            score=length_score,
            weight=self.weights["length_match"],
            explanation=f"Query: {query_length} words, Text: {text_length} words, Optimal: {optimal_length} → {length_score:.3f}",
        )

    def calculate_structural_score(self, result: SearchResult) -> RankingSignal:
        """Calculate structural importance score"""
        structural_score = 0.5  # Default

        if hasattr(result, "metadata") and result.metadata:
            # Boost for headers and summaries
            if result.metadata.get("header"):
                structural_score += 0.3

            # Boost for important sections
            structure_info = result.metadata.get("structure_info", {})
            if structure_info.get("header_type") == "chapter":
                structural_score += 0.2
            elif structure_info.get("header_type") == "section":
                structural_score += 0.1

            # Boost for special content
            if structure_info.get("contains_strategy"):
                structural_score += 0.2
            if structure_info.get("contains_example"):
                structural_score += 0.1

        structural_score = min(1.0, structural_score)

        return RankingSignal(
            name="structural",
            score=structural_score,
            weight=self.weights["structural"],
            explanation=f"Structural importance: {structural_score:.3f}",
        )

    def rank_results(
        self,
        query: str,
        semantic_results: list[SearchResult],
        text_results: list[SearchResult] = None,
    ) -> list[RankingResult]:
        """
        Rank search results using hybrid scoring.

        Args:
            query: Original search query
            semantic_results: Results from vector search
            text_results: Results from text search (optional)

        Returns:
            Ranked results with scoring details
        """
        # Combine and deduplicate results
        all_results = {}

        # Add semantic results
        for result in semantic_results:
            result_key = f"{result.book_id}_{result.chunk_index}"
            if result_key not in all_results:
                all_results[result_key] = result

        # Add text results
        if text_results:
            for result in text_results:
                result_key = f"{result.book_id}_{result.chunk_index}"
                if result_key not in all_results:
                    # Text search results might not have similarity scores
                    if not hasattr(result, "score") or result.score is None:
                        result.score = 0.1  # Low semantic score for text-only matches
                    all_results[result_key] = result

        # Rank each result
        ranked_results = []

        for result in all_results.values():
            signals = [
                self.calculate_semantic_similarity_score(result),
                self.calculate_exact_match_score(query, result),
                self.calculate_content_type_score(query, result),
                self.calculate_recency_score(result),
                self.calculate_authority_score(result),
                self.calculate_interaction_score(result),
                self.calculate_length_match_score(query, result),
                self.calculate_structural_score(result),
            ]

            # Calculate weighted final score
            final_score = sum(signal.score * signal.weight for signal in signals)

            # Create ranking explanation
            explanation_parts = []
            for signal in signals:
                contribution = signal.score * signal.weight
                explanation_parts.append(f"{signal.name}: {contribution:.3f}")

            ranking_explanation = (
                " + ".join(explanation_parts) + f" = {final_score:.3f}"
            )

            ranked_result = RankingResult(
                result=result,
                final_score=final_score,
                signals=signals,
                ranking_explanation=ranking_explanation,
            )

            ranked_results.append(ranked_result)

        # Sort by final score
        ranked_results.sort(key=lambda x: x.final_score, reverse=True)

        logger.info(f"Ranked {len(ranked_results)} results for query: '{query}'")

        return ranked_results

    def update_interaction_feedback(
        self, result_id: str, clicked: bool, dwell_time: float
    ):
        """Update interaction metrics based on user feedback"""
        # Simple exponential moving average
        alpha = 0.1  # Learning rate

        if clicked:
            self.click_through_rates[result_id] = (
                alpha * 1.0 + (1 - alpha) * self.click_through_rates[result_id]
            )

        if dwell_time > 0:
            self.dwell_times[result_id] = (
                alpha * dwell_time + (1 - alpha) * self.dwell_times[result_id]
            )

    def update_book_authority(self, book_id: str, authority_score: float):
        """Update book authority score"""
        alpha = 0.05  # Slower learning for authority
        self.book_authority[book_id] = (
            alpha * authority_score + (1 - alpha) * self.book_authority[book_id]
        )

    def get_ranking_stats(self) -> dict[str, Any]:
        """Get ranking engine statistics"""
        return {
            "weights": self.weights,
            "num_books_with_authority": len(self.book_authority),
            "num_results_with_interactions": len(self.click_through_rates),
            "avg_ctr": (
                sum(self.click_through_rates.values()) / len(self.click_through_rates)
                if self.click_through_rates
                else 0
            ),
            "avg_dwell_time": (
                sum(self.dwell_times.values()) / len(self.dwell_times)
                if self.dwell_times
                else 0
            ),
        }


# Global ranking engine instance
_ranking_engine = None


def get_ranking_engine() -> HybridSearchRanker:
    """Get the global ranking engine instance"""
    global _ranking_engine
    if _ranking_engine is None:
        _ranking_engine = HybridSearchRanker()
    return _ranking_engine
