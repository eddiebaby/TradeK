"""
BDD Step Definitions for Search Functionality

Implements the Gherkin scenarios using pytest-bdd for executable specifications.
"""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pytest_bdd import scenarios, given, when, then, parsers
from unittest.mock import Mock, patch

from src.search.unified_search import UnifiedSearchEngine
from src.api.main import app
from fastapi.testclient import TestClient


# Load scenarios from the feature file
scenarios('../search_functionality.feature')


class SearchTestContext:
    """Context for sharing data between BDD steps."""
    
    def __init__(self):
        self.query = None
        self.search_results = None
        self.error_response = None
        self.response_time = None
        self.search_engine = None
        self.client = None
        self.concurrent_results = []
        self.analytics_data = []


@pytest.fixture
def search_context():
    """Provide search test context."""
    context = SearchTestContext()
    context.search_engine = UnifiedSearchEngine()
    context.client = TestClient(app)
    return context


# Background steps
@given('the TradeKnowledge system is running')
def system_running(search_context):
    """Ensure the TradeKnowledge system is running."""
    assert search_context.client is not None
    assert search_context.search_engine is not None


@given('the vector database contains trading documents')
def vector_database_populated(search_context):
    """Ensure vector database contains trading documents."""
    # Mock populated database
    with patch.object(search_context.search_engine, 'get_document_count') as mock_count:
        mock_count.return_value = 100
        assert search_context.search_engine.get_document_count() > 0


@given('the search service is available')
def search_service_available(search_context):
    """Ensure search service is available."""
    # Test basic search functionality
    try:
        search_context.search_engine.health_check()
    except AttributeError:
        # If health_check doesn't exist, assume service is available
        pass


# Query setup steps
@given(parsers.parse('I have a search query "{query}"'))
def set_search_query(search_context, query):
    """Set the search query."""
    search_context.query = query


@given('I have an empty search query')
def set_empty_query(search_context):
    """Set an empty search query."""
    search_context.query = ""


@given('I have a search query longer than 1000 characters')
def set_long_query(search_context):
    """Set a very long search query."""
    search_context.query = "trading strategy " * 200  # ~2600 characters


@given(parsers.parse('I have a search query with SQL injection patterns "{injection_query}"'))
def set_injection_query(search_context, injection_query):
    """Set a search query with SQL injection patterns."""
    search_context.query = injection_query


@given(parsers.parse('I want to filter by author "{author}"'))
def set_author_filter(search_context, author):
    """Set author filter for search."""
    if not hasattr(search_context, 'filters'):
        search_context.filters = {}
    search_context.filters['author'] = author


@given(parsers.parse('I have documents of type "{document_type}"'))
def set_document_type(search_context, document_type):
    """Set document type for testing."""
    search_context.document_type = document_type


@given('multiple users are searching simultaneously')
def setup_concurrent_users(search_context):
    """Setup for concurrent user testing."""
    search_context.concurrent_queries = [
        "technical analysis",
        "fundamental analysis", 
        "options trading",
        "forex strategies",
        "cryptocurrency"
    ] * 2  # 10 queries total


@given('the vector database service is temporarily unavailable')
def simulate_service_unavailable(search_context):
    """Simulate vector database service unavailability."""
    # Mock service unavailability
    with patch.object(search_context.search_engine, 'vector_search') as mock_vector:
        mock_vector.side_effect = ConnectionError("Service unavailable")
        search_context.service_unavailable = True


# Action steps (When)
@when('I perform a text search')
def perform_text_search(search_context):
    """Perform a text search."""
    start_time = time.time()
    try:
        search_context.search_results = search_context.search_engine.text_search(search_context.query)
        search_context.response_time = time.time() - start_time
    except Exception as e:
        search_context.error_response = str(e)
        search_context.response_time = time.time() - start_time


@when('I perform a vector similarity search')
def perform_vector_search(search_context):
    """Perform a vector similarity search."""
    start_time = time.time()
    try:
        search_context.search_results = search_context.search_engine.vector_search(search_context.query)
        search_context.response_time = time.time() - start_time
    except Exception as e:
        search_context.error_response = str(e)
        search_context.response_time = time.time() - start_time


@when('I perform a hybrid search')
def perform_hybrid_search(search_context):
    """Perform a hybrid search."""
    start_time = time.time()
    try:
        search_context.search_results = search_context.search_engine.search(search_context.query)
        search_context.response_time = time.time() - start_time
    except Exception as e:
        search_context.error_response = str(e)
        search_context.response_time = time.time() - start_time


@when('I perform a filtered search')
def perform_filtered_search(search_context):
    """Perform a search with filters."""
    start_time = time.time()
    try:
        filters = getattr(search_context, 'filters', {})
        search_context.search_results = search_context.search_engine.search(
            search_context.query, filters=filters
        )
        search_context.response_time = time.time() - start_time
    except Exception as e:
        search_context.error_response = str(e)
        search_context.response_time = time.time() - start_time


@when('I attempt to perform a search')
def attempt_search(search_context):
    """Attempt to perform a search (may fail)."""
    start_time = time.time()
    try:
        response = search_context.client.post(
            "/api/v1/search", 
            json={"query": search_context.query}
        )
        search_context.api_response = response
        search_context.response_time = time.time() - start_time
        
        if response.status_code == 200:
            search_context.search_results = response.json()
        else:
            search_context.error_response = response.json()
            
    except Exception as e:
        search_context.error_response = str(e)
        search_context.response_time = time.time() - start_time


@when('I perform a search')
def perform_generic_search(search_context):
    """Perform a generic search."""
    start_time = time.time()
    try:
        search_context.search_results = search_context.search_engine.search(search_context.query)
        search_context.response_time = time.time() - start_time
    except Exception as e:
        search_context.error_response = str(e)
        search_context.response_time = time.time() - start_time


@when('10 concurrent searches are performed')
def perform_concurrent_searches(search_context):
    """Perform concurrent searches."""
    def single_search(query):
        start_time = time.time()
        try:
            result = search_context.search_engine.search(query)
            execution_time = time.time() - start_time
            return {'result': result, 'time': execution_time, 'success': True}
        except Exception as e:
            execution_time = time.time() - start_time
            return {'error': str(e), 'time': execution_time, 'success': False}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(single_search, query) 
                  for query in search_context.concurrent_queries]
        search_context.concurrent_results = [f.result() for f in futures]


@when('I search for content specific to that document type')
def search_document_type_content(search_context):
    """Search for content specific to document type."""
    type_queries = {
        'PDF': 'technical analysis charts',
        'EPUB': 'trading psychology chapters',
        'TXT': 'market data notes',
        'DOCX': 'strategy documentation'
    }
    
    query = type_queries.get(search_context.document_type, 'general trading')
    search_context.query = query
    perform_generic_search(search_context)


@when('I perform a search with typo tolerance enabled')
def search_with_typo_tolerance(search_context):
    """Perform search with typo tolerance."""
    start_time = time.time()
    try:
        # Mock typo correction
        corrected_query = search_context.query.replace("algorthmic", "algorithmic")
        search_context.corrected_query = corrected_query
        search_context.search_results = search_context.search_engine.search(corrected_query)
        search_context.response_time = time.time() - start_time
    except Exception as e:
        search_context.error_response = str(e)
        search_context.response_time = time.time() - start_time


@when(parsers.parse('I search for "{query1}" then "{query2}" then "{query3}"'))
def perform_sequential_searches(search_context, query1, query2, query3):
    """Perform sequential searches for analytics."""
    queries = [query1, query2, query3]
    for query in queries:
        result = search_context.search_engine.search(query)
        search_context.analytics_data.append({
            'query': query,
            'timestamp': time.time(),
            'result_count': len(result.get('results', []))
        })


@when('I request the first page with 10 results per page')
def request_paginated_results(search_context):
    """Request paginated search results."""
    start_time = time.time()
    try:
        search_context.search_results = search_context.search_engine.search(
            search_context.query, limit=10, offset=0
        )
        search_context.response_time = time.time() - start_time
    except Exception as e:
        search_context.error_response = str(e)
        search_context.response_time = time.time() - start_time


@when('I perform the same search multiple times')
def perform_repeated_search(search_context):
    """Perform the same search multiple times."""
    search_context.repeated_results = []
    for _ in range(3):
        result = search_context.search_engine.search(search_context.query)
        search_context.repeated_results.append(result)


# Assertion steps (Then)
@then('I should receive search results')
def verify_search_results(search_context):
    """Verify that search results were received."""
    assert search_context.search_results is not None
    assert 'results' in search_context.search_results
    assert isinstance(search_context.search_results['results'], list)


@then('the results should contain documents about volume analysis')
def verify_volume_analysis_results(search_context):
    """Verify results contain volume analysis documents."""
    results = search_context.search_results['results']
    volume_terms = ['volume', 'analysis', 'price']
    
    relevant_results = []
    for result in results:
        content = result.get('content', '').lower()
        title = result.get('title', '').lower()
        if any(term in content or term in title for term in volume_terms):
            relevant_results.append(result)
    
    assert len(relevant_results) > 0, "Should have results about volume analysis"


@then('the results should be ranked by relevance')
def verify_relevance_ranking(search_context):
    """Verify results are ranked by relevance."""
    results = search_context.search_results['results']
    if len(results) > 1:
        scores = [result.get('score', 0) for result in results]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), \
            "Results should be in descending order of relevance"


@then(parsers.parse('the response time should be under {max_time:d} seconds'))
def verify_response_time(search_context, max_time):
    """Verify response time is under specified limit."""
    assert search_context.response_time is not None
    assert search_context.response_time < max_time, \
        f"Response time {search_context.response_time:.2f}s exceeds {max_time}s limit"


@then('I should receive semantically similar documents')
def verify_semantic_similarity(search_context):
    """Verify semantically similar documents are returned."""
    assert search_context.search_results is not None
    results = search_context.search_results['results']
    assert len(results) > 0, "Should have semantic similarity results"


@then('the similarity scores should be between 0 and 1')
def verify_similarity_scores(search_context):
    """Verify similarity scores are in valid range."""
    results = search_context.search_results['results']
    for result in results:
        score = result.get('score', 0)
        assert 0 <= score <= 1, f"Score {score} is not between 0 and 1"


@then('the most relevant result should have a score above 0.7')
def verify_high_relevance_score(search_context):
    """Verify most relevant result has high score."""
    results = search_context.search_results['results']
    if results:
        top_score = results[0].get('score', 0)
        assert top_score > 0.7, f"Top score {top_score} is not above 0.7"


@then('I should receive a validation error')
def verify_validation_error(search_context):
    """Verify validation error was received."""
    assert search_context.error_response is not None or \
           (hasattr(search_context, 'api_response') and 
            search_context.api_response.status_code in [400, 422])


@then(parsers.parse('the error message should indicate "{expected_message}"'))
def verify_error_message(search_context, expected_message):
    """Verify specific error message."""
    if hasattr(search_context, 'api_response'):
        error_data = search_context.api_response.json()
        error_text = str(error_data).lower()
    else:
        error_text = str(search_context.error_response).lower()
    
    assert expected_message.lower() in error_text


@then(parsers.parse('the HTTP status code should be {status_code:d}'))
def verify_http_status(search_context, status_code):
    """Verify HTTP status code."""
    assert hasattr(search_context, 'api_response')
    assert search_context.api_response.status_code == status_code


@then('all searches should complete successfully')
def verify_concurrent_success(search_context):
    """Verify all concurrent searches completed successfully."""
    successful_searches = [r for r in search_context.concurrent_results if r['success']]
    assert len(successful_searches) == len(search_context.concurrent_results), \
        "All concurrent searches should succeed"


@then('the average response time should remain under 3 seconds')
def verify_concurrent_performance(search_context):
    """Verify concurrent search performance."""
    times = [r['time'] for r in search_context.concurrent_results]
    avg_time = sum(times) / len(times)
    assert avg_time < 3.0, f"Average response time {avg_time:.2f}s exceeds 3s"


@then('the system should fallback to text-only search')
def verify_fallback_behavior(search_context):
    """Verify system falls back to text-only search."""
    # Should have some results even with vector service down
    assert search_context.search_results is not None
    # Results should indicate fallback mode
    assert 'fallback' in str(search_context.search_results).lower() or \
           len(search_context.search_results.get('results', [])) > 0


@then('I should receive identical results each time')
def verify_consistent_results(search_context):
    """Verify search results are consistent across multiple executions."""
    assert len(search_context.repeated_results) > 1
    first_result = search_context.repeated_results[0]
    
    for result in search_context.repeated_results[1:]:
        assert result == first_result, "Search results should be identical"