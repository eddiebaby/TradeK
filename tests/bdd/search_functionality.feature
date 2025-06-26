Feature: Trade Knowledge Search Functionality
  As a trader using the TradeKnowledge system
  I want to search for trading information and strategies
  So that I can make informed trading decisions

  Background:
    Given the TradeKnowledge system is running
    And the vector database contains trading documents
    And the search service is available

  Scenario: Basic text search returns relevant results
    Given I have a search query "volume price analysis"
    When I perform a text search
    Then I should receive search results
    And the results should contain documents about volume analysis
    And the results should be ranked by relevance
    And the response time should be under 2 seconds

  Scenario: Vector similarity search finds semantically related content
    Given I have a query about "market volatility patterns"
    When I perform a vector similarity search
    Then I should receive semantically similar documents
    And the similarity scores should be between 0 and 1
    And the most relevant result should have a score above 0.7
    And the results should include different variations of volatility concepts

  Scenario: Hybrid search combines text and vector results effectively
    Given I have a complex query "algorithmic trading machine learning strategies"
    When I perform a hybrid search
    Then I should receive results from both text and vector search
    And the results should be merged and deduplicated
    And the combined relevance scoring should prioritize exact matches
    And vector similarity should boost semantically related content

  Scenario: Search with filters narrows results appropriately
    Given I have a search query "technical analysis"
    And I want to filter by author "Brian Shannon"
    When I perform a filtered search
    Then I should receive only results from the specified author
    And the results should still be relevant to technical analysis
    And the total result count should be less than unfiltered search

  Scenario: Empty search query returns validation error
    Given I have an empty search query
    When I attempt to perform a search
    Then I should receive a validation error
    And the error message should indicate "Query cannot be empty"
    And the HTTP status code should be 400

  Scenario: Extremely long search query is truncated safely
    Given I have a search query longer than 1000 characters
    When I perform a search
    Then the query should be truncated to a safe length
    And I should receive relevant results based on the truncated query
    And no system errors should occur

  Scenario: Search handles special characters and injection attempts
    Given I have a search query with SQL injection patterns "'; DROP TABLE books; --"
    When I perform a search
    Then the query should be safely sanitized
    And I should receive search results or an empty result set
    And no database errors should occur
    And the system should remain secure

  Scenario: Concurrent searches maintain performance
    Given multiple users are searching simultaneously
    When 10 concurrent searches are performed
    Then all searches should complete successfully
    And the average response time should remain under 3 seconds
    And no search should timeout or fail due to concurrency

  Scenario: Search gracefully handles service unavailability
    Given the vector database service is temporarily unavailable
    When I perform a search
    Then the system should fallback to text-only search
    And I should receive a partial results warning
    And the search should still return some relevant results
    And the system should not crash or return 500 errors

  Scenario Outline: Search supports different document types
    Given I have documents of type "<document_type>"
    When I search for content specific to that document type
    Then I should receive results from "<document_type>" documents
    And the content should be properly parsed and indexed
    And the search should work across different file formats

    Examples:
      | document_type |
      | PDF           |
      | EPUB          |
      | TXT           |
      | DOCX          |

  Scenario: Search provides query suggestions for typos
    Given I have a search query with a typo "algorthmic trading"
    When I perform a search with typo tolerance enabled
    Then I should receive a suggested correction "algorithmic trading"
    And I should see results for both the original and corrected terms
    And the system should indicate the auto-correction

  Scenario: Search analytics track user behavior
    Given I perform multiple searches during a session
    When I search for "options trading" then "put call ratio" then "volatility skew"
    Then the system should track the search progression
    And query analytics should capture search patterns
    And no personal information should be logged
    And the analytics should help improve search relevance

  Scenario: Search results include proper metadata
    Given I perform a search for "momentum indicators"
    When I receive search results
    Then each result should include document metadata
    And metadata should contain title, author, and publication date
    And chunk information should show the source location
    And relevance scores should be included for each result

  Scenario: Search supports pagination for large result sets
    Given I have a search query that returns more than 10 results
    When I request the first page with 10 results per page
    Then I should receive exactly 10 results
    And the response should include pagination information
    And I should be able to request subsequent pages
    And the total result count should be accurate

  Scenario: Search maintains consistent results for same query
    Given I have a specific search query "fibonacci retracements"
    When I perform the same search multiple times
    Then I should receive identical results each time
    And the result ordering should be consistent
    And the relevance scores should not vary
    And no random elements should affect deterministic results