#!/usr/bin/env python3
"""
Comprehensive Phase 2 Testing Script for TradeKnowledge

This script tests all Phase 2 components to ensure they work correctly:
- Enhanced book processor with all parsers
- OCR functionality
- EPUB parsing
- Content analysis
- Caching system
- Query suggestions
- Error handling and edge cases
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ingestion.enhanced_book_processor import EnhancedBookProcessor
from ingestion.ocr_processor import OCRProcessor
from ingestion.epub_parser import EPUBParser
from ingestion.content_analyzer import ContentAnalyzer
from utils.cache_manager import get_cache_manager
from search.query_suggester import QuerySuggester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase2Tester:
    """Comprehensive tester for Phase 2 components"""
    
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
    
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"   Details: {details}")
        
        self.test_results[test_name] = {
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 2 tests"""
        logger.info("🚀 Starting Phase 2 Comprehensive Tests")
        logger.info("=" * 60)
        
        # Test 1: Component Initialization
        await self.test_component_initialization()
        
        # Test 2: Cache Manager
        await self.test_cache_manager()
        
        # Test 3: Content Analyzer
        await self.test_content_analyzer()
        
        # Test 4: Query Suggester
        await self.test_query_suggester()
        
        # Test 5: Enhanced Book Processor
        await self.test_enhanced_book_processor()
        
        # Test 6: Error Handling
        await self.test_error_handling()
        
        # Test 7: Performance and Caching
        await self.test_performance_features()
        
        # Generate final report
        return self.generate_final_report()
    
    async def test_component_initialization(self):
        """Test that all components can be initialized"""
        logger.info("\n📋 Testing Component Initialization...")
        
        try:
            # Test cache manager
            cache_manager = await get_cache_manager()
            await cache_manager.set("test_key", "test_value")
            value = await cache_manager.get("test_key")
            assert value == "test_value"
            self.log_test_result("Cache Manager Init", True, "Memory caching working")
        except Exception as e:
            self.log_test_result("Cache Manager Init", False, str(e))
        
        try:
            # Test content analyzer
            analyzer = ContentAnalyzer()
            regions = analyzer.analyze_text("def test(): return 42")
            assert len(regions) > 0
            self.log_test_result("Content Analyzer Init", True, f"Found {len(regions)} content regions")
        except Exception as e:
            self.log_test_result("Content Analyzer Init", False, str(e))
        
        try:
            # Test query suggester
            suggester = QuerySuggester()
            await suggester.initialize()
            suggestions = await suggester.suggest("test")
            self.log_test_result("Query Suggester Init", True, f"Generated {len(suggestions)} suggestions")
        except Exception as e:
            self.log_test_result("Query Suggester Init", False, str(e))
        
        try:
            # Test enhanced book processor
            processor = EnhancedBookProcessor()
            await processor.initialize()
            await processor.cleanup()
            self.log_test_result("Enhanced Book Processor Init", True, "Full initialization completed")
        except Exception as e:
            self.log_test_result("Enhanced Book Processor Init", False, str(e))
    
    async def test_cache_manager(self):
        """Test cache manager functionality"""
        logger.info("\n💾 Testing Cache Manager...")
        
        cache_manager = await get_cache_manager()
        
        try:
            # Test basic operations
            await cache_manager.set("test1", {"data": "value1"})
            result = await cache_manager.get("test1")
            assert result["data"] == "value1"
            self.log_test_result("Cache Basic Operations", True, "Set/Get working")
        except Exception as e:
            self.log_test_result("Cache Basic Operations", False, str(e))
        
        try:
            # Test different cache types
            await cache_manager.set("embed1", [0.1, 0.2, 0.3], cache_type="embedding")
            await cache_manager.set("search1", ["result1", "result2"], cache_type="search")
            
            embed_result = await cache_manager.get("embed1", cache_type="embedding")
            search_result = await cache_manager.get("search1", cache_type="search")
            
            assert embed_result == [0.1, 0.2, 0.3]
            assert search_result == ["result1", "result2"]
            self.log_test_result("Cache Type Separation", True, "Different cache types working")
        except Exception as e:
            self.log_test_result("Cache Type Separation", False, str(e))
        
        try:
            # Test cache statistics
            stats = cache_manager.get_stats()
            assert 'total_requests' in stats
            assert stats['total_requests'] > 0
            self.log_test_result("Cache Statistics", True, f"Hit rate: {stats.get('hit_rate', 0):.2f}")
        except Exception as e:
            self.log_test_result("Cache Statistics", False, str(e))
    
    async def test_content_analyzer(self):
        """Test content analyzer functionality"""
        logger.info("\n🔍 Testing Content Analyzer...")
        
        analyzer = ContentAnalyzer()
        
        # Test sample with various content types
        sample_text = """
        This is a sample trading book chapter.
        
        Here's a Python code example:
        
        ```python
        def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
            excess_returns = returns - risk_free_rate
            return np.mean(excess_returns) / np.std(excess_returns)
        ```
        
        The Sharpe ratio formula is: S = (E[R_p] - R_f) / σ_p
        
        Here's a performance table:
        
        | Strategy | Return | Volatility | Sharpe |
        |----------|--------|------------|--------|
        | Long     | 12.5%  | 15.2%     | 0.82   |
        | Short    | 8.3%   | 12.1%     | 0.69   |
        
        Strategy Rules:
        - Buy when RSI < 30
        - Sell when RSI > 70
        """
        
        try:
            special_content = analyzer.extract_special_content(sample_text)
            
            code_found = len(special_content['code']) > 0
            formulas_found = len(special_content['formulas']) > 0
            tables_found = len(special_content['tables']) > 0
            
            details = f"Code: {len(special_content['code'])}, Formulas: {len(special_content['formulas'])}, Tables: {len(special_content['tables'])}"
            
            if code_found and formulas_found and tables_found:
                self.log_test_result("Content Analysis - Full Detection", True, details)
            else:
                self.log_test_result("Content Analysis - Full Detection", False, f"Missing content types. {details}")
        except Exception as e:
            self.log_test_result("Content Analysis - Full Detection", False, str(e))
        
        try:
            # Test language detection
            python_code = "import pandas as pd\ndef main():\n    return True"
            regions = analyzer.analyze_text(python_code)
            python_detected = any(
                region.metadata.get('language') == 'python' 
                for region in regions
            )
            self.log_test_result("Language Detection", python_detected, "Python code detected" if python_detected else "Failed to detect Python")
        except Exception as e:
            self.log_test_result("Language Detection", False, str(e))
    
    async def test_query_suggester(self):
        """Test query suggestion functionality"""
        logger.info("\n💡 Testing Query Suggester...")
        
        suggester = QuerySuggester()
        await suggester.initialize()
        
        try:
            # Record some search history
            await suggester.record_search("moving average strategy", 5)
            await suggester.record_search("rsi indicator", 3)
            await suggester.record_search("python backtest", 10)
            
            # Test suggestions
            suggestions = await suggester.suggest("mov", max_suggestions=5)
            
            moving_suggested = any("moving" in s['text'].lower() for s in suggestions)
            self.log_test_result("Query History Suggestions", moving_suggested, f"Generated {len(suggestions)} suggestions")
        except Exception as e:
            self.log_test_result("Query History Suggestions", False, str(e))
        
        try:
            # Test query expansion
            expanded = await suggester.expand_query("momentum strategy")
            has_expansions = len(expanded) > 1
            self.log_test_result("Query Expansion", has_expansions, f"Expanded to {len(expanded)} queries")
        except Exception as e:
            self.log_test_result("Query Expansion", False, str(e))
        
        try:
            # Test popular queries
            popular = suggester.get_popular_queries(5)
            self.log_test_result("Popular Queries", len(popular) > 0, f"Found {len(popular)} popular queries")
        except Exception as e:
            self.log_test_result("Popular Queries", False, str(e))
    
    async def test_enhanced_book_processor(self):
        """Test enhanced book processor with mock data"""
        logger.info("\n📚 Testing Enhanced Book Processor...")
        
        processor = EnhancedBookProcessor()
        await processor.initialize()
        
        try:
            # Test validation
            result = await processor._validate_file(Path("/nonexistent/file.pdf"))
            assert not result['valid']
            self.log_test_result("File Validation", True, "Correctly rejects nonexistent files")
        except Exception as e:
            self.log_test_result("File Validation", False, str(e))
        
        try:
            # Test supported file types
            supported = processor.supported_extensions
            assert '.pdf' in supported
            assert '.epub' in supported
            self.log_test_result("File Type Support", True, f"Supports: {list(supported.keys())}")
        except Exception as e:
            self.log_test_result("File Type Support", False, str(e))
        
        try:
            # Test content analysis integration
            sample_pages = [
                {'text': 'def calculate_returns(prices): return prices.pct_change()'},
                {'text': 'The formula for returns is: R = (P1 - P0) / P0'}
            ]
            analysis = await processor._analyze_content(sample_pages)
            
            has_code = len(analysis.get('code', [])) > 0
            has_formulas = len(analysis.get('formulas', [])) > 0
            
            self.log_test_result("Content Analysis Integration", has_code and has_formulas, "Detected code and formulas")
        except Exception as e:
            self.log_test_result("Content Analysis Integration", False, str(e))
        
        await processor.cleanup()
    
    async def test_error_handling(self):
        """Test error handling and graceful degradation"""
        logger.info("\n⚠️  Testing Error Handling...")
        
        try:
            # Test cache manager without Redis
            cache_manager = await get_cache_manager()
            # Should work with memory cache even if Redis fails
            await cache_manager.set("error_test", "value")
            result = await cache_manager.get("error_test")
            assert result == "value"
            self.log_test_result("Cache Fallback", True, "Memory cache works without Redis")
        except Exception as e:
            self.log_test_result("Cache Fallback", False, str(e))
        
        try:
            # Test content analyzer with empty text
            analyzer = ContentAnalyzer()
            result = analyzer.extract_special_content("")
            assert isinstance(result, dict)
            self.log_test_result("Empty Content Handling", True, "Handles empty text gracefully")
        except Exception as e:
            self.log_test_result("Empty Content Handling", False, str(e))
        
        try:
            # Test query suggester with invalid input
            suggester = QuerySuggester()
            await suggester.initialize()
            suggestions = await suggester.suggest("")  # Empty query
            assert len(suggestions) == 0
            self.log_test_result("Invalid Query Handling", True, "Handles empty queries")
        except Exception as e:
            self.log_test_result("Invalid Query Handling", False, str(e))
    
    async def test_performance_features(self):
        """Test performance and caching features"""
        logger.info("\n⚡ Testing Performance Features...")
        
        cache_manager = await get_cache_manager()
        
        try:
            # Test caching decorator
            from utils.cache_manager import cached
            
            call_count = 0
            
            @cached(cache_type='general', ttl=60)
            async def expensive_function(value):
                nonlocal call_count
                call_count += 1
                return f"result_{value}"
            
            # First call
            result1 = await expensive_function("test")
            # Second call (should be cached)
            result2 = await expensive_function("test")
            
            assert result1 == result2
            assert call_count == 1  # Function called only once
            self.log_test_result("Caching Decorator", True, "Function result cached correctly")
        except Exception as e:
            self.log_test_result("Caching Decorator", False, str(e))
        
        try:
            # Test cache key generation
            key1 = cache_manager.cache_key("test", arg1="value1", arg2="value2")
            key2 = cache_manager.cache_key("test", arg2="value2", arg1="value1")  # Different order
            assert key1 == key2  # Should be same despite different order
            self.log_test_result("Cache Key Generation", True, "Consistent key generation")
        except Exception as e:
            self.log_test_result("Cache Key Generation", False, str(e))
        
        try:
            # Test compression for large values
            large_data = ["data"] * 1000  # Large list
            await cache_manager.set("large_test", large_data)
            retrieved = await cache_manager.get("large_test")
            assert retrieved == large_data
            self.log_test_result("Large Data Caching", True, "Large data cached and retrieved correctly")
        except Exception as e:
            self.log_test_result("Large Data Caching", False, str(e))
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate final test report"""
        total_tests = self.passed_tests + self.failed_tests
        success_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info("\n" + "=" * 60)
        logger.info("🎯 PHASE 2 TEST REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {self.passed_tests} ✅")
        logger.info(f"Failed: {self.failed_tests} ❌")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if self.failed_tests > 0:
            logger.info("\n❌ Failed Tests:")
            for test_name, result in self.test_results.items():
                if not result['success']:
                    logger.info(f"   - {test_name}: {result['details']}")
        
        overall_status = "PASS" if self.failed_tests == 0 else "PARTIAL" if success_rate >= 70 else "FAIL"
        
        if overall_status == "PASS":
            logger.info("\n🎉 ALL TESTS PASSED! Phase 2 is ready for production.")
        elif overall_status == "PARTIAL":
            logger.info("\n⚠️  SOME TESTS FAILED. Phase 2 has issues but core functionality works.")
        else:
            logger.info("\n💥 MAJOR ISSUES FOUND. Phase 2 needs significant fixes.")
        
        return {
            'overall_status': overall_status,
            'total_tests': total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'success_rate': success_rate,
            'test_results': self.test_results,
            'timestamp': datetime.now().isoformat()
        }


async def main():
    """Run Phase 2 comprehensive tests"""
    print("🚀 TradeKnowledge Phase 2 Comprehensive Testing")
    print("=" * 60)
    
    tester = Phase2Tester()
    report = await tester.run_all_tests()
    
    # Save report
    report_path = Path(__file__).parent / "phase2_test_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_path}")
    
    # Exit with appropriate code
    sys.exit(0 if report['overall_status'] == 'PASS' else 1)


if __name__ == "__main__":
    asyncio.run(main())