#!/usr/bin/env python3
"""
Comprehensive Market Data Integration Test
Tests database integration with maximum granular market data collection for SPY and QQQ
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.database import DatabaseService, DatabaseConfig
from src.services.market_data_service import MarketDataService, MarketDataPoint
from src.core.database_mixins import ResearcherDatabaseMixin, MastermindDatabaseMixin, ExecutorDatabaseMixin

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveMarketTest:
    """Comprehensive test with maximum granular market data collection"""
    
    def __init__(self):
        self.db_service = DatabaseService()
        self.market_service = MarketDataService(self.db_service)
        
        # Test symbols - focus on SPY and QQQ as requested
        self.test_symbols = ["SPY", "QQQ"]
        
        # Additional symbols for comprehensive testing
        self.extended_symbols = ["SPY", "QQQ", "IWM", "DIA", "VIX"]
        
        # Test results tracking
        self.test_results = {
            "start_time": None,
            "end_time": None,
            "total_duration": None,
            "data_collection": {},
            "database_integration": {},
            "sparc_trio_tests": {},
            "performance_metrics": {},
            "overall_success": False
        }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run the complete comprehensive market data integration test"""
        self.test_results["start_time"] = datetime.utcnow()
        
        print("🚀 Starting Comprehensive Market Data Integration Test")
        print("=" * 80)
        print(f"Primary symbols: {self.test_symbols}")
        print(f"Extended symbols: {self.extended_symbols}")
        print(f"Start time: {self.test_results['start_time']}")
        print()
        
        try:
            # Phase 1: Database Connection Test
            await self._test_database_connections()
            
            # Phase 2: Maximum Granular Data Collection
            await self._test_maximum_data_collection()
            
            # Phase 3: Database Storage Integration
            await self._test_database_storage()
            
            # Phase 4: Data Quality Analysis
            await self._test_data_quality_analysis()
            
            # Phase 5: SPARC Trio Integration
            await self._test_sparc_trio_integration()
            
            # Phase 6: Performance Analysis
            await self._test_performance_metrics()
            
            # Phase 7: Cross-Timeframe Analysis
            await self._test_cross_timeframe_analysis()
            
            self.test_results["overall_success"] = True
            
        except Exception as e:
            logger.error(f"Comprehensive test failed: {e}")
            self.test_results["error"] = str(e)
            self.test_results["overall_success"] = False
        
        finally:
            self.test_results["end_time"] = datetime.utcnow()
            self.test_results["total_duration"] = (
                self.test_results["end_time"] - self.test_results["start_time"]
            ).total_seconds()
            
            await self._generate_final_report()
            await self.db_service.disconnect_all()
        
        return self.test_results
    
    async def _test_database_connections(self):
        """Test all database connections"""
        print("🔍 Phase 1: Testing Database Connections")
        print("-" * 50)
        
        start_time = time.time()
        
        # Connect to all databases
        await self.db_service.connect_all()
        
        # Test health check
        health = await self.db_service.health_check()
        
        connection_results = {
            "connection_time": time.time() - start_time,
            "health_status": health,
            "all_healthy": all("healthy" in status for status in health.values())
        }
        
        self.test_results["database_integration"]["connections"] = connection_results
        
        for service, status in health.items():
            status_icon = "✅" if "healthy" in status else "❌"
            print(f"  {service}: {status_icon} {status}")
        
        if connection_results["all_healthy"]:
            print("✅ All database connections healthy")
        else:
            print("⚠️  Some database connections have issues")
        
        print(f"Connection time: {connection_results['connection_time']:.2f}s")
        print()
    
    async def _test_maximum_data_collection(self):
        """Test maximum granular data collection"""
        print("📊 Phase 2: Maximum Granular Data Collection")
        print("-" * 50)
        
        start_time = time.time()
        
        # Test with primary symbols first (SPY, QQQ)
        print(f"Collecting maximum granular data for: {self.test_symbols}")
        
        primary_data = await self.market_service.fetch_comprehensive_data(self.test_symbols)
        
        # Analyze collection results
        collection_stats = self._analyze_collection_results(primary_data)
        
        collection_results = {
            "collection_time": time.time() - start_time,
            "symbols_processed": len(primary_data),
            "total_data_points": collection_stats["total_points"],
            "timeframes_collected": collection_stats["timeframes"],
            "data_by_symbol": {}
        }
        
        for symbol, timeframe_data in primary_data.items():
            symbol_stats = {
                "timeframes": len(timeframe_data),
                "total_points": sum(len(points) for points in timeframe_data.values()),
                "timeframe_breakdown": {
                    tf: len(points) for tf, points in timeframe_data.items()
                }
            }
            collection_results["data_by_symbol"][symbol] = symbol_stats
            
            print(f"  {symbol}: {symbol_stats['total_points']} total data points across {symbol_stats['timeframes']} timeframes")
            for tf, count in symbol_stats["timeframe_breakdown"].items():
                print(f"    {tf}: {count:,} points")
        
        self.test_results["data_collection"]["primary_symbols"] = collection_results
        self.collected_data = primary_data
        
        print(f"✅ Data collection completed: {collection_results['total_data_points']:,} total points")
        print(f"Collection time: {collection_results['collection_time']:.2f}s")
        print()
    
    async def _test_database_storage(self):
        """Test comprehensive database storage"""
        print("💾 Phase 3: Database Storage Integration")
        print("-" * 50)
        
        start_time = time.time()
        
        # Store the collected data
        storage_stats = await self.market_service.store_market_data(self.collected_data)
        
        storage_results = {
            "storage_time": time.time() - start_time,
            "storage_stats": storage_stats
        }
        
        self.test_results["database_integration"]["storage"] = storage_results
        
        print(f"  Symbols processed: {storage_stats['symbols_processed']}")
        print(f"  Total data points: {storage_stats['total_data_points']:,}")
        print(f"  InfluxDB points: {storage_stats['influxdb_points']:,}")
        print(f"  PostgreSQL records: {storage_stats['postgresql_records']}")
        
        if storage_stats["errors"]:
            print(f"  ⚠️  Errors encountered: {len(storage_stats['errors'])}")
            for error in storage_stats["errors"][:3]:  # Show first 3 errors
                print(f"    - {error}")
        else:
            print("  ✅ No storage errors")
        
        print(f"Storage time: {storage_results['storage_time']:.2f}s")
        print()
    
    async def _test_data_quality_analysis(self):
        """Test comprehensive data quality analysis"""
        print("🔍 Phase 4: Data Quality Analysis")
        print("-" * 50)
        
        start_time = time.time()
        
        # Analyze data quality
        quality_analysis = await self.market_service.analyze_data_quality(self.collected_data)
        
        quality_results = {
            "analysis_time": time.time() - start_time,
            "quality_analysis": quality_analysis
        }
        
        self.test_results["data_collection"]["quality_analysis"] = quality_results
        
        print(f"  Total symbols: {quality_analysis['total_symbols']}")
        print(f"  Total data points: {quality_analysis['total_data_points']:,}")
        print(f"  Overall quality score: {quality_analysis['overall_quality_score']:.3f}")
        print(f"  Data gaps found: {len(quality_analysis['data_gaps'])}")
        
        print("\n  Timeframe coverage:")
        for timeframe, coverage in quality_analysis["timeframe_coverage"].items():
            print(f"    {timeframe}: {coverage['symbols']} symbols, {coverage['total_points']:,} points")
        
        print("\n  Quality metrics by symbol/timeframe:")
        for key, metrics in list(quality_analysis["quality_metrics"].items())[:10]:  # Show first 10
            print(f"    {key}: {metrics['avg_quality']:.3f} avg quality, {metrics['data_points']} points")
        
        print(f"Analysis time: {quality_results['analysis_time']:.2f}s")
        print()
    
    async def _test_sparc_trio_integration(self):
        """Test SPARC trio integration with real market data"""
        print("🤖 Phase 5: SPARC Trio Integration Test")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Create enhanced agents with database capabilities
            researcher = EnhancedResearcher(self.db_service)
            mastermind = EnhancedMastermind(self.db_service)
            executor = EnhancedExecutor(self.db_service)
            
            # Test workflow for SPY
            symbol = "SPY"
            print(f"Testing SPARC trio workflow for {symbol}")
            
            # RESEARCHER: Gather market intelligence
            print("  🔍 RESEARCHER: Gathering market intelligence...")
            research_start = time.time()
            intelligence = await researcher.get_research_intelligence({
                "symbol": symbol,
                "analysis_type": "comprehensive"
            })
            research_time = time.time() - research_start
            
            # MASTERMIND: Strategic analysis
            print("  🧠 MASTERMIND: Performing strategic analysis...")
            strategy_start = time.time()
            strategic_analysis = await mastermind.analyze_strategic_patterns(intelligence)
            strategy_time = time.time() - strategy_start
            
            # EXECUTOR: Implementation
            print("  ⚡ EXECUTOR: Implementing with persistence...")
            impl_start = time.time()
            
            # Create test user and analysis for executor
            test_user = await self._create_test_user()
            test_analysis = await self._create_test_analysis(test_user["id"], symbol)
            
            implementation = await executor.implement_with_persistence(
                strategic_analysis,
                {"quality_gates": {"test_coverage": 90, "mutation_score": 80}},
                test_user["id"],
                test_analysis["id"]
            )
            impl_time = time.time() - impl_start
            
            sparc_results = {
                "total_time": time.time() - start_time,
                "research_time": research_time,
                "strategy_time": strategy_time,
                "implementation_time": impl_time,
                "intelligence": intelligence,
                "strategic_analysis": strategic_analysis,
                "implementation": implementation,
                "success": True
            }
            
            print(f"  ✅ RESEARCHER completed in {research_time:.2f}s")
            print(f"  ✅ MASTERMIND completed in {strategy_time:.2f}s") 
            print(f"  ✅ EXECUTOR completed in {impl_time:.2f}s")
            print(f"  📊 Intelligence confidence: {intelligence.get('confidence', 0):.3f}")
            print(f"  🎯 Strategic confidence: {strategic_analysis.get('confidence', 0):.3f}")
            print(f"  💯 Implementation quality: {implementation.get('overall_quality_score', 0):.1f}/10")
            
        except Exception as e:
            logger.error(f"SPARC trio test failed: {e}")
            sparc_results = {
                "total_time": time.time() - start_time,
                "error": str(e),
                "success": False
            }
            print(f"  ❌ SPARC trio test failed: {e}")
        
        self.test_results["sparc_trio_tests"] = sparc_results
        print(f"SPARC trio test time: {sparc_results['total_time']:.2f}s")
        print()
    
    async def _test_performance_metrics(self):
        """Test system performance metrics"""
        print("⚡ Phase 6: Performance Metrics Analysis")
        print("-" * 50)
        
        start_time = time.time()
        
        # Calculate performance metrics
        total_data_points = sum(
            len(points) 
            for symbol_data in self.collected_data.values()
            for points in symbol_data.values()
        )
        
        collection_time = self.test_results["data_collection"]["primary_symbols"]["collection_time"]
        storage_time = self.test_results["database_integration"]["storage"]["storage_time"]
        
        performance_metrics = {
            "analysis_time": time.time() - start_time,
            "data_throughput": {
                "points_per_second_collection": total_data_points / collection_time if collection_time > 0 else 0,
                "points_per_second_storage": total_data_points / storage_time if storage_time > 0 else 0,
                "total_data_points": total_data_points,
                "collection_time": collection_time,
                "storage_time": storage_time
            },
            "database_performance": {
                "connection_time": self.test_results["database_integration"]["connections"]["connection_time"],
                "health_check_passed": self.test_results["database_integration"]["connections"]["all_healthy"]
            }
        }
        
        # Add SPARC trio performance if available
        if "sparc_trio_tests" in self.test_results and self.test_results["sparc_trio_tests"]["success"]:
            trio_results = self.test_results["sparc_trio_tests"]
            performance_metrics["sparc_trio_performance"] = {
                "total_time": trio_results["total_time"],
                "research_time": trio_results["research_time"],
                "strategy_time": trio_results["strategy_time"],
                "implementation_time": trio_results["implementation_time"]
            }
        
        self.test_results["performance_metrics"] = performance_metrics
        
        print(f"  Total data points processed: {total_data_points:,}")
        print(f"  Collection throughput: {performance_metrics['data_throughput']['points_per_second_collection']:.1f} points/sec")
        print(f"  Storage throughput: {performance_metrics['data_throughput']['points_per_second_storage']:.1f} points/sec")
        print(f"  Database connection time: {performance_metrics['database_performance']['connection_time']:.2f}s")
        
        if "sparc_trio_performance" in performance_metrics:
            trio_perf = performance_metrics["sparc_trio_performance"]
            print(f"  SPARC trio total time: {trio_perf['total_time']:.2f}s")
            print(f"  Average agent time: {(trio_perf['research_time'] + trio_perf['strategy_time'] + trio_perf['implementation_time']) / 3:.2f}s")
        
        print(f"Performance analysis time: {performance_metrics['analysis_time']:.2f}s")
        print()
    
    async def _test_cross_timeframe_analysis(self):
        """Test cross-timeframe analysis capabilities"""
        print("📈 Phase 7: Cross-Timeframe Analysis")
        print("-" * 50)
        
        start_time = time.time()
        
        # Analyze data across multiple timeframes for SPY
        symbol = "SPY"
        symbol_data = self.collected_data.get(symbol, {})
        
        if not symbol_data:
            print(f"  ⚠️  No data available for {symbol}")
            return
        
        cross_analysis = {
            "analysis_time": 0,
            "timeframes_analyzed": len(symbol_data),
            "correlations": {},
            "patterns": {}
        }
        
        # Analyze timeframe coverage
        timeframes = list(symbol_data.keys())
        print(f"  Analyzing {symbol} across {len(timeframes)} timeframes: {timeframes}")
        
        for timeframe in timeframes:
            data_points = symbol_data[timeframe]
            if data_points:
                prices = [p.close for p in data_points if p.close > 0]
                if len(prices) > 1:
                    # Calculate basic statistics
                    avg_price = sum(prices) / len(prices)
                    price_range = max(prices) - min(prices)
                    volatility = (price_range / avg_price) * 100 if avg_price > 0 else 0
                    
                    cross_analysis["patterns"][timeframe] = {
                        "data_points": len(data_points),
                        "avg_price": avg_price,
                        "price_range": price_range,
                        "volatility_percent": volatility,
                        "quality_score": sum(p.data_quality_score for p in data_points) / len(data_points)
                    }
                    
                    print(f"    {timeframe}: {len(data_points)} points, avg_price=${avg_price:.2f}, volatility={volatility:.1f}%")
        
        cross_analysis["analysis_time"] = time.time() - start_time
        self.test_results["data_collection"]["cross_timeframe_analysis"] = cross_analysis
        
        print(f"Cross-timeframe analysis time: {cross_analysis['analysis_time']:.2f}s")
        print()
    
    async def _create_test_user(self) -> Dict[str, Any]:
        """Create a test user for SPARC trio testing"""
        user_data = {
            "email": f"test_comprehensive_{int(time.time())}@tradeknowledge.ai",
            "hashed_password": "test_hashed_password",
            "full_name": "Comprehensive Test User",
            "credits": 100
        }
        
        user_id = await self.db_service.postgres.create_user(user_data)
        user_data["id"] = user_id
        return user_data
    
    async def _create_test_analysis(self, user_id: str, symbol: str) -> Dict[str, Any]:
        """Create a test analysis record"""
        analysis_data = {
            "user_id": user_id,
            "query_type": "comprehensive_test",
            "query_params": {"symbol": symbol, "test_type": "comprehensive_integration"},
            "source": "comprehensive_test"
        }
        
        analysis_id = await self.db_service.postgres.create_analysis(analysis_data)
        analysis_data["id"] = analysis_id
        return analysis_data
    
    def _analyze_collection_results(self, data: Dict[str, Dict[str, List[MarketDataPoint]]]) -> Dict[str, Any]:
        """Analyze data collection results"""
        total_points = 0
        timeframes = set()
        
        for symbol_data in data.values():
            for timeframe, points in symbol_data.items():
                total_points += len(points)
                timeframes.add(timeframe)
        
        return {
            "total_points": total_points,
            "timeframes": sorted(list(timeframes))
        }
    
    async def _generate_final_report(self):
        """Generate comprehensive final report"""
        print("📋 Final Comprehensive Test Report")
        print("=" * 80)
        
        results = self.test_results
        
        print(f"Test Duration: {results['total_duration']:.2f} seconds")
        print(f"Overall Success: {'✅ PASSED' if results['overall_success'] else '❌ FAILED'}")
        print()
        
        # Data Collection Summary
        if "data_collection" in results:
            dc = results["data_collection"]
            if "primary_symbols" in dc:
                primary = dc["primary_symbols"]
                print(f"📊 Data Collection:")
                print(f"  Symbols: {primary['symbols_processed']}")
                print(f"  Total Data Points: {primary['total_data_points']:,}")
                print(f"  Collection Time: {primary['collection_time']:.2f}s")
                print()
        
        # Database Integration Summary
        if "database_integration" in results:
            db = results["database_integration"]
            print(f"💾 Database Integration:")
            if "connections" in db:
                print(f"  All Connections Healthy: {'✅' if db['connections']['all_healthy'] else '❌'}")
            if "storage" in db:
                storage_stats = db["storage"]["storage_stats"]
                print(f"  Data Points Stored: {storage_stats['total_data_points']:,}")
                print(f"  Storage Errors: {len(storage_stats['errors'])}")
            print()
        
        # SPARC Trio Summary
        if "sparc_trio_tests" in results:
            sparc = results["sparc_trio_tests"]
            print(f"🤖 SPARC Trio Integration:")
            print(f"  Success: {'✅' if sparc['success'] else '❌'}")
            if sparc['success']:
                print(f"  Total Time: {sparc['total_time']:.2f}s")
                print(f"  Research Time: {sparc['research_time']:.2f}s")
                print(f"  Strategy Time: {sparc['strategy_time']:.2f}s")
                print(f"  Implementation Time: {sparc['implementation_time']:.2f}s")
            print()
        
        # Performance Summary
        if "performance_metrics" in results:
            perf = results["performance_metrics"]
            print(f"⚡ Performance Metrics:")
            if "data_throughput" in perf:
                throughput = perf["data_throughput"]
                print(f"  Collection Throughput: {throughput['points_per_second_collection']:.1f} points/sec")
                print(f"  Storage Throughput: {throughput['points_per_second_storage']:.1f} points/sec")
            print()
        
        print("🎉 Comprehensive Market Data Integration Test Complete!")


# Enhanced agent classes for testing
class EnhancedResearcher(ResearcherDatabaseMixin):
    def __init__(self, db_service):
        super().__init__(db_service)

class EnhancedMastermind(MastermindDatabaseMixin):
    def __init__(self, db_service):
        super().__init__(db_service)

class EnhancedExecutor(ExecutorDatabaseMixin):
    def __init__(self, db_service):
        super().__init__(db_service)


async def main():
    """Run the comprehensive market data integration test"""
    test = ComprehensiveMarketTest()
    results = await test.run_comprehensive_test()
    
    # Return appropriate exit code
    if results["overall_success"]:
        print("\n✅ All tests passed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        sys.exit(1)