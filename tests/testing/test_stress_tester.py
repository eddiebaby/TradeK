"""
Tests for Comprehensive Stress Testing Framework.

This module tests the stress testing capabilities including
CPU stress, memory pressure, I/O saturation, and concurrent users.
"""

import pytest
import asyncio
import time
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.testing.stress_tester import (
    StressTester,
    StressTestConfig,
    StressTestType,
    StressTestResult,
    StressTestMetrics,
    create_cpu_stress_config,
    create_memory_stress_config,
    create_io_stress_config,
    create_concurrent_users_config,
    run_comprehensive_stress_suite
)


class TestStressTestConfig:
    """Test stress test configuration"""
    
    def test_basic_config_creation(self):
        """Test basic configuration creation"""
        config = StressTestConfig(
            test_name="test_stress",
            test_type=StressTestType.CPU_INTENSIVE,
            duration_seconds=30,
            intensity_level=3
        )
        
        assert config.test_name == "test_stress"
        assert config.test_type == StressTestType.CPU_INTENSIVE
        assert config.duration_seconds == 30
        assert config.intensity_level == 3
        assert config.concurrent_operations == 100  # default
    
    def test_config_with_custom_parameters(self):
        """Test configuration with custom parameters"""
        custom_params = {"target_memory_mb": 500, "file_size_mb": 20}
        config = StressTestConfig(
            test_name="custom_test",
            test_type=StressTestType.MEMORY_PRESSURE,
            custom_parameters=custom_params
        )
        
        assert config.custom_parameters["target_memory_mb"] == 500
        assert config.custom_parameters["file_size_mb"] == 20


class TestStressTestMetrics:
    """Test stress test metrics collection"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        start_time = datetime.now()
        metrics = StressTestMetrics(
            test_name="test_metrics",
            start_time=start_time
        )
        
        assert metrics.test_name == "test_metrics"
        assert metrics.start_time == start_time
        assert metrics.total_operations == 0
        assert metrics.successful_operations == 0
        assert metrics.failed_operations == 0
        assert metrics.system_stability_score == 1.0
        assert not metrics.breaking_point_reached


class TestStressTestResult:
    """Test stress test result analysis"""
    
    def test_successful_result(self):
        """Test successful stress test result"""
        config = StressTestConfig(
            test_name="success_test",
            test_type=StressTestType.CPU_INTENSIVE,
            failure_threshold=0.1
        )
        
        metrics = StressTestMetrics(
            test_name="success_test",
            start_time=datetime.now()
        )
        metrics.total_operations = 100
        metrics.successful_operations = 95
        metrics.failed_operations = 5
        metrics.system_stability_score = 0.8
        
        result = StressTestResult(config, metrics)
        
        assert result.success is True
        assert len(result.recommendations) > 0
        assert "Ready for production" in result.recommendations[0]
    
    def test_failed_result_high_failure_rate(self):
        """Test failed result due to high failure rate"""
        config = StressTestConfig(
            test_name="failure_test",
            test_type=StressTestType.CPU_INTENSIVE,
            failure_threshold=0.1
        )
        
        metrics = StressTestMetrics(
            test_name="failure_test",
            start_time=datetime.now()
        )
        metrics.total_operations = 100
        metrics.successful_operations = 70
        metrics.failed_operations = 30  # 30% failure rate
        metrics.system_stability_score = 0.8
        
        result = StressTestResult(config, metrics)
        
        assert result.success is False
        assert any("High failure rate" in rec for rec in result.recommendations)
    
    def test_failed_result_breaking_point(self):
        """Test failed result due to breaking point reached"""
        config = StressTestConfig(
            test_name="breaking_point_test",
            test_type=StressTestType.RESOURCE_EXHAUSTION,
            failure_threshold=0.1
        )
        
        metrics = StressTestMetrics(
            test_name="breaking_point_test",
            start_time=datetime.now()
        )
        metrics.total_operations = 100
        metrics.successful_operations = 95
        metrics.failed_operations = 5
        metrics.breaking_point_reached = True
        metrics.breaking_point_metric = "memory_usage"
        metrics.system_stability_score = 0.8
        
        result = StressTestResult(config, metrics)
        
        assert result.success is False
        assert any("Breaking point reached" in rec for rec in result.recommendations)
    
    def test_recommendations_cpu_usage(self):
        """Test recommendations for high CPU usage"""
        config = StressTestConfig(
            test_name="cpu_test",
            test_type=StressTestType.CPU_INTENSIVE
        )
        
        metrics = StressTestMetrics(
            test_name="cpu_test",
            start_time=datetime.now()
        )
        metrics.total_operations = 100
        metrics.successful_operations = 100
        metrics.peak_cpu_usage = 95.0  # High CPU usage
        metrics.system_stability_score = 0.8
        
        result = StressTestResult(config, metrics)
        
        assert any("CPU usage exceeded 90%" in rec for rec in result.recommendations)
    
    def test_recommendations_memory_usage(self):
        """Test recommendations for high memory usage"""
        config = StressTestConfig(
            test_name="memory_test",
            test_type=StressTestType.MEMORY_PRESSURE
        )
        
        metrics = StressTestMetrics(
            test_name="memory_test",
            start_time=datetime.now()
        )
        metrics.total_operations = 100
        metrics.successful_operations = 100
        metrics.peak_memory_usage = 95.0  # High memory usage
        metrics.system_stability_score = 0.8
        
        result = StressTestResult(config, metrics)
        
        assert any("Memory usage exceeded 90%" in rec for rec in result.recommendations)
    
    def test_recommendations_slow_response(self):
        """Test recommendations for slow response times"""
        config = StressTestConfig(
            test_name="response_test",
            test_type=StressTestType.CONCURRENT_USERS
        )
        
        metrics = StressTestMetrics(
            test_name="response_test",
            start_time=datetime.now()
        )
        metrics.total_operations = 100
        metrics.successful_operations = 100
        metrics.average_response_time = 2000.0  # 2 seconds
        metrics.system_stability_score = 0.8
        
        result = StressTestResult(config, metrics)
        
        assert any("response time exceeds 1 second" in rec for rec in result.recommendations)


class TestStressTester:
    """Test stress tester functionality"""
    
    @pytest.fixture
    def stress_tester(self):
        """Create stress tester instance"""
        return StressTester()
    
    @pytest.fixture
    def basic_config(self):
        """Create basic stress test configuration"""
        return StressTestConfig(
            test_name="basic_test",
            test_type=StressTestType.CONCURRENT_USERS,
            duration_seconds=1,  # Very short for testing
            concurrent_operations=5,
            ramp_up_seconds=0,
            cool_down_seconds=0
        )
    
    @pytest.mark.asyncio
    async def test_generic_stress_test(self, stress_tester, basic_config):
        """Test generic stress test execution"""
        result = await stress_tester.run_stress_test(basic_config)
        
        assert isinstance(result, StressTestResult)
        assert result.config.test_name == "basic_test"
        assert result.metrics.total_operations > 0
        assert result.metrics.duration_seconds >= 1.0
        assert result.metrics.end_time is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_users_test(self, stress_tester):
        """Test concurrent users stress test"""
        config = StressTestConfig(
            test_name="concurrent_users_test",
            test_type=StressTestType.CONCURRENT_USERS,
            duration_seconds=2,
            concurrent_operations=3,
            ramp_up_seconds=0,
            cool_down_seconds=0
        )
        
        result = await stress_tester.run_stress_test(config)
        
        assert result.metrics.total_operations > 0
        assert result.metrics.successful_operations >= 0
        assert result.metrics.failed_operations >= 0
    
    @pytest.mark.asyncio
    async def test_cpu_stress_test(self, stress_tester):
        """Test CPU stress test"""
        config = StressTestConfig(
            test_name="cpu_stress_test",
            test_type=StressTestType.CPU_INTENSIVE,
            duration_seconds=2,
            intensity_level=2,  # Low intensity for testing
            concurrent_operations=2,
            ramp_up_seconds=0,
            cool_down_seconds=0
        )
        
        result = await stress_tester.run_stress_test(config)
        
        assert result.metrics.total_operations > 0
        assert result.metrics.peak_cpu_usage >= 0
    
    @pytest.mark.asyncio
    async def test_memory_stress_test(self, stress_tester):
        """Test memory stress test"""
        config = StressTestConfig(
            test_name="memory_stress_test",
            test_type=StressTestType.MEMORY_PRESSURE,
            duration_seconds=2,
            custom_parameters={"target_memory_mb": 10},  # Small amount for testing
            ramp_up_seconds=0,
            cool_down_seconds=0
        )
        
        result = await stress_tester.run_stress_test(config)
        
        assert result.metrics.total_operations > 0
        assert result.metrics.peak_memory_mb >= 0
    
    @pytest.mark.asyncio
    async def test_io_stress_test(self, stress_tester):
        """Test I/O stress test"""
        config = StressTestConfig(
            test_name="io_stress_test",
            test_type=StressTestType.IO_SATURATION,
            duration_seconds=2,
            intensity_level=2,  # Low intensity for testing
            custom_parameters={"file_size_mb": 1},  # Small files for testing
            ramp_up_seconds=0,
            cool_down_seconds=0
        )
        
        result = await stress_tester.run_stress_test(config)
        
        assert result.metrics.total_operations >= 0  # May be 0 if operations are slow
        # I/O operations might not complete in 2 seconds, so we just check it runs
    
    @pytest.mark.asyncio
    async def test_network_stress_test_mock(self, stress_tester):
        """Test network stress test with mocked HTTP calls"""
        config = StressTestConfig(
            test_name="network_stress_test",
            test_type=StressTestType.NETWORK_STRESS,
            duration_seconds=1,
            concurrent_operations=3,
            custom_parameters={"target_url": "http://test.example.com"},
            ramp_up_seconds=0,
            cool_down_seconds=0
        )
        
        # Mock aiohttp to avoid actual network calls
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            # Mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text.return_value = "OK"
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            result = await stress_tester.run_stress_test(config)
            
            assert result.metrics.total_operations > 0
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_test(self, stress_tester):
        """Test resource exhaustion test"""
        config = StressTestConfig(
            test_name="resource_exhaustion_test",
            test_type=StressTestType.RESOURCE_EXHAUSTION,
            duration_seconds=1,  # Very short to avoid actually exhausting resources
            ramp_up_seconds=0,
            cool_down_seconds=0
        )
        
        result = await stress_tester.run_stress_test(config)
        
        # Should complete without error (may or may not reach breaking point)
        assert result.metrics.total_operations >= 0
    
    @pytest.mark.asyncio
    async def test_system_limits_test(self, stress_tester):
        """Test system limits test"""
        config = StressTestConfig(
            test_name="system_limits_test",
            test_type=StressTestType.SYSTEM_LIMITS,
            duration_seconds=1,  # Very short to avoid creating too many threads
            ramp_up_seconds=0,
            cool_down_seconds=0
        )
        
        result = await stress_tester.run_stress_test(config)
        
        # Should complete without error
        assert result.metrics.total_operations >= 0
    
    @pytest.mark.asyncio
    async def test_ramp_up_phase(self, stress_tester):
        """Test ramp-up phase execution"""
        config = StressTestConfig(
            test_name="ramp_up_test",
            test_type=StressTestType.CONCURRENT_USERS,
            duration_seconds=1,
            concurrent_operations=10,
            ramp_up_seconds=1,  # Include ramp-up
            cool_down_seconds=0
        )
        
        result = await stress_tester.run_stress_test(config)
        
        assert result.metrics.total_operations > 0
        assert result.metrics.duration_seconds >= 2.0  # Should include ramp-up time
    
    @pytest.mark.asyncio
    async def test_cool_down_phase(self, stress_tester):
        """Test cool-down phase execution"""
        config = StressTestConfig(
            test_name="cool_down_test",
            test_type=StressTestType.CONCURRENT_USERS,
            duration_seconds=1,
            concurrent_operations=5,
            ramp_up_seconds=0,
            cool_down_seconds=1  # Include cool-down
        )
        
        result = await stress_tester.run_stress_test(config)
        
        assert result.metrics.total_operations > 0
        assert result.metrics.duration_seconds >= 2.0  # Should include cool-down time
    
    @pytest.mark.asyncio
    async def test_system_monitoring(self, stress_tester):
        """Test system resource monitoring during stress test"""
        config = StressTestConfig(
            test_name="monitoring_test",
            test_type=StressTestType.CONCURRENT_USERS,
            duration_seconds=2,
            concurrent_operations=3,
            ramp_up_seconds=0,
            cool_down_seconds=0
        )
        
        result = await stress_tester.run_stress_test(config)
        
        # Should have collected resource usage samples
        assert len(result.metrics.resource_usage_samples) > 0
        assert result.metrics.peak_cpu_usage >= 0
        assert result.metrics.peak_memory_usage >= 0
        assert result.metrics.peak_memory_mb >= 0
        
        # Check sample format
        sample = result.metrics.resource_usage_samples[0]
        assert "timestamp" in sample
        assert "cpu_percent" in sample
        assert "memory_percent" in sample
        assert "memory_mb" in sample
    
    def test_record_response_time(self, stress_tester):
        """Test response time recording"""
        metrics = StressTestMetrics(
            test_name="response_time_test",
            start_time=datetime.now()
        )
        
        # Record first response time
        metrics.successful_operations = 1
        stress_tester._record_response_time(metrics, 100.0)
        assert metrics.average_response_time == 100.0
        
        # Record second response time
        metrics.successful_operations = 2
        stress_tester._record_response_time(metrics, 200.0)
        assert metrics.average_response_time == 150.0  # (100 + 200) / 2
    
    def test_record_error(self, stress_tester):
        """Test error recording"""
        metrics = StressTestMetrics(
            test_name="error_test",
            start_time=datetime.now()
        )
        
        stress_tester._record_error(metrics, "ConnectionError: Connection refused")
        stress_tester._record_error(metrics, "TimeoutError: Request timeout")
        stress_tester._record_error(metrics, "ConnectionError: Network unreachable")
        
        assert metrics.error_distribution["ConnectionError"] == 2
        assert metrics.error_distribution["TimeoutError"] == 1
    
    def test_analyze_metrics(self, stress_tester):
        """Test metrics analysis"""
        metrics = StressTestMetrics(
            test_name="analysis_test",
            start_time=datetime.now()
        )
        
        # Set up test data
        metrics.total_operations = 100
        metrics.successful_operations = 90
        metrics.failed_operations = 10
        metrics.peak_cpu_usage = 70.0
        metrics.peak_memory_usage = 75.0
        
        stress_tester._analyze_metrics(metrics)
        
        # Should calculate stability score
        assert 0 <= metrics.system_stability_score <= 1.0
        
        # Test with breaking point reached
        metrics.breaking_point_reached = True
        stress_tester._analyze_metrics(metrics)
        
        # Stability score should be penalized
        assert metrics.system_stability_score <= 0.5


class TestStressTestConfigFactories:
    """Test stress test configuration factory functions"""
    
    def test_create_cpu_stress_config(self):
        """Test CPU stress configuration factory"""
        config = create_cpu_stress_config(duration=120, intensity=7)
        
        assert config.test_name == "cpu_stress_intensity_7"
        assert config.test_type == StressTestType.CPU_INTENSIVE
        assert config.duration_seconds == 120
        assert config.intensity_level == 7
    
    def test_create_memory_stress_config(self):
        """Test memory stress configuration factory"""
        config = create_memory_stress_config(duration=90, target_memory_mb=2000)
        
        assert config.test_name == "memory_stress_2000mb"
        assert config.test_type == StressTestType.MEMORY_PRESSURE
        assert config.duration_seconds == 90
        assert config.custom_parameters["target_memory_mb"] == 2000
    
    def test_create_io_stress_config(self):
        """Test I/O stress configuration factory"""
        config = create_io_stress_config(duration=75, file_size_mb=50)
        
        assert config.test_name == "io_stress_50mb_files"
        assert config.test_type == StressTestType.IO_SATURATION
        assert config.duration_seconds == 75
        assert config.custom_parameters["file_size_mb"] == 50
    
    def test_create_concurrent_users_config(self):
        """Test concurrent users configuration factory"""
        config = create_concurrent_users_config(duration=180, num_users=500)
        
        assert config.test_name == "concurrent_users_500"
        assert config.test_type == StressTestType.CONCURRENT_USERS
        assert config.duration_seconds == 180
        assert config.concurrent_operations == 500


class TestComprehensiveStressSuite:
    """Test comprehensive stress testing suite"""
    
    @pytest.mark.asyncio
    async def test_comprehensive_stress_suite_basic(self):
        """Test basic comprehensive stress suite"""
        # Mock time-consuming operations for faster testing
        with patch('src.testing.stress_tester.StressTester._cpu_stress_test') as mock_cpu, \
             patch('src.testing.stress_tester.StressTester._memory_stress_test') as mock_memory, \
             patch('src.testing.stress_tester.StressTester._io_stress_test') as mock_io, \
             patch('src.testing.stress_tester.StressTester._concurrent_users_test') as mock_users:
            
            # Mock all stress test methods to complete quickly
            async def quick_test(config, metrics):
                metrics.total_operations = 10
                metrics.successful_operations = 9
                metrics.failed_operations = 1
                await asyncio.sleep(0.1)
            
            mock_cpu.side_effect = quick_test
            mock_memory.side_effect = quick_test
            mock_io.side_effect = quick_test
            mock_users.side_effect = quick_test
            
            results = await run_comprehensive_stress_suite()
            
            # Should run 4 tests (CPU, memory, I/O, concurrent users)
            assert len(results) == 4
            assert all(isinstance(result, StressTestResult) for result in results)
            
            # Check that all expected test types are covered
            test_types = [result.config.test_type for result in results]
            assert StressTestType.CPU_INTENSIVE in test_types
            assert StressTestType.MEMORY_PRESSURE in test_types
            assert StressTestType.IO_SATURATION in test_types
            assert StressTestType.CONCURRENT_USERS in test_types
    
    @pytest.mark.asyncio
    async def test_comprehensive_stress_suite_with_network(self):
        """Test comprehensive stress suite with network testing"""
        target_url = "http://test.example.com"
        
        # Mock all operations for faster testing
        with patch('src.testing.stress_tester.StressTester._cpu_stress_test') as mock_cpu, \
             patch('src.testing.stress_tester.StressTester._memory_stress_test') as mock_memory, \
             patch('src.testing.stress_tester.StressTester._io_stress_test') as mock_io, \
             patch('src.testing.stress_tester.StressTester._concurrent_users_test') as mock_users, \
             patch('src.testing.stress_tester.StressTester._network_stress_test') as mock_network:
            
            async def quick_test(config, metrics):
                metrics.total_operations = 10
                metrics.successful_operations = 9
                metrics.failed_operations = 1
                await asyncio.sleep(0.1)
            
            mock_cpu.side_effect = quick_test
            mock_memory.side_effect = quick_test
            mock_io.side_effect = quick_test
            mock_users.side_effect = quick_test
            mock_network.side_effect = quick_test
            
            results = await run_comprehensive_stress_suite(target_url=target_url)
            
            # Should run 5 tests (including network)
            assert len(results) == 5
            
            # Check that network test is included
            test_types = [result.config.test_type for result in results]
            assert StressTestType.NETWORK_STRESS in test_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])