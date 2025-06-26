"""
Tests for Scalability Validation Framework.

This module tests the scalability validation capabilities including
throughput scaling, user growth simulation, and latency under load.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.testing.scalability_validator import (
    ScalabilityValidator,
    ScalabilityTestConfig,
    ScalabilityTestType,
    ScalabilityResult,
    ScalabilityMetrics,
    ScalabilityDataPoint,
    create_throughput_scaling_config,
    create_user_growth_config,
    create_latency_under_load_config,
    run_comprehensive_scalability_suite
)


class TestScalabilityTestConfig:
    """Test scalability test configuration"""
    
    def test_basic_config_creation(self):
        """Test basic configuration creation"""
        config = ScalabilityTestConfig(
            test_name="test_scalability",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING,
            baseline_load=10,
            max_load=100,
            scaling_steps=5
        )
        
        assert config.test_name == "test_scalability"
        assert config.test_type == ScalabilityTestType.THROUGHPUT_SCALING
        assert config.baseline_load == 10
        assert config.max_load == 100
        assert config.scaling_steps == 5
        assert config.scaling_pattern == "exponential"  # default
    
    def test_config_with_target_function(self):
        """Test configuration with target function"""
        async def test_function():
            return "test_result"
        
        config = ScalabilityTestConfig(
            test_name="function_test",
            test_type=ScalabilityTestType.USER_GROWTH_SIMULATION,
            baseline_load=5,
            max_load=50,
            target_function=test_function
        )
        
        assert config.target_function == test_function
        assert config.test_type == ScalabilityTestType.USER_GROWTH_SIMULATION


class TestScalabilityDataPoint:
    """Test scalability data point"""
    
    def test_data_point_creation(self):
        """Test data point creation with all fields"""
        timestamp = datetime.now()
        data_point = ScalabilityDataPoint(
            load_level=50,
            timestamp=timestamp,
            response_time_ms=150.5,
            throughput_ops_per_sec=25.0,
            success_rate=0.95,
            cpu_usage_percent=70.0,
            memory_usage_percent=60.0,
            memory_usage_mb=1024.0,
            error_count=2,
            concurrent_operations=50
        )
        
        assert data_point.load_level == 50
        assert data_point.timestamp == timestamp
        assert data_point.response_time_ms == 150.5
        assert data_point.throughput_ops_per_sec == 25.0
        assert data_point.success_rate == 0.95
        assert data_point.cpu_usage_percent == 70.0
        assert data_point.memory_usage_percent == 60.0
        assert data_point.memory_usage_mb == 1024.0
        assert data_point.error_count == 2
        assert data_point.concurrent_operations == 50


class TestScalabilityResult:
    """Test scalability result analysis"""
    
    def test_excellent_scalability_grade(self):
        """Test excellent scalability grade calculation"""
        config = ScalabilityTestConfig(
            test_name="excellent_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING
        )
        
        metrics = ScalabilityMetrics(
            test_name="excellent_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING,
            start_time=datetime.now()
        )
        metrics.scaling_efficiency = 0.95  # Excellent
        
        result = ScalabilityResult(config, metrics)
        assert result.scalability_grade == "A"
    
    def test_poor_scalability_grade(self):
        """Test poor scalability grade calculation"""
        config = ScalabilityTestConfig(
            test_name="poor_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING
        )
        
        metrics = ScalabilityMetrics(
            test_name="poor_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING,
            start_time=datetime.now()
        )
        metrics.scaling_efficiency = 0.5  # Poor
        
        result = ScalabilityResult(config, metrics)
        assert result.scalability_grade == "F"
    
    def test_recommendations_poor_scalability(self):
        """Test recommendations for poor scalability"""
        config = ScalabilityTestConfig(
            test_name="rec_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING
        )
        
        metrics = ScalabilityMetrics(
            test_name="rec_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING,
            start_time=datetime.now()
        )
        metrics.scaling_efficiency = 0.6  # Poor
        metrics.latency_degradation_factor = 3.0  # High latency degradation
        metrics.resource_efficiency_score = 0.5  # Poor resource efficiency
        metrics.throughput_scaling_factor = 0.4  # Poor throughput scaling
        
        result = ScalabilityResult(config, metrics)
        
        assert any("poor scalability" in rec.lower() for rec in result.recommendations)
        assert any("latency increases significantly" in rec.lower() for rec in result.recommendations)
        assert any("resource utilization is inefficient" in rec.lower() for rec in result.recommendations)
        assert any("throughput does not scale well" in rec.lower() for rec in result.recommendations)
    
    def test_recommendations_with_breaking_point(self):
        """Test recommendations when breaking point is reached"""
        config = ScalabilityTestConfig(
            test_name="breaking_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING
        )
        
        metrics = ScalabilityMetrics(
            test_name="breaking_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING,
            start_time=datetime.now()
        )
        metrics.scaling_efficiency = 0.8
        metrics.breaking_point = ScalabilityDataPoint(
            load_level=200,
            timestamp=datetime.now(),
            response_time_ms=1000,
            throughput_ops_per_sec=10,
            success_rate=0.3,
            cpu_usage_percent=95,
            memory_usage_percent=90,
            memory_usage_mb=2048,
            error_count=50,
            concurrent_operations=200
        )
        
        result = ScalabilityResult(config, metrics)
        
        assert any("breaking point at 200" in rec for rec in result.recommendations)
    
    def test_scaling_limits_identification(self):
        """Test scaling limits identification"""
        config = ScalabilityTestConfig(
            test_name="limits_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING
        )
        
        metrics = ScalabilityMetrics(
            test_name="limits_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING,
            start_time=datetime.now()
        )
        
        # Add baseline performance
        metrics.baseline_performance = ScalabilityDataPoint(
            load_level=10,
            timestamp=datetime.now(),
            response_time_ms=100,
            throughput_ops_per_sec=50,
            success_rate=1.0,
            cpu_usage_percent=20,
            memory_usage_percent=30,
            memory_usage_mb=512,
            error_count=0,
            concurrent_operations=10
        )
        
        # Add data points showing resource bottlenecks
        metrics.data_points = [
            metrics.baseline_performance,
            ScalabilityDataPoint(
                load_level=50,
                timestamp=datetime.now(),
                response_time_ms=150,
                throughput_ops_per_sec=100,
                success_rate=0.98,
                cpu_usage_percent=95,  # CPU bottleneck
                memory_usage_percent=70,
                memory_usage_mb=1024,
                error_count=1,
                concurrent_operations=50
            ),
            ScalabilityDataPoint(
                load_level=100,
                timestamp=datetime.now(),
                response_time_ms=250,  # Latency bottleneck (>2x baseline)
                throughput_ops_per_sec=150,
                success_rate=0.95,
                cpu_usage_percent=98,
                memory_usage_percent=95,  # Memory bottleneck
                memory_usage_mb=2048,
                error_count=5,
                concurrent_operations=100
            )
        ]
        
        result = ScalabilityResult(config, metrics)
        
        assert result.scaling_limits["cpu_bottleneck_load"] == 50
        assert result.scaling_limits["memory_bottleneck_load"] == 100
        assert result.scaling_limits["latency_bottleneck_load"] == 100
        assert result.scaling_limits["max_tested_load"] == 100


class TestScalabilityValidator:
    """Test scalability validator functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create scalability validator instance"""
        return ScalabilityValidator()
    
    @pytest.fixture
    def basic_config(self):
        """Create basic scalability test configuration"""
        async def test_function():
            await asyncio.sleep(0.01)  # Simulate some work
            return "test_result"
        
        return ScalabilityTestConfig(
            test_name="basic_scalability_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING,
            baseline_load=2,
            max_load=8,
            scaling_steps=3,
            step_duration_seconds=1,  # Very short for testing
            warmup_duration_seconds=0,
            cooldown_duration_seconds=0,
            target_function=test_function,
            scaling_pattern="linear"
        )
    
    def test_generate_load_levels_linear(self, validator):
        """Test linear load level generation"""
        config = ScalabilityTestConfig(
            test_name="linear_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING,
            baseline_load=10,
            max_load=100,
            scaling_steps=4,
            scaling_pattern="linear"
        )
        
        load_levels = validator._generate_load_levels(config)
        
        expected = [10, 32, 55, 77, 100]  # Linear progression
        assert len(load_levels) == 5  # scaling_steps + 1
        assert load_levels[0] == 10
        assert load_levels[-1] == 100
        # Check that progression is roughly linear
        for i in range(1, len(load_levels)):
            assert load_levels[i] > load_levels[i-1]
    
    def test_generate_load_levels_exponential(self, validator):
        """Test exponential load level generation"""
        config = ScalabilityTestConfig(
            test_name="exponential_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING,
            baseline_load=10,
            max_load=160,
            scaling_steps=3,
            scaling_pattern="exponential"
        )
        
        load_levels = validator._generate_load_levels(config)
        
        assert len(load_levels) == 4  # scaling_steps + 1
        assert load_levels[0] == 10
        assert load_levels[-1] <= 160
        # Check exponential growth
        for i in range(1, len(load_levels)):
            assert load_levels[i] > load_levels[i-1]
    
    def test_generate_load_levels_logarithmic(self, validator):
        """Test logarithmic load level generation"""
        config = ScalabilityTestConfig(
            test_name="log_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING,
            baseline_load=10,
            max_load=100,
            scaling_steps=4,
            scaling_pattern="logarithmic"
        )
        
        load_levels = validator._generate_load_levels(config)
        
        assert len(load_levels) == 5
        assert load_levels[0] == 10
        assert load_levels[-1] == 100
        # Check logarithmic growth (slower than linear)
        for i in range(1, len(load_levels)):
            assert load_levels[i] > load_levels[i-1]
    
    @pytest.mark.asyncio
    async def test_default_load_function(self, validator):
        """Test default load function"""
        result = await validator._default_load_function()
        assert result == "default_result"
    
    @pytest.mark.asyncio
    async def test_execute_target_function_async(self, validator):
        """Test executing async target function"""
        async def async_test_function():
            await asyncio.sleep(0.01)
            return "async_result"
        
        result = await validator._execute_target_function(async_test_function)
        assert result == "async_result"
    
    @pytest.mark.asyncio
    async def test_execute_target_function_sync(self, validator):
        """Test executing sync target function"""
        def sync_test_function():
            return "sync_result"
        
        result = await validator._execute_target_function(sync_test_function)
        assert result == "sync_result"
    
    def test_is_breaking_point(self, validator):
        """Test breaking point detection"""
        config = ScalabilityTestConfig(
            test_name="breaking_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING
        )
        
        # Normal data point - should not be breaking point
        normal_point = ScalabilityDataPoint(
            load_level=50,
            timestamp=datetime.now(),
            response_time_ms=150,
            throughput_ops_per_sec=40,
            success_rate=0.98,
            cpu_usage_percent=70,
            memory_usage_percent=60,
            memory_usage_mb=1024,
            error_count=1,
            concurrent_operations=50
        )
        
        assert not validator._is_breaking_point(normal_point, config)
        
        # Breaking point - low success rate
        breaking_point = ScalabilityDataPoint(
            load_level=200,
            timestamp=datetime.now(),
            response_time_ms=2000,
            throughput_ops_per_sec=10,
            success_rate=0.3,  # Low success rate
            cpu_usage_percent=95,
            memory_usage_percent=90,
            memory_usage_mb=4096,
            error_count=150,
            concurrent_operations=200
        )
        
        assert validator._is_breaking_point(breaking_point, config)
        
        # Breaking point - high CPU
        cpu_breaking_point = ScalabilityDataPoint(
            load_level=150,
            timestamp=datetime.now(),
            response_time_ms=500,
            throughput_ops_per_sec=20,
            success_rate=0.8,
            cpu_usage_percent=96,  # Very high CPU
            memory_usage_percent=70,
            memory_usage_mb=2048,
            error_count=30,
            concurrent_operations=150
        )
        
        assert validator._is_breaking_point(cpu_breaking_point, config)
    
    def test_calculate_correlation(self, validator):
        """Test correlation calculation"""
        # Perfect positive correlation
        x1 = [1, 2, 3, 4, 5]
        y1 = [2, 4, 6, 8, 10]
        correlation1 = validator._calculate_correlation(x1, y1)
        assert abs(correlation1 - 1.0) < 0.01
        
        # Perfect negative correlation
        x2 = [1, 2, 3, 4, 5]
        y2 = [10, 8, 6, 4, 2]
        correlation2 = validator._calculate_correlation(x2, y2)
        assert abs(correlation2 - (-1.0)) < 0.01
        
        # No correlation
        x3 = [1, 2, 3, 4, 5]
        y3 = [3, 3, 3, 3, 3]
        correlation3 = validator._calculate_correlation(x3, y3)
        assert correlation3 == 0.0
    
    def test_analyze_scalability_metrics(self, validator):
        """Test scalability metrics analysis"""
        metrics = ScalabilityMetrics(
            test_name="analysis_test",
            test_type=ScalabilityTestType.THROUGHPUT_SCALING,
            start_time=datetime.now()
        )
        
        # Set up baseline performance
        metrics.baseline_performance = ScalabilityDataPoint(
            load_level=10,
            timestamp=datetime.now(),
            response_time_ms=100,
            throughput_ops_per_sec=50,
            success_rate=1.0,
            cpu_usage_percent=20,
            memory_usage_percent=30,
            memory_usage_mb=512,
            error_count=0,
            concurrent_operations=10
        )
        
        # Add data points
        metrics.data_points = [
            metrics.baseline_performance,
            ScalabilityDataPoint(
                load_level=20,
                timestamp=datetime.now(),
                response_time_ms=120,
                throughput_ops_per_sec=90,  # Good scaling
                success_rate=0.98,
                cpu_usage_percent=40,
                memory_usage_percent=50,
                memory_usage_mb=1024,
                error_count=1,
                concurrent_operations=20
            ),
            ScalabilityDataPoint(
                load_level=40,
                timestamp=datetime.now(),
                response_time_ms=150,
                throughput_ops_per_sec=160,  # Still good scaling
                success_rate=0.95,
                cpu_usage_percent=70,
                memory_usage_percent=80,
                memory_usage_mb=2048,
                error_count=3,
                concurrent_operations=40
            )
        ]
        
        validator._analyze_scalability_metrics(metrics)
        
        # Check that metrics were calculated
        assert metrics.scaling_efficiency > 0
        assert metrics.throughput_scaling_factor > 0
        assert metrics.latency_degradation_factor > 0
        assert metrics.linear_scalability_score >= 0
        assert metrics.resource_efficiency_score >= 0
    
    @pytest.mark.asyncio
    async def test_monitor_resources_during_test(self, validator):
        """Test resource monitoring during test"""
        # Mock psutil to return predictable values
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_cpu.return_value = 75.0
            
            mock_memory_obj = MagicMock()
            mock_memory_obj.percent = 60.0
            mock_memory_obj.used = 1024 * 1024 * 1024  # 1GB
            mock_memory.return_value = mock_memory_obj
            
            resource_usage = await validator._monitor_resources_during_test(1)
            
            assert "avg_cpu" in resource_usage
            assert "avg_memory_percent" in resource_usage
            assert "avg_memory_mb" in resource_usage
            assert resource_usage["avg_cpu"] == 75.0
            assert resource_usage["avg_memory_percent"] == 60.0
    
    @pytest.mark.asyncio
    async def test_basic_scalability_validation(self, validator, basic_config):
        """Test basic scalability validation"""
        # Mock resource monitoring
        with patch.object(validator, '_monitor_resources_during_test') as mock_monitor:
            mock_monitor.return_value = {
                "avg_cpu": 50.0,
                "avg_memory_percent": 40.0,
                "avg_memory_mb": 1024.0
            }
            
            result = await validator.validate_scalability(basic_config)
            
            assert isinstance(result, ScalabilityResult)
            assert result.config.test_name == "basic_scalability_test"
            assert len(result.metrics.data_points) > 0
            assert result.metrics.baseline_performance is not None
            assert result.scalability_grade in ["A", "B", "C", "D", "F"]
            assert len(result.recommendations) > 0


class TestScalabilityConfigFactories:
    """Test scalability configuration factory functions"""
    
    def test_create_throughput_scaling_config(self):
        """Test throughput scaling configuration factory"""
        async def test_function():
            return "test"
        
        config = create_throughput_scaling_config(
            target_function=test_function,
            baseline_load=20,
            max_load=400
        )
        
        assert config.test_name == "throughput_scaling"
        assert config.test_type == ScalabilityTestType.THROUGHPUT_SCALING
        assert config.baseline_load == 20
        assert config.max_load == 400
        assert config.target_function == test_function
        assert config.scaling_pattern == "exponential"
    
    def test_create_user_growth_config(self):
        """Test user growth configuration factory"""
        def test_function():
            return "test"
        
        config = create_user_growth_config(
            target_function=test_function,
            baseline_users=100,
            max_users=2000
        )
        
        assert config.test_name == "user_growth_simulation"
        assert config.test_type == ScalabilityTestType.USER_GROWTH_SIMULATION
        assert config.baseline_load == 100
        assert config.max_load == 2000
        assert config.target_function == test_function
    
    def test_create_latency_under_load_config(self):
        """Test latency under load configuration factory"""
        async def test_function():
            return "test"
        
        config = create_latency_under_load_config(
            target_function=test_function,
            baseline_load=15,
            max_load=300
        )
        
        assert config.test_name == "latency_under_load"
        assert config.test_type == ScalabilityTestType.LATENCY_UNDER_LOAD
        assert config.baseline_load == 15
        assert config.max_load == 300
        assert config.acceptable_degradation == 0.5
        assert config.scaling_pattern == "linear"


class TestComprehensiveScalabilitySuite:
    """Test comprehensive scalability testing suite"""
    
    @pytest.mark.asyncio
    async def test_comprehensive_scalability_suite(self):
        """Test comprehensive scalability suite"""
        async def test_function():
            await asyncio.sleep(0.01)
            return "test_result"
        
        # Mock the validator to complete quickly
        with patch('src.testing.scalability_validator.ScalabilityValidator.validate_scalability') as mock_validate:
            # Create mock results
            mock_result = MagicMock()
            mock_result.scalability_grade = "B"
            mock_result.metrics.scaling_efficiency = 0.8
            mock_validate.return_value = mock_result
            
            results = await run_comprehensive_scalability_suite(test_function)
            
            # Should run 3 tests (throughput, user growth, latency)
            assert len(results) == 3
            assert mock_validate.call_count == 3
            
            # Check that different test types were called
            call_args = [call[0][0] for call in mock_validate.call_args_list]
            test_names = [config.test_name for config in call_args]
            
            assert "throughput_scaling" in test_names
            assert "user_growth_simulation" in test_names
            assert "latency_under_load" in test_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])