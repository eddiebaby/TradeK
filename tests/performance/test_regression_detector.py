"""
Tests for Performance Regression Detection.

This module tests the regression detection capabilities including
baseline establishment, trend analysis, and automated alerting.
"""

import pytest
import asyncio
import time
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.performance.regression_detector import (
    PerformanceRegressionDetector,
    PerformanceBaseline,
    PerformanceMeasurement,
    RegressionAlert,
    TrendAnalysis,
    RegressionSeverity,
    ChangeDirection,
    performance_monitor,
    get_regression_detector,
    establish_baseline_for_operation,
    check_performance_regression
)


class TestPerformanceBaseline:
    """Test performance baseline functionality"""
    
    def test_baseline_creation(self):
        """Test baseline creation with all fields"""
        baseline = PerformanceBaseline(
            metric_name="response_time",
            operation_name="search_query",
            baseline_value=150.5,
            sample_size=100,
            confidence_interval=(140.0, 161.0),
            created_at=datetime.now(),
            last_updated=datetime.now(),
            standard_deviation=25.0,
            percentiles={"p50": 145.0, "p95": 180.0},
            metadata={"version": "1.0"}
        )
        
        assert baseline.metric_name == "response_time"
        assert baseline.operation_name == "search_query"
        assert baseline.baseline_value == 150.5
        assert baseline.sample_size == 100
        assert baseline.confidence_interval == (140.0, 161.0)
        assert baseline.standard_deviation == 25.0
        assert baseline.percentiles["p95"] == 180.0
        assert baseline.metadata["version"] == "1.0"


class TestPerformanceMeasurement:
    """Test performance measurement functionality"""
    
    def test_measurement_creation(self):
        """Test measurement creation with all fields"""
        measurement = PerformanceMeasurement(
            metric_name="response_time",
            operation_name="search_query",
            value=175.5,
            timestamp=datetime.now(),
            duration_ms=175.5,
            context={"user_id": "123", "query_complexity": "high"},
            tags={"version": "1.0", "environment": "production"}
        )
        
        assert measurement.metric_name == "response_time"
        assert measurement.operation_name == "search_query"
        assert measurement.value == 175.5
        assert measurement.duration_ms == 175.5
        assert measurement.context["user_id"] == "123"
        assert measurement.tags["environment"] == "production"


class TestRegressionAlert:
    """Test regression alert functionality"""
    
    def test_alert_creation(self):
        """Test alert creation with all fields"""
        alert = RegressionAlert(
            alert_id="alert_123",
            metric_name="response_time",
            operation_name="search_query",
            severity=RegressionSeverity.HIGH,
            change_direction=ChangeDirection.DEGRADATION,
            baseline_value=150.0,
            current_value=195.0,
            change_percentage=0.3,
            detection_time=datetime.now(),
            confidence_level=0.95,
            sample_size=50,
            statistical_significance=0.01,
            description="Performance degradation detected",
            remediation_suggestions=["Check database performance", "Review recent code changes"]
        )
        
        assert alert.alert_id == "alert_123"
        assert alert.severity == RegressionSeverity.HIGH
        assert alert.change_direction == ChangeDirection.DEGRADATION
        assert alert.change_percentage == 0.3
        assert len(alert.remediation_suggestions) == 2


class TestPerformanceRegressionDetector:
    """Test performance regression detector functionality"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def detector(self, temp_db):
        """Create detector instance for testing"""
        return PerformanceRegressionDetector(db_path=temp_db)
    
    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector.baseline_sample_size == 100
        assert detector.regression_threshold == 0.15
        assert detector.confidence_level == 0.95
        assert len(detector.baselines) == 0
        assert len(detector.recent_measurements) == 0
        assert len(detector.alerts) == 0
    
    def test_record_measurement(self, detector):
        """Test recording a performance measurement"""
        detector.record_measurement(
            metric_name="response_time",
            operation_name="test_operation",
            value=150.0,
            duration_ms=150.0,
            context={"test": True},
            tags={"env": "test"}
        )
        
        key = "response_time:test_operation"
        assert key in detector.recent_measurements
        assert len(detector.recent_measurements[key]) == 1
        
        measurement = detector.recent_measurements[key][0]
        assert measurement.value == 150.0
        assert measurement.context["test"] is True
        assert measurement.tags["env"] == "test"
    
    def test_establish_baseline_with_measurements(self, detector):
        """Test establishing baseline with provided measurements"""
        measurements = [100.0, 110.0, 120.0, 105.0, 115.0, 125.0, 95.0, 130.0, 108.0, 112.0, 118.0, 122.0]
        
        baseline = detector.establish_baseline(
            metric_name="response_time",
            operation_name="test_operation",
            measurements=measurements
        )
        
        assert baseline.metric_name == "response_time"
        assert baseline.operation_name == "test_operation"
        assert baseline.sample_size == len(measurements)
        assert 110.0 < baseline.baseline_value < 120.0  # Should be around the mean
        assert baseline.standard_deviation > 0
        assert len(baseline.percentiles) == 5  # p50, p75, p90, p95, p99
        
        # Should be stored in detector
        key = "response_time:test_operation"
        assert key in detector.baselines
    
    def test_establish_baseline_insufficient_data(self, detector):
        """Test establishing baseline with insufficient data"""
        measurements = [100.0, 110.0, 120.0]  # Only 3 measurements
        
        with pytest.raises(ValueError, match="Insufficient measurements"):
            detector.establish_baseline(
                "response_time",
                "test_operation",
                measurements
            )
    
    def test_establish_baseline_from_recent_measurements(self, detector):
        """Test establishing baseline from recent measurements"""
        # Record enough measurements
        for i in range(15):
            detector.record_measurement(
                "response_time",
                "test_operation",
                100.0 + i * 5  # Values from 100 to 170
            )
        
        baseline = detector.establish_baseline("response_time", "test_operation")
        
        assert baseline.sample_size == 15
        assert 120.0 < baseline.baseline_value < 140.0  # Should be around the mean
    
    def test_regression_detection_degradation(self, detector):
        """Test detection of performance degradation"""
        # Establish baseline
        baseline_measurements = [100.0] * 20  # Stable baseline of 100ms
        detector.establish_baseline("response_time", "test_operation", baseline_measurements)
        
        # Record measurements to build recent history
        for i in range(10):
            detector.record_measurement("response_time", "test_operation", 100.0)
        
        # Record a significant degradation (50% increase)
        detector.record_measurement("response_time", "test_operation", 150.0)
        
        # Should have generated an alert
        alerts = detector.get_alerts(limit=1)
        assert len(alerts) > 0
        
        alert = alerts[0]
        assert alert.change_direction == ChangeDirection.DEGRADATION
        assert alert.severity in [RegressionSeverity.MEDIUM, RegressionSeverity.HIGH]
        assert alert.baseline_value == 100.0
        assert alert.current_value == 150.0
    
    def test_regression_detection_improvement(self, detector):
        """Test detection of performance improvement"""
        # Establish baseline
        baseline_measurements = [200.0] * 20  # Baseline of 200ms
        detector.establish_baseline("response_time", "test_operation", baseline_measurements)
        
        # Record a significant improvement (50% decrease)
        detector.record_measurement("response_time", "test_operation", 100.0)
        
        # Should have generated an improvement alert
        alerts = detector.get_alerts(limit=1)
        assert len(alerts) > 0
        
        alert = alerts[0]
        assert alert.change_direction == ChangeDirection.IMPROVEMENT
        assert alert.severity == RegressionSeverity.LOW  # Improvements are low priority
    
    def test_no_regression_within_threshold(self, detector):
        """Test that small changes don't trigger alerts"""
        # Establish baseline
        baseline_measurements = [100.0] * 20
        detector.establish_baseline("response_time", "test_operation", baseline_measurements)
        
        # Record a small change (10% increase, below 15% threshold)
        detector.record_measurement("response_time", "test_operation", 110.0)
        
        # Should not generate an alert
        alerts = detector.get_alerts(since=datetime.now() - timedelta(minutes=1))
        assert len(alerts) == 0
    
    def test_severity_calculation(self, detector):
        """Test severity calculation for different change percentages"""
        # Test critical severity (50%+ change)
        severity = detector._calculate_severity(0.6)
        assert severity == RegressionSeverity.CRITICAL
        
        # Test high severity (30-49% change)
        severity = detector._calculate_severity(0.35)
        assert severity == RegressionSeverity.HIGH
        
        # Test medium severity (20-29% change)
        severity = detector._calculate_severity(0.25)
        assert severity == RegressionSeverity.MEDIUM
        
        # Test low severity (15-19% change)
        severity = detector._calculate_severity(0.17)
        assert severity == RegressionSeverity.LOW
    
    def test_remediation_suggestions_response_time(self, detector):
        """Test remediation suggestions for response time degradation"""
        measurement = PerformanceMeasurement(
            metric_name="response_time_ms",
            operation_name="test_op",
            value=200.0,
            timestamp=datetime.now()
        )
        
        baseline = PerformanceBaseline(
            metric_name="response_time_ms",
            operation_name="test_op",
            baseline_value=100.0,
            sample_size=20,
            confidence_interval=(90.0, 110.0),
            created_at=datetime.now(),
            last_updated=datetime.now(),
            standard_deviation=10.0
        )
        
        suggestions = detector._generate_remediation_suggestions(
            measurement, baseline, ChangeDirection.DEGRADATION, RegressionSeverity.HIGH
        )
        
        assert len(suggestions) > 0
        assert any("load" in suggestion.lower() for suggestion in suggestions)
        assert any("database" in suggestion.lower() for suggestion in suggestions)
        assert any("caching" in suggestion.lower() for suggestion in suggestions)
    
    @pytest.mark.skipif(True, reason="Requires scipy for trend analysis")
    def test_trend_analysis(self, detector):
        """Test trend analysis functionality"""
        # This test requires scipy, skip if not available
        try:
            from scipy import stats
        except ImportError:
            pytest.skip("scipy not available")
        
        # Record measurements with a clear upward trend
        base_time = datetime.now() - timedelta(hours=2)
        for i in range(20):
            timestamp = base_time + timedelta(minutes=i * 5)
            value = 100.0 + i * 2  # Linear increase
            
            # Mock the database insert
            with patch.object(detector, '_save_measurement'):
                measurement = PerformanceMeasurement(
                    metric_name="response_time",
                    operation_name="test_operation",
                    value=value,
                    timestamp=timestamp
                )
                detector.recent_measurements.setdefault("response_time:test_operation", []).append(measurement)
        
        # Mock database query for trend analysis
        with patch('sqlite3.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                (100.0 + i * 2, (base_time + timedelta(minutes=i * 5)).isoformat())
                for i in range(20)
            ]
            mock_connect.return_value.__enter__.return_value.execute.return_value = mock_cursor
            
            trend = detector.analyze_trends("response_time", "test_operation")
            
            assert trend.trend_direction == ChangeDirection.DEGRADATION  # Upward trend is degradation
            assert trend.trend_strength > 0.8  # Should have strong correlation
            assert trend.slope > 0  # Positive slope
            assert trend.data_points == 20
    
    def test_get_alerts_filtering(self, detector):
        """Test alert filtering functionality"""
        # Create some test alerts directly
        now = datetime.now()
        
        alert1 = RegressionAlert(
            alert_id="alert1",
            metric_name="response_time",
            operation_name="op1",
            severity=RegressionSeverity.CRITICAL,
            change_direction=ChangeDirection.DEGRADATION,
            baseline_value=100.0,
            current_value=200.0,
            change_percentage=1.0,
            detection_time=now - timedelta(hours=1),
            confidence_level=0.95,
            sample_size=20,
            statistical_significance=0.01,
            description="Critical regression"
        )
        
        alert2 = RegressionAlert(
            alert_id="alert2",
            metric_name="throughput",
            operation_name="op2",
            severity=RegressionSeverity.LOW,
            change_direction=ChangeDirection.IMPROVEMENT,
            baseline_value=50.0,
            current_value=40.0,
            change_percentage=-0.2,
            detection_time=now - timedelta(hours=2),
            confidence_level=0.95,
            sample_size=15,
            statistical_significance=0.05,
            description="Minor improvement"
        )
        
        # Save alerts to database
        detector._save_alert(alert1)
        detector._save_alert(alert2)
        
        # Test filtering by severity
        critical_alerts = detector.get_alerts(severity=RegressionSeverity.CRITICAL)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].alert_id == "alert1"
        
        # Test filtering by time
        recent_alerts = detector.get_alerts(since=now - timedelta(minutes=90))
        assert len(recent_alerts) == 1
        assert recent_alerts[0].alert_id == "alert1"
        
        # Test limit
        limited_alerts = detector.get_alerts(limit=1)
        assert len(limited_alerts) == 1
    
    def test_get_baseline(self, detector):
        """Test getting baseline for a metric"""
        # No baseline initially
        baseline = detector.get_baseline("response_time", "test_operation")
        assert baseline is None
        
        # Establish baseline
        measurements = [100.0] * 20
        detector.establish_baseline("response_time", "test_operation", measurements)
        
        # Should now return baseline
        baseline = detector.get_baseline("response_time", "test_operation")
        assert baseline is not None
        assert baseline.metric_name == "response_time"
        assert baseline.operation_name == "test_operation"
    
    def test_get_recent_measurements(self, detector):
        """Test getting recent measurements"""
        # Record some measurements
        for i in range(5):
            detector.record_measurement("response_time", "test_operation", 100.0 + i)
        
        measurements = detector.get_recent_measurements("response_time", "test_operation", limit=3)
        assert len(measurements) <= 3
        
        # Should be ordered by timestamp (most recent first)
        if len(measurements) > 1:
            assert measurements[0].timestamp >= measurements[1].timestamp
    
    def test_performance_report_generation(self, detector):
        """Test comprehensive performance report generation"""
        # Set up some test data
        detector.establish_baseline("response_time", "search", [100.0] * 20)
        detector.establish_baseline("throughput", "api", [50.0] * 20)
        
        # Generate report
        report = detector.generate_performance_report()
        
        assert "generated_at" in report
        assert "summary" in report
        assert "baselines" in report
        assert "trends" in report
        assert "alerts" in report
        assert "recommendations" in report
        
        assert report["summary"]["total_baselines"] == 2
        assert len(report["baselines"]) == 2
        assert len(report["recommendations"]) > 0
    
    def test_measurement_retention(self, detector):
        """Test that old measurements are cleaned up"""
        # Mock old measurement
        old_time = datetime.now() - timedelta(days=31)  # Older than retention period
        old_measurement = PerformanceMeasurement(
            metric_name="response_time",
            operation_name="test_operation",
            value=100.0,
            timestamp=old_time
        )
        
        key = "response_time:test_operation"
        detector.recent_measurements[key] = [old_measurement]
        
        # Record new measurement (should trigger cleanup)
        detector.record_measurement("response_time", "test_operation", 110.0)
        
        # Old measurement should be removed
        assert len(detector.recent_measurements[key]) == 1
        assert detector.recent_measurements[key][0].timestamp > old_time


class TestPerformanceMonitorDecorator:
    """Test performance monitor decorator"""
    
    @pytest.fixture
    def temp_detector(self):
        """Create temporary detector for testing"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        detector = PerformanceRegressionDetector(db_path=path)
        
        # Patch global detector
        with patch('src.performance.regression_detector._global_detector', detector):
            yield detector
        
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    
    @pytest.mark.asyncio
    async def test_async_function_monitoring(self, temp_detector):
        """Test monitoring async functions"""
        @performance_monitor("response_time", "async_test")
        async def async_test_function():
            await asyncio.sleep(0.01)  # Small delay
            return "test_result"
        
        result = await async_test_function()
        
        assert result == "test_result"
        
        # Should have recorded a measurement
        measurements = temp_detector.get_recent_measurements("response_time", "async_test")
        assert len(measurements) == 1
        assert measurements[0].value > 0  # Should have some duration
    
    def test_sync_function_monitoring(self, temp_detector):
        """Test monitoring sync functions"""
        @performance_monitor("response_time", "sync_test")
        def sync_test_function():
            time.sleep(0.01)  # Small delay
            return "test_result"
        
        result = sync_test_function()
        
        assert result == "test_result"
        
        # Should have recorded a measurement
        measurements = temp_detector.get_recent_measurements("response_time", "sync_test")
        assert len(measurements) == 1
        assert measurements[0].value > 0  # Should have some duration
    
    @pytest.mark.asyncio
    async def test_function_monitoring_with_exception(self, temp_detector):
        """Test monitoring functions that raise exceptions"""
        @performance_monitor("response_time", "error_test")
        async def failing_function():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await failing_function()
        
        # Should have recorded an error measurement
        measurements = temp_detector.get_recent_measurements("response_time_error", "error_test")
        assert len(measurements) == 1
        assert measurements[0].value > 0


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.fixture
    def temp_detector(self):
        """Create temporary detector for testing"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        detector = PerformanceRegressionDetector(db_path=path)
        
        # Patch global detector
        with patch('src.performance.regression_detector._global_detector', detector):
            yield detector
        
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    
    def test_establish_baseline_for_operation(self, temp_detector):
        """Test convenience function for establishing baselines"""
        measurements = [100.0, 110.0, 120.0, 105.0, 115.0, 125.0, 95.0, 130.0, 108.0, 112.0]
        
        baseline = establish_baseline_for_operation(
            "test_operation",
            "response_time_ms",
            measurements
        )
        
        assert baseline.operation_name == "test_operation"
        assert baseline.metric_name == "response_time_ms"
        assert baseline.sample_size == len(measurements)
    
    def test_check_performance_regression(self, temp_detector):
        """Test convenience function for checking regression"""
        # Establish baseline first
        measurements = [100.0] * 20
        establish_baseline_for_operation("test_operation", "response_time_ms", measurements)
        
        # Check for regression
        alert = check_performance_regression("test_operation", 150.0, "response_time_ms")
        
        # Should detect regression (50% increase)
        assert alert is not None
        assert alert.change_direction == ChangeDirection.DEGRADATION
        assert alert.current_value == 150.0


class TestDatabasePersistence:
    """Test database persistence functionality"""
    
    @pytest.fixture
    def temp_detector(self):
        """Create temporary detector for testing"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        detector = PerformanceRegressionDetector(db_path=path)
        yield detector, path
        
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    
    def test_baseline_persistence(self, temp_detector):
        """Test that baselines are persisted and loaded correctly"""
        detector, db_path = temp_detector
        
        # Establish baseline
        measurements = [100.0] * 20
        baseline = detector.establish_baseline("response_time", "test_operation", measurements)
        
        # Create new detector instance with same database
        detector2 = PerformanceRegressionDetector(db_path=db_path)
        
        # Should load existing baseline
        loaded_baseline = detector2.get_baseline("response_time", "test_operation")
        assert loaded_baseline is not None
        assert loaded_baseline.baseline_value == baseline.baseline_value
        assert loaded_baseline.sample_size == baseline.sample_size
    
    def test_measurement_persistence(self, temp_detector):
        """Test that measurements are persisted correctly"""
        detector, _ = temp_detector
        
        # Record measurement
        detector.record_measurement("response_time", "test_operation", 150.0)
        
        # Should be able to retrieve from database
        measurements = detector.get_recent_measurements("response_time", "test_operation")
        assert len(measurements) == 1
        assert measurements[0].value == 150.0
    
    def test_alert_persistence(self, temp_detector):
        """Test that alerts are persisted correctly"""
        detector, _ = temp_detector
        
        # Create baseline and trigger alert
        measurements = [100.0] * 20
        detector.establish_baseline("response_time", "test_operation", measurements)
        detector.record_measurement("response_time", "test_operation", 200.0)  # 100% increase
        
        # Should have persisted alert
        alerts = detector.get_alerts()
        assert len(alerts) > 0
        assert alerts[0].current_value == 200.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])