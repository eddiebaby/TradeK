"""
Tests for Metrics Collection and Monitoring System.

This module tests the metrics collection, alerting, and monitoring
functionality for TradeKnowledge.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.monitoring.metrics_collector import (
    MetricsCollector,
    MetricDefinition,
    MetricType,
    MetricValue,
    AlertManager,
    AlertRule,
    Alert,
    console_alert_handler,
    log_alert_handler,
    get_metrics_collector,
    get_alert_manager,
    setup_monitoring
)


class TestMetricsCollector:
    """Test metrics collection functionality"""
    
    @pytest.fixture
    def collector(self):
        """Create metrics collector for testing"""
        return MetricsCollector(max_retention_hours=1, max_series_length=100)
    
    def test_metric_registration(self, collector):
        """Test metric definition registration"""
        metric_def = MetricDefinition(
            name="test.metric",
            metric_type=MetricType.GAUGE,
            description="Test metric for unit tests",
            unit="units",
            tags={"component": "test"}
        )
        
        collector.register_metric(metric_def)
        
        assert "test.metric" in collector.metric_definitions
        assert collector.metric_definitions["test.metric"].description == "Test metric for unit tests"
    
    def test_record_and_retrieve_metrics(self, collector):
        """Test recording and retrieving metric values"""
        now = datetime.now()
        
        # Record some metric values
        collector.record_metric("test.counter", 10, tags={"env": "test"}, timestamp=now)
        collector.record_metric("test.counter", 15, tags={"env": "test"}, timestamp=now + timedelta(seconds=30))
        collector.record_metric("test.counter", 20, tags={"env": "test"}, timestamp=now + timedelta(seconds=60))
        
        # Retrieve all values
        values = collector.get_metric_values("test.counter")
        assert len(values) == 3
        assert values[0].value == 10
        assert values[1].value == 15
        assert values[2].value == 20
        
        # Retrieve with time filter
        filtered_values = collector.get_metric_values(
            "test.counter",
            start_time=now + timedelta(seconds=20),
            end_time=now + timedelta(seconds=50)
        )
        assert len(filtered_values) == 1
        assert filtered_values[0].value == 15
        
        # Retrieve with tag filter
        tagged_values = collector.get_metric_values(
            "test.counter",
            tags_filter={"env": "test"}
        )
        assert len(tagged_values) == 3
        
        # Non-matching tag filter
        no_match_values = collector.get_metric_values(
            "test.counter",
            tags_filter={"env": "production"}
        )
        assert len(no_match_values) == 0
    
    def test_metric_summary_statistics(self, collector):
        """Test metric summary statistics calculation"""
        now = datetime.now()
        
        # Record test data
        test_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for i, value in enumerate(test_values):
            collector.record_metric(
                "test.histogram",
                value,
                timestamp=now + timedelta(seconds=i)
            )
        
        summary = collector.get_metric_summary("test.histogram")
        
        assert summary["count"] == 10
        assert summary["min"] == 10
        assert summary["max"] == 100
        assert summary["mean"] == 55.0
        assert summary["latest"] == 100
        assert "p50" in summary
        assert "p95" in summary
        assert "p99" in summary
        
        # Test with no data
        empty_summary = collector.get_metric_summary("nonexistent.metric")
        assert empty_summary["count"] == 0
        assert "error" in empty_summary
    
    def test_counter_rate_calculation(self, collector):
        """Test automatic rate calculation for counters"""
        # Register a counter metric
        counter_def = MetricDefinition(
            name="test.requests",
            metric_type=MetricType.COUNTER,
            description="Test request counter"
        )
        collector.register_metric(counter_def)
        
        now = datetime.now()
        
        # Record counter values
        collector.record_metric("test.requests", 100, timestamp=now)
        collector.record_metric("test.requests", 110, timestamp=now + timedelta(seconds=10))
        collector.record_metric("test.requests", 125, timestamp=now + timedelta(seconds=20))
        
        # Check that rate metrics were automatically created
        rate_values = collector.get_metric_values("test.requests.rate")
        assert len(rate_values) >= 1
        
        # Rate should be approximately 1.0 requests/second (10 requests / 10 seconds)
        assert abs(rate_values[0].value - 1.0) < 0.1
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_system_metrics_collection(self, mock_disk, mock_memory, mock_cpu, collector):
        """Test system metrics collection"""
        # Mock system metrics
        mock_cpu.return_value = 75.5
        
        mock_memory_obj = MagicMock()
        mock_memory_obj.used = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory_obj.percent = 80.0
        mock_memory.return_value = mock_memory_obj
        
        mock_disk_obj = MagicMock()
        mock_disk_obj.used = 500 * 1024 * 1024 * 1024  # 500GB
        mock_disk_obj.total = 1000 * 1024 * 1024 * 1024  # 1TB
        mock_disk.return_value = mock_disk_obj
        
        # Collect system metrics
        collector.collect_system_metrics()
        
        # Verify metrics were recorded
        cpu_values = collector.get_metric_values("system.cpu.percent")
        assert len(cpu_values) == 1
        assert cpu_values[0].value == 75.5
        
        memory_values = collector.get_metric_values("system.memory.percent")
        assert len(memory_values) == 1
        assert memory_values[0].value == 80.0
        
        disk_values = collector.get_metric_values("system.disk.used_percent")
        assert len(disk_values) == 1
        assert disk_values[0].value == 50.0  # 500GB / 1TB * 100
    
    def test_metrics_cleanup(self, collector):
        """Test cleanup of old metrics"""
        now = datetime.now()
        old_time = now - timedelta(hours=25)  # Older than retention
        recent_time = now - timedelta(minutes=30)  # Within retention
        
        # Record old and recent metrics
        collector.record_metric("test.cleanup", 1, timestamp=old_time)
        collector.record_metric("test.cleanup", 2, timestamp=recent_time)
        collector.record_metric("test.cleanup", 3, timestamp=now)
        
        # Before cleanup
        values_before = collector.get_metric_values("test.cleanup")
        assert len(values_before) == 3
        
        # Cleanup with 24 hour retention
        collector.cleanup_old_metrics(older_than_hours=24)
        
        # After cleanup
        values_after = collector.get_metric_values("test.cleanup")
        assert len(values_after) == 2  # Only recent values should remain
        assert values_after[0].value == 2
        assert values_after[1].value == 3
    
    def test_metrics_export_json(self, collector):
        """Test JSON export of metrics"""
        now = datetime.now()
        
        collector.record_metric("test.export", 42, tags={"format": "json"}, timestamp=now)
        
        json_export = collector.export_metrics("json")
        
        assert '"test.export"' in json_export
        assert '"value": 42' in json_export
        assert '"format": "json"' in json_export
        
        # Test invalid format
        with pytest.raises(ValueError):
            collector.export_metrics("invalid_format")
    
    def test_metrics_export_prometheus(self, collector):
        """Test Prometheus export of metrics"""
        now = datetime.now()
        
        # Register metric with description
        metric_def = MetricDefinition(
            name="test.prometheus",
            metric_type=MetricType.GAUGE,
            description="Test metric for Prometheus export"
        )
        collector.register_metric(metric_def)
        
        collector.record_metric("test.prometheus", 99, tags={"service": "test"}, timestamp=now)
        
        prometheus_export = collector.export_metrics("prometheus")
        
        assert "# HELP test_prometheus Test metric for Prometheus export" in prometheus_export
        assert "# TYPE test_prometheus gauge" in prometheus_export
        assert 'test_prometheus{service="test"} 99' in prometheus_export


class TestAlertManager:
    """Test alert management functionality"""
    
    @pytest.fixture
    def collector(self):
        """Create metrics collector for testing"""
        return MetricsCollector(max_retention_hours=1, max_series_length=100)
    
    @pytest.fixture
    def alert_manager(self, collector):
        """Create alert manager for testing"""
        return AlertManager(collector)
    
    def test_alert_rule_management(self, alert_manager):
        """Test adding and removing alert rules"""
        rule = AlertRule(
            name="test_rule",
            metric_name="test.metric",
            condition="greater_than",
            threshold=50.0,
            duration_minutes=1,
            severity="warning",
            description="Test alert rule"
        )
        
        alert_manager.add_alert_rule(rule)
        assert "test_rule" in alert_manager.alert_rules
        
        alert_manager.remove_alert_rule("test_rule")
        assert "test_rule" not in alert_manager.alert_rules
    
    def test_alert_condition_evaluation(self, alert_manager):
        """Test alert condition evaluation logic"""
        # Test different conditions
        assert alert_manager._check_condition(75, "greater_than", 50) == True
        assert alert_manager._check_condition(25, "greater_than", 50) == False
        
        assert alert_manager._check_condition(25, "less_than", 50) == True
        assert alert_manager._check_condition(75, "less_than", 50) == False
        
        assert alert_manager._check_condition(50, "equals", 50) == True
        assert alert_manager._check_condition(51, "equals", 50) == False
        
        assert alert_manager._check_condition(51, "not_equals", 50) == True
        assert alert_manager._check_condition(50, "not_equals", 50) == False
    
    def test_alert_triggering_and_resolution(self, alert_manager, collector):
        """Test alert triggering and resolution"""
        # Set up alert rule
        rule = AlertRule(
            name="high_cpu",
            metric_name="test.cpu",
            condition="greater_than",
            threshold=80.0,
            duration_minutes=1,
            severity="warning",
            description="High CPU usage"
        )
        alert_manager.add_alert_rule(rule)
        
        now = datetime.now()
        
        # Record high CPU values that should trigger alert
        for i in range(5):
            collector.record_metric(
                "test.cpu",
                85.0,  # Above threshold
                timestamp=now + timedelta(seconds=i*10)
            )
        
        # Evaluate alert rules
        alert_manager.evaluate_alert_rules()
        
        # Alert should be active
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].rule_name == "high_cpu"
        assert active_alerts[0].current_value == 85.0
        
        # Record normal CPU values within the duration window
        for i in range(5):
            collector.record_metric(
                "test.cpu",
                60.0,  # Below threshold
                timestamp=now + timedelta(seconds=50+i*2)  # Within 1 minute window
            )
        
        # Evaluate again with a later evaluation time
        # Simulate evaluation happening after the normal CPU values
        evaluation_time = now + timedelta(seconds=65)
        alert_manager._evaluate_single_rule(rule, evaluation_time)
        
        # Alert should be resolved
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 0
    
    def test_alert_handlers(self, alert_manager, collector):
        """Test alert handler functionality"""
        # Create a mock handler
        mock_handler = MagicMock()
        alert_manager.add_alert_handler(mock_handler)
        
        # Set up alert rule
        rule = AlertRule(
            name="test_handler",
            metric_name="test.metric",
            condition="greater_than",
            threshold=100.0,
            duration_minutes=1,
            severity="critical"
        )
        alert_manager.add_alert_rule(rule)
        
        now = datetime.now()
        
        # Record metrics that trigger alert
        for i in range(3):
            collector.record_metric(
                "test.metric",
                150.0,
                timestamp=now + timedelta(seconds=i*10)
            )
        
        # Evaluate alert rules
        alert_manager.evaluate_alert_rules()
        
        # Handler should have been called
        mock_handler.assert_called_once()
        
        # Verify alert object passed to handler
        alert_arg = mock_handler.call_args[0][0]
        assert isinstance(alert_arg, Alert)
        assert alert_arg.rule_name == "test_handler"
        assert alert_arg.severity == "critical"
    
    def test_alert_summary(self, alert_manager, collector):
        """Test alert summary generation"""
        # Set up multiple alert rules
        rules = [
            AlertRule("warning_rule", "test.metric1", "greater_than", 50, severity="warning"),
            AlertRule("critical_rule", "test.metric2", "greater_than", 90, severity="critical")
        ]
        
        for rule in rules:
            alert_manager.add_alert_rule(rule)
        
        now = datetime.now()
        
        # Trigger both alerts
        collector.record_metric("test.metric1", 75, timestamp=now)
        collector.record_metric("test.metric2", 95, timestamp=now)
        
        alert_manager.evaluate_alert_rules()
        
        summary = alert_manager.get_alert_summary()
        
        assert summary["total_active_alerts"] == 2
        assert summary["severity_breakdown"]["warning"] == 1
        assert summary["severity_breakdown"]["critical"] == 1
        assert summary["total_rules"] == len(alert_manager.alert_rules)
        assert summary["firing_rules"] == 2
    
    def test_default_alert_rules(self, alert_manager):
        """Test that default alert rules are created"""
        # Default rules should include system monitoring
        rule_names = list(alert_manager.alert_rules.keys())
        
        expected_rules = [
            "high_cpu_usage",
            "high_memory_usage", 
            "high_disk_usage",
            "api_error_rate_high",
            "search_response_time_slow"
        ]
        
        for expected_rule in expected_rules:
            assert expected_rule in rule_names


class TestMonitoringIntegration:
    """Test monitoring system integration"""
    
    def test_global_instances(self):
        """Test global instance getters"""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        # Should return the same instance
        assert collector1 is collector2
        
        alert_manager1 = get_alert_manager()
        alert_manager2 = get_alert_manager()
        
        # Should return the same instance
        assert alert_manager1 is alert_manager2
        
        # Alert manager should use the same collector
        assert alert_manager1.metrics_collector is collector1
    
    def test_setup_monitoring(self):
        """Test monitoring system setup"""
        # This test would need to be carefully designed to avoid interfering
        # with global state in other tests
        collector = MetricsCollector()
        alert_manager = AlertManager(collector)
        
        # Verify that basic setup works
        assert len(alert_manager.alert_rules) > 0
        assert len(alert_manager._alert_handlers) == 0  # No handlers by default
    
    @pytest.mark.asyncio
    async def test_automatic_collection(self):
        """Test automatic metrics collection"""
        collector = MetricsCollector()
        
        # Start automatic collection with very short interval
        collector.start_automatic_collection(interval_seconds=0.1)
        
        # Wait briefly
        await asyncio.sleep(0.3)
        
        # Stop collection
        collector.stop_automatic_collection()
        
        # Should have collected some system metrics
        cpu_values = collector.get_metric_values("system.cpu.percent")
        assert len(cpu_values) > 0
    
    @pytest.mark.asyncio
    async def test_automatic_alert_evaluation(self):
        """Test automatic alert evaluation"""
        collector = MetricsCollector()
        alert_manager = AlertManager(collector)
        
        # Add a test rule
        rule = AlertRule(
            name="auto_eval_test",
            metric_name="test.auto",
            condition="greater_than",
            threshold=50.0,
            duration_minutes=1
        )
        alert_manager.add_alert_rule(rule)
        
        # Record a metric that should trigger the alert
        now = datetime.now()
        for i in range(3):
            collector.record_metric(
                "test.auto",
                75.0,
                timestamp=now + timedelta(seconds=i*10)
            )
        
        # Start automatic evaluation with very short interval
        alert_manager.start_alert_evaluation(interval_seconds=0.1)
        
        # Wait briefly for evaluation
        await asyncio.sleep(0.3)
        
        # Stop evaluation
        alert_manager.stop_alert_evaluation()
        
        # Alert should have been triggered
        active_alerts = alert_manager.get_active_alerts()
        assert any(alert.rule_name == "auto_eval_test" for alert in active_alerts)


class TestAlertHandlers:
    """Test built-in alert handlers"""
    
    def test_console_alert_handler(self, capsys):
        """Test console alert handler"""
        alert = Alert(
            rule_name="test_console",
            metric_name="test.metric",
            current_value=95.0,
            threshold=80.0,
            condition="greater_than",
            severity="warning",
            description="Test console alert",
            started_at=datetime.now()
        )
        
        console_alert_handler(alert)
        
        captured = capsys.readouterr()
        assert "test_console" in captured.out
        assert "Test console alert" in captured.out
        assert "95.0" in captured.out
        assert "80.0" in captured.out
    
    def test_log_alert_handler(self, caplog):
        """Test log alert handler"""
        alert = Alert(
            rule_name="test_log",
            metric_name="test.metric",
            current_value=95.0,
            threshold=80.0,
            condition="greater_than",
            severity="critical",
            description="Test log alert",
            started_at=datetime.now()
        )
        
        log_alert_handler(alert)
        
        assert "test_log" in caplog.text
        assert "Test log alert" in caplog.text
        assert "95.0" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])