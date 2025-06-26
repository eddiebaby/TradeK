"""
Comprehensive tests for monitoring and observability system
Tests metrics collection, alerting, health monitoring, and notification systems
"""

import pytest
import asyncio
import tempfile
import os
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

from src.core.monitoring_service import (
    MonitoringService, MetricsCollector, AlertManager, HealthMonitor,
    MetricPoint, Alert, HealthCheck, monitoring_service
)
from src.monitoring.alerts import (
    AlertNotificationManager, EmailNotificationHandler, WebhookNotificationHandler,
    SlackNotificationHandler, FileLogNotificationHandler, default_alert_handler
)
from src.api.middleware.monitoring_middleware import (
    MonitoringMiddleware, add_monitoring_headers, request_size_monitor
)


class TestMetricsCollector:
    """Test metrics collection functionality"""
    
    @pytest.fixture
    def collector(self):
        """Create test metrics collector"""
        return MetricsCollector(retention_minutes=5)
    
    def test_record_metric(self, collector):
        """Test recording metrics"""
        collector.record_metric("test_metric", 42.5)
        
        assert "test_metric" in collector.metrics
        assert len(collector.metrics["test_metric"]) == 1
        
        point = collector.metrics["test_metric"][0]
        assert point.value == 42.5
        assert isinstance(point.timestamp, datetime)
    
    def test_record_metric_with_labels(self, collector):
        """Test recording metrics with labels"""
        labels = {"service": "api", "endpoint": "/search"}
        collector.record_metric("response_time", 150.0, labels)
        
        point = collector.metrics["response_time"][0]
        assert point.labels == labels
    
    def test_metric_retention(self, collector):
        """Test metric retention limits"""
        # Set very small retention for testing
        collector.metrics["test_metric"].maxlen = 3
        
        # Add more metrics than retention limit
        for i in range(5):
            collector.record_metric("test_metric", float(i))
        
        # Should only keep the last 3
        assert len(collector.metrics["test_metric"]) == 3
        values = [point.value for point in collector.metrics["test_metric"]]
        assert values == [2.0, 3.0, 4.0]
    
    def test_get_metric_history(self, collector):
        """Test retrieving metric history"""
        # Record metrics with different timestamps
        now = datetime.utcnow()
        
        # Mock timestamps
        with patch('src.core.monitoring_service.datetime') as mock_dt:
            mock_dt.utcnow.side_effect = [
                now - timedelta(minutes=10),
                now - timedelta(minutes=5),
                now - timedelta(minutes=1),
                now
            ]
            
            collector.record_metric("test_metric", 1.0)
            collector.record_metric("test_metric", 2.0)
            collector.record_metric("test_metric", 3.0)
            
            # Reset to current time for the query
            mock_dt.utcnow.return_value = now
            
            # Get last 2 minutes - should only include the last metric
            recent_history = collector.get_metric_history("test_metric", minutes=2)
            assert len(recent_history) == 1
            assert recent_history[0].value == 3.0
    
    def test_get_metric_summary(self, collector):
        """Test metric summary statistics"""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            collector.record_metric("test_metric", value)
        
        summary = collector.get_metric_summary("test_metric", minutes=60)
        
        assert summary["current"] == 50.0
        assert summary["avg"] == 30.0
        assert summary["min"] == 10.0
        assert summary["max"] == 50.0
        assert summary["count"] == 5
    
    def test_get_summary_empty_metric(self, collector):
        """Test summary for non-existent metric"""
        summary = collector.get_metric_summary("nonexistent", minutes=5)
        assert summary == {}


class TestAlertManager:
    """Test alert management functionality"""
    
    @pytest.fixture
    def alert_manager(self):
        """Create test alert manager"""
        return AlertManager()
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample alert"""
        return Alert(
            id="test_alert",
            name="Test Alert",
            description="Test alert description",
            severity="warning",
            condition="cpu_usage_percent greater_than",
            threshold=80.0
        )
    
    def test_register_alert(self, alert_manager, sample_alert):
        """Test alert registration"""
        alert_manager.register_alert(sample_alert)
        
        assert "test_alert" in alert_manager.alerts
        assert alert_manager.alerts["test_alert"] == sample_alert
    
    def test_register_handler(self, alert_manager):
        """Test alert handler registration"""
        mock_handler = Mock()
        alert_manager.register_handler(mock_handler)
        
        assert mock_handler in alert_manager.alert_handlers
    
    @pytest.mark.asyncio
    async def test_evaluate_alerts_trigger(self, alert_manager, sample_alert):
        """Test alert triggering"""
        # Setup metrics collector with high CPU usage
        collector = MetricsCollector()
        collector.record_metric("cpu_usage_percent", 90.0)  # Above threshold
        
        alert_manager.set_metrics_collector(collector)
        alert_manager.register_alert(sample_alert)
        
        # Mock handler
        mock_handler = AsyncMock()
        alert_manager.register_handler(mock_handler)
        
        await alert_manager.evaluate_alerts()
        
        # Alert should be triggered
        assert sample_alert.status == "triggered"
        assert sample_alert.triggered_at is not None
        mock_handler.assert_called_once_with(sample_alert)
    
    @pytest.mark.asyncio
    async def test_evaluate_alerts_resolve(self, alert_manager, sample_alert):
        """Test alert resolution"""
        # Setup triggered alert
        sample_alert.status = "triggered"
        sample_alert.triggered_at = datetime.utcnow()
        
        # Setup metrics collector with normal CPU usage
        collector = MetricsCollector()
        collector.record_metric("cpu_usage_percent", 50.0)  # Below threshold
        
        alert_manager.set_metrics_collector(collector)
        alert_manager.register_alert(sample_alert)
        
        await alert_manager.evaluate_alerts()
        
        # Alert should be resolved
        assert sample_alert.status == "resolved"
        assert sample_alert.resolved_at is not None


class TestHealthMonitor:
    """Test health monitoring functionality"""
    
    @pytest.fixture
    def health_monitor(self):
        """Create test health monitor"""
        return HealthMonitor()
    
    @pytest.mark.asyncio
    async def test_register_health_check(self, health_monitor):
        """Test health check registration"""
        async def mock_check():
            return {"status": "healthy"}
        
        health_monitor.register_health_check("test_component", mock_check)
        
        assert "test_component" in health_monitor.check_functions
    
    @pytest.mark.asyncio
    async def test_run_health_checks_success(self, health_monitor):
        """Test successful health checks"""
        async def healthy_check():
            return {
                "status": "healthy",
                "metadata": {"version": "1.0.0"}
            }
        
        health_monitor.register_health_check("test_service", healthy_check)
        
        results = await health_monitor.run_health_checks()
        
        assert "test_service" in results
        check = results["test_service"]
        assert check.status == "healthy"
        assert check.response_time_ms > 0
        assert check.metadata["version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_run_health_checks_failure(self, health_monitor):
        """Test failed health checks"""
        async def failing_check():
            raise Exception("Service unavailable")
        
        health_monitor.register_health_check("failing_service", failing_check)
        
        results = await health_monitor.run_health_checks()
        
        assert "failing_service" in results
        check = results["failing_service"]
        assert check.status == "unhealthy"
        assert "Service unavailable" in check.error_message
    
    def test_get_overall_health_healthy(self, health_monitor):
        """Test overall health when all components are healthy"""
        health_monitor.health_checks = {
            "service1": HealthCheck("service1", "healthy", datetime.utcnow(), 10.0),
            "service2": HealthCheck("service2", "healthy", datetime.utcnow(), 15.0)
        }
        
        assert health_monitor.get_overall_health() == "healthy"
    
    def test_get_overall_health_degraded(self, health_monitor):
        """Test overall health when some components are degraded"""
        health_monitor.health_checks = {
            "service1": HealthCheck("service1", "healthy", datetime.utcnow(), 10.0),
            "service2": HealthCheck("service2", "degraded", datetime.utcnow(), 15.0)
        }
        
        assert health_monitor.get_overall_health() == "degraded"
    
    def test_get_overall_health_unhealthy(self, health_monitor):
        """Test overall health when any component is unhealthy"""
        health_monitor.health_checks = {
            "service1": HealthCheck("service1", "healthy", datetime.utcnow(), 10.0),
            "service2": HealthCheck("service2", "unhealthy", datetime.utcnow(), 15.0)
        }
        
        assert health_monitor.get_overall_health() == "unhealthy"


class TestMonitoringService:
    """Test complete monitoring service"""
    
    @pytest.fixture
    async def monitoring_service(self):
        """Create test monitoring service"""
        service = MonitoringService()
        # Don't initialize to avoid starting background tasks
        service.metrics_collector = MetricsCollector()
        service.alert_manager = AlertManager()
        service.health_monitor = HealthMonitor()
        return service
    
    @pytest.mark.asyncio
    async def test_collect_system_metrics(self, monitoring_service):
        """Test system metrics collection"""
        await monitoring_service._collect_system_metrics()
        
        # Check that system metrics were recorded
        metrics = monitoring_service.metrics_collector.metrics
        assert "cpu_usage_percent" in metrics
        assert "memory_usage_percent" in metrics
        assert "disk_usage_percent" in metrics
    
    def test_record_request(self, monitoring_service):
        """Test API request recording"""
        monitoring_service.record_request(150.0, 200)
        
        # Check request metrics were recorded
        assert len(monitoring_service.request_times) == 1
        assert monitoring_service.request_times[0] == 150.0
        
        # Check metrics were recorded
        metrics = monitoring_service.metrics_collector.metrics
        assert "response_time_ms" in metrics
        assert "requests_total" in metrics
    
    def test_record_request_error(self, monitoring_service):
        """Test API error request recording"""
        monitoring_service.record_request(200.0, 500)
        
        # Check error tracking
        assert monitoring_service.error_counts[500] == 1
        
        # Check error metrics were recorded
        metrics = monitoring_service.metrics_collector.metrics
        assert "errors_total" in metrics
        assert "error_rate_percent" in metrics
    
    def test_get_metrics_summary(self, monitoring_service):
        """Test metrics summary generation"""
        # Record some test metrics
        monitoring_service.metrics_collector.record_metric("cpu_usage_percent", 45.0)
        monitoring_service.metrics_collector.record_metric("memory_usage_percent", 60.0)
        
        summary = monitoring_service.get_metrics_summary()
        
        assert "cpu_usage_percent" in summary
        assert "memory_usage_percent" in summary
        assert summary["cpu_usage_percent"]["current"] == 45.0
    
    def test_get_dashboard_data(self, monitoring_service):
        """Test dashboard data generation"""
        # Setup some test data
        monitoring_service.health_monitor.health_checks = {
            "test_service": HealthCheck("test_service", "healthy", datetime.utcnow(), 10.0)
        }
        
        dashboard = monitoring_service.get_dashboard_data()
        
        assert "system_health" in dashboard
        assert "health_checks" in dashboard
        assert "metrics_summary" in dashboard
        assert "system_info" in dashboard
        assert "generated_at" in dashboard


class TestEmailNotificationHandler:
    """Test email notification handler"""
    
    @pytest.fixture
    def email_handler(self):
        """Create test email handler"""
        config = {
            "host": "localhost",
            "port": 587,
            "username": "test@example.com",
            "password": "password",
            "from_email": "alerts@tradeknowledge.com",
            "to_emails": ["admin@example.com"],
            "use_tls": True
        }
        return EmailNotificationHandler(config)
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample alert"""
        alert = Alert(
            id="test_alert",
            name="High CPU Usage",
            description="CPU usage is above threshold",
            severity="warning",
            condition="cpu_usage_percent greater_than",
            threshold=80.0
        )
        alert.triggered_at = datetime.utcnow()
        return alert
    
    def test_create_alert_email_html(self, email_handler, sample_alert):
        """Test HTML email content creation"""
        html_content = email_handler._create_alert_email_html(sample_alert)
        
        assert "High CPU Usage" in html_content
        assert "warning" in html_content.lower()
        assert "CPU usage is above threshold" in html_content
        assert str(sample_alert.threshold) in html_content
    
    def test_create_alert_email_text(self, email_handler, sample_alert):
        """Test plain text email content creation"""
        text_content = email_handler._create_alert_email_text(sample_alert)
        
        assert "High CPU Usage" in text_content
        assert "WARNING" in text_content
        assert "CPU usage is above threshold" in text_content
        assert str(sample_alert.threshold) in text_content


class TestWebhookNotificationHandler:
    """Test webhook notification handler"""
    
    @pytest.fixture
    def webhook_handler(self):
        """Create test webhook handler"""
        config = {
            "url": "https://webhook.example.com/alerts",
            "headers": {"Authorization": "Bearer token123"},
            "timeout": 30
        }
        return WebhookNotificationHandler(config)
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample alert"""
        alert = Alert(
            id="test_alert",
            name="High Memory Usage",
            description="Memory usage is above threshold",
            severity="critical",
            condition="memory_usage_percent greater_than",
            threshold=90.0
        )
        alert.triggered_at = datetime.utcnow()
        return alert
    
    @pytest.mark.asyncio
    async def test_send_webhook_notification(self, webhook_handler, sample_alert):
        """Test webhook notification sending"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            await webhook_handler.send_alert_notification(sample_alert)
            
            # Verify webhook was called
            mock_session.return_value.__aenter__.return_value.post.assert_called_once()
            
            # Check the payload structure
            call_args = mock_session.return_value.__aenter__.return_value.post.call_args
            payload = call_args.kwargs['json']
            
            assert payload['alert_id'] == 'test_alert'
            assert payload['name'] == 'High Memory Usage'
            assert payload['severity'] == 'critical'


class TestFileLogNotificationHandler:
    """Test file logging notification handler"""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file"""
        fd, path = tempfile.mkstemp(suffix='.log')
        os.close(fd)
        yield path
        os.unlink(path)
    
    @pytest.fixture
    def file_handler(self, temp_log_file):
        """Create test file handler"""
        config = {
            "file_path": temp_log_file,
            "max_file_size_mb": 1,
            "backup_count": 3
        }
        return FileLogNotificationHandler(config)
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample alert"""
        alert = Alert(
            id="test_alert",
            name="Disk Space Low",
            description="Available disk space is below threshold",
            severity="warning",
            condition="disk_usage_percent greater_than",
            threshold=85.0
        )
        alert.triggered_at = datetime.utcnow()
        return alert
    
    @pytest.mark.asyncio
    async def test_log_alert_to_file(self, file_handler, sample_alert):
        """Test logging alert to file"""
        await file_handler.send_alert_notification(sample_alert)
        
        # Verify log file was created and contains alert
        assert file_handler.log_file.exists()
        
        with open(file_handler.log_file, 'r') as f:
            content = f.read()
            assert 'test_alert' in content
            assert 'Disk Space Low' in content
            assert 'warning' in content


class TestAlertNotificationManager:
    """Test alert notification manager"""
    
    @pytest.fixture
    async def notification_manager(self):
        """Create test notification manager"""
        manager = AlertNotificationManager()
        # Mock configuration to avoid dependencies
        manager.config = Mock()
        return manager
    
    @pytest.mark.asyncio
    async def test_add_custom_handler(self, notification_manager):
        """Test adding custom handler"""
        mock_handler = Mock()
        mock_handler.send_alert_notification = AsyncMock()
        
        notification_manager.add_custom_handler(mock_handler)
        
        assert mock_handler in notification_manager.handlers
    
    @pytest.mark.asyncio
    async def test_handle_alert(self, notification_manager):
        """Test handling alert with multiple handlers"""
        # Add mock handlers
        handler1 = Mock()
        handler1.send_alert_notification = AsyncMock()
        handler2 = Mock()
        handler2.send_alert_notification = AsyncMock()
        
        notification_manager.handlers = [handler1, handler2]
        
        alert = Alert(
            id="test_alert",
            name="Test Alert",
            description="Test description",
            severity="info",
            condition="test_condition",
            threshold=50.0
        )
        
        await notification_manager.handle_alert(alert)
        
        # Both handlers should be called
        handler1.send_alert_notification.assert_called_once_with(alert)
        handler2.send_alert_notification.assert_called_once_with(alert)


class TestMonitoringMiddleware:
    """Test monitoring middleware"""
    
    @pytest.fixture
    def mock_app(self):
        """Create mock ASGI app"""
        async def app(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": []
            })
            await send({
                "type": "http.response.body",
                "body": b"OK"
            })
        return app
    
    @pytest.fixture
    def monitoring_middleware(self, mock_app):
        """Create monitoring middleware"""
        return MonitoringMiddleware(mock_app)
    
    @pytest.mark.asyncio
    async def test_middleware_tracks_request(self, monitoring_middleware):
        """Test that middleware tracks requests"""
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test"
        }
        
        async def receive():
            return {"type": "http.request"}
        
        messages = []
        async def send(message):
            messages.append(message)
        
        with patch('src.api.middleware.monitoring_middleware.get_monitoring_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service
            
            await monitoring_middleware(scope, receive, send)
            
            # Service should have recorded the request
            mock_service.record_request.assert_called_once()
            
            # Check response was sent
            assert len(messages) == 2
            assert messages[0]["type"] == "http.response.start"
            assert messages[1]["type"] == "http.response.body"


class TestIntegration:
    """Integration tests for the complete monitoring system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_flow(self):
        """Test complete monitoring workflow"""
        # Initialize monitoring service
        service = MonitoringService()
        service.metrics_collector = MetricsCollector()
        service.alert_manager = AlertManager()
        service.health_monitor = HealthMonitor()
        
        # Setup alert manager
        service.alert_manager.set_metrics_collector(service.metrics_collector)
        
        # Register a test alert
        alert = Alert(
            id="test_integration_alert",
            name="Integration Test Alert",
            description="Test alert for integration testing",
            severity="warning",
            condition="test_metric greater_than",
            threshold=50.0
        )
        service.alert_manager.register_alert(alert)
        
        # Register mock alert handler
        mock_handler = AsyncMock()
        service.alert_manager.register_handler(mock_handler)
        
        # Record metrics that should trigger the alert
        service.metrics_collector.record_metric("test_metric", 75.0)  # Above threshold
        
        # Evaluate alerts
        await service.alert_manager.evaluate_alerts()
        
        # Verify alert was triggered
        assert alert.status == "triggered"
        mock_handler.assert_called_once_with(alert)
        
        # Record metrics that should resolve the alert
        service.metrics_collector.record_metric("test_metric", 25.0)  # Below threshold
        
        # Evaluate alerts again
        await service.alert_manager.evaluate_alerts()
        
        # Verify alert was resolved
        assert alert.status == "resolved"
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test performance monitoring integration"""
        service = MonitoringService()
        service.metrics_collector = MetricsCollector()
        
        # Simulate API requests
        request_times = [50.0, 75.0, 100.0, 125.0, 200.0]
        status_codes = [200, 200, 200, 404, 500]
        
        for duration, status in zip(request_times, status_codes):
            service.record_request(duration, status)
        
        # Get metrics summary
        summary = service.get_metrics_summary()
        
        # Verify request metrics
        assert "response_time_ms" in summary
        assert "error_rate_percent" in summary
        
        response_time_summary = summary["response_time_ms"]
        assert response_time_summary["avg"] == 110.0  # Average of request times
        assert response_time_summary["max"] == 200.0
        
        # Verify error rate calculation
        error_rate_summary = summary["error_rate_percent"]
        assert error_rate_summary["current"] == 40.0  # 2 errors out of 5 requests


# Global test cleanup
@pytest.fixture(autouse=True)
async def cleanup_monitoring_state():
    """Clean up monitoring state between tests"""
    yield
    
    # Reset global monitoring service
    if hasattr(monitoring_service, 'metrics_collector'):
        monitoring_service.metrics_collector.metrics.clear()
    if hasattr(monitoring_service, 'alert_manager'):
        monitoring_service.alert_manager.alerts.clear()
        monitoring_service.alert_manager.alert_handlers.clear()
    if hasattr(monitoring_service, 'health_monitor'):
        monitoring_service.health_monitor.health_checks.clear()
        monitoring_service.health_monitor.check_functions.clear()