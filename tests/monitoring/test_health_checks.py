"""
Tests for Health Check System.

This module tests the health monitoring capabilities including
system health checks, dependency monitoring, and status reporting.
"""

import pytest
import asyncio
import tempfile
import os
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.monitoring.health_checks import (
    HealthCheckManager,
    HealthCheckRegistry,
    HealthCheckResult,
    DependencyStatus,
    SystemMetrics,
    HealthStatus,
    CheckPriority,
    health_check,
    get_health_manager,
    get_system_health,
    run_health_checks,
    add_dependency_check
)


class TestHealthStatus:
    """Test health status enumeration"""
    
    def test_health_status_values(self):
        """Test health status enum values"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestCheckPriority:
    """Test check priority enumeration"""
    
    def test_check_priority_values(self):
        """Test check priority enum values"""
        assert CheckPriority.CRITICAL.value == "critical"
        assert CheckPriority.HIGH.value == "high"
        assert CheckPriority.MEDIUM.value == "medium"
        assert CheckPriority.LOW.value == "low"


class TestHealthCheckResult:
    """Test health check result data structure"""
    
    def test_result_creation(self):
        """Test health check result creation"""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All systems operational",
            timestamp=datetime.now(),
            duration_ms=150.5,
            priority=CheckPriority.HIGH,
            details={"metric1": "value1"},
            remediation_steps=["Step 1", "Step 2"]
        )
        
        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All systems operational"
        assert result.duration_ms == 150.5
        assert result.priority == CheckPriority.HIGH
        assert result.details["metric1"] == "value1"
        assert len(result.remediation_steps) == 2
        assert result.error is None


class TestDependencyStatus:
    """Test dependency status data structure"""
    
    def test_dependency_status_creation(self):
        """Test dependency status creation"""
        status = DependencyStatus(
            name="api_service",
            url="http://api.example.com",
            status=HealthStatus.HEALTHY,
            response_time_ms=75.2,
            version="1.2.3",
            error_count=0,
            consecutive_failures=0
        )
        
        assert status.name == "api_service"
        assert status.url == "http://api.example.com"
        assert status.status == HealthStatus.HEALTHY
        assert status.response_time_ms == 75.2
        assert status.version == "1.2.3"
        assert status.error_count == 0
        assert status.consecutive_failures == 0


class TestSystemMetrics:
    """Test system metrics data structure"""
    
    def test_system_metrics_creation(self):
        """Test system metrics creation"""
        metrics = SystemMetrics(
            cpu_percent=45.2,
            memory_percent=67.8,
            disk_percent=89.1,
            disk_usage_gb=234.5,
            available_memory_gb=4.2,
            load_average=1.5,
            uptime_seconds=123456.7,
            process_count=145,
            open_files=23,
            network_connections=15
        )
        
        assert metrics.cpu_percent == 45.2
        assert metrics.memory_percent == 67.8
        assert metrics.disk_percent == 89.1
        assert metrics.disk_usage_gb == 234.5
        assert metrics.available_memory_gb == 4.2
        assert metrics.load_average == 1.5
        assert metrics.uptime_seconds == 123456.7
        assert metrics.process_count == 145
        assert metrics.open_files == 23
        assert metrics.network_connections == 15


class TestHealthCheckRegistry:
    """Test health check registry functionality"""
    
    @pytest.fixture
    def registry(self):
        """Create health check registry for testing"""
        return HealthCheckRegistry()
    
    def test_register_health_check(self, registry):
        """Test registering a health check"""
        def test_check():
            return HealthCheckResult(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.MEDIUM
            )
        
        registry.register(
            "test_check",
            test_check,
            CheckPriority.HIGH,
            timeout_seconds=10.0,
            interval_seconds=30.0
        )
        
        checks = registry.get_checks()
        assert "test_check" in checks
        assert checks["test_check"]["function"] == test_check
        assert checks["test_check"]["priority"] == CheckPriority.HIGH
        assert checks["test_check"]["timeout"] == 10.0
        assert checks["test_check"]["interval"] == 30.0
        assert checks["test_check"]["enabled"] is True
    
    def test_unregister_health_check(self, registry):
        """Test unregistering a health check"""
        def test_check():
            return HealthCheckResult(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.MEDIUM
            )
        
        registry.register("test_check", test_check)
        assert "test_check" in registry.get_checks()
        
        registry.unregister("test_check")
        assert "test_check" not in registry.get_checks()
    
    def test_enable_disable_check(self, registry):
        """Test enabling and disabling health checks"""
        def test_check():
            return HealthCheckResult(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.MEDIUM
            )
        
        registry.register("test_check", test_check)
        
        # Disable check
        registry.disable_check("test_check")
        checks = registry.get_checks()
        assert "test_check" not in checks
        
        # Enable check
        registry.enable_check("test_check")
        checks = registry.get_checks()
        assert "test_check" in checks
    
    def test_filter_by_priority(self, registry):
        """Test filtering checks by priority"""
        def high_check():
            return HealthCheckResult(
                name="high",
                status=HealthStatus.HEALTHY,
                message="OK",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.HIGH
            )
        
        def medium_check():
            return HealthCheckResult(
                name="medium",
                status=HealthStatus.HEALTHY,
                message="OK",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.MEDIUM
            )
        
        registry.register("high_check", high_check, CheckPriority.HIGH)
        registry.register("medium_check", medium_check, CheckPriority.MEDIUM)
        
        high_checks = registry.get_checks(CheckPriority.HIGH)
        assert len(high_checks) == 1
        assert "high_check" in high_checks
        
        medium_checks = registry.get_checks(CheckPriority.MEDIUM)
        assert len(medium_checks) == 1
        assert "medium_check" in medium_checks


class TestHealthCheckManager:
    """Test health check manager functionality"""
    
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
    def manager(self, temp_db):
        """Create health check manager for testing"""
        return HealthCheckManager(db_path=temp_db, check_interval=0.1)
    
    def test_manager_initialization(self, manager):
        """Test health check manager initialization"""
        assert manager.check_interval == 0.1
        assert isinstance(manager.registry, HealthCheckRegistry)
        assert len(manager.dependencies) == 0
        assert len(manager.recent_results) == 0
        assert manager._running is False
    
    @pytest.mark.asyncio
    async def test_single_health_check_execution(self, manager):
        """Test executing a single health check"""
        check_called = False
        
        async def test_check():
            nonlocal check_called
            check_called = True
            return HealthCheckResult(
                name="test_check",
                status=HealthStatus.HEALTHY,
                message="Test passed",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.MEDIUM
            )
        
        manager.registry.register("test_check", test_check)
        
        results = await manager.run_all_checks()
        
        assert check_called is True
        assert "test_check" in results
        assert results["test_check"].status == HealthStatus.HEALTHY
        assert results["test_check"].message == "Test passed"
        # Should have at least our test result plus default checks
        assert len(manager.recent_results) >= 1
        # Check that our specific test result is in the recent results
        test_results = [r for r in manager.recent_results if r.name == "test_check"]
        assert len(test_results) == 1
        assert test_results[0].status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_multiple_health_checks(self, manager):
        """Test executing multiple health checks concurrently"""
        check1_called = False
        check2_called = False
        
        async def check1():
            nonlocal check1_called
            check1_called = True
            await asyncio.sleep(0.01)
            return HealthCheckResult(
                name="check1",
                status=HealthStatus.HEALTHY,
                message="Check 1 OK",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.HIGH
            )
        
        async def check2():
            nonlocal check2_called
            check2_called = True
            await asyncio.sleep(0.01)
            return HealthCheckResult(
                name="check2",
                status=HealthStatus.DEGRADED,
                message="Check 2 degraded",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.MEDIUM
            )
        
        manager.registry.register("check1", check1, CheckPriority.HIGH)
        manager.registry.register("check2", check2, CheckPriority.MEDIUM)
        
        results = await manager.run_all_checks()
        
        assert check1_called is True
        assert check2_called is True
        assert len(results) == 2
        assert results["check1"].status == HealthStatus.HEALTHY
        assert results["check2"].status == HealthStatus.DEGRADED
        assert len(manager.recent_results) == 2
    
    @pytest.mark.asyncio
    async def test_health_check_timeout(self, manager):
        """Test health check timeout handling"""
        async def slow_check():
            await asyncio.sleep(1.0)  # Longer than timeout
            return HealthCheckResult(
                name="slow_check",
                status=HealthStatus.HEALTHY,
                message="Should not reach here",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.MEDIUM
            )
        
        manager.registry.register("slow_check", slow_check, timeout_seconds=0.1)
        
        results = await manager.run_all_checks()
        
        assert "slow_check" in results
        assert results["slow_check"].status == HealthStatus.UNHEALTHY
        assert "timed out" in results["slow_check"].message.lower()
        assert results["slow_check"].error == "TimeoutError"
    
    @pytest.mark.asyncio
    async def test_health_check_exception_handling(self, manager):
        """Test health check exception handling"""
        async def failing_check():
            raise ValueError("Test error")
        
        manager.registry.register("failing_check", failing_check)
        
        results = await manager.run_all_checks()
        
        assert "failing_check" in results
        assert results["failing_check"].status == HealthStatus.UNHEALTHY
        assert "Test error" in results["failing_check"].message
        assert results["failing_check"].error == "Test error"
    
    def test_overall_health_calculation(self, manager):
        """Test overall health status calculation"""
        # No results initially
        health = manager.get_overall_health()
        assert health["status"] == HealthStatus.UNKNOWN.value
        
        # Add some results
        manager.recent_results = [
            HealthCheckResult(
                name="critical_check",
                status=HealthStatus.HEALTHY,
                message="OK",
                timestamp=datetime.now(),
                duration_ms=10.0,
                priority=CheckPriority.CRITICAL
            ),
            HealthCheckResult(
                name="high_check",
                status=HealthStatus.DEGRADED,
                message="Degraded",
                timestamp=datetime.now(),
                duration_ms=20.0,
                priority=CheckPriority.HIGH
            ),
            HealthCheckResult(
                name="medium_check",
                status=HealthStatus.HEALTHY,
                message="OK",
                timestamp=datetime.now(),
                duration_ms=15.0,
                priority=CheckPriority.MEDIUM
            )
        ]
        
        health = manager.get_overall_health()
        assert health["status"] == HealthStatus.DEGRADED.value
        assert health["summary"]["total_checks"] == 3
        assert health["summary"]["healthy"] == 2
        assert health["summary"]["degraded"] == 1
        assert health["summary"]["unhealthy"] == 0
    
    def test_overall_health_critical_failure(self, manager):
        """Test overall health with critical system failure"""
        manager.recent_results = [
            HealthCheckResult(
                name="critical_check",
                status=HealthStatus.UNHEALTHY,
                message="Critical failure",
                timestamp=datetime.now(),
                duration_ms=10.0,
                priority=CheckPriority.CRITICAL
            ),
            HealthCheckResult(
                name="medium_check",
                status=HealthStatus.HEALTHY,
                message="OK",
                timestamp=datetime.now(),
                duration_ms=15.0,
                priority=CheckPriority.MEDIUM
            )
        ]
        
        health = manager.get_overall_health()
        assert health["status"] == HealthStatus.UNHEALTHY.value
        assert "critical system" in health["message"].lower()
    
    @pytest.mark.asyncio
    async def test_monitoring_start_stop(self, manager):
        """Test starting and stopping continuous monitoring"""
        assert manager._running is False
        
        await manager.start_monitoring()
        assert manager._running is True
        assert manager._background_task is not None
        
        # Let it run briefly
        await asyncio.sleep(0.05)
        
        await manager.stop_monitoring()
        assert manager._running is False
    
    def test_health_history_retrieval(self, manager):
        """Test retrieving health check history"""
        # Add some test data to database
        now = datetime.now()
        test_results = [
            HealthCheckResult(
                name="test_check",
                status=HealthStatus.HEALTHY,
                message="OK 1",
                timestamp=now - timedelta(minutes=5),
                duration_ms=10.0,
                priority=CheckPriority.MEDIUM
            ),
            HealthCheckResult(
                name="test_check",
                status=HealthStatus.DEGRADED,
                message="Warning",
                timestamp=now - timedelta(minutes=3),
                duration_ms=15.0,
                priority=CheckPriority.MEDIUM
            ),
            HealthCheckResult(
                name="other_check",
                status=HealthStatus.HEALTHY,
                message="OK 2",
                timestamp=now - timedelta(minutes=1),
                duration_ms=8.0,
                priority=CheckPriority.HIGH
            )
        ]
        
        for result in test_results:
            manager._store_result(result)
        
        # Get all history
        history = manager.get_health_history()
        assert len(history) >= 3
        
        # Get history for specific check
        test_history = manager.get_health_history(check_name="test_check")
        assert len(test_history) == 2
        assert all(h["name"] == "test_check" for h in test_history)
        
        # Get recent history
        recent_history = manager.get_health_history(since=now - timedelta(minutes=2))
        assert len(recent_history) >= 2
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.boot_time')
    @patch('psutil.pids')
    def test_system_metrics_collection(self, mock_pids, mock_boot_time, 
                                      mock_disk_usage, mock_memory, mock_cpu, manager):
        """Test system metrics collection"""
        # Mock psutil responses
        mock_cpu.return_value = 45.2
        
        mock_memory_obj = MagicMock()
        mock_memory_obj.percent = 67.8
        mock_memory_obj.available = 4.2 * (1024**3)
        mock_memory.return_value = mock_memory_obj
        
        mock_disk_obj = MagicMock()
        mock_disk_obj.total = 1000 * (1024**3)
        mock_disk_obj.used = 890 * (1024**3)
        mock_disk_usage.return_value = mock_disk_obj
        
        mock_boot_time.return_value = time.time() - 3600  # 1 hour ago
        mock_pids.return_value = list(range(150))  # 150 processes
        
        metrics = manager._get_system_metrics()
        
        assert metrics.cpu_percent == 45.2
        assert metrics.memory_percent == 67.8
        assert abs(metrics.available_memory_gb - 4.2) < 0.1
        assert metrics.disk_percent == 89.0
        assert 3500 < metrics.uptime_seconds < 3700  # Around 1 hour
        assert metrics.process_count == 150
    
    @pytest.mark.asyncio
    async def test_system_resources_check(self, manager):
        """Test system resources health check"""
        with patch.object(manager, '_get_system_metrics') as mock_metrics:
            # Mock healthy system
            mock_metrics.return_value = SystemMetrics(
                cpu_percent=50.0,
                memory_percent=70.0,
                disk_percent=80.0,
                disk_usage_gb=200.0,
                available_memory_gb=2.0,
                load_average=1.0,
                uptime_seconds=3600.0,
                process_count=100,
                open_files=50,
                network_connections=20
            )
            
            result = await manager._check_system_resources()
            
            assert result.name == "system_resources"
            assert result.status == HealthStatus.HEALTHY
            assert "healthy" in result.message.lower()
            assert result.details["cpu_percent"] == 50.0
            assert result.details["memory_percent"] == 70.0
    
    @pytest.mark.asyncio
    async def test_system_resources_check_critical(self, manager):
        """Test system resources health check with critical usage"""
        with patch.object(manager, '_get_system_metrics') as mock_metrics:
            # Mock overloaded system
            mock_metrics.return_value = SystemMetrics(
                cpu_percent=95.0,  # Critical
                memory_percent=98.0,  # Critical
                disk_percent=97.0,  # Critical
                disk_usage_gb=950.0,
                available_memory_gb=0.1,
                load_average=8.0,
                uptime_seconds=3600.0,
                process_count=500,
                open_files=1000,
                network_connections=200
            )
            
            result = await manager._check_system_resources()
            
            assert result.name == "system_resources"
            assert result.status == HealthStatus.UNHEALTHY
            assert "critical" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_database_connection_check(self, manager):
        """Test database connection health check"""
        result = await manager._check_database_connection()
        
        assert result.name == "database_connection"
        # Should be healthy since we create a valid database
        assert result.status == HealthStatus.HEALTHY
        assert "connected" in result.message.lower()
        assert result.details["database_type"] == "SQLite"
    
    @pytest.mark.asyncio
    async def test_file_system_check(self, manager):
        """Test file system health check"""
        result = await manager._check_file_system()
        
        assert result.name == "file_system"
        # Should be healthy in test environment
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert result.details["checked_directories"] is not None


class TestHealthCheckDecorator:
    """Test health check decorator functionality"""
    
    def test_health_check_decorator(self):
        """Test health check decorator registration"""
        # Clear existing registrations
        manager = get_health_manager()
        if "decorated_check" in manager.registry.checks:
            manager.registry.unregister("decorated_check")
        
        @health_check("decorated_check", CheckPriority.HIGH, timeout_seconds=20.0)
        async def decorated_check():
            return HealthCheckResult(
                name="decorated_check",
                status=HealthStatus.HEALTHY,
                message="Decorator test",
                timestamp=datetime.now(),
                duration_ms=0.0,
                priority=CheckPriority.HIGH
            )
        
        # Should be registered
        checks = manager.registry.get_checks()
        assert "decorated_check" in checks
        assert checks["decorated_check"]["priority"] == CheckPriority.HIGH
        assert checks["decorated_check"]["timeout"] == 20.0


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.mark.asyncio
    async def test_get_system_health(self):
        """Test get_system_health convenience function"""
        health = await get_system_health()
        
        assert "status" in health
        assert "message" in health
        assert "timestamp" in health
        assert "checks" in health
        assert "summary" in health
    
    @pytest.mark.asyncio
    async def test_run_health_checks(self):
        """Test run_health_checks convenience function"""
        results = await run_health_checks()
        
        assert isinstance(results, dict)
        # Should have at least the default checks
        assert len(results) > 0
    
    def test_add_dependency_check(self):
        """Test add_dependency_check convenience function"""
        manager = get_health_manager()
        
        # Clear existing dependency checks
        existing_deps = [name for name in manager.registry.checks.keys() 
                        if name.startswith("dependency_")]
        for dep in existing_deps:
            manager.registry.unregister(dep)
        
        add_dependency_check("test_api", "http://api.example.com/health", timeout=10.0)
        
        checks = manager.registry.get_checks()
        assert "dependency_test_api" in checks
        assert checks["dependency_test_api"]["priority"] == CheckPriority.HIGH


class TestHealthCheckIntegration:
    """Test integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_health_monitoring_cycle(self):
        """Test complete health monitoring cycle"""
        # Create temporary manager for isolation
        fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        try:
            manager = HealthCheckManager(db_path=db_path, check_interval=0.1)
            
            # Add custom check
            check_called = False
            
            async def custom_check():
                nonlocal check_called
                check_called = True
                return HealthCheckResult(
                    name="custom_check",
                    status=HealthStatus.HEALTHY,
                    message="Custom check OK",
                    timestamp=datetime.now(),
                    duration_ms=5.0,
                    priority=CheckPriority.MEDIUM
                )
            
            manager.registry.register("custom_check", custom_check)
            
            # Start monitoring
            await manager.start_monitoring()
            
            # Wait for a few cycles
            await asyncio.sleep(0.25)
            
            # Stop monitoring
            await manager.stop_monitoring()
            
            # Verify results
            assert check_called is True
            assert len(manager.recent_results) > 0
            
            # Check overall health
            health = manager.get_overall_health()
            assert health["status"] in [
                HealthStatus.HEALTHY.value, 
                HealthStatus.DEGRADED.value
            ]
            assert health["summary"]["total_checks"] > 0
        
        finally:
            try:
                os.unlink(db_path)
            except FileNotFoundError:
                pass
    
    @pytest.mark.asyncio
    async def test_high_load_health_checks(self):
        """Test health checks under high load"""
        fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        try:
            manager = HealthCheckManager(db_path=db_path)
            
            # Register many quick checks
            for i in range(20):
                async def quick_check(check_id=i):
                    await asyncio.sleep(0.001)  # Very fast check
                    return HealthCheckResult(
                        name=f"quick_check_{check_id}",
                        status=HealthStatus.HEALTHY,
                        message=f"Quick check {check_id} OK",
                        timestamp=datetime.now(),
                        duration_ms=1.0,
                        priority=CheckPriority.LOW
                    )
                
                manager.registry.register(f"quick_check_{i}", quick_check, CheckPriority.LOW)
            
            # Run all checks
            start_time = time.time()
            results = await manager.run_all_checks()
            end_time = time.time()
            
            # Should complete quickly and successfully
            assert len(results) == 20
            assert (end_time - start_time) < 2.0  # Should be much faster
            assert all(r.status == HealthStatus.HEALTHY for r in results.values())
        
        finally:
            try:
                os.unlink(db_path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])