"""
Tests for Load Testing Infrastructure.

This module tests the load testing framework including:
- Load test configuration and execution
- Different load test types (constant, ramp-up, spike, stress)
- Result collection and analysis
- System metrics monitoring
- Export functionality
"""

import pytest
import asyncio
import aiohttp
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import json
import tempfile
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.testing.load_tester import (
    LoadTestRunner,
    LoadTestConfig,
    LoadTestType,
    LoadTestResult,
    LoadTestSummary,
    create_search_api_test_config,
    create_stress_test_config,
    run_basic_load_test
)


class MockResponse:
    """Mock HTTP response for testing"""
    
    def __init__(self, status=200, text="mock response", size=100):
        self.status = status
        self._text = text
        self._size = size
    
    async def text(self):
        return self._text
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestLoadTestRunner:
    """Test load test runner functionality"""
    
    @pytest.fixture
    def runner(self):
        """Create load test runner for testing"""
        return LoadTestRunner()
    
    @pytest.fixture
    def basic_config(self):
        """Create basic load test configuration"""
        return LoadTestConfig(
            test_name="test_load",
            test_type=LoadTestType.CONSTANT_LOAD,
            target_url="http://localhost:8000/api/test",
            duration_seconds=2,  # Short for testing
            concurrent_users=2,
            think_time_seconds=0.1,  # Very short for testing
            headers={"Content-Type": "application/json"}
        )
    
    def test_load_test_config_creation(self):
        """Test load test configuration creation"""
        config = LoadTestConfig(
            test_name="api_test",
            test_type=LoadTestType.RAMP_UP,
            target_url="http://example.com/api",
            duration_seconds=60,
            concurrent_users=10,
            ramp_up_seconds=30,
            think_time_seconds=2.0
        )
        
        assert config.test_name == "api_test"
        assert config.test_type == LoadTestType.RAMP_UP
        assert config.target_url == "http://example.com/api"
        assert config.duration_seconds == 60
        assert config.concurrent_users == 10
        assert config.ramp_up_seconds == 30
        assert config.think_time_seconds == 2.0
    
    def test_generate_test_request_data(self, runner, basic_config):
        """Test test request data generation"""
        data = runner._generate_test_request_data(basic_config)
        
        assert "query" in data
        assert "max_results" in data
        assert "intent" in data
        assert isinstance(data["query"], str)
        assert isinstance(data["max_results"], int)
        assert data["intent"] in ["research", "learning", "quick_lookup"]
    
    def test_generate_volume_test_data(self, runner):
        """Test volume test data generation"""
        data = runner._generate_volume_test_data()
        
        assert "query" in data
        assert "max_results" in data
        assert "intent" in data
        assert "filters" in data
        assert len(data["query"]) > 50  # Volume queries should be longer
        assert data["max_results"] >= 100  # Volume should request more results
    
    @pytest.mark.asyncio
    async def test_make_request_success(self, runner):
        """Test successful HTTP request"""
        # Mock session and response
        mock_session = AsyncMock()
        mock_response = MockResponse(status=200, text='{"result": "success"}')
        mock_session.post.return_value = mock_response
        
        result = await runner._make_request(
            mock_session,
            "http://test.com/api",
            {"Content-Type": "application/json"},
            {"query": "test"},
            "test_user"
        )
        
        assert isinstance(result, LoadTestResult)
        assert result.success is True
        assert result.status_code == 200
        assert result.response_time_ms > 0
        assert result.user_id == "test_user"
        assert result.error_message is None
    
    @pytest.mark.asyncio
    async def test_make_request_failure(self, runner):
        """Test HTTP request failure"""
        mock_session = AsyncMock()
        mock_response = MockResponse(status=500, text="Internal Server Error")
        mock_session.post.return_value = mock_response
        
        result = await runner._make_request(
            mock_session,
            "http://test.com/api",
            {"Content-Type": "application/json"},
            {"query": "test"},
            "test_user"
        )
        
        assert isinstance(result, LoadTestResult)
        assert result.success is False
        assert result.status_code == 500
        assert result.response_time_ms > 0
        assert result.user_id == "test_user"
    
    @pytest.mark.asyncio
    async def test_make_request_timeout(self, runner):
        """Test HTTP request timeout"""
        mock_session = AsyncMock()
        mock_session.post.side_effect = asyncio.TimeoutError()
        
        result = await runner._make_request(
            mock_session,
            "http://test.com/api",
            {"Content-Type": "application/json"},
            {"query": "test"},
            "test_user"
        )
        
        assert isinstance(result, LoadTestResult)
        assert result.success is False
        assert result.status_code == 0
        assert result.error_message == "Request timeout"
        assert result.user_id == "test_user"
    
    @pytest.mark.asyncio
    async def test_make_request_exception(self, runner):
        """Test HTTP request with general exception"""
        mock_session = AsyncMock()
        mock_session.post.side_effect = Exception("Connection error")
        
        result = await runner._make_request(
            mock_session,
            "http://test.com/api",
            {"Content-Type": "application/json"},
            {"query": "test"},
            "test_user"
        )
        
        assert isinstance(result, LoadTestResult)
        assert result.success is False
        assert result.status_code == 0
        assert "Connection error" in result.error_message
        assert result.user_id == "test_user"
    
    def test_percentile_calculation(self, runner):
        """Test percentile calculation"""
        data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        assert runner._percentile(data, 50) == 55.0  # Median
        assert runner._percentile(data, 90) == 91.0  # 90th percentile
        assert runner._percentile(data, 95) == 95.5  # 95th percentile
        assert runner._percentile([], 50) == 0       # Empty data
    
    @pytest.mark.asyncio
    async def test_simulate_user(self, runner, basic_config):
        """Test user simulation"""
        # Mock aiohttp session
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            # Mock successful responses
            mock_response = MockResponse(status=200, text='{"success": true}')
            mock_session.post.return_value = mock_response
            
            # Run user simulation for short duration
            runner.running = True
            
            # Start simulation task
            task = asyncio.create_task(
                runner._simulate_user(basic_config, "test_user")
            )
            
            # Let it run briefly
            await asyncio.sleep(0.3)
            
            # Stop simulation
            runner.running = False
            await task
            
            # Check that requests were made
            assert len(runner.results) > 0
            assert all(r.user_id == "test_user" for r in runner.results)
    
    @pytest.mark.asyncio
    async def test_constant_load_test(self, runner, basic_config):
        """Test constant load test execution"""
        # Mock HTTP calls
        with patch.object(runner, '_make_request') as mock_request:
            mock_result = LoadTestResult(
                timestamp=datetime.now(),
                status_code=200,
                response_time_ms=50.0,
                success=True,
                user_id="test_user"
            )
            mock_request.return_value = mock_result
            
            # Run short constant load test
            summary = await runner.run_load_test(basic_config)
            
            assert isinstance(summary, LoadTestSummary)
            assert summary.test_name == "test_load"
            assert summary.test_type == LoadTestType.CONSTANT_LOAD
            assert summary.total_requests > 0
            assert summary.success_rate >= 0.0
    
    @pytest.mark.asyncio
    async def test_ramp_up_test(self, runner):
        """Test ramp-up load test"""
        config = LoadTestConfig(
            test_name="ramp_test",
            test_type=LoadTestType.RAMP_UP,
            target_url="http://localhost:8000/api/test",
            duration_seconds=2,
            concurrent_users=4,
            ramp_up_seconds=1,
            think_time_seconds=0.1
        )
        
        # Mock HTTP calls
        with patch.object(runner, '_make_request') as mock_request:
            mock_result = LoadTestResult(
                timestamp=datetime.now(),
                status_code=200,
                response_time_ms=30.0,
                success=True,
                user_id="ramp_user"
            )
            mock_request.return_value = mock_result
            
            summary = await runner.run_load_test(config)
            
            assert summary.test_type == LoadTestType.RAMP_UP
            assert summary.total_requests > 0
    
    @pytest.mark.asyncio
    async def test_spike_test(self, runner):
        """Test spike load test"""
        config = LoadTestConfig(
            test_name="spike_test",
            test_type=LoadTestType.SPIKE,
            target_url="http://localhost:8000/api/test",
            duration_seconds=3,  # Short test
            concurrent_users=6,
            think_time_seconds=0.1
        )
        
        # Mock HTTP calls
        with patch.object(runner, '_make_request') as mock_request:
            mock_result = LoadTestResult(
                timestamp=datetime.now(),
                status_code=200,
                response_time_ms=40.0,
                success=True,
                user_id="spike_user"
            )
            mock_request.return_value = mock_result
            
            summary = await runner.run_load_test(config)
            
            assert summary.test_type == LoadTestType.SPIKE
            assert summary.total_requests > 0
    
    @pytest.mark.asyncio
    async def test_volume_test(self, runner):
        """Test volume load test"""
        config = LoadTestConfig(
            test_name="volume_test",
            test_type=LoadTestType.VOLUME,
            target_url="http://localhost:8000/api/test",
            duration_seconds=2,
            concurrent_users=3,
            think_time_seconds=0.2
        )
        
        # Mock HTTP calls
        with patch.object(runner, '_make_request') as mock_request:
            mock_result = LoadTestResult(
                timestamp=datetime.now(),
                status_code=200,
                response_time_ms=100.0,  # Slower for volume
                success=True,
                response_size_bytes=5000,  # Larger response
                user_id="volume_user"
            )
            mock_request.return_value = mock_result
            
            summary = await runner.run_load_test(config)
            
            assert summary.test_type == LoadTestType.VOLUME
            assert summary.total_requests > 0
    
    @pytest.mark.asyncio
    async def test_stress_test_short(self, runner):
        """Test stress test with early termination"""
        config = LoadTestConfig(
            test_name="stress_test",
            test_type=LoadTestType.STRESS,
            target_url="http://localhost:8000/api/test",
            duration_seconds=10,  # Short for testing
            concurrent_users=4,   # Starting point
            think_time_seconds=0.1
        )
        
        # Mock HTTP calls with increasing failure rate
        call_count = 0
        def mock_request_with_failures(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Simulate increasing failure rate
            if call_count > 10:  # Fail after some successful requests
                success = False
                status_code = 500
            else:
                success = True
                status_code = 200
            
            return LoadTestResult(
                timestamp=datetime.now(),
                status_code=status_code,
                response_time_ms=50.0,
                success=success,
                user_id="stress_user"
            )
        
        with patch.object(runner, '_make_request', side_effect=mock_request_with_failures):
            summary = await runner.run_load_test(config)
            
            assert summary.test_type == LoadTestType.STRESS
            assert summary.total_requests > 0
            # Should have some failures due to our mock
            assert summary.failed_requests >= 0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.net_io_counters')
    @patch('psutil.net_connections')
    @patch('psutil.pids')
    def test_system_monitoring(self, mock_pids, mock_net_conn, mock_net_io, 
                               mock_disk_io, mock_memory, mock_cpu, runner):
        """Test system metrics monitoring"""
        # Mock system metrics
        mock_cpu.return_value = 75.5
        
        mock_memory_obj = MagicMock()
        mock_memory_obj.percent = 60.0
        mock_memory_obj.used = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.return_value = mock_memory_obj
        
        mock_disk_obj = MagicMock()
        mock_disk_obj.read_bytes = 1000 * 1024 * 1024  # 1GB
        mock_disk_obj.write_bytes = 500 * 1024 * 1024   # 500MB
        mock_disk_io.return_value = mock_disk_obj
        
        mock_net_obj = MagicMock()
        mock_net_obj.bytes_sent = 100 * 1024 * 1024     # 100MB
        mock_net_obj.bytes_recv = 200 * 1024 * 1024     # 200MB
        mock_net_io.return_value = mock_net_obj
        
        mock_net_conn.return_value = [1, 2, 3]  # 3 connections
        mock_pids.return_value = list(range(150))  # 150 processes
        
        async def test_monitoring():
            runner.running = True
            
            # Start monitoring
            monitor_task = asyncio.create_task(runner._monitor_system())
            
            # Let it collect a few samples
            await asyncio.sleep(0.1)
            
            # Stop monitoring
            runner.running = False
            monitor_task.cancel()
            
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            
            # Check collected metrics
            assert len(runner.system_metrics) > 0
            
            metrics = runner.system_metrics[0]
            assert "cpu_percent" in metrics
            assert "memory_percent" in metrics
            assert "memory_used_mb" in metrics
            assert "timestamp" in metrics
        
        asyncio.run(test_monitoring())
    
    def test_generate_summary(self, runner):
        """Test load test summary generation"""
        # Add mock results
        start_time = datetime.now()
        
        mock_results = [
            LoadTestResult(
                timestamp=start_time + timedelta(seconds=1),
                status_code=200,
                response_time_ms=50.0,
                success=True,
                response_size_bytes=1000,
                user_id="user1"
            ),
            LoadTestResult(
                timestamp=start_time + timedelta(seconds=2),
                status_code=200,
                response_time_ms=75.0,
                success=True,
                response_size_bytes=1200,
                user_id="user2"
            ),
            LoadTestResult(
                timestamp=start_time + timedelta(seconds=3),
                status_code=500,
                response_time_ms=100.0,
                success=False,
                error_message="Server error",
                user_id="user1"
            )
        ]
        
        runner.results = mock_results
        
        config = LoadTestConfig(
            test_name="summary_test",
            test_type=LoadTestType.CONSTANT_LOAD,
            target_url="http://test.com",
            duration_seconds=10,
            concurrent_users=2
        )
        
        end_time = start_time + timedelta(seconds=5)
        summary = runner._generate_summary(config, start_time, end_time)
        
        assert summary.test_name == "summary_test"
        assert summary.total_requests == 3
        assert summary.successful_requests == 2
        assert summary.failed_requests == 1
        assert summary.success_rate == 2/3
        assert summary.response_time_stats["min"] == 50.0
        assert summary.response_time_stats["max"] == 100.0
        assert summary.status_code_distribution[200] == 2
        assert summary.status_code_distribution[500] == 1
        assert summary.error_distribution["Server error"] == 1
    
    def test_export_json(self, runner):
        """Test JSON export functionality"""
        # Create mock summary
        summary = LoadTestSummary(
            test_name="export_test",
            test_type=LoadTestType.CONSTANT_LOAD,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=60),
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            success_rate=0.95,
            total_duration_seconds=60.0,
            requests_per_second=1.67,
            response_time_stats={"mean": 45.0, "p95": 85.0},
            status_code_distribution={200: 95, 500: 5},
            error_distribution={"Timeout": 3, "Server Error": 2},
            system_metrics={"avg_cpu_percent": 45.0}
        )
        
        # Add some mock results
        runner.results = [
            LoadTestResult(
                timestamp=datetime.now(),
                status_code=200,
                response_time_ms=45.0,
                success=True,
                user_id="test_user"
            )
        ]
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        runner._export_json(summary, temp_path)
        
        # Verify exported data
        with open(temp_path, 'r') as f:
            exported_data = json.load(f)
        
        assert "summary" in exported_data
        assert "detailed_results" in exported_data
        assert exported_data["summary"]["test_name"] == "export_test"
        assert exported_data["summary"]["total_requests"] == 100
        assert len(exported_data["detailed_results"]) == 1
        
        # Cleanup
        Path(temp_path).unlink()
    
    def test_export_csv(self, runner):
        """Test CSV export functionality"""
        # Create mock summary
        summary = LoadTestSummary(
            test_name="csv_test",
            test_type=LoadTestType.CONSTANT_LOAD,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=30),
            total_requests=50,
            successful_requests=48,
            failed_requests=2,
            success_rate=0.96,
            total_duration_seconds=30.0,
            requests_per_second=1.67,
            response_time_stats={"mean": 40.0},
            status_code_distribution={200: 48, 500: 2},
            error_distribution={},
            system_metrics={}
        )
        
        # Add mock results
        runner.results = [
            LoadTestResult(
                timestamp=datetime.now(),
                status_code=200,
                response_time_ms=40.0,
                success=True,
                response_size_bytes=800,
                user_id="csv_user"
            ),
            LoadTestResult(
                timestamp=datetime.now(),
                status_code=500,
                response_time_ms=200.0,
                success=False,
                error_message="Server Error",
                user_id="csv_user"
            )
        ]
        
        # Export to temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        runner._export_csv(summary, temp_path)
        
        # Verify CSV content
        import csv
        with open(temp_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2
        assert rows[0]['status_code'] == '200'
        assert rows[0]['success'] == 'True'
        assert rows[1]['status_code'] == '500'
        assert rows[1]['success'] == 'False'
        
        # Cleanup
        Path(temp_path).unlink()


class TestLoadTestConfigurations:
    """Test load test configuration helpers"""
    
    def test_create_search_api_test_config(self):
        """Test search API test configuration creation"""
        config = create_search_api_test_config("http://localhost:8080")
        
        assert config.test_name == "search_api_load_test"
        assert config.test_type == LoadTestType.CONSTANT_LOAD
        assert config.target_url == "http://localhost:8080/api/v1/search/query"
        assert config.duration_seconds == 60
        assert config.concurrent_users == 10
        assert config.headers["Content-Type"] == "application/json"
    
    def test_create_stress_test_config(self):
        """Test stress test configuration creation"""
        config = create_stress_test_config("http://test.example.com")
        
        assert config.test_name == "search_api_stress_test"
        assert config.test_type == LoadTestType.STRESS
        assert config.target_url == "http://test.example.com/api/v1/search/query"
        assert config.duration_seconds == 300
        assert config.concurrent_users == 20
        assert config.think_time_seconds == 1.0


class TestLoadTestIntegration:
    """Test load test integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_run_basic_load_test_mock(self):
        """Test basic load test with mocked HTTP responses"""
        # Mock the actual HTTP calls to avoid needing a running server
        with patch('src.testing.load_tester.LoadTestRunner.run_load_test') as mock_run:
            # Create a mock summary
            mock_summary = LoadTestSummary(
                test_name="search_api_load_test",
                test_type=LoadTestType.CONSTANT_LOAD,
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(seconds=60),
                total_requests=100,
                successful_requests=98,
                failed_requests=2,
                success_rate=0.98,
                total_duration_seconds=60.0,
                requests_per_second=1.67,
                response_time_stats={"mean": 42.0, "p95": 78.0, "p99": 95.0},
                status_code_distribution={200: 98, 500: 2},
                error_distribution={"Timeout": 2},
                system_metrics={"avg_cpu_percent": 35.0, "avg_memory_percent": 55.0}
            )
            
            mock_run.return_value = mock_summary
            
            # Mock file operations for export
            with patch('builtins.open'), patch('json.dump'):
                summary = await run_basic_load_test("http://localhost:8000")
            
            assert summary.test_name == "search_api_load_test"
            assert summary.success_rate == 0.98
            assert summary.total_requests == 100
            mock_run.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])