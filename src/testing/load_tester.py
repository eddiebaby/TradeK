"""
Load Testing Infrastructure for TradeKnowledge.

This module provides comprehensive load testing capabilities including:
- Concurrent user simulation
- API endpoint stress testing
- Search performance under load
- Resource usage monitoring during tests
- Load test reporting and analysis
"""

import asyncio
import json
import logging
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import aiohttp
import psutil

logger = logging.getLogger(__name__)


class LoadTestType(Enum):
    """Types of load tests"""

    CONSTANT_LOAD = "constant_load"  # Steady number of concurrent users
    RAMP_UP = "ramp_up"  # Gradually increasing load
    SPIKE = "spike"  # Sudden increase in load
    STRESS = "stress"  # Push to breaking point
    VOLUME = "volume"  # Large amounts of data


@dataclass
class LoadTestConfig:
    """Configuration for load testing"""

    test_name: str
    test_type: LoadTestType
    target_url: str
    duration_seconds: int
    concurrent_users: int
    requests_per_user: int = 0  # 0 means run for duration
    ramp_up_seconds: int = 0  # For gradual load increase
    think_time_seconds: float = 1.0  # Delay between requests
    timeout_seconds: float = 30.0  # Request timeout
    headers: dict[str, str] = field(default_factory=dict)
    auth_token: str | None = None


@dataclass
class LoadTestResult:
    """Result of a single request during load testing"""

    timestamp: datetime
    status_code: int
    response_time_ms: float
    success: bool
    error_message: str | None = None
    response_size_bytes: int = 0
    user_id: str = ""


@dataclass
class LoadTestSummary:
    """Summary of load test results"""

    test_name: str
    test_type: LoadTestType
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    total_duration_seconds: float
    requests_per_second: float
    response_time_stats: dict[str, float]
    status_code_distribution: dict[int, int]
    error_distribution: dict[str, int]
    system_metrics: dict[str, Any]


class LoadTestRunner:
    """
    Load test execution engine that simulates concurrent users
    and measures system performance under various load conditions.
    """

    def __init__(self):
        self.results: list[LoadTestResult] = []
        self.system_metrics: list[dict[str, Any]] = []
        self.running = False
        self._monitoring_task: asyncio.Task | None = None

    async def run_load_test(self, config: LoadTestConfig) -> LoadTestSummary:
        """
        Run a load test according to the provided configuration.

        Args:
            config: Load test configuration

        Returns:
            LoadTestSummary with results and metrics
        """
        logger.info(
            f"Starting load test '{config.test_name}' - {config.test_type.value}"
        )
        logger.info(f"Target: {config.target_url}")
        logger.info(
            f"Users: {config.concurrent_users}, Duration: {config.duration_seconds}s"
        )

        # Reset state
        self.results.clear()
        self.system_metrics.clear()
        self.running = True

        start_time = datetime.now()

        # Start system monitoring
        self._monitoring_task = asyncio.create_task(self._monitor_system())

        try:
            # Execute the appropriate load test type
            if config.test_type == LoadTestType.CONSTANT_LOAD:
                await self._run_constant_load_test(config)
            elif config.test_type == LoadTestType.RAMP_UP:
                await self._run_ramp_up_test(config)
            elif config.test_type == LoadTestType.SPIKE:
                await self._run_spike_test(config)
            elif config.test_type == LoadTestType.STRESS:
                await self._run_stress_test(config)
            elif config.test_type == LoadTestType.VOLUME:
                await self._run_volume_test(config)
            else:
                raise ValueError(f"Unsupported test type: {config.test_type}")

        finally:
            self.running = False
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass

        end_time = datetime.now()

        # Generate summary
        summary = self._generate_summary(config, start_time, end_time)

        logger.info(
            f"Load test completed: {summary.total_requests} requests, "
            f"{summary.success_rate:.2%} success rate, "
            f"{summary.requests_per_second:.2f} RPS"
        )

        return summary

    async def _run_constant_load_test(self, config: LoadTestConfig):
        """Run constant load test with steady concurrent users"""
        # Create user tasks
        user_tasks = []

        for user_id in range(config.concurrent_users):
            task = asyncio.create_task(self._simulate_user(config, f"user_{user_id}"))
            user_tasks.append(task)

        # Wait for all users to complete or timeout
        await asyncio.gather(*user_tasks, return_exceptions=True)

    async def _run_ramp_up_test(self, config: LoadTestConfig):
        """Run ramp-up test with gradually increasing load"""
        if config.ramp_up_seconds <= 0:
            config.ramp_up_seconds = config.duration_seconds // 2

        users_per_interval = max(1, config.concurrent_users // 10)
        interval_seconds = config.ramp_up_seconds / (
            config.concurrent_users // users_per_interval
        )

        user_tasks = []

        for batch in range(0, config.concurrent_users, users_per_interval):
            # Start a batch of users
            for user_id in range(
                batch, min(batch + users_per_interval, config.concurrent_users)
            ):
                task = asyncio.create_task(
                    self._simulate_user(config, f"user_{user_id}")
                )
                user_tasks.append(task)

            # Wait before starting next batch
            if batch + users_per_interval < config.concurrent_users:
                await asyncio.sleep(interval_seconds)

        # Wait for all users to complete
        await asyncio.gather(*user_tasks, return_exceptions=True)

    async def _run_spike_test(self, config: LoadTestConfig):
        """Run spike test with sudden load increase"""
        # Run with 25% load for 1/3 of duration
        spike_start = config.duration_seconds // 3
        spike_duration = config.duration_seconds // 3

        # Low load phase
        low_load_users = max(1, config.concurrent_users // 4)
        low_load_config = LoadTestConfig(
            test_name=config.test_name + "_low",
            test_type=LoadTestType.CONSTANT_LOAD,
            target_url=config.target_url,
            duration_seconds=spike_start,
            concurrent_users=low_load_users,
            think_time_seconds=config.think_time_seconds,
            headers=config.headers,
            auth_token=config.auth_token,
        )

        # Start low load
        low_load_tasks = []
        for user_id in range(low_load_users):
            task = asyncio.create_task(
                self._simulate_user(low_load_config, f"low_user_{user_id}")
            )
            low_load_tasks.append(task)

        # Wait for spike time
        await asyncio.sleep(spike_start)

        # Start spike load
        spike_users = config.concurrent_users - low_load_users
        spike_tasks = []
        for user_id in range(spike_users):
            task = asyncio.create_task(
                self._simulate_user(config, f"spike_user_{user_id}")
            )
            spike_tasks.append(task)

        # Wait for all tasks
        all_tasks = low_load_tasks + spike_tasks
        await asyncio.gather(*all_tasks, return_exceptions=True)

    async def _run_stress_test(self, config: LoadTestConfig):
        """Run stress test to find breaking point"""
        # Start with base load and increase until failure rate is high
        base_users = config.concurrent_users
        max_users = base_users * 5  # Cap at 5x the configured load
        step_size = max(1, base_users // 4)

        current_users = base_users

        while current_users <= max_users and self.running:
            logger.info(f"Stress test: Testing with {current_users} users")

            # Run a short test with current user count
            stress_config = LoadTestConfig(
                test_name=f"{config.test_name}_stress_{current_users}",
                test_type=LoadTestType.CONSTANT_LOAD,
                target_url=config.target_url,
                duration_seconds=min(30, config.duration_seconds),  # Short bursts
                concurrent_users=current_users,
                think_time_seconds=config.think_time_seconds * 0.5,  # Faster requests
                headers=config.headers,
                auth_token=config.auth_token,
            )

            # Run test batch
            user_tasks = []
            for user_id in range(current_users):
                task = asyncio.create_task(
                    self._simulate_user(
                        stress_config, f"stress_user_{current_users}_{user_id}"
                    )
                )
                user_tasks.append(task)

            await asyncio.gather(*user_tasks, return_exceptions=True)

            # Check if we should continue (based on recent failure rate)
            recent_results = [
                r
                for r in self.results
                if (datetime.now() - r.timestamp).total_seconds() < 60
            ]
            if recent_results:
                recent_failure_rate = len(
                    [r for r in recent_results if not r.success]
                ) / len(recent_results)
                if recent_failure_rate > 0.2:  # Stop if >20% failure rate
                    logger.warning(
                        f"Stress test stopped at {current_users} users due to high failure rate"
                    )
                    break

            current_users += step_size
            await asyncio.sleep(5)  # Brief pause between stress levels

    async def _run_volume_test(self, config: LoadTestConfig):
        """Run volume test with large amounts of data"""
        # Volume test focuses on data-intensive operations
        # For search endpoints, this means complex queries and large result sets

        volume_config = LoadTestConfig(
            test_name=config.test_name + "_volume",
            test_type=LoadTestType.CONSTANT_LOAD,
            target_url=config.target_url,
            duration_seconds=config.duration_seconds,
            concurrent_users=min(config.concurrent_users, 10),  # Fewer users for volume
            think_time_seconds=config.think_time_seconds * 2,  # Longer think time
            headers=config.headers,
            auth_token=config.auth_token,
        )

        # Run with volume-focused user simulation
        user_tasks = []
        for user_id in range(volume_config.concurrent_users):
            task = asyncio.create_task(
                self._simulate_volume_user(volume_config, f"volume_user_{user_id}")
            )
            user_tasks.append(task)

        await asyncio.gather(*user_tasks, return_exceptions=True)

    async def _simulate_user(self, config: LoadTestConfig, user_id: str):
        """Simulate a single user's behavior"""
        session_timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)

        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            start_time = time.time()
            request_count = 0

            while self.running:
                # Check if we should stop
                elapsed = time.time() - start_time
                if elapsed >= config.duration_seconds:
                    break

                if (
                    config.requests_per_user > 0
                    and request_count >= config.requests_per_user
                ):
                    break

                # Prepare request
                headers = config.headers.copy()
                if config.auth_token:
                    headers["Authorization"] = f"Bearer {config.auth_token}"

                # Generate test data
                test_data = self._generate_test_request_data(config)

                # Make request
                result = await self._make_request(
                    session, config.target_url, headers, test_data, user_id
                )

                self.results.append(result)
                request_count += 1

                # Think time
                if config.think_time_seconds > 0:
                    think_time = config.think_time_seconds + random.uniform(-0.1, 0.1)
                    await asyncio.sleep(max(0, think_time))

    async def _simulate_volume_user(self, config: LoadTestConfig, user_id: str):
        """Simulate a user focused on volume/data-intensive operations"""
        session_timeout = aiohttp.ClientTimeout(
            total=config.timeout_seconds * 2
        )  # Longer timeout

        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            start_time = time.time()

            while self.running and (time.time() - start_time) < config.duration_seconds:
                # Prepare volume-intensive request
                headers = config.headers.copy()
                if config.auth_token:
                    headers["Authorization"] = f"Bearer {config.auth_token}"

                # Generate volume test data (large queries, many results)
                volume_data = self._generate_volume_test_data()

                # Make request
                result = await self._make_request(
                    session, config.target_url, headers, volume_data, user_id
                )

                self.results.append(result)

                # Longer think time for volume operations
                await asyncio.sleep(config.think_time_seconds)

    def _generate_test_request_data(self, config: LoadTestConfig) -> dict[str, Any]:
        """Generate test request data"""
        # For search API testing
        search_queries = [
            "algorithmic trading strategies",
            "risk management techniques",
            "machine learning in finance",
            "technical analysis indicators",
            "portfolio optimization methods",
            "quantitative trading models",
            "market microstructure analysis",
            "derivatives pricing models",
            "behavioral finance theory",
            "high frequency trading",
        ]

        return {
            "query": random.choice(search_queries),
            "max_results": random.choice([5, 10, 20, 50]),
            "intent": random.choice(["research", "learning", "quick_lookup"]),
        }

    def _generate_volume_test_data(self) -> dict[str, Any]:
        """Generate volume-intensive test data"""
        # Large, complex queries for volume testing
        complex_queries = [
            "comprehensive analysis of momentum trading strategies with machine learning applications",
            "detailed study of quantitative risk management techniques across different asset classes",
            "in-depth examination of algorithmic trading implementation using Python and statistical models",
            "extensive research on market microstructure effects on high frequency trading performance",
            "thorough investigation of portfolio optimization methodologies including Black-Litterman and mean reversion",
        ]

        return {
            "query": random.choice(complex_queries),
            "max_results": random.choice([100, 200, 500]),  # Large result sets
            "intent": "research",
            "filters": {
                "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                "content_type": random.choice(["book", "paper", "tutorial"]),
            },
        }

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: dict[str, str],
        data: dict[str, Any],
        user_id: str,
    ) -> LoadTestResult:
        """Make HTTP request and record result"""
        start_time = time.perf_counter()
        timestamp = datetime.now()

        try:
            async with session.post(url, headers=headers, json=data) as response:
                end_time = time.perf_counter()
                response_time_ms = (end_time - start_time) * 1000

                # Read response
                response_text = await response.text()
                response_size = len(response_text.encode("utf-8"))

                success = 200 <= response.status < 300

                return LoadTestResult(
                    timestamp=timestamp,
                    status_code=response.status,
                    response_time_ms=response_time_ms,
                    success=success,
                    response_size_bytes=response_size,
                    user_id=user_id,
                )

        except TimeoutError:
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            return LoadTestResult(
                timestamp=timestamp,
                status_code=0,
                response_time_ms=response_time_ms,
                success=False,
                error_message="Request timeout",
                user_id=user_id,
            )

        except Exception as e:
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            return LoadTestResult(
                timestamp=timestamp,
                status_code=0,
                response_time_ms=response_time_ms,
                success=False,
                error_message=str(e),
                user_id=user_id,
            )

    async def _monitor_system(self):
        """Monitor system metrics during load test"""
        while self.running:
            try:
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_used_mb": psutil.virtual_memory().used / 1024 / 1024,
                    "disk_io_read_mb": (
                        psutil.disk_io_counters().read_bytes / 1024 / 1024
                        if psutil.disk_io_counters()
                        else 0
                    ),
                    "disk_io_write_mb": (
                        psutil.disk_io_counters().write_bytes / 1024 / 1024
                        if psutil.disk_io_counters()
                        else 0
                    ),
                    "network_sent_mb": psutil.net_io_counters().bytes_sent
                    / 1024
                    / 1024,
                    "network_recv_mb": psutil.net_io_counters().bytes_recv
                    / 1024
                    / 1024,
                    "active_connections": len(psutil.net_connections()),
                    "process_count": len(psutil.pids()),
                }

                self.system_metrics.append(metrics)
                await asyncio.sleep(1)  # Collect metrics every second

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(1)

    def _generate_summary(
        self, config: LoadTestConfig, start_time: datetime, end_time: datetime
    ) -> LoadTestSummary:
        """Generate load test summary"""
        if not self.results:
            # Return empty summary if no results
            return LoadTestSummary(
                test_name=config.test_name,
                test_type=config.test_type,
                start_time=start_time,
                end_time=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                success_rate=0.0,
                total_duration_seconds=0.0,
                requests_per_second=0.0,
                response_time_stats={},
                status_code_distribution={},
                error_distribution={},
                system_metrics={},
            )

        # Calculate basic metrics
        total_requests = len(self.results)
        successful_requests = len([r for r in self.results if r.success])
        failed_requests = total_requests - successful_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0

        total_duration = (end_time - start_time).total_seconds()
        requests_per_second = (
            total_requests / total_duration if total_duration > 0 else 0
        )

        # Response time statistics
        response_times = [r.response_time_ms for r in self.results]
        response_time_stats = {
            "min": min(response_times) if response_times else 0,
            "max": max(response_times) if response_times else 0,
            "mean": statistics.mean(response_times) if response_times else 0,
            "median": statistics.median(response_times) if response_times else 0,
            "p95": self._percentile(response_times, 95) if response_times else 0,
            "p99": self._percentile(response_times, 99) if response_times else 0,
        }

        # Status code distribution
        status_codes = defaultdict(int)
        for result in self.results:
            status_codes[result.status_code] += 1

        # Error distribution
        errors = defaultdict(int)
        for result in self.results:
            if not result.success and result.error_message:
                errors[result.error_message] += 1

        # System metrics summary
        system_summary = {}
        if self.system_metrics:
            system_summary = {
                "avg_cpu_percent": statistics.mean(
                    [m["cpu_percent"] for m in self.system_metrics]
                ),
                "max_cpu_percent": max([m["cpu_percent"] for m in self.system_metrics]),
                "avg_memory_percent": statistics.mean(
                    [m["memory_percent"] for m in self.system_metrics]
                ),
                "max_memory_percent": max(
                    [m["memory_percent"] for m in self.system_metrics]
                ),
                "avg_memory_used_mb": statistics.mean(
                    [m["memory_used_mb"] for m in self.system_metrics]
                ),
                "max_memory_used_mb": max(
                    [m["memory_used_mb"] for m in self.system_metrics]
                ),
                "total_network_sent_mb": max(
                    [m["network_sent_mb"] for m in self.system_metrics]
                )
                - min([m["network_sent_mb"] for m in self.system_metrics]),
                "total_network_recv_mb": max(
                    [m["network_recv_mb"] for m in self.system_metrics]
                )
                - min([m["network_recv_mb"] for m in self.system_metrics]),
            }

        return LoadTestSummary(
            test_name=config.test_name,
            test_type=config.test_type,
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=success_rate,
            total_duration_seconds=total_duration,
            requests_per_second=requests_per_second,
            response_time_stats=response_time_stats,
            status_code_distribution=dict(status_codes),
            error_distribution=dict(errors),
            system_metrics=system_summary,
        )

    def _percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0

        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)

        if index == int(index):
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def export_results(
        self, summary: LoadTestSummary, output_path: str, format_type: str = "json"
    ):
        """Export load test results"""
        if format_type == "json":
            self._export_json(summary, output_path)
        elif format_type == "csv":
            self._export_csv(summary, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _export_json(self, summary: LoadTestSummary, output_path: str):
        """Export results as JSON"""
        export_data = {
            "summary": {
                "test_name": summary.test_name,
                "test_type": summary.test_type.value,
                "start_time": summary.start_time.isoformat(),
                "end_time": summary.end_time.isoformat(),
                "total_requests": summary.total_requests,
                "successful_requests": summary.successful_requests,
                "failed_requests": summary.failed_requests,
                "success_rate": summary.success_rate,
                "total_duration_seconds": summary.total_duration_seconds,
                "requests_per_second": summary.requests_per_second,
                "response_time_stats": summary.response_time_stats,
                "status_code_distribution": summary.status_code_distribution,
                "error_distribution": summary.error_distribution,
                "system_metrics": summary.system_metrics,
            },
            "detailed_results": [
                {
                    "timestamp": result.timestamp.isoformat(),
                    "status_code": result.status_code,
                    "response_time_ms": result.response_time_ms,
                    "success": result.success,
                    "error_message": result.error_message,
                    "response_size_bytes": result.response_size_bytes,
                    "user_id": result.user_id,
                }
                for result in self.results
            ],
            "system_metrics": self.system_metrics,
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Load test results exported to {output_path}")

    def _export_csv(self, summary: LoadTestSummary, output_path: str):
        """Export results as CSV"""
        import csv

        with open(output_path, "w", newline="") as csvfile:
            fieldnames = [
                "timestamp",
                "status_code",
                "response_time_ms",
                "success",
                "error_message",
                "response_size_bytes",
                "user_id",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                writer.writerow(
                    {
                        "timestamp": result.timestamp.isoformat(),
                        "status_code": result.status_code,
                        "response_time_ms": result.response_time_ms,
                        "success": result.success,
                        "error_message": result.error_message or "",
                        "response_size_bytes": result.response_size_bytes,
                        "user_id": result.user_id,
                    }
                )

        logger.info(f"Load test results exported to CSV: {output_path}")


def create_search_api_test_config(
    base_url: str = "http://localhost:8000",
) -> LoadTestConfig:
    """Create a standard load test configuration for search API"""
    return LoadTestConfig(
        test_name="search_api_load_test",
        test_type=LoadTestType.CONSTANT_LOAD,
        target_url=f"{base_url}/api/v1/search/query",
        duration_seconds=60,
        concurrent_users=10,
        think_time_seconds=2.0,
        headers={"Content-Type": "application/json"},
    )


def create_stress_test_config(
    base_url: str = "http://localhost:8000",
) -> LoadTestConfig:
    """Create a stress test configuration"""
    return LoadTestConfig(
        test_name="search_api_stress_test",
        test_type=LoadTestType.STRESS,
        target_url=f"{base_url}/api/v1/search/query",
        duration_seconds=300,  # 5 minutes
        concurrent_users=20,  # Starting point
        think_time_seconds=1.0,
        headers={"Content-Type": "application/json"},
    )


async def run_basic_load_test(
    base_url: str = "http://localhost:8000",
) -> LoadTestSummary:
    """Run a basic load test against the search API"""
    config = create_search_api_test_config(base_url)
    runner = LoadTestRunner()

    summary = await runner.run_load_test(config)

    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"load_test_results_{timestamp}.json"
    runner.export_results(summary, output_path)

    return summary
