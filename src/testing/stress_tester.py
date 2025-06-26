"""
Comprehensive Stress Testing Framework for TradeKnowledge.

This module provides advanced stress testing capabilities including
memory pressure, CPU stress, network saturation, and system limit testing.
"""

import asyncio
import concurrent.futures
import gc
import logging
import random
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class StressTestType(Enum):
    """Types of stress tests"""

    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_PRESSURE = "memory_pressure"
    IO_SATURATION = "io_saturation"
    NETWORK_STRESS = "network_stress"
    CONCURRENT_USERS = "concurrent_users"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SYSTEM_LIMITS = "system_limits"


@dataclass
class StressTestConfig:
    """Configuration for stress tests"""

    test_name: str
    test_type: StressTestType
    duration_seconds: int = 60
    intensity_level: int = 5  # 1-10 scale
    target_resource_usage: float = 0.8  # Target 80% resource usage
    concurrent_operations: int = 100
    ramp_up_seconds: int = 10
    cool_down_seconds: int = 5
    failure_threshold: float = 0.1  # 10% failure rate threshold
    timeout_seconds: float = 30.0
    custom_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestMetrics:
    """Metrics collected during stress testing"""

    test_name: str
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    timeout_operations: int = 0
    peak_cpu_usage: float = 0.0
    peak_memory_usage: float = 0.0
    peak_memory_mb: float = 0.0
    average_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    system_stability_score: float = 1.0
    breaking_point_reached: bool = False
    breaking_point_metric: str | None = None
    resource_usage_samples: list[dict[str, float]] = field(default_factory=list)
    error_distribution: dict[str, int] = field(default_factory=dict)


class StressTestResult:
    """Result of a stress test execution"""

    def __init__(self, config: StressTestConfig, metrics: StressTestMetrics):
        self.config = config
        self.metrics = metrics
        self.success = self._determine_success()
        self.recommendations = self._generate_recommendations()

    def _determine_success(self) -> bool:
        """Determine if the stress test was successful"""
        failure_rate = self.metrics.failed_operations / max(
            1, self.metrics.total_operations
        )

        return (
            failure_rate <= self.config.failure_threshold
            and not self.metrics.breaking_point_reached
            and self.metrics.system_stability_score >= 0.7
        )

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        failure_rate = self.metrics.failed_operations / max(
            1, self.metrics.total_operations
        )

        if failure_rate > self.config.failure_threshold:
            recommendations.append(
                f"High failure rate ({failure_rate:.2%}). Consider implementing circuit breakers or rate limiting."
            )

        if self.metrics.peak_cpu_usage > 0.9:
            recommendations.append(
                "CPU usage exceeded 90%. Consider horizontal scaling or CPU optimization."
            )

        if self.metrics.peak_memory_usage > 0.9:
            recommendations.append(
                "Memory usage exceeded 90%. Implement memory management or increase resources."
            )

        if self.metrics.average_response_time > 1000:  # 1 second
            recommendations.append(
                "Average response time exceeds 1 second. Optimize database queries and algorithms."
            )

        if self.metrics.breaking_point_reached:
            recommendations.append(
                f"Breaking point reached in {self.metrics.breaking_point_metric}. "
                f"System limits need to be addressed before production deployment."
            )

        if not recommendations:
            recommendations.append(
                "System performed well under stress. Ready for production load."
            )

        return recommendations


class StressTester:
    """
    Comprehensive stress testing framework that pushes the system
    to its limits to identify breaking points and performance bottlenecks.
    """

    def __init__(self):
        self.running = False
        self.metrics_collector: Callable | None = None
        self._monitoring_task: asyncio.Task | None = None

    async def run_stress_test(self, config: StressTestConfig) -> StressTestResult:
        """
        Execute a comprehensive stress test.

        Args:
            config: Stress test configuration

        Returns:
            StressTestResult with metrics and analysis
        """
        logger.info(
            f"Starting stress test: {config.test_name} ({config.test_type.value})"
        )

        metrics = StressTestMetrics(
            test_name=config.test_name, start_time=datetime.now()
        )

        self.running = True

        try:
            # Start system monitoring
            self._monitoring_task = asyncio.create_task(
                self._monitor_system_resources(metrics)
            )

            # Ramp up phase
            if config.ramp_up_seconds > 0:
                await self._ramp_up_phase(config, metrics)

            # Main stress phase
            await self._execute_stress_test(config, metrics)

            # Cool down phase
            if config.cool_down_seconds > 0:
                await self._cool_down_phase(config, metrics)

        finally:
            self.running = False

            # Stop monitoring
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass

            metrics.end_time = datetime.now()
            metrics.duration_seconds = (
                metrics.end_time - metrics.start_time
            ).total_seconds()

        # Analyze results
        self._analyze_metrics(metrics)

        result = StressTestResult(config, metrics)

        logger.info(f"Stress test completed: {config.test_name}")
        logger.info(
            f"Success: {result.success}, Stability Score: {metrics.system_stability_score:.2f}"
        )

        return result

    async def _ramp_up_phase(
        self, config: StressTestConfig, metrics: StressTestMetrics
    ):
        """Gradually increase load during ramp-up"""
        logger.info(f"Ramp-up phase: {config.ramp_up_seconds}s")

        steps = min(10, config.ramp_up_seconds)
        step_duration = config.ramp_up_seconds / steps

        for step in range(steps):
            load_factor = (step + 1) / steps
            operations = int(config.concurrent_operations * load_factor)

            await self._execute_operations(config, metrics, operations, step_duration)

    async def _execute_stress_test(
        self, config: StressTestConfig, metrics: StressTestMetrics
    ):
        """Execute the main stress test phase"""
        logger.info(f"Main stress phase: {config.duration_seconds}s")

        if config.test_type == StressTestType.CPU_INTENSIVE:
            await self._cpu_stress_test(config, metrics)
        elif config.test_type == StressTestType.MEMORY_PRESSURE:
            await self._memory_stress_test(config, metrics)
        elif config.test_type == StressTestType.IO_SATURATION:
            await self._io_stress_test(config, metrics)
        elif config.test_type == StressTestType.NETWORK_STRESS:
            await self._network_stress_test(config, metrics)
        elif config.test_type == StressTestType.CONCURRENT_USERS:
            await self._concurrent_users_test(config, metrics)
        elif config.test_type == StressTestType.RESOURCE_EXHAUSTION:
            await self._resource_exhaustion_test(config, metrics)
        elif config.test_type == StressTestType.SYSTEM_LIMITS:
            await self._system_limits_test(config, metrics)
        else:
            await self._generic_stress_test(config, metrics)

    async def _cool_down_phase(
        self, config: StressTestConfig, metrics: StressTestMetrics
    ):
        """Cool down phase to observe system recovery"""
        logger.info(f"Cool-down phase: {config.cool_down_seconds}s")
        await asyncio.sleep(config.cool_down_seconds)

    async def _cpu_stress_test(
        self, config: StressTestConfig, metrics: StressTestMetrics
    ):
        """CPU-intensive stress test"""

        def cpu_intensive_work(n: int) -> int:
            # CPU-bound calculation
            total = 0
            for i in range(n * 1000):
                total += i**2
                if i % 10000 == 0:
                    # Periodically check if we should stop
                    if not self.running:
                        break
            return total

        # Calculate work size based on intensity
        work_size = config.intensity_level * 10000

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(config.concurrent_operations, psutil.cpu_count() * 2)
        ) as executor:

            start_time = time.time()
            futures = []

            while time.time() - start_time < config.duration_seconds and self.running:
                # Submit CPU work
                future = executor.submit(cpu_intensive_work, work_size)
                futures.append((future, time.time()))
                metrics.total_operations += 1

                # Clean up completed futures
                completed_futures = [(f, start_t) for f, start_t in futures if f.done()]

                for future, start_t in completed_futures:
                    try:
                        future.result(timeout=0.1)
                        metrics.successful_operations += 1
                        response_time = (time.time() - start_t) * 1000
                        self._record_response_time(metrics, response_time)
                    except Exception as e:
                        metrics.failed_operations += 1
                        self._record_error(metrics, str(e))

                    futures.remove((future, start_t))

                # Control load intensity
                await asyncio.sleep(0.01)

            # Wait for remaining futures
            for future, start_t in futures:
                try:
                    future.result(timeout=config.timeout_seconds)
                    metrics.successful_operations += 1
                except Exception as e:
                    metrics.failed_operations += 1
                    self._record_error(metrics, str(e))

    async def _memory_stress_test(
        self, config: StressTestConfig, metrics: StressTestMetrics
    ):
        """Memory pressure stress test"""
        memory_blocks = []
        target_memory_mb = config.custom_parameters.get("target_memory_mb", 1000)
        block_size_mb = 10

        try:
            start_time = time.time()

            while (
                time.time() - start_time < config.duration_seconds
                and self.running
                and len(memory_blocks) * block_size_mb < target_memory_mb
            ):

                # Allocate memory block
                block = bytearray(block_size_mb * 1024 * 1024)
                # Fill with random data to prevent optimization
                for i in range(0, len(block), 1024):
                    block[i] = random.randint(0, 255)

                memory_blocks.append(block)
                metrics.total_operations += 1
                metrics.successful_operations += 1

                # Check memory usage
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 95:
                    metrics.breaking_point_reached = True
                    metrics.breaking_point_metric = "memory_usage"
                    logger.warning("Memory usage exceeded 95%, stopping test")
                    break

                await asyncio.sleep(0.1)

        finally:
            # Clean up memory
            memory_blocks.clear()
            gc.collect()

    async def _io_stress_test(
        self, config: StressTestConfig, metrics: StressTestMetrics
    ):
        """I/O saturation stress test"""
        temp_dir = tempfile.mkdtemp(prefix="stress_test_")
        file_size_mb = config.custom_parameters.get("file_size_mb", 10)
        num_files = config.intensity_level * 10

        try:
            start_time = time.time()

            # Create multiple I/O operations
            tasks = []
            for i in range(num_files):
                task = asyncio.create_task(
                    self._io_operation(
                        temp_dir, f"test_file_{i}", file_size_mb, metrics
                    )
                )
                tasks.append(task)

            # Wait for completion or timeout
            done, pending = await asyncio.wait(
                tasks,
                timeout=config.duration_seconds,
                return_when=asyncio.ALL_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            await asyncio.gather(*pending, return_exceptions=True)

        finally:
            # Clean up temporary files
            try:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")

    async def _io_operation(
        self, temp_dir: str, filename: str, size_mb: int, metrics: StressTestMetrics
    ):
        """Perform I/O operation"""
        filepath = Path(temp_dir) / filename
        data = b"x" * (1024 * 1024)  # 1MB chunk

        start_time = time.time()

        try:
            # Write file
            with open(filepath, "wb") as f:
                for _ in range(size_mb):
                    f.write(data)
                    await asyncio.sleep(0)  # Yield control

            # Read file
            with open(filepath, "rb") as f:
                while f.read(1024 * 1024):  # Read in chunks
                    await asyncio.sleep(0)  # Yield control

            # Delete file
            filepath.unlink()

            metrics.total_operations += 1
            metrics.successful_operations += 1
            response_time = (time.time() - start_time) * 1000
            self._record_response_time(metrics, response_time)

        except Exception as e:
            metrics.total_operations += 1
            metrics.failed_operations += 1
            self._record_error(metrics, str(e))

    async def _network_stress_test(
        self, config: StressTestConfig, metrics: StressTestMetrics
    ):
        """Network stress test (if target URL provided)"""
        target_url = config.custom_parameters.get(
            "target_url", "http://localhost:8000/health"
        )

        import aiohttp

        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            tasks = []

            while time.time() - start_time < config.duration_seconds and self.running:
                # Create network request
                task = asyncio.create_task(
                    self._network_request(session, target_url, metrics)
                )
                tasks.append(task)

                # Limit concurrent requests
                if len(tasks) >= config.concurrent_operations:
                    done, tasks = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    tasks = list(tasks)

                await asyncio.sleep(0.01)

            # Wait for remaining tasks
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _network_request(
        self, session: "aiohttp.ClientSession", url: str, metrics: StressTestMetrics
    ):
        """Perform network request"""
        start_time = time.time()

        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                await response.text()

                metrics.total_operations += 1
                if response.status < 400:
                    metrics.successful_operations += 1
                else:
                    metrics.failed_operations += 1
                    self._record_error(metrics, f"HTTP {response.status}")

                response_time = (time.time() - start_time) * 1000
                self._record_response_time(metrics, response_time)

        except Exception as e:
            metrics.total_operations += 1
            metrics.failed_operations += 1
            self._record_error(metrics, str(e))

    async def _concurrent_users_test(
        self, config: StressTestConfig, metrics: StressTestMetrics
    ):
        """Simulate many concurrent users"""
        user_tasks = []

        for user_id in range(config.concurrent_operations):
            task = asyncio.create_task(
                self._simulate_user_session(user_id, config, metrics)
            )
            user_tasks.append(task)

        # Wait for all user sessions or timeout
        done, pending = await asyncio.wait(
            user_tasks,
            timeout=config.duration_seconds,
            return_when=asyncio.ALL_COMPLETED,
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()

        await asyncio.gather(*pending, return_exceptions=True)

    async def _simulate_user_session(
        self, user_id: int, config: StressTestConfig, metrics: StressTestMetrics
    ):
        """Simulate a user session"""
        session_duration = random.uniform(5, config.duration_seconds)
        start_time = time.time()

        while time.time() - start_time < session_duration and self.running:
            try:
                # Simulate user operations
                operation_time = random.uniform(0.1, 2.0)
                await asyncio.sleep(operation_time)

                metrics.total_operations += 1

                # Random chance of operation failure
                if random.random() < 0.05:  # 5% failure rate
                    metrics.failed_operations += 1
                    self._record_error(metrics, "Simulated user operation failure")
                else:
                    metrics.successful_operations += 1
                    self._record_response_time(metrics, operation_time * 1000)

                # Think time between operations
                await asyncio.sleep(random.uniform(0.5, 3.0))

            except Exception as e:
                metrics.failed_operations += 1
                self._record_error(metrics, str(e))

    async def _resource_exhaustion_test(
        self, config: StressTestConfig, metrics: StressTestMetrics
    ):
        """Test resource exhaustion scenarios"""
        # Test file descriptor exhaustion
        file_handles = []

        try:
            start_time = time.time()

            while time.time() - start_time < config.duration_seconds and self.running:
                try:
                    # Open file handles
                    handle = open("/dev/null")
                    file_handles.append(handle)
                    metrics.total_operations += 1
                    metrics.successful_operations += 1

                    if len(file_handles) % 100 == 0:
                        logger.debug(f"Opened {len(file_handles)} file handles")

                    await asyncio.sleep(0.001)

                except OSError as e:
                    if "Too many open files" in str(e):
                        metrics.breaking_point_reached = True
                        metrics.breaking_point_metric = "file_descriptors"
                        logger.warning(
                            f"File descriptor limit reached: {len(file_handles)}"
                        )
                        break
                    else:
                        metrics.failed_operations += 1
                        self._record_error(metrics, str(e))

        finally:
            # Clean up file handles
            for handle in file_handles:
                try:
                    handle.close()
                except:
                    pass

    async def _system_limits_test(
        self, config: StressTestConfig, metrics: StressTestMetrics
    ):
        """Test system limits and boundaries"""
        # Test thread creation limits
        threads = []

        def dummy_work():
            time.sleep(config.duration_seconds)

        try:
            start_time = time.time()

            while time.time() - start_time < config.duration_seconds and self.running:
                try:
                    thread = threading.Thread(target=dummy_work)
                    thread.start()
                    threads.append(thread)

                    metrics.total_operations += 1
                    metrics.successful_operations += 1

                    if len(threads) % 50 == 0:
                        logger.debug(f"Created {len(threads)} threads")

                    await asyncio.sleep(0.01)

                except Exception as e:
                    if "can't start new thread" in str(e).lower():
                        metrics.breaking_point_reached = True
                        metrics.breaking_point_metric = "thread_limit"
                        logger.warning(f"Thread limit reached: {len(threads)}")
                        break
                    else:
                        metrics.failed_operations += 1
                        self._record_error(metrics, str(e))

        finally:
            # Signal threads to stop (they'll exit naturally)
            self.running = False

            # Wait for threads to complete (with timeout)
            for thread in threads:
                thread.join(timeout=1.0)

    async def _generic_stress_test(
        self, config: StressTestConfig, metrics: StressTestMetrics
    ):
        """Generic stress test for custom scenarios"""
        await self._execute_operations(
            config, metrics, config.concurrent_operations, config.duration_seconds
        )

    async def _execute_operations(
        self,
        config: StressTestConfig,
        metrics: StressTestMetrics,
        num_operations: int,
        duration: float,
    ):
        """Execute a number of operations for specified duration"""
        tasks = []
        start_time = time.time()

        while time.time() - start_time < duration and self.running:
            # Create operations up to the limit
            while len(tasks) < num_operations:
                task = asyncio.create_task(self._generic_operation(metrics))
                tasks.append(task)

            # Wait for some operations to complete
            if tasks:
                done, tasks = await asyncio.wait(
                    tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                )
                tasks = list(tasks)

            await asyncio.sleep(0.01)

        # Wait for remaining tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _generic_operation(self, metrics: StressTestMetrics):
        """Generic operation for stress testing"""
        start_time = time.time()

        try:
            # Simulate work
            await asyncio.sleep(random.uniform(0.01, 0.1))

            metrics.total_operations += 1
            metrics.successful_operations += 1

            response_time = (time.time() - start_time) * 1000
            self._record_response_time(metrics, response_time)

        except Exception as e:
            metrics.total_operations += 1
            metrics.failed_operations += 1
            self._record_error(metrics, str(e))

    async def _monitor_system_resources(self, metrics: StressTestMetrics):
        """Monitor system resources during stress test"""
        while self.running:
            try:
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()

                metrics.peak_cpu_usage = max(metrics.peak_cpu_usage, cpu_percent)
                metrics.peak_memory_usage = max(
                    metrics.peak_memory_usage, memory.percent
                )
                metrics.peak_memory_mb = max(
                    metrics.peak_memory_mb, memory.used / 1024 / 1024
                )

                # Record sample
                sample = {
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_mb": memory.used / 1024 / 1024,
                }
                metrics.resource_usage_samples.append(sample)

                await asyncio.sleep(1)  # Sample every second

            except Exception as e:
                logger.warning(f"Error monitoring resources: {e}")
                await asyncio.sleep(1)

    def _record_response_time(
        self, metrics: StressTestMetrics, response_time_ms: float
    ):
        """Record response time for metrics calculation"""
        # Simple running average (could be improved with proper statistics)
        total_responses = metrics.successful_operations
        if total_responses == 1:
            metrics.average_response_time = response_time_ms
        else:
            metrics.average_response_time = (
                metrics.average_response_time * (total_responses - 1) + response_time_ms
            ) / total_responses

    def _record_error(self, metrics: StressTestMetrics, error_message: str):
        """Record error for metrics tracking"""
        error_type = (
            error_message.split(":")[0] if ":" in error_message else error_message
        )
        metrics.error_distribution[error_type] = (
            metrics.error_distribution.get(error_type, 0) + 1
        )

    def _analyze_metrics(self, metrics: StressTestMetrics):
        """Analyze collected metrics and calculate stability score"""
        if metrics.total_operations == 0:
            metrics.system_stability_score = 0.0
            return

        # Calculate failure rate
        failure_rate = metrics.failed_operations / metrics.total_operations

        # Calculate resource stability (penalize high resource usage)
        cpu_stability = (
            max(0, 1 - (metrics.peak_cpu_usage - 80) / 20)
            if metrics.peak_cpu_usage > 80
            else 1
        )
        memory_stability = (
            max(0, 1 - (metrics.peak_memory_usage - 80) / 20)
            if metrics.peak_memory_usage > 80
            else 1
        )

        # Calculate overall stability score
        metrics.system_stability_score = (
            (1 - failure_rate) * 0.5 + cpu_stability * 0.25 + memory_stability * 0.25
        )

        # Penalize if breaking point was reached
        if metrics.breaking_point_reached:
            metrics.system_stability_score *= 0.5


# Convenience functions for common stress test scenarios
def create_cpu_stress_config(
    duration: int = 60, intensity: int = 5
) -> StressTestConfig:
    """Create CPU stress test configuration"""
    return StressTestConfig(
        test_name=f"cpu_stress_intensity_{intensity}",
        test_type=StressTestType.CPU_INTENSIVE,
        duration_seconds=duration,
        intensity_level=intensity,
        concurrent_operations=psutil.cpu_count() * 2,
    )


def create_memory_stress_config(
    duration: int = 60, target_memory_mb: int = 1000
) -> StressTestConfig:
    """Create memory stress test configuration"""
    return StressTestConfig(
        test_name=f"memory_stress_{target_memory_mb}mb",
        test_type=StressTestType.MEMORY_PRESSURE,
        duration_seconds=duration,
        intensity_level=5,
        custom_parameters={"target_memory_mb": target_memory_mb},
    )


def create_io_stress_config(
    duration: int = 60, file_size_mb: int = 10
) -> StressTestConfig:
    """Create I/O stress test configuration"""
    return StressTestConfig(
        test_name=f"io_stress_{file_size_mb}mb_files",
        test_type=StressTestType.IO_SATURATION,
        duration_seconds=duration,
        intensity_level=5,
        custom_parameters={"file_size_mb": file_size_mb},
    )


def create_concurrent_users_config(
    duration: int = 60, num_users: int = 100
) -> StressTestConfig:
    """Create concurrent users stress test configuration"""
    return StressTestConfig(
        test_name=f"concurrent_users_{num_users}",
        test_type=StressTestType.CONCURRENT_USERS,
        duration_seconds=duration,
        concurrent_operations=num_users,
        intensity_level=5,
    )


async def run_comprehensive_stress_suite(
    target_url: str | None = None,
) -> list[StressTestResult]:
    """
    Run a comprehensive suite of stress tests.

    Args:
        target_url: Optional URL for network stress testing

    Returns:
        List of stress test results
    """
    stress_tester = StressTester()
    results = []

    # CPU stress test
    cpu_config = create_cpu_stress_config(duration=30, intensity=3)
    cpu_result = await stress_tester.run_stress_test(cpu_config)
    results.append(cpu_result)

    # Memory stress test
    memory_config = create_memory_stress_config(duration=30, target_memory_mb=500)
    memory_result = await stress_tester.run_stress_test(memory_config)
    results.append(memory_result)

    # I/O stress test
    io_config = create_io_stress_config(duration=30, file_size_mb=5)
    io_result = await stress_tester.run_stress_test(io_config)
    results.append(io_result)

    # Concurrent users test
    users_config = create_concurrent_users_config(duration=30, num_users=50)
    users_result = await stress_tester.run_stress_test(users_config)
    results.append(users_result)

    # Network stress test (if URL provided)
    if target_url:
        network_config = StressTestConfig(
            test_name="network_stress",
            test_type=StressTestType.NETWORK_STRESS,
            duration_seconds=30,
            concurrent_operations=20,
            custom_parameters={"target_url": target_url},
        )
        network_result = await stress_tester.run_stress_test(network_config)
        results.append(network_result)

    return results
