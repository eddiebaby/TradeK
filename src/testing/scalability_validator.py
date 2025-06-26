"""
Scalability Validation Framework for TradeKnowledge.

This module provides comprehensive scalability testing and validation
capabilities to ensure the system can handle growth in users, data, and load.
"""

import asyncio
import logging
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ScalabilityTestType(Enum):
    """Types of scalability tests"""

    HORIZONTAL_SCALING = "horizontal_scaling"
    VERTICAL_SCALING = "vertical_scaling"
    DATA_VOLUME_SCALING = "data_volume_scaling"
    USER_GROWTH_SIMULATION = "user_growth_simulation"
    THROUGHPUT_SCALING = "throughput_scaling"
    LATENCY_UNDER_LOAD = "latency_under_load"
    RESOURCE_EFFICIENCY = "resource_efficiency"


@dataclass
class ScalabilityTestConfig:
    """Configuration for scalability tests"""

    test_name: str
    test_type: ScalabilityTestType
    baseline_load: int = 10  # Starting load level
    max_load: int = 1000  # Maximum load to test
    scaling_steps: int = 10  # Number of scaling steps
    step_duration_seconds: int = 30  # Duration for each step
    warmup_duration_seconds: int = 10  # Warmup before measurements
    cooldown_duration_seconds: int = 5  # Cooldown between steps
    target_function: Callable | None = None  # Function to test
    success_criteria: dict[str, float] = field(default_factory=dict)
    scaling_pattern: str = "exponential"  # "linear", "exponential", "logarithmic"
    acceptable_degradation: float = 0.2  # 20% performance degradation threshold


@dataclass
class ScalabilityDataPoint:
    """Single data point in scalability test"""

    load_level: int
    timestamp: datetime
    response_time_ms: float
    throughput_ops_per_sec: float
    success_rate: float
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_usage_mb: float
    error_count: int
    concurrent_operations: int


@dataclass
class ScalabilityMetrics:
    """Comprehensive scalability metrics"""

    test_name: str
    test_type: ScalabilityTestType
    start_time: datetime
    end_time: datetime | None = None
    data_points: list[ScalabilityDataPoint] = field(default_factory=list)
    baseline_performance: ScalabilityDataPoint | None = None
    peak_performance: ScalabilityDataPoint | None = None
    breaking_point: ScalabilityDataPoint | None = None
    scaling_efficiency: float = 0.0  # How well it scales (0-1)
    linear_scalability_score: float = 0.0  # How close to linear scaling
    throughput_scaling_factor: float = 0.0  # Throughput increase factor
    latency_degradation_factor: float = 0.0  # Latency increase factor
    resource_efficiency_score: float = 0.0  # Resource usage efficiency


class ScalabilityResult:
    """Result of scalability validation"""

    def __init__(self, config: ScalabilityTestConfig, metrics: ScalabilityMetrics):
        self.config = config
        self.metrics = metrics
        self.scalability_grade = self._calculate_scalability_grade()
        self.recommendations = self._generate_recommendations()
        self.scaling_limits = self._identify_scaling_limits()

    def _calculate_scalability_grade(self) -> str:
        """Calculate overall scalability grade"""
        efficiency = self.metrics.scaling_efficiency

        if efficiency >= 0.9:
            return "A"  # Excellent scalability
        elif efficiency >= 0.8:
            return "B"  # Good scalability
        elif efficiency >= 0.7:
            return "C"  # Acceptable scalability
        elif efficiency >= 0.6:
            return "D"  # Poor scalability
        else:
            return "F"  # Unacceptable scalability

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on scalability analysis"""
        recommendations = []

        if self.metrics.scaling_efficiency < 0.7:
            recommendations.append(
                "System shows poor scalability. Consider architectural improvements, "
                "database optimization, or horizontal scaling strategies."
            )

        if self.metrics.latency_degradation_factor > 2.0:
            recommendations.append(
                "Latency increases significantly under load. Implement caching, "
                "optimize database queries, or consider load balancing."
            )

        if self.metrics.resource_efficiency_score < 0.6:
            recommendations.append(
                "Resource utilization is inefficient. Optimize algorithms, "
                "implement connection pooling, or review resource allocation."
            )

        if self.metrics.breaking_point:
            breaking_load = self.metrics.breaking_point.load_level
            recommendations.append(
                f"System reaches breaking point at {breaking_load} concurrent operations. "
                f"Plan capacity accordingly and implement circuit breakers."
            )

        if self.metrics.throughput_scaling_factor < 0.5:
            recommendations.append(
                "Throughput does not scale well. Consider asynchronous processing, "
                "microservices architecture, or database sharding."
            )

        if not recommendations:
            recommendations.append(
                "System demonstrates good scalability characteristics. "
                "Monitor performance as load increases in production."
            )

        return recommendations

    def _identify_scaling_limits(self) -> dict[str, Any]:
        """Identify scaling limits and bottlenecks"""
        limits = {
            "max_tested_load": max(
                [dp.load_level for dp in self.metrics.data_points], default=0
            ),
            "recommended_max_load": None,
            "cpu_bottleneck_load": None,
            "memory_bottleneck_load": None,
            "latency_bottleneck_load": None,
            "throughput_plateau_load": None,
        }

        # Find where CPU usage becomes problematic (>90%)
        for dp in self.metrics.data_points:
            if dp.cpu_usage_percent > 90 and limits["cpu_bottleneck_load"] is None:
                limits["cpu_bottleneck_load"] = dp.load_level

        # Find where memory usage becomes problematic (>90%)
        for dp in self.metrics.data_points:
            if (
                dp.memory_usage_percent > 90
                and limits["memory_bottleneck_load"] is None
            ):
                limits["memory_bottleneck_load"] = dp.load_level

        # Find where latency becomes unacceptable (>2x baseline)
        if self.metrics.baseline_performance:
            baseline_latency = self.metrics.baseline_performance.response_time_ms
            for dp in self.metrics.data_points:
                if (
                    dp.response_time_ms > baseline_latency * 2
                    and limits["latency_bottleneck_load"] is None
                ):
                    limits["latency_bottleneck_load"] = dp.load_level

        # Find recommended max load (80% of breaking point or last good performance)
        if self.metrics.breaking_point:
            limits["recommended_max_load"] = int(
                self.metrics.breaking_point.load_level * 0.8
            )
        else:
            # Use load where performance is still acceptable
            acceptable_points = [
                dp
                for dp in self.metrics.data_points
                if dp.success_rate > 0.95
                and dp.response_time_ms
                < (
                    self.metrics.baseline_performance.response_time_ms * 1.5
                    if self.metrics.baseline_performance
                    else 1000
                )
            ]
            if acceptable_points:
                limits["recommended_max_load"] = max(
                    [dp.load_level for dp in acceptable_points]
                )

        return limits


class ScalabilityValidator:
    """
    Comprehensive scalability validation framework that tests how well
    the system scales with increasing load, users, and data volume.
    """

    def __init__(self):
        self.running = False
        self._monitoring_task: asyncio.Task | None = None

    async def validate_scalability(
        self, config: ScalabilityTestConfig
    ) -> ScalabilityResult:
        """
        Run comprehensive scalability validation.

        Args:
            config: Scalability test configuration

        Returns:
            ScalabilityResult with analysis and recommendations
        """
        logger.info(f"Starting scalability validation: {config.test_name}")

        metrics = ScalabilityMetrics(
            test_name=config.test_name,
            test_type=config.test_type,
            start_time=datetime.now(),
        )

        self.running = True

        try:
            # Generate load levels based on scaling pattern
            load_levels = self._generate_load_levels(config)

            # Execute scalability test for each load level
            for i, load_level in enumerate(load_levels):
                logger.info(
                    f"Testing load level {load_level} ({i+1}/{len(load_levels)})"
                )

                # Warmup phase
                await self._warmup_phase(config, load_level)

                # Measurement phase
                data_point = await self._measure_performance_at_load(config, load_level)
                metrics.data_points.append(data_point)

                # Set baseline from first measurement
                if metrics.baseline_performance is None:
                    metrics.baseline_performance = data_point

                # Check for breaking point
                if self._is_breaking_point(data_point, config):
                    metrics.breaking_point = data_point
                    logger.warning(f"Breaking point reached at load level {load_level}")
                    break

                # Update peak performance
                if (
                    metrics.peak_performance is None
                    or data_point.throughput_ops_per_sec
                    > metrics.peak_performance.throughput_ops_per_sec
                ):
                    metrics.peak_performance = data_point

                # Cooldown between tests
                if i < len(load_levels) - 1:
                    await asyncio.sleep(config.cooldown_duration_seconds)

        finally:
            self.running = False
            metrics.end_time = datetime.now()

        # Analyze results
        self._analyze_scalability_metrics(metrics)

        result = ScalabilityResult(config, metrics)

        logger.info(
            f"Scalability validation completed: Grade {result.scalability_grade}"
        )
        logger.info(f"Scaling efficiency: {metrics.scaling_efficiency:.2f}")

        return result

    def _generate_load_levels(self, config: ScalabilityTestConfig) -> list[int]:
        """Generate load levels based on scaling pattern"""
        load_levels = []

        if config.scaling_pattern == "linear":
            step_size = (config.max_load - config.baseline_load) / config.scaling_steps
            for i in range(config.scaling_steps + 1):
                load_levels.append(int(config.baseline_load + i * step_size))

        elif config.scaling_pattern == "exponential":
            # Exponential growth: load_i = baseline * (max_load/baseline)^(i/steps)
            growth_factor = (config.max_load / config.baseline_load) ** (
                1 / config.scaling_steps
            )
            for i in range(config.scaling_steps + 1):
                load_level = int(config.baseline_load * (growth_factor**i))
                load_levels.append(min(load_level, config.max_load))

        elif config.scaling_pattern == "logarithmic":
            # Logarithmic growth
            for i in range(config.scaling_steps + 1):
                if i == 0:
                    load_levels.append(config.baseline_load)
                else:
                    progress = i / config.scaling_steps
                    log_progress = math.log(1 + progress * (math.e - 1))
                    load_level = int(
                        config.baseline_load
                        + (config.max_load - config.baseline_load) * log_progress
                    )
                    load_levels.append(load_level)

        else:
            # Default to linear
            load_levels = self._generate_load_levels(
                config._replace(scaling_pattern="linear")
            )

        return load_levels

    async def _warmup_phase(self, config: ScalabilityTestConfig, load_level: int):
        """Warmup phase before measurement"""
        if config.warmup_duration_seconds <= 0:
            return

        logger.debug(
            f"Warmup phase: {config.warmup_duration_seconds}s at load {load_level}"
        )

        # Run light load to warm up the system
        warmup_tasks = []
        for _ in range(min(load_level, 10)):  # Light warmup load
            if config.target_function:
                task = asyncio.create_task(
                    self._execute_target_function(config.target_function)
                )
                warmup_tasks.append(task)

        # Wait for warmup to complete
        if warmup_tasks:
            await asyncio.gather(*warmup_tasks, return_exceptions=True)
        else:
            await asyncio.sleep(config.warmup_duration_seconds)

    async def _measure_performance_at_load(
        self, config: ScalabilityTestConfig, load_level: int
    ) -> ScalabilityDataPoint:
        """Measure performance at specific load level"""
        start_time = time.time()
        timestamp = datetime.now()

        # Start resource monitoring
        resource_monitor = asyncio.create_task(
            self._monitor_resources_during_test(config.step_duration_seconds)
        )

        # Execute load test
        response_times = []
        success_count = 0
        error_count = 0

        # Create tasks for the specified load level
        tasks = []
        task_start_times = {}

        for i in range(load_level):
            if config.target_function:
                task = asyncio.create_task(
                    self._execute_target_function(config.target_function)
                )
            else:
                task = asyncio.create_task(self._default_load_function())

            tasks.append(task)
            task_start_times[task] = time.time()

        # Execute tasks with controlled timing
        completed_tasks = 0
        test_end_time = start_time + config.step_duration_seconds

        while time.time() < test_end_time and tasks:
            # Wait for some tasks to complete
            done, pending = await asyncio.wait(
                tasks, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
            )

            # Process completed tasks
            for task in done:
                try:
                    await task
                    success_count += 1
                    response_time = (time.time() - task_start_times[task]) * 1000
                    response_times.append(response_time)
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Task failed: {e}")

                completed_tasks += 1
                tasks.remove(task)

                # Start new task if still within test duration
                if time.time() < test_end_time:
                    if config.target_function:
                        new_task = asyncio.create_task(
                            self._execute_target_function(config.target_function)
                        )
                    else:
                        new_task = asyncio.create_task(self._default_load_function())

                    tasks.append(new_task)
                    task_start_times[new_task] = time.time()

        # Cancel remaining tasks
        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

        # Get resource usage
        resource_usage = await resource_monitor

        # Calculate metrics
        total_operations = success_count + error_count
        success_rate = success_count / total_operations if total_operations > 0 else 0
        avg_response_time = statistics.mean(response_times) if response_times else 0
        throughput = total_operations / config.step_duration_seconds

        return ScalabilityDataPoint(
            load_level=load_level,
            timestamp=timestamp,
            response_time_ms=avg_response_time,
            throughput_ops_per_sec=throughput,
            success_rate=success_rate,
            cpu_usage_percent=resource_usage["avg_cpu"],
            memory_usage_percent=resource_usage["avg_memory_percent"],
            memory_usage_mb=resource_usage["avg_memory_mb"],
            error_count=error_count,
            concurrent_operations=load_level,
        )

    async def _monitor_resources_during_test(
        self, duration_seconds: int
    ) -> dict[str, float]:
        """Monitor system resources during test"""
        import psutil

        cpu_samples = []
        memory_samples = []
        memory_mb_samples = []

        end_time = time.time() + duration_seconds

        while time.time() < end_time:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()

                cpu_samples.append(cpu_percent)
                memory_samples.append(memory.percent)
                memory_mb_samples.append(memory.used / 1024 / 1024)

                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error monitoring resources: {e}")

        return {
            "avg_cpu": statistics.mean(cpu_samples) if cpu_samples else 0,
            "avg_memory_percent": (
                statistics.mean(memory_samples) if memory_samples else 0
            ),
            "avg_memory_mb": (
                statistics.mean(memory_mb_samples) if memory_mb_samples else 0
            ),
        }

    async def _execute_target_function(self, target_function: Callable) -> Any:
        """Execute the target function being tested"""
        if asyncio.iscoroutinefunction(target_function):
            return await target_function()
        else:
            return target_function()

    async def _default_load_function(self) -> str:
        """Default load function when no target function provided"""
        # Simulate some work
        await asyncio.sleep(0.01 + (time.time() % 0.1))
        return "default_result"

    def _is_breaking_point(
        self, data_point: ScalabilityDataPoint, config: ScalabilityTestConfig
    ) -> bool:
        """Determine if this data point represents a breaking point"""
        # Check multiple criteria for breaking point
        breaking_conditions = [
            data_point.success_rate < 0.5,  # More than 50% failures
            data_point.cpu_usage_percent > 95,  # CPU maxed out
            data_point.memory_usage_percent > 95,  # Memory maxed out
            data_point.error_count > data_point.load_level * 0.5,  # High error rate
        ]

        # Also check against acceptable degradation
        if hasattr(config, "baseline_performance") and config.baseline_performance:
            baseline_response_time = config.baseline_performance.response_time_ms
            if data_point.response_time_ms > baseline_response_time * (
                1 + config.acceptable_degradation * 2
            ):
                breaking_conditions.append(True)

        return any(breaking_conditions)

    def _analyze_scalability_metrics(self, metrics: ScalabilityMetrics):
        """Analyze collected metrics and calculate scalability scores"""
        if len(metrics.data_points) < 2:
            return

        baseline = metrics.baseline_performance
        if not baseline:
            return

        # Calculate scaling efficiency
        # Compare actual throughput scaling to ideal linear scaling
        max_load_point = max(metrics.data_points, key=lambda dp: dp.load_level)
        load_increase_factor = max_load_point.load_level / baseline.load_level
        throughput_increase_factor = (
            max_load_point.throughput_ops_per_sec / baseline.throughput_ops_per_sec
        )

        metrics.throughput_scaling_factor = throughput_increase_factor
        metrics.scaling_efficiency = min(
            1.0, throughput_increase_factor / load_increase_factor
        )

        # Calculate linear scalability score
        # How close is the scaling to linear (perfect would be 1.0)
        ideal_throughputs = []
        actual_throughputs = []

        for dp in metrics.data_points:
            load_factor = dp.load_level / baseline.load_level
            ideal_throughput = baseline.throughput_ops_per_sec * load_factor
            ideal_throughputs.append(ideal_throughput)
            actual_throughputs.append(dp.throughput_ops_per_sec)

        if ideal_throughputs and actual_throughputs:
            # Calculate correlation coefficient as linear scalability score
            try:
                correlation = self._calculate_correlation(
                    ideal_throughputs, actual_throughputs
                )
                metrics.linear_scalability_score = max(0, correlation)
            except:
                metrics.linear_scalability_score = 0.0

        # Calculate latency degradation factor
        max_latency = max([dp.response_time_ms for dp in metrics.data_points])
        metrics.latency_degradation_factor = max_latency / baseline.response_time_ms

        # Calculate resource efficiency score
        # How well are resources utilized relative to throughput increase
        max_cpu = max([dp.cpu_usage_percent for dp in metrics.data_points])
        max_memory = max([dp.memory_usage_percent for dp in metrics.data_points])

        resource_utilization = (max_cpu + max_memory) / 200  # Normalize to 0-1
        metrics.resource_efficiency_score = min(
            1.0, throughput_increase_factor / max(1, resource_utilization)
        )

    def _calculate_correlation(self, x: list[float], y: list[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))

        denominator = math.sqrt(sum_sq_x * sum_sq_y)

        return numerator / denominator if denominator != 0 else 0.0


# Convenience functions for common scalability tests
def create_throughput_scaling_config(
    target_function: Callable, baseline_load: int = 10, max_load: int = 200
) -> ScalabilityTestConfig:
    """Create throughput scaling test configuration"""
    return ScalabilityTestConfig(
        test_name="throughput_scaling",
        test_type=ScalabilityTestType.THROUGHPUT_SCALING,
        baseline_load=baseline_load,
        max_load=max_load,
        scaling_steps=8,
        step_duration_seconds=30,
        target_function=target_function,
        scaling_pattern="exponential",
    )


def create_user_growth_config(
    target_function: Callable, baseline_users: int = 50, max_users: int = 1000
) -> ScalabilityTestConfig:
    """Create user growth simulation configuration"""
    return ScalabilityTestConfig(
        test_name="user_growth_simulation",
        test_type=ScalabilityTestType.USER_GROWTH_SIMULATION,
        baseline_load=baseline_users,
        max_load=max_users,
        scaling_steps=10,
        step_duration_seconds=45,
        target_function=target_function,
        scaling_pattern="exponential",
    )


def create_latency_under_load_config(
    target_function: Callable, baseline_load: int = 10, max_load: int = 500
) -> ScalabilityTestConfig:
    """Create latency under load test configuration"""
    return ScalabilityTestConfig(
        test_name="latency_under_load",
        test_type=ScalabilityTestType.LATENCY_UNDER_LOAD,
        baseline_load=baseline_load,
        max_load=max_load,
        scaling_steps=12,
        step_duration_seconds=20,
        target_function=target_function,
        acceptable_degradation=0.5,  # Allow 50% latency increase
        scaling_pattern="linear",
    )


async def run_comprehensive_scalability_suite(
    target_function: Callable,
) -> list[ScalabilityResult]:
    """
    Run comprehensive scalability validation suite.

    Args:
        target_function: Function to test for scalability

    Returns:
        List of scalability results
    """
    validator = ScalabilityValidator()
    results = []

    # Throughput scaling test
    throughput_config = create_throughput_scaling_config(target_function)
    throughput_result = await validator.validate_scalability(throughput_config)
    results.append(throughput_result)

    # User growth simulation
    user_growth_config = create_user_growth_config(
        target_function, baseline_users=25, max_users=500
    )
    user_growth_result = await validator.validate_scalability(user_growth_config)
    results.append(user_growth_result)

    # Latency under load test
    latency_config = create_latency_under_load_config(target_function)
    latency_result = await validator.validate_scalability(latency_config)
    results.append(latency_result)

    return results
