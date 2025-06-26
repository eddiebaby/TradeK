"""
MCP Monitoring and Metrics Tools for Comprehensive Observability

These tools provide advanced monitoring, metrics collection, alerting,
and observability capabilities for production systems.
"""

import asyncio
import json
import time
import statistics
import sqlite3
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import numpy as np
import psutil
import requests
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, Summary


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition and state."""
    alert_id: str
    name: str
    description: str
    severity: str
    condition: str
    threshold: Union[int, float]
    duration: int  # seconds
    state: str  # "pending", "firing", "resolved"
    triggered_at: Optional[float] = None
    resolved_at: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check definition and result."""
    check_id: str
    name: str
    description: str
    check_type: str
    endpoint: Optional[str] = None
    timeout: int = 30
    interval: int = 60
    retries: int = 3
    last_result: Optional[Dict[str, Any]] = None
    status: str = "unknown"  # "healthy", "unhealthy", "unknown"


@dataclass
class PerformanceProfile:
    """Performance profiling data."""
    profile_id: str
    function_name: str
    execution_time: float
    memory_usage: int
    cpu_usage: float
    call_count: int
    peak_memory: int
    gc_collections: int
    timestamp: float


class MetricsCollector(ABC):
    """Abstract base class for metrics collectors."""
    
    @abstractmethod
    async def collect(self) -> List[MetricPoint]:
        """Collect metrics and return data points."""
        pass
    
    @abstractmethod
    def get_collector_info(self) -> Dict[str, Any]:
        """Get information about this collector."""
        pass


class AlertManager(ABC):
    """Abstract base class for alert management."""
    
    @abstractmethod
    async def evaluate_alerts(self, metrics: List[MetricPoint]) -> List[Alert]:
        """Evaluate alert conditions against metrics."""
        pass
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert notification."""
        pass


class MCPMonitoringMetrics:
    """Comprehensive monitoring and metrics system."""
    
    def __init__(self):
        self.metrics_collectors: Dict[str, MetricsCollector] = {}
        self.alert_managers: Dict[str, AlertManager] = {}
        self.metric_storage = self._initialize_metric_storage()
        self.alert_rules: List[Dict[str, Any]] = []
        self.health_checks: Dict[str, HealthCheck] = {}
        self.performance_profiles: List[PerformanceProfile] = []
        
        # Prometheus registry for custom metrics
        self.prometheus_registry = CollectorRegistry()
        self.custom_metrics: Dict[str, Any] = {}
        
        # System monitoring
        self.system_metrics_enabled = True
        self.application_metrics_enabled = True
        self.business_metrics_enabled = True
        
        # Initialize built-in collectors and alert managers
        self._register_builtin_collectors()
        self._register_builtin_alert_managers()
    
    async def comprehensive_monitoring_setup(self,
                                           application_spec: Dict[str, Any],
                                           monitoring_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Setup comprehensive monitoring for application and infrastructure.
        
        Args:
            application_spec: Application monitoring configuration
            monitoring_requirements: Monitoring requirements and SLAs
            
        Returns:
            Dict[str, Any]: Complete monitoring setup configuration
        """
        monitoring_config = {
            "setup_timestamp": time.time(),
            "application_spec": application_spec,
            "monitoring_requirements": monitoring_requirements,
            "metrics_collection": await self._setup_metrics_collection(application_spec, monitoring_requirements),
            "alerting_rules": await self._setup_alerting_rules(application_spec, monitoring_requirements),
            "health_monitoring": await self._setup_health_monitoring(application_spec),
            "performance_monitoring": await self._setup_performance_monitoring(application_spec),
            "business_metrics": await self._setup_business_metrics(application_spec),
            "dashboards": await self._generate_monitoring_dashboards(application_spec, monitoring_requirements),
            "log_aggregation": await self._setup_log_aggregation(application_spec),
            "distributed_tracing": await self._setup_distributed_tracing(application_spec),
            "anomaly_detection": await self._setup_anomaly_detection(application_spec, monitoring_requirements),
            "capacity_planning": await self._setup_capacity_planning_monitoring(application_spec)
        }
        
        return monitoring_config
    
    async def real_time_metrics_collection(self,
                                         collection_interval: int = 60,
                                         duration: int = 3600) -> Dict[str, Any]:
        """
        Collect real-time metrics across all registered collectors.
        
        Args:
            collection_interval: Interval between collections in seconds
            duration: Total duration to collect metrics in seconds
            
        Returns:
            Dict[str, Any]: Collected metrics and analysis
        """
        collection_start = time.time()
        collected_metrics: List[MetricPoint] = []
        collection_stats = {
            "start_time": collection_start,
            "end_time": 0,
            "total_points": 0,
            "collections": 0,
            "errors": 0,
            "collector_performance": {}
        }
        
        while time.time() - collection_start < duration:
            collection_round_start = time.time()
            
            # Collect from all registered collectors
            for collector_name, collector in self.metrics_collectors.items():
                try:
                    collector_start = time.time()
                    points = await collector.collect()
                    collector_duration = time.time() - collector_start
                    
                    collected_metrics.extend(points)
                    
                    # Track collector performance
                    if collector_name not in collection_stats["collector_performance"]:
                        collection_stats["collector_performance"][collector_name] = {
                            "total_duration": 0,
                            "collections": 0,
                            "points_collected": 0,
                            "errors": 0
                        }
                    
                    perf_stats = collection_stats["collector_performance"][collector_name]
                    perf_stats["total_duration"] += collector_duration
                    perf_stats["collections"] += 1
                    perf_stats["points_collected"] += len(points)
                    
                except Exception as e:
                    collection_stats["errors"] += 1
                    if collector_name in collection_stats["collector_performance"]:
                        collection_stats["collector_performance"][collector_name]["errors"] += 1
            
            collection_stats["collections"] += 1
            
            # Store metrics
            await self._store_metrics_batch(collected_metrics[-100:])  # Store last 100 points
            
            # Evaluate alerts
            alerts = await self._evaluate_all_alerts(collected_metrics[-100:])
            await self._process_alerts(alerts)
            
            # Wait for next collection interval
            collection_duration = time.time() - collection_round_start
            sleep_time = max(0, collection_interval - collection_duration)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        collection_stats["end_time"] = time.time()
        collection_stats["total_points"] = len(collected_metrics)
        
        # Analyze collected metrics
        analysis = await self._analyze_metrics_collection(collected_metrics, collection_stats)
        
        return {
            "collection_stats": collection_stats,
            "metrics_analysis": analysis,
            "alerts_generated": await self._get_recent_alerts(duration),
            "performance_insights": await self._generate_performance_insights(collected_metrics),
            "recommendations": await self._generate_monitoring_recommendations(analysis)
        }
    
    async def advanced_alerting_system(self,
                                     alert_rules: List[Dict[str, Any]],
                                     notification_channels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Setup advanced alerting system with intelligent routing.
        
        Args:
            alert_rules: Alert rule definitions
            notification_channels: Notification channel configurations
            
        Returns:
            Dict[str, Any]: Alerting system configuration and status
        """
        alerting_config = {
            "setup_timestamp": time.time(),
            "alert_rules": [],
            "notification_channels": [],
            "alert_routing": {},
            "escalation_policies": {},
            "suppression_rules": {},
            "alert_correlation": {},
            "intelligent_grouping": {}
        }
        
        # Process and validate alert rules
        for rule in alert_rules:
            processed_rule = await self._process_alert_rule(rule)
            alerting_config["alert_rules"].append(processed_rule)
        
        # Setup notification channels
        for channel in notification_channels:
            channel_config = await self._setup_notification_channel(channel)
            alerting_config["notification_channels"].append(channel_config)
        
        # Configure alert routing
        alerting_config["alert_routing"] = await self._configure_alert_routing(
            alert_rules, notification_channels
        )
        
        # Setup escalation policies
        alerting_config["escalation_policies"] = await self._setup_escalation_policies(
            alert_rules, notification_channels
        )
        
        # Configure alert suppression
        alerting_config["suppression_rules"] = await self._configure_alert_suppression()
        
        # Setup alert correlation
        alerting_config["alert_correlation"] = await self._setup_alert_correlation()
        
        # Configure intelligent grouping
        alerting_config["intelligent_grouping"] = await self._setup_intelligent_alert_grouping()
        
        return alerting_config
    
    async def performance_profiling_analysis(self,
                                           target_functions: List[str],
                                           profiling_duration: int = 300,
                                           sampling_rate: float = 0.1) -> Dict[str, Any]:
        """
        Perform detailed performance profiling and analysis.
        
        Args:
            target_functions: Functions to profile
            profiling_duration: Duration of profiling in seconds
            sampling_rate: Sampling rate for profiling
            
        Returns:
            Dict[str, Any]: Performance profiling results
        """
        profiling_start = time.time()
        profiling_results = {
            "profiling_metadata": {
                "start_time": profiling_start,
                "duration": profiling_duration,
                "sampling_rate": sampling_rate,
                "target_functions": target_functions
            },
            "function_profiles": [],
            "memory_analysis": {},
            "cpu_analysis": {},
            "io_analysis": {},
            "performance_bottlenecks": [],
            "optimization_recommendations": []
        }
        
        # Profile each target function
        for function_name in target_functions:
            profile = await self._profile_function_performance(
                function_name, profiling_duration, sampling_rate
            )
            profiling_results["function_profiles"].append(profile)
        
        # Analyze memory usage patterns
        profiling_results["memory_analysis"] = await self._analyze_memory_patterns(
            profiling_results["function_profiles"]
        )
        
        # Analyze CPU usage patterns
        profiling_results["cpu_analysis"] = await self._analyze_cpu_patterns(
            profiling_results["function_profiles"]
        )
        
        # Analyze I/O patterns
        profiling_results["io_analysis"] = await self._analyze_io_patterns(
            profiling_results["function_profiles"]
        )
        
        # Identify performance bottlenecks
        profiling_results["performance_bottlenecks"] = await self._identify_performance_bottlenecks(
            profiling_results["function_profiles"]
        )
        
        # Generate optimization recommendations
        profiling_results["optimization_recommendations"] = await self._generate_optimization_recommendations(
            profiling_results
        )
        
        return profiling_results
    
    async def business_metrics_tracking(self,
                                      business_kpis: List[Dict[str, Any]],
                                      data_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Setup business metrics tracking and KPI monitoring.
        
        Args:
            business_kpis: Business KPI definitions
            data_sources: Data sources for business metrics
            
        Returns:
            Dict[str, Any]: Business metrics configuration
        """
        business_config = {
            "setup_timestamp": time.time(),
            "kpi_definitions": [],
            "data_source_configs": [],
            "metric_calculations": {},
            "trend_analysis": {},
            "forecasting_models": {},
            "business_dashboards": {},
            "executive_reports": {}
        }
        
        # Process KPI definitions
        for kpi in business_kpis:
            kpi_config = await self._process_business_kpi(kpi)
            business_config["kpi_definitions"].append(kpi_config)
        
        # Setup data sources
        for source in data_sources:
            source_config = await self._setup_business_data_source(source)
            business_config["data_source_configs"].append(source_config)
        
        # Configure metric calculations
        business_config["metric_calculations"] = await self._configure_business_metric_calculations(
            business_kpis, data_sources
        )
        
        # Setup trend analysis
        business_config["trend_analysis"] = await self._setup_business_trend_analysis(business_kpis)
        
        # Configure forecasting models
        business_config["forecasting_models"] = await self._setup_business_forecasting(business_kpis)
        
        # Generate business dashboards
        business_config["business_dashboards"] = await self._generate_business_dashboards(business_kpis)
        
        # Setup executive reporting
        business_config["executive_reports"] = await self._setup_executive_reporting(business_kpis)
        
        return business_config
    
    async def anomaly_detection_system(self,
                                     metrics_patterns: List[Dict[str, Any]],
                                     detection_algorithms: List[str],
                                     sensitivity_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Setup intelligent anomaly detection system.
        
        Args:
            metrics_patterns: Historical metrics patterns
            detection_algorithms: Algorithms to use for detection
            sensitivity_settings: Sensitivity configuration
            
        Returns:
            Dict[str, Any]: Anomaly detection configuration
        """
        anomaly_config = {
            "setup_timestamp": time.time(),
            "detection_models": {},
            "baseline_patterns": {},
            "anomaly_thresholds": {},
            "correlation_analysis": {},
            "predictive_models": {},
            "anomaly_alerts": {},
            "false_positive_reduction": {}
        }
        
        # Train detection models
        for algorithm in detection_algorithms:
            model = await self._train_anomaly_detection_model(
                algorithm, metrics_patterns, sensitivity_settings
            )
            anomaly_config["detection_models"][algorithm] = model
        
        # Establish baseline patterns
        anomaly_config["baseline_patterns"] = await self._establish_baseline_patterns(metrics_patterns)
        
        # Configure anomaly thresholds
        anomaly_config["anomaly_thresholds"] = await self._configure_anomaly_thresholds(
            metrics_patterns, sensitivity_settings
        )
        
        # Setup correlation analysis
        anomaly_config["correlation_analysis"] = await self._setup_anomaly_correlation_analysis()
        
        # Train predictive models
        anomaly_config["predictive_models"] = await self._train_predictive_anomaly_models(metrics_patterns)
        
        # Configure anomaly alerts
        anomaly_config["anomaly_alerts"] = await self._configure_anomaly_alerts(sensitivity_settings)
        
        # Setup false positive reduction
        anomaly_config["false_positive_reduction"] = await self._setup_false_positive_reduction()
        
        return anomaly_config
    
    async def capacity_planning_analytics(self,
                                        historical_metrics: List[MetricPoint],
                                        growth_projections: Dict[str, Any],
                                        resource_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform capacity planning analysis and recommendations.
        
        Args:
            historical_metrics: Historical metrics data
            growth_projections: Expected growth patterns
            resource_constraints: Resource limitations and constraints
            
        Returns:
            Dict[str, Any]: Capacity planning analysis
        """
        capacity_analysis = {
            "analysis_timestamp": time.time(),
            "current_utilization": {},
            "trend_analysis": {},
            "growth_projections": {},
            "capacity_recommendations": {},
            "scaling_strategies": {},
            "cost_projections": {},
            "risk_assessment": {}
        }
        
        # Analyze current utilization
        capacity_analysis["current_utilization"] = await self._analyze_current_utilization(historical_metrics)
        
        # Perform trend analysis
        capacity_analysis["trend_analysis"] = await self._analyze_capacity_trends(historical_metrics)
        
        # Project growth scenarios
        capacity_analysis["growth_projections"] = await self._project_capacity_growth(
            historical_metrics, growth_projections
        )
        
        # Generate capacity recommendations
        capacity_analysis["capacity_recommendations"] = await self._generate_capacity_recommendations(
            capacity_analysis["growth_projections"], resource_constraints
        )
        
        # Develop scaling strategies
        capacity_analysis["scaling_strategies"] = await self._develop_scaling_strategies(
            capacity_analysis["capacity_recommendations"]
        )
        
        # Project costs
        capacity_analysis["cost_projections"] = await self._project_capacity_costs(
            capacity_analysis["scaling_strategies"]
        )
        
        # Assess risks
        capacity_analysis["risk_assessment"] = await self._assess_capacity_risks(
            capacity_analysis["growth_projections"], resource_constraints
        )
        
        return capacity_analysis
    
    # Core monitoring setup methods
    
    async def _setup_metrics_collection(self, app_spec: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Setup comprehensive metrics collection."""
        return {
            "system_metrics": await self._configure_system_metrics_collection(),
            "application_metrics": await self._configure_application_metrics_collection(app_spec),
            "custom_metrics": await self._configure_custom_metrics_collection(app_spec, requirements),
            "business_metrics": await self._configure_business_metrics_collection(app_spec),
            "security_metrics": await self._configure_security_metrics_collection(app_spec),
            "performance_metrics": await self._configure_performance_metrics_collection(app_spec),
            "infrastructure_metrics": await self._configure_infrastructure_metrics_collection(app_spec)
        }
    
    async def _setup_alerting_rules(self, app_spec: Dict[str, Any], requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Setup comprehensive alerting rules."""
        rules = []
        
        # System resource alerts
        rules.extend(await self._generate_system_resource_alerts(requirements))
        
        # Application performance alerts
        rules.extend(await self._generate_application_performance_alerts(app_spec, requirements))
        
        # Error rate alerts
        rules.extend(await self._generate_error_rate_alerts(app_spec, requirements))
        
        # Security alerts
        rules.extend(await self._generate_security_alerts(app_spec, requirements))
        
        # Business metrics alerts
        rules.extend(await self._generate_business_metrics_alerts(app_spec, requirements))
        
        # Custom alerts
        if requirements.get("custom_alerts"):
            rules.extend(await self._generate_custom_alerts(requirements["custom_alerts"]))
        
        return rules
    
    async def _setup_health_monitoring(self, app_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Setup comprehensive health monitoring."""
        health_config = {
            "endpoint_health_checks": [],
            "dependency_health_checks": [],
            "deep_health_checks": [],
            "synthetic_monitoring": [],
            "health_aggregation": {}
        }
        
        # Setup endpoint health checks
        if app_spec.get("endpoints"):
            for endpoint in app_spec["endpoints"]:
                health_check = await self._create_endpoint_health_check(endpoint)
                health_config["endpoint_health_checks"].append(health_check)
        
        # Setup dependency health checks
        if app_spec.get("dependencies"):
            for dependency in app_spec["dependencies"]:
                health_check = await self._create_dependency_health_check(dependency)
                health_config["dependency_health_checks"].append(health_check)
        
        # Setup deep health checks
        health_config["deep_health_checks"] = await self._setup_deep_health_checks(app_spec)
        
        # Setup synthetic monitoring
        health_config["synthetic_monitoring"] = await self._setup_synthetic_monitoring(app_spec)
        
        # Configure health aggregation
        health_config["health_aggregation"] = await self._configure_health_aggregation(app_spec)
        
        return health_config
    
    # Metrics storage and analysis
    
    def _initialize_metric_storage(self) -> sqlite3.Connection:
        """Initialize metrics storage database."""
        db_path = Path("agents/data/metrics.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(db_path))
        
        # Create metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                metric_name TEXT,
                value REAL,
                labels TEXT,
                metadata TEXT
            )
        """)
        
        # Create alerts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE,
                name TEXT,
                severity TEXT,
                state TEXT,
                triggered_at REAL,
                resolved_at REAL,
                labels TEXT,
                annotations TEXT
            )
        """)
        
        # Create health checks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS health_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_id TEXT,
                timestamp REAL,
                status TEXT,
                result TEXT
            )
        """)
        
        conn.commit()
        return conn
    
    async def _store_metrics_batch(self, metrics: List[MetricPoint]):
        """Store batch of metrics in database."""
        if not metrics:
            return
        
        cursor = self.metric_storage.cursor()
        
        for metric in metrics:
            cursor.execute("""
                INSERT INTO metrics (timestamp, metric_name, value, labels, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                metric.timestamp,
                "metric",  # Would extract actual metric name
                metric.value,
                json.dumps(metric.labels),
                json.dumps(metric.metadata)
            ))
        
        self.metric_storage.commit()
    
    async def _analyze_metrics_collection(self, metrics: List[MetricPoint], stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze collected metrics and provide insights."""
        if not metrics:
            return {"status": "no_metrics"}
        
        values = [m.value for m in metrics if isinstance(m.value, (int, float))]
        
        analysis = {
            "total_points": len(metrics),
            "time_range": {
                "start": min(m.timestamp for m in metrics),
                "end": max(m.timestamp for m in metrics)
            },
            "value_statistics": {
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "mean": statistics.mean(values) if values else 0,
                "median": statistics.median(values) if values else 0,
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0
            },
            "trend_analysis": await self._analyze_metric_trends(metrics),
            "anomaly_detection": await self._detect_metric_anomalies(metrics),
            "quality_assessment": await self._assess_metric_quality(stats)
        }
        
        return analysis
    
    # Built-in collectors and alert managers
    
    def _register_builtin_collectors(self):
        """Register built-in metrics collectors."""
        # Would register actual collector implementations
        pass
    
    def _register_builtin_alert_managers(self):
        """Register built-in alert managers."""
        # Would register actual alert manager implementations
        pass
    
    # Placeholder implementations for complex methods
    # These would be fully implemented in a production system
    
    async def _evaluate_all_alerts(self, metrics: List[MetricPoint]) -> List[Alert]:
        """Evaluate all alert rules against current metrics."""
        return []
    
    async def _process_alerts(self, alerts: List[Alert]):
        """Process and send alerts."""
        pass
    
    async def _get_recent_alerts(self, duration: int) -> List[Alert]:
        """Get recent alerts within specified duration."""
        return []
    
    async def _generate_performance_insights(self, metrics: List[MetricPoint]) -> Dict[str, Any]:
        """Generate performance insights from metrics."""
        return {"insights": []}
    
    async def _generate_monitoring_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate monitoring recommendations based on analysis."""
        return ["Increase metrics collection frequency", "Add more detailed labels"]
    
    async def _analyze_metric_trends(self, metrics: List[MetricPoint]) -> Dict[str, Any]:
        """Analyze trends in metric data."""
        return {"trend": "stable"}
    
    async def _detect_metric_anomalies(self, metrics: List[MetricPoint]) -> Dict[str, Any]:
        """Detect anomalies in metric data."""
        return {"anomalies": []}
    
    async def _assess_metric_quality(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of metric collection."""
        return {"quality_score": 8.5}


# Built-in system metrics collector
class SystemMetricsCollector(MetricsCollector):
    """Collector for system-level metrics."""
    
    async def collect(self) -> List[MetricPoint]:
        """Collect system metrics."""
        timestamp = time.time()
        metrics = []
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(MetricPoint(
            timestamp=timestamp,
            value=cpu_percent,
            labels={"metric": "cpu_usage_percent", "type": "system"}
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(MetricPoint(
            timestamp=timestamp,
            value=memory.percent,
            labels={"metric": "memory_usage_percent", "type": "system"}
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.append(MetricPoint(
            timestamp=timestamp,
            value=disk.percent,
            labels={"metric": "disk_usage_percent", "type": "system"}
        ))
        
        return metrics
    
    def get_collector_info(self) -> Dict[str, Any]:
        """Get collector information."""
        return {
            "name": "SystemMetricsCollector",
            "version": "1.0.0",
            "metrics": ["cpu_usage_percent", "memory_usage_percent", "disk_usage_percent"],
            "collection_interval": 60
        }


# Global monitoring metrics instance
monitoring_metrics = MCPMonitoringMetrics()