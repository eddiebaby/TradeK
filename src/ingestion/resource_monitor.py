"""
Resource Monitor for TradeKnowledge

Monitors system resources during PDF processing to prevent crashes.
"""

import logging
import psutil
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ResourceLimits:
    """Resource limits configuration"""
    max_memory_percent: float = 80.0  # Maximum memory usage percentage
    max_memory_mb: int = 2048  # Maximum memory in MB for WSL2
    check_interval: float = 5.0  # Check every 5 seconds
    warning_threshold: float = 70.0  # Warn at 70% memory usage

class ResourceMonitor:
    """
    Monitors system resources during processing to prevent crashes.
    
    Features:
    - Memory usage monitoring
    - Automatic process pausing when limits are approached
    - Resource cleanup recommendations
    - WSL2-specific optimizations
    """
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        """Initialize resource monitor"""
        self.limits = limits or ResourceLimits()
        self.monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []
        
        # Track resource usage over time
        self.usage_history: List[Dict[str, Any]] = []
        self.max_history_size = 50  # Keep last 50 measurements
        
        # Current status
        self.current_memory_percent = 0.0
        self.current_memory_mb = 0.0
        self.is_memory_critical = False
        self.warnings_issued = 0
        
    async def start_monitoring(self):
        """Start resource monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Resource monitoring started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")
    
    def add_callback(self, callback):
        """Add callback for resource events"""
        self._callbacks.append(callback)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'system_total_mb': memory.total // 1024 // 1024,
            'system_available_mb': memory.available // 1024 // 1024,
            'system_used_percent': memory.percent,
            'process_memory_mb': process.memory_info().rss // 1024 // 1024,
            'process_memory_percent': process.memory_percent(),
            'timestamp': datetime.now()
        }
    
    def check_memory_limits(self) -> Dict[str, Any]:
        """Check if memory limits are being approached"""
        usage = self.get_memory_usage()
        
        # Update current status
        self.current_memory_percent = usage['system_used_percent']
        self.current_memory_mb = usage['process_memory_mb']
        
        # Add to history
        self.usage_history.append(usage)
        if len(self.usage_history) > self.max_history_size:
            self.usage_history.pop(0)
        
        # Check limits
        memory_critical = (
            usage['system_used_percent'] > self.limits.max_memory_percent or
            usage['process_memory_mb'] > self.limits.max_memory_mb
        )
        
        memory_warning = usage['system_used_percent'] > self.limits.warning_threshold
        
        result = {
            'memory_ok': not memory_critical,
            'memory_warning': memory_warning,
            'memory_critical': memory_critical,
            'usage': usage,
            'recommendations': []
        }
        
        # Generate recommendations
        if memory_critical:
            result['recommendations'].extend([
                'Stop processing immediately',
                'Clear caches and run garbage collection',
                'Consider reducing batch sizes',
                'Close other applications'
            ])
        elif memory_warning:
            result['recommendations'].extend([
                'Consider reducing batch sizes',
                'Enable aggressive garbage collection',
                'Monitor closely'
            ])
        
        # Update critical status
        self.is_memory_critical = memory_critical
        
        return result
    
    async def wait_for_memory_available(self, timeout: float = 60.0):
        """Wait for memory to become available"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            check = self.check_memory_limits()
            
            if check['memory_ok']:
                return True
            
            logger.warning(f"Memory usage critical: {self.current_memory_percent:.1f}% - waiting...")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Wait before checking again
            await asyncio.sleep(self.limits.check_interval)
        
        logger.error(f"Memory did not become available within {timeout} seconds")
        return False
    
    def get_processing_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for processing large files"""
        memory_info = self.get_memory_usage()
        
        # Calculate safe batch sizes based on available memory
        available_mb = memory_info['system_available_mb']
        
        # Conservative batch sizes for different operations
        pdf_batch_size = max(5, min(20, available_mb // 50))  # 50MB per batch
        embedding_batch_size = max(10, min(100, available_mb // 20))  # 20MB per batch
        
        recommendations = {
            'pdf_batch_size': pdf_batch_size,
            'embedding_batch_size': embedding_batch_size,
            'enable_aggressive_gc': available_mb < 1000,
            'pause_between_batches': available_mb < 500,
            'memory_status': 'critical' if available_mb < 500 else 'warning' if available_mb < 1000 else 'ok'
        }
        
        return recommendations
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                check = self.check_memory_limits()
                
                # Call registered callbacks
                for callback in self._callbacks:
                    try:
                        await callback(check)
                    except Exception as e:
                        logger.error(f"Error in resource monitor callback: {e}")
                
                # Log warnings
                if check['memory_critical'] and self.warnings_issued % 5 == 0:  # Every 5th check
                    logger.error(f"CRITICAL: Memory usage at {self.current_memory_percent:.1f}%")
                    logger.error(f"Recommendations: {', '.join(check['recommendations'])}")
                    
                elif check['memory_warning'] and self.warnings_issued % 10 == 0:  # Every 10th check
                    logger.warning(f"Memory usage at {self.current_memory_percent:.1f}%")
                
                if check['memory_warning'] or check['memory_critical']:
                    self.warnings_issued += 1
                
                await asyncio.sleep(self.limits.check_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(self.limits.check_interval)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage during processing"""
        if not self.usage_history:
            return {'error': 'No usage data available'}
        
        memory_values = [entry['system_used_percent'] for entry in self.usage_history]
        process_memory_values = [entry['process_memory_mb'] for entry in self.usage_history]
        
        return {
            'duration_minutes': (
                self.usage_history[-1]['timestamp'] - self.usage_history[0]['timestamp']
            ).total_seconds() / 60,
            'memory_stats': {
                'min_percent': min(memory_values),
                'max_percent': max(memory_values),
                'avg_percent': sum(memory_values) / len(memory_values),
                'final_percent': memory_values[-1]
            },
            'process_memory_stats': {
                'min_mb': min(process_memory_values),
                'max_mb': max(process_memory_values),
                'avg_mb': sum(process_memory_values) / len(process_memory_values),
                'final_mb': process_memory_values[-1]
            },
            'warnings_issued': self.warnings_issued,
            'peak_memory_exceeded': max(memory_values) > self.limits.max_memory_percent
        }

# Global monitor instance
_global_monitor: Optional[ResourceMonitor] = None

def get_resource_monitor() -> ResourceMonitor:
    """Get global resource monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ResourceMonitor()
    return _global_monitor

async def monitor_processing(func, *args, **kwargs):
    """Decorator/wrapper to monitor resource usage during processing"""
    monitor = get_resource_monitor()
    
    # Start monitoring
    await monitor.start_monitoring()
    
    try:
        # Check if memory is available before starting
        if not await monitor.wait_for_memory_available(timeout=30):
            raise RuntimeError("Insufficient memory to start processing")
        
        # Run the function
        result = await func(*args, **kwargs)
        
        return result
        
    finally:
        # Stop monitoring and get summary
        await monitor.stop_monitoring()
        summary = monitor.get_usage_summary()
        logger.info(f"Processing completed. Memory usage summary: {summary}")