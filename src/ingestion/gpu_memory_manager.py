"""
GPU Memory Management for NVIDIA 1080 Optimization

This module provides intelligent GPU memory management for optimal
embedding generation performance on desktop GPUs.
"""

import asyncio
import logging
import platform
from dataclasses import dataclass
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU information and capabilities"""

    name: str
    memory_total_gb: float
    memory_available_gb: float
    compute_capability: str
    driver_version: str
    cuda_version: str


@dataclass
class BatchSizeConfig:
    """Optimal batch sizes for different chunk types"""

    small_chunks: int  # < 200 characters
    medium_chunks: int  # 200-500 characters
    large_chunks: int  # 500+ characters
    embedding_dimension: int


class GPUMemoryManager:
    """
    Intelligent GPU memory management for embedding generation.

    Features:
    - Automatic GPU detection and memory profiling
    - Dynamic batch size optimization
    - Memory pressure monitoring
    - Fallback to CPU when needed
    """

    def __init__(self):
        self.gpu_info: GPUInfo | None = None
        self.batch_config: BatchSizeConfig | None = None
        self.system_info = self._get_system_info()
        self._initialize_gpu_info()

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information"""
        return {
            "platform": platform.system(),
            "processor": platform.processor(),
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_cores": psutil.cpu_count(),
            "cpu_threads": psutil.cpu_count(logical=True),
        }

    def _initialize_gpu_info(self):
        """Initialize GPU information"""
        try:
            # Try to import CUDA-related libraries
            self.gpu_info = self._detect_nvidia_gpu()
            if self.gpu_info:
                self.batch_config = self._calculate_optimal_batch_sizes()
                logger.info(
                    f"GPU detected: {self.gpu_info.name} ({self.gpu_info.memory_total_gb:.1f}GB)"
                )
            else:
                logger.info("No compatible GPU detected, using CPU-optimized settings")
                self.batch_config = self._get_cpu_batch_sizes()
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            self.batch_config = self._get_cpu_batch_sizes()

    def _detect_nvidia_gpu(self) -> GPUInfo | None:
        """Detect NVIDIA GPU information"""
        try:
            import re
            import subprocess

            # Try nvidia-smi first
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\\n")
                if lines and lines[0]:
                    parts = lines[0].split(", ")
                    if len(parts) >= 4:
                        name = parts[0].strip()
                        memory_total_mb = float(parts[1].strip())
                        memory_free_mb = float(parts[2].strip())
                        driver_version = parts[3].strip()

                        # Get CUDA version
                        cuda_version = "Unknown"
                        try:
                            cuda_result = subprocess.run(
                                ["nvcc", "--version"],
                                capture_output=True,
                                text=True,
                                timeout=5,
                            )
                            if cuda_result.returncode == 0:
                                cuda_match = re.search(
                                    r"V(\\d+\\.\\d+)", cuda_result.stdout
                                )
                                if cuda_match:
                                    cuda_version = cuda_match.group(1)
                        except:
                            pass

                        # Determine compute capability for GTX 1080
                        compute_capability = "6.1"  # GTX 1080 default
                        if "1080" in name:
                            compute_capability = "6.1"
                        elif "1070" in name:
                            compute_capability = "6.1"
                        elif "1060" in name:
                            compute_capability = "6.1"
                        elif "RTX" in name:
                            compute_capability = "7.5"  # RTX 20xx series

                        return GPUInfo(
                            name=name,
                            memory_total_gb=memory_total_mb / 1024,
                            memory_available_gb=memory_free_mb / 1024,
                            compute_capability=compute_capability,
                            driver_version=driver_version,
                            cuda_version=cuda_version,
                        )
        except Exception as e:
            logger.debug(f"nvidia-smi detection failed: {e}")

        # Fallback: try PyTorch CUDA detection
        try:
            import torch

            if torch.cuda.is_available():
                device = torch.cuda.get_device_properties(0)
                memory_gb = device.total_memory / (1024**3)

                return GPUInfo(
                    name=device.name,
                    memory_total_gb=memory_gb,
                    memory_available_gb=memory_gb * 0.8,  # Estimate
                    compute_capability=f"{device.major}.{device.minor}",
                    driver_version="Unknown",
                    cuda_version="PyTorch",
                )
        except Exception as e:
            logger.debug(f"PyTorch GPU detection failed: {e}")

        return None

    def _calculate_optimal_batch_sizes(self) -> BatchSizeConfig:
        """Calculate optimal batch sizes based on GPU memory"""
        if not self.gpu_info:
            return self._get_cpu_batch_sizes()

        memory_gb = self.gpu_info.memory_available_gb

        # Conservative batch sizes for NVIDIA GTX 1080 (8GB total, ~6GB usable)
        if memory_gb <= 2:
            # Low memory GPU
            return BatchSizeConfig(
                small_chunks=16,
                medium_chunks=8,
                large_chunks=4,
                embedding_dimension=384,
            )
        elif memory_gb <= 4:
            # Medium memory GPU (GTX 1060, etc.)
            return BatchSizeConfig(
                small_chunks=32,
                medium_chunks=16,
                large_chunks=8,
                embedding_dimension=384,
            )
        elif memory_gb <= 8:
            # High memory GPU (GTX 1080, RTX 2070, etc.)
            return BatchSizeConfig(
                small_chunks=64,
                medium_chunks=32,
                large_chunks=16,
                embedding_dimension=384,
            )
        else:
            # Very high memory GPU (RTX 3080+, etc.)
            return BatchSizeConfig(
                small_chunks=128,
                medium_chunks=64,
                large_chunks=32,
                embedding_dimension=384,
            )

    def _get_cpu_batch_sizes(self) -> BatchSizeConfig:
        """Get CPU-optimized batch sizes"""
        cpu_cores = self.system_info["cpu_threads"]

        # Scale batch size with CPU cores
        base_size = min(8, max(2, cpu_cores // 2))

        return BatchSizeConfig(
            small_chunks=base_size * 2,
            medium_chunks=base_size,
            large_chunks=max(1, base_size // 2),
            embedding_dimension=384,
        )

    def get_optimal_batch_size(self, text_length: int) -> int:
        """Get optimal batch size for given text length"""
        if not self.batch_config:
            return 8  # Safe default

        if text_length < 200:
            return self.batch_config.small_chunks
        elif text_length < 500:
            return self.batch_config.medium_chunks
        else:
            return self.batch_config.large_chunks

    def get_adaptive_batch_size(self, chunks: list) -> int:
        """Get adaptive batch size based on chunk characteristics"""
        if not chunks:
            return 8

        # Analyze chunk sizes
        avg_length = sum(
            len(chunk.text) if hasattr(chunk, "text") else len(str(chunk))
            for chunk in chunks
        ) / len(chunks)

        # Get base batch size
        base_batch_size = self.get_optimal_batch_size(avg_length)

        # Adjust for GPU memory pressure
        if self.gpu_info:
            try:
                current_memory = self._get_current_gpu_memory_usage()
                if current_memory > 0.8:  # 80% memory usage
                    base_batch_size = max(1, base_batch_size // 2)
                elif current_memory < 0.5:  # 50% memory usage
                    base_batch_size = min(base_batch_size * 2, 256)
            except:
                pass  # Ignore memory check errors

        return base_batch_size

    def _get_current_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage ratio (0.0 - 1.0)"""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\\n")
                if lines and lines[0]:
                    used, total = lines[0].split(", ")
                    return float(used) / float(total)
        except:
            pass
        return 0.5  # Default assumption

    async def monitor_memory_pressure(self) -> dict[str, Any]:
        """Monitor GPU memory pressure and system resources"""
        info = {
            "timestamp": asyncio.get_event_loop().time(),
            "gpu_available": self.gpu_info is not None,
            "system": self.system_info,
        }

        if self.gpu_info:
            try:
                gpu_usage = self._get_current_gpu_memory_usage()
                info["gpu"] = {
                    "name": self.gpu_info.name,
                    "memory_total_gb": self.gpu_info.memory_total_gb,
                    "memory_usage_ratio": gpu_usage,
                    "memory_used_gb": self.gpu_info.memory_total_gb * gpu_usage,
                    "compute_capability": self.gpu_info.compute_capability,
                }
            except Exception as e:
                info["gpu_error"] = str(e)

        # System memory
        memory = psutil.virtual_memory()
        info["system_memory"] = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "usage_ratio": memory.percent / 100,
        }

        # CPU usage
        info["cpu_usage"] = psutil.cpu_percent(interval=1)

        return info

    def get_recommended_settings(self) -> dict[str, Any]:
        """Get recommended settings for current hardware"""
        settings = {
            "batch_size_small": self.batch_config.small_chunks,
            "batch_size_medium": self.batch_config.medium_chunks,
            "batch_size_large": self.batch_config.large_chunks,
            "embedding_dimension": self.batch_config.embedding_dimension,
            "max_concurrent_requests": 4,
            "use_gpu": self.gpu_info is not None,
        }

        if self.gpu_info:
            settings.update(
                {
                    "gpu_name": self.gpu_info.name,
                    "gpu_memory_gb": self.gpu_info.memory_total_gb,
                    "compute_capability": self.gpu_info.compute_capability,
                    "recommended_gpu_memory_fraction": 0.8,
                    "max_concurrent_requests": min(
                        8, max(2, int(self.gpu_info.memory_total_gb))
                    ),
                }
            )
        else:
            settings.update(
                {
                    "cpu_workers": self.system_info["cpu_threads"],
                    "max_concurrent_requests": min(self.system_info["cpu_cores"], 4),
                }
            )

        return settings

    def log_system_info(self):
        """Log comprehensive system information"""
        logger.info("=== GPU Memory Manager Initialization ===")
        logger.info(f"Platform: {self.system_info['platform']}")
        logger.info(
            f"CPU: {self.system_info['cpu_cores']} cores, {self.system_info['cpu_threads']} threads"
        )
        logger.info(f"RAM: {self.system_info['ram_gb']:.1f}GB")

        if self.gpu_info:
            logger.info(f"GPU: {self.gpu_info.name}")
            logger.info(
                f"GPU Memory: {self.gpu_info.memory_total_gb:.1f}GB total, {self.gpu_info.memory_available_gb:.1f}GB available"
            )
            logger.info(f"Compute Capability: {self.gpu_info.compute_capability}")
            logger.info(f"CUDA Version: {self.gpu_info.cuda_version}")
            logger.info(f"Driver Version: {self.gpu_info.driver_version}")
        else:
            logger.info("GPU: Not available or not supported")

        settings = self.get_recommended_settings()
        logger.info(
            f"Recommended batch sizes: Small={settings['batch_size_small']}, Medium={settings['batch_size_medium']}, Large={settings['batch_size_large']}"
        )
        logger.info(f"Max concurrent requests: {settings['max_concurrent_requests']}")
        logger.info("==========================================")


# Global instance
_gpu_memory_manager = None


def get_gpu_memory_manager() -> GPUMemoryManager:
    """Get the global GPU memory manager instance"""
    global _gpu_memory_manager
    if _gpu_memory_manager is None:
        _gpu_memory_manager = GPUMemoryManager()
        _gpu_memory_manager.log_system_info()
    return _gpu_memory_manager


async def optimize_batch_processing(
    chunks: list, processing_func, progress_callback=None
) -> list:
    """
    Optimized batch processing with GPU memory management.

    Args:
        chunks: List of chunks to process
        processing_func: Async function to process each batch
        progress_callback: Optional callback for progress updates

    Returns:
        List of processed results
    """
    manager = get_gpu_memory_manager()

    # Get adaptive batch size
    batch_size = manager.get_adaptive_batch_size(chunks)

    results = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    logger.info(
        f"Processing {len(chunks)} chunks in {total_batches} batches (batch_size={batch_size})"
    )

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1

        try:
            # Process batch
            batch_results = await processing_func(batch)
            results.extend(batch_results)

            # Progress callback
            if progress_callback:
                await progress_callback(batch_num, total_batches, len(batch))

            # Memory pressure check
            if manager.gpu_info and batch_num % 5 == 0:  # Check every 5 batches
                memory_info = await manager.monitor_memory_pressure()
                gpu_usage = memory_info.get("gpu", {}).get("memory_usage_ratio", 0)
                if gpu_usage > 0.9:  # 90% memory usage
                    logger.warning(f"High GPU memory usage: {gpu_usage:.1%}")
                    await asyncio.sleep(1)  # Brief pause

        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {e}")
            # Continue with next batch rather than failing completely
            continue

    return results
