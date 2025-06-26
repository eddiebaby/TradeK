"""
Parallel Book Processing Pipeline

This module implements concurrent book processing for improved performance
while managing system resources effectively.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .enhanced_book_processor import EnhancedBookProcessor
from .gpu_memory_manager import get_gpu_memory_manager

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of book processing operation"""

    book_path: Path
    success: bool
    book_id: str | None = None
    chunks_created: int = 0
    processing_time: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = None


class ParallelBookProcessor:
    """
    Parallel book processing pipeline with intelligent resource management.

    Features:
    - Concurrent book processing with configurable limits
    - GPU memory-aware batch sizing
    - Progress tracking and error recovery
    - Resource monitoring and throttling
    """

    def __init__(self, max_workers: int | None = None):
        """Initialize parallel processor"""
        self.gpu_manager = get_gpu_memory_manager()

        # Determine optimal worker count
        if max_workers is None:
            recommended = self.gpu_manager.get_recommended_settings()
            max_workers = recommended["max_concurrent_requests"]

        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)

        # Progress tracking
        self.total_books = 0
        self.completed_books = 0
        self.failed_books = 0
        self.start_time = None

        logger.info(f"Parallel processor initialized with {max_workers} workers")

    async def process_books_parallel(
        self,
        book_paths: list[Path],
        progress_callback: Callable | None = None,
        error_callback: Callable | None = None,
    ) -> list[ProcessingResult]:
        """
        Process multiple books in parallel.

        Args:
            book_paths: List of book file paths to process
            progress_callback: Optional callback for progress updates
            error_callback: Optional callback for error handling

        Returns:
            List of ProcessingResult objects
        """
        self.total_books = len(book_paths)
        self.completed_books = 0
        self.failed_books = 0
        self.start_time = time.time()

        logger.info(f"Starting parallel processing of {self.total_books} books")

        # Create tasks for all books
        tasks = []
        for book_path in book_paths:
            task = self._process_single_book_with_semaphore(
                book_path, progress_callback, error_callback
            )
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ProcessingResult(
                    book_path=book_paths[i], success=False, error=str(result)
                )
                processed_results.append(error_result)
                self.failed_books += 1
            else:
                processed_results.append(result)

        # Final summary
        elapsed_time = time.time() - self.start_time
        success_count = sum(1 for r in processed_results if r.success)

        logger.info("Parallel processing completed:")
        logger.info(f"  Total: {self.total_books} books")
        logger.info(f"  Success: {success_count} books")
        logger.info(f"  Failed: {self.failed_books} books")
        logger.info(f"  Time: {elapsed_time:.1f} seconds")
        logger.info(f"  Rate: {self.total_books/elapsed_time:.1f} books/second")

        return processed_results

    async def _process_single_book_with_semaphore(
        self,
        book_path: Path,
        progress_callback: Callable | None,
        error_callback: Callable | None,
    ) -> ProcessingResult:
        """Process a single book with semaphore-based concurrency control"""
        async with self.semaphore:
            return await self._process_single_book(
                book_path, progress_callback, error_callback
            )

    async def _process_single_book(
        self,
        book_path: Path,
        progress_callback: Callable | None,
        error_callback: Callable | None,
    ) -> ProcessingResult:
        """Process a single book and track results"""
        start_time = time.time()

        try:
            logger.debug(f"Starting processing: {book_path.name}")

            # Create processor instance for this book
            processor = EnhancedBookProcessor()
            await processor.initialize()

            try:
                # Process the book
                result = await processor.add_book(str(book_path))

                processing_time = time.time() - start_time

                if result.get("success"):
                    self.completed_books += 1

                    processing_result = ProcessingResult(
                        book_path=book_path,
                        success=True,
                        book_id=result.get("book_id"),
                        chunks_created=result.get("chunks_created", 0),
                        processing_time=processing_time,
                        metadata=result.get("metadata", {}),
                    )

                    logger.info(
                        f"✅ {book_path.name}: {result.get('chunks_created', 0)} chunks ({processing_time:.1f}s)"
                    )
                else:
                    self.failed_books += 1
                    error = result.get("error", "Unknown error")

                    processing_result = ProcessingResult(
                        book_path=book_path,
                        success=False,
                        error=error,
                        processing_time=processing_time,
                    )

                    logger.error(f"❌ {book_path.name}: {error}")

                    if error_callback:
                        await error_callback(book_path, error)

                # Progress callback
                if progress_callback:
                    await progress_callback(
                        self.completed_books + self.failed_books,
                        self.total_books,
                        processing_result,
                    )

                return processing_result

            finally:
                # Cleanup processor
                await processor.cleanup()

        except Exception as e:
            self.failed_books += 1
            processing_time = time.time() - start_time
            error_str = str(e)

            logger.error(
                f"❌ {book_path.name}: Processing failed with exception: {error_str}"
            )

            if error_callback:
                try:
                    await error_callback(book_path, error_str)
                except:
                    pass  # Don't let callback errors stop processing

            return ProcessingResult(
                book_path=book_path,
                success=False,
                error=error_str,
                processing_time=processing_time,
            )

    async def process_with_resource_monitoring(
        self, book_paths: list[Path], monitor_interval: float = 30.0
    ) -> list[ProcessingResult]:
        """
        Process books with continuous resource monitoring.

        Args:
            book_paths: List of book paths to process
            monitor_interval: Seconds between resource checks

        Returns:
            List of processing results
        """

        async def resource_monitor():
            """Background task to monitor system resources"""
            while True:
                try:
                    memory_info = await self.gpu_manager.monitor_memory_pressure()

                    # Log resource usage periodically
                    if self.gpu_manager.gpu_info:
                        gpu_usage = memory_info.get("gpu", {}).get(
                            "memory_usage_ratio", 0
                        )
                        logger.info(
                            f"GPU Memory: {gpu_usage:.1%}, System Memory: {memory_info['system_memory']['usage_ratio']:.1%}"
                        )

                        # Adjust processing if memory pressure is high
                        if gpu_usage > 0.9:
                            logger.warning(
                                "High GPU memory pressure detected - reducing concurrency"
                            )
                            # Could dynamically adjust semaphore here

                    await asyncio.sleep(monitor_interval)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
                    await asyncio.sleep(monitor_interval)

        # Start resource monitoring
        monitor_task = asyncio.create_task(resource_monitor())

        try:
            # Process books with monitoring
            results = await self.process_books_parallel(book_paths)
            return results
        finally:
            # Stop monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    def get_processing_stats(self) -> dict[str, Any]:
        """Get current processing statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            "total_books": self.total_books,
            "completed_books": self.completed_books,
            "failed_books": self.failed_books,
            "remaining_books": max(
                0, self.total_books - self.completed_books - self.failed_books
            ),
            "success_rate": self.completed_books
            / max(1, self.completed_books + self.failed_books),
            "elapsed_time": elapsed,
            "processing_rate": (self.completed_books + self.failed_books)
            / max(1, elapsed),
            "estimated_completion": elapsed
            * self.total_books
            / max(1, self.completed_books + self.failed_books)
            - elapsed,
            "max_workers": self.max_workers,
        }


class BatchedBookProcessor:
    """
    Batched processing for very large book collections.

    Processes books in smaller batches to manage memory and provide
    regular progress checkpoints.
    """

    def __init__(self, batch_size: int = 5, max_workers_per_batch: int = 3):
        """Initialize batched processor"""
        self.batch_size = batch_size
        self.max_workers_per_batch = max_workers_per_batch

    async def process_books_in_batches(
        self, book_paths: list[Path], progress_callback: Callable | None = None
    ) -> list[ProcessingResult]:
        """
        Process books in smaller batches.

        Args:
            book_paths: All book paths to process
            progress_callback: Progress callback for overall progress

        Returns:
            Combined results from all batches
        """
        total_books = len(book_paths)
        all_results = []

        logger.info(f"Processing {total_books} books in batches of {self.batch_size}")

        for batch_start in range(0, total_books, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_books)
            batch_paths = book_paths[batch_start:batch_end]
            batch_num = batch_start // self.batch_size + 1
            total_batches = (total_books + self.batch_size - 1) // self.batch_size

            logger.info(
                f"Processing batch {batch_num}/{total_batches}: {len(batch_paths)} books"
            )

            # Process this batch
            processor = ParallelBookProcessor(max_workers=self.max_workers_per_batch)
            batch_results = await processor.process_books_parallel(batch_paths)
            all_results.extend(batch_results)

            # Overall progress callback
            if progress_callback:
                await progress_callback(len(all_results), total_books, batch_results)

            # Brief pause between batches
            if batch_end < total_books:
                await asyncio.sleep(1)

        return all_results


# Convenience functions for common use cases


async def process_books_fast(
    book_paths: list[Path], max_workers: int | None = None
) -> list[ProcessingResult]:
    """
    Fast parallel processing of books with optimal settings.

    Args:
        book_paths: List of book paths to process
        max_workers: Optional worker limit (auto-detected if None)

    Returns:
        List of processing results
    """
    processor = ParallelBookProcessor(max_workers=max_workers)
    return await processor.process_books_parallel(book_paths)


async def process_books_safe(
    book_paths: list[Path], batch_size: int = 3, max_workers_per_batch: int = 2
) -> list[ProcessingResult]:
    """
    Safe batched processing for large book collections or limited resources.

    Args:
        book_paths: List of book paths to process
        batch_size: Number of books per batch
        max_workers_per_batch: Workers per batch

    Returns:
        List of processing results
    """
    processor = BatchedBookProcessor(
        batch_size=batch_size, max_workers_per_batch=max_workers_per_batch
    )
    return await processor.process_books_in_batches(book_paths)


async def process_books_with_monitoring(
    book_paths: list[Path],
) -> list[ProcessingResult]:
    """
    Process books with comprehensive resource monitoring.

    Args:
        book_paths: List of book paths to process

    Returns:
        List of processing results
    """
    processor = ParallelBookProcessor()
    return await processor.process_with_resource_monitoring(book_paths)
