"""
Application Initialization with Persistent Memory System

This module handles application startup, crash recovery, and persistent memory
system initialization for the TradeKnowledge application.

Features:
- Automatic crash detection and recovery
- Persistent state initialization
- Agent state restoration
- System health validation
- Clean shutdown handling
"""

import asyncio
import atexit
import logging
import signal
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.config import get_config
from src.core.crash_recovery import (
    get_recovery_system,
    mark_clean_shutdown,
    perform_startup_recovery,
)
from src.core.persistent_state import get_state_manager, shutdown_state_manager

logger = logging.getLogger(__name__)


class ApplicationLifecycleManager:
    """
    Manages the complete application lifecycle with persistent memory support.

    Handles startup, runtime monitoring, and graceful shutdown with
    state preservation and crash recovery capabilities.
    """

    def __init__(self):
        self.config = get_config()
        self.state_manager = None
        self.recovery_system = None
        self.shutdown_handlers = []
        self.startup_complete = False
        self.shutdown_in_progress = False

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        logger.info("ApplicationLifecycleManager initialized")

    async def initialize_application(self) -> dict[str, Any]:
        """
        Initialize the application with persistent memory and crash recovery.

        Returns:
            Dict containing initialization results and system status
        """
        try:
            logger.info("Starting TradeKnowledge application initialization...")

            # 1. Check if persistence is enabled
            if not self.config.api.agents.persistence.enabled:
                logger.info("Persistent memory disabled in configuration")
                return {
                    "success": True,
                    "persistence_enabled": False,
                    "message": "Application started without persistent memory",
                }

            # 2. Initialize persistent memory system
            initialization_result = await self._initialize_persistent_memory()

            # 3. Perform crash recovery if needed
            recovery_result = await self._perform_crash_recovery()

            # 4. Initialize agents with persistent capabilities
            agent_result = await self._initialize_agents()

            # 5. Validate system health
            health_result = await self._validate_system_health()

            # 6. Register shutdown handlers
            self._register_shutdown_handlers()

            self.startup_complete = True

            result = {
                "success": True,
                "persistence_enabled": True,
                "initialization": initialization_result,
                "recovery": recovery_result,
                "agents": agent_result,
                "health": health_result,
                "startup_timestamp": datetime.now(UTC).isoformat(),
                "configuration": {
                    "base_path": self.config.api.agents.persistence.base_path,
                    "auto_backup_interval": self.config.api.agents.persistence.auto_backup_interval_minutes,
                    "crash_recovery_enabled": self.config.api.agents.persistence.enable_crash_recovery,
                },
            }

            logger.info(
                "TradeKnowledge application initialization completed successfully"
            )
            return result

        except Exception as e:
            logger.error(f"Application initialization failed: {e}")
            return {"success": False, "error": str(e), "persistence_enabled": False}

    async def _initialize_persistent_memory(self) -> dict[str, Any]:
        """Initialize the persistent memory system"""
        try:
            logger.info("Initializing persistent memory system...")

            # Get state manager (this initializes it)
            self.state_manager = get_state_manager()

            # Configure based on settings
            config = self.config.api.agents.persistence
            self.state_manager.max_memory_signals = config.max_memory_signals
            self.state_manager.max_memory_lines = config.max_memory_lines
            self.state_manager.backup_interval_minutes = (
                config.auto_backup_interval_minutes
            )
            self.state_manager.backup_retention_hours = config.backup_retention_hours

            # Get system status
            status = self.state_manager.get_system_status()

            # Record initialization
            self.state_manager.add_signal(
                source_agent="application_lifecycle",
                event_type="system_event",
                summary="Persistent memory system initialized",
                context={"configuration": config.__dict__},
                metadata={"initialization": True},
            )

            logger.info("Persistent memory system initialized successfully")
            return {
                "success": True,
                "system_status": status,
                "configuration": config.__dict__,
            }

        except Exception as e:
            logger.error(f"Failed to initialize persistent memory: {e}")
            return {"success": False, "error": str(e)}

    async def _perform_crash_recovery(self) -> dict[str, Any]:
        """Perform crash detection and recovery"""
        try:
            if not self.config.api.agents.persistence.enable_crash_recovery:
                logger.info("Crash recovery disabled in configuration")
                return {"recovery_performed": False, "reason": "disabled_in_config"}

            logger.info("Performing crash recovery check...")

            # Get recovery system
            self.recovery_system = get_recovery_system()

            # Perform startup recovery
            recovery_report = perform_startup_recovery()

            result = {
                "recovery_performed": True,
                "shutdown_type": recovery_report.shutdown_type,
                "agents_recovered": recovery_report.agents_recovered,
                "workflows_restored": recovery_report.workflows_restored,
                "documents_validated": recovery_report.documents_validated,
                "errors_encountered": recovery_report.errors_encountered,
                "recovery_duration_seconds": recovery_report.recovery_duration_seconds,
                "system_health_score": recovery_report.system_health_score,
            }

            if recovery_report.system_health_score < 0.8:
                logger.warning(
                    f"System health score after recovery: {recovery_report.system_health_score}"
                )
            else:
                logger.info(
                    f"Crash recovery completed successfully (health: {recovery_report.system_health_score})"
                )

            return result

        except Exception as e:
            logger.error(f"Crash recovery failed: {e}")
            return {"recovery_performed": False, "error": str(e)}

    async def _initialize_agents(self) -> dict[str, Any]:
        """Initialize agents with persistent capabilities"""
        try:
            logger.info("Initializing agents with persistent capabilities...")

            # This would normally initialize your specific agents
            # For now, we'll just return a placeholder
            initialized_agents = []

            # In a real implementation, you would:
            # 1. Import your agent classes
            # 2. Initialize them with persistent capabilities
            # 3. Restore their states if available
            # 4. Register them with the communication system

            logger.info("Agent initialization completed")
            return {
                "success": True,
                "agents_initialized": initialized_agents,
                "total_agents": len(initialized_agents),
            }

        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            return {"success": False, "error": str(e)}

    async def _validate_system_health(self) -> dict[str, Any]:
        """Validate overall system health"""
        try:
            logger.info("Validating system health...")

            health_checks = {
                "persistent_storage": False,
                "state_manager": False,
                "recovery_system": False,
                "configuration": False,
            }

            # Check persistent storage
            if self.state_manager:
                storage_path = Path(self.state_manager.base_path)
                health_checks["persistent_storage"] = (
                    storage_path.exists() and storage_path.is_dir()
                )
                health_checks["state_manager"] = True

            # Check recovery system
            if self.recovery_system:
                health_checks["recovery_system"] = True

            # Check configuration
            health_checks["configuration"] = self.config is not None

            # Calculate overall health score
            passed_checks = sum(health_checks.values())
            total_checks = len(health_checks)
            health_score = passed_checks / total_checks

            result = {
                "health_score": health_score,
                "checks": health_checks,
                "passed_checks": passed_checks,
                "total_checks": total_checks,
                "status": (
                    "healthy"
                    if health_score >= 0.8
                    else "degraded" if health_score >= 0.6 else "unhealthy"
                ),
            }

            if health_score < 0.8:
                logger.warning(f"System health degraded: {health_score:.2f}")
            else:
                logger.info(f"System health validation passed: {health_score:.2f}")

            return result

        except Exception as e:
            logger.error(f"System health validation failed: {e}")
            return {"health_score": 0.0, "status": "error", "error": str(e)}

    def _register_shutdown_handlers(self):
        """Register handlers for graceful shutdown"""
        try:
            # Register atexit handler
            atexit.register(self.graceful_shutdown)

            # Add to internal handlers list
            self.shutdown_handlers.append(self._shutdown_persistent_memory)
            self.shutdown_handlers.append(self._shutdown_agents)

            logger.info("Shutdown handlers registered")

        except Exception as e:
            logger.error(f"Failed to register shutdown handlers: {e}")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.graceful_shutdown())

        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal_handler)

    async def graceful_shutdown(self):
        """Perform graceful application shutdown"""
        if self.shutdown_in_progress:
            logger.info("Shutdown already in progress, ignoring duplicate request")
            return

        self.shutdown_in_progress = True
        logger.info("Starting graceful application shutdown...")

        try:
            # Mark clean shutdown before doing anything else
            if self.recovery_system:
                mark_clean_shutdown()

            # Execute shutdown handlers in reverse order
            for handler in reversed(self.shutdown_handlers):
                try:
                    await handler()
                except Exception as e:
                    logger.error(f"Error in shutdown handler: {e}")

            logger.info("Graceful shutdown completed")

        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
        finally:
            self.shutdown_in_progress = False

    async def _shutdown_agents(self):
        """Shutdown agents gracefully"""
        try:
            logger.info("Shutting down agents...")

            # This would normally iterate through your agents and shut them down
            # For each agent:
            # 1. Save current state
            # 2. Complete any critical tasks
            # 3. Record shutdown events

            logger.info("Agent shutdown completed")

        except Exception as e:
            logger.error(f"Error shutting down agents: {e}")

    async def _shutdown_persistent_memory(self):
        """Shutdown persistent memory system"""
        try:
            logger.info("Shutting down persistent memory system...")

            if self.state_manager:
                # Record shutdown event
                self.state_manager.add_signal(
                    source_agent="application_lifecycle",
                    event_type="system_event",
                    summary="Application shutting down gracefully",
                    context={"clean_shutdown": True},
                    metadata={"shutdown": True},
                )

                # Shutdown state manager
                shutdown_state_manager()

            logger.info("Persistent memory system shutdown completed")

        except Exception as e:
            logger.error(f"Error shutting down persistent memory: {e}")

    def add_shutdown_handler(self, handler: callable):
        """Add a custom shutdown handler"""
        self.shutdown_handlers.append(handler)

    def get_system_status(self) -> dict[str, Any]:
        """Get current system status"""
        status = {
            "startup_complete": self.startup_complete,
            "shutdown_in_progress": self.shutdown_in_progress,
            "persistence_enabled": self.config.api.agents.persistence.enabled,
            "crash_recovery_enabled": self.config.api.agents.persistence.enable_crash_recovery,
        }

        if self.state_manager:
            status["state_manager"] = self.state_manager.get_system_status()

        if self.recovery_system:
            status["recovery_system"] = self.recovery_system.get_system_health_status()

        return status


# Global instance
_lifecycle_manager: ApplicationLifecycleManager | None = None


def get_lifecycle_manager() -> ApplicationLifecycleManager:
    """Get the global lifecycle manager instance"""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = ApplicationLifecycleManager()
    return _lifecycle_manager


async def initialize_application() -> dict[str, Any]:
    """
    Initialize the TradeKnowledge application with persistent memory.

    This should be called at application startup before starting any agents
    or processing requests.

    Returns:
        Dict containing initialization results
    """
    lifecycle_manager = get_lifecycle_manager()
    return await lifecycle_manager.initialize_application()


async def shutdown_application():
    """
    Gracefully shutdown the application.

    This should be called during application shutdown to ensure
    all state is properly saved and cleanup is performed.
    """
    global _lifecycle_manager
    if _lifecycle_manager:
        await _lifecycle_manager.graceful_shutdown()
        _lifecycle_manager = None


def get_application_status() -> dict[str, Any]:
    """
    Get current application status.

    Returns:
        Dict containing application status information
    """
    lifecycle_manager = get_lifecycle_manager()
    return lifecycle_manager.get_system_status()
