"""
Dependency Injection Container for London School TDD

This replaces the singleton pattern with proper dependency injection,
making components testable and following SOLID principles.
"""

from typing import Dict, Type, Any, Optional, Callable, TypeVar, Generic
from abc import ABC, abstractmethod
import asyncio
from contextlib import asynccontextmanager
import inspect
from dataclasses import dataclass

T = TypeVar('T')


class DependencyResolutionError(Exception):
    """Raised when a dependency cannot be resolved."""
    pass


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""
    pass


@dataclass
class ServiceDefinition:
    """Definition of a service in the container"""
    service_type: Type
    factory: Optional[Callable] = None
    singleton: bool = True
    async_factory: bool = False
    dependencies: Optional[Dict[str, Type]] = None


class IContainer(ABC):
    """Interface for dependency injection container"""
    
    @abstractmethod
    def register(
        self,
        service_type: Type[T],
        factory: Optional[Callable[..., T]] = None,
        singleton: bool = True,
        async_factory: bool = False,
        **dependencies
    ) -> None:
        """Register a service with the container"""
        pass
    
    @abstractmethod
    async def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service from the container"""
        pass
    
    @abstractmethod
    async def resolve_all(self, service_type: Type[T]) -> list[T]:
        """Resolve all implementations of a service type"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up all managed resources"""
        pass


class Container(IContainer):
    """
    Dependency Injection Container implementing London School TDD principles
    
    Features:
    - Constructor injection support
    - Singleton and transient lifetimes
    - Async factory support
    - Circular dependency detection
    - Proper resource cleanup
    - Interface-based registration
    """
    
    def __init__(self):
        self._services: Dict[Type, ServiceDefinition] = {}
        self._instances: Dict[Type, Any] = {}
        self._resolution_stack: list[Type] = []
        self._cleanup_tasks: list[Callable] = []
    
    def register(
        self,
        service_type: Type[T],
        factory: Optional[Callable[..., T]] = None,
        singleton: bool = True,
        async_factory: bool = False,
        **dependencies
    ) -> None:
        """
        Register a service with the container
        
        Args:
            service_type: The type/interface to register
            factory: Optional factory function to create the service
            singleton: Whether to create a single instance (default: True)
            async_factory: Whether the factory is async (default: False)
            **dependencies: Named dependencies to inject
        """
        self._services[service_type] = ServiceDefinition(
            service_type=service_type,
            factory=factory,
            singleton=singleton,
            async_factory=async_factory,
            dependencies=dependencies or {}
        )
    
    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """Register a pre-created instance"""
        self._services[service_type] = ServiceDefinition(
            service_type=service_type,
            singleton=True
        )
        self._instances[service_type] = instance
    
    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[..., T],
        singleton: bool = True,
        async_factory: bool = False
    ) -> None:
        """Register a factory function for a service"""
        self.register(
            service_type=service_type,
            factory=factory,
            singleton=singleton,
            async_factory=async_factory
        )
    
    async def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service from the container
        
        Args:
            service_type: The type to resolve
            
        Returns:
            An instance of the requested type
            
        Raises:
            DependencyResolutionError: If the service cannot be resolved
            CircularDependencyError: If circular dependencies are detected
        """
        # Check for circular dependencies
        if service_type in self._resolution_stack:
            cycle = " -> ".join([t.__name__ for t in self._resolution_stack])
            cycle += f" -> {service_type.__name__}"
            raise CircularDependencyError(f"Circular dependency detected: {cycle}")
        
        # Check if we already have a singleton instance
        if service_type in self._instances:
            return self._instances[service_type]
        
        # Check if service is registered
        if service_type not in self._services:
            raise DependencyResolutionError(f"Service {service_type.__name__} is not registered")
        
        service_def = self._services[service_type]
        
        # Add to resolution stack
        self._resolution_stack.append(service_type)
        
        try:
            # Create the instance
            if service_def.factory:
                instance = await self._create_from_factory(service_def)
            else:
                instance = await self._create_from_constructor(service_def)
            
            # Store singleton instances
            if service_def.singleton:
                self._instances[service_type] = instance
            
            # Add cleanup if instance has cleanup method
            if hasattr(instance, 'cleanup') and callable(getattr(instance, 'cleanup')):
                self._cleanup_tasks.append(instance.cleanup)
            
            return instance
            
        finally:
            # Remove from resolution stack
            self._resolution_stack.pop()
    
    async def resolve_all(self, service_type: Type[T]) -> list[T]:
        """Resolve all implementations of a service type"""
        implementations = []
        
        for registered_type, service_def in self._services.items():
            if issubclass(registered_type, service_type) or registered_type == service_type:
                implementations.append(await self.resolve(registered_type))
        
        return implementations
    
    async def _create_from_factory(self, service_def: ServiceDefinition) -> Any:
        """Create an instance using a factory function"""
        factory_args = {}
        
        # Resolve factory dependencies
        if service_def.dependencies:
            for arg_name, dep_type in service_def.dependencies.items():
                factory_args[arg_name] = await self.resolve(dep_type)
        
        # Call factory
        if service_def.async_factory:
            instance = await service_def.factory(**factory_args)
        else:
            instance = service_def.factory(**factory_args)
        
        return instance
    
    async def _create_from_constructor(self, service_def: ServiceDefinition) -> Any:
        """Create an instance using constructor injection"""
        constructor = service_def.service_type
        
        # Get constructor signature
        sig = inspect.signature(constructor.__init__)
        constructor_args = {}
        
        # Resolve constructor dependencies
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Check if dependency is explicitly specified
            if param_name in service_def.dependencies:
                dep_type = service_def.dependencies[param_name]
                constructor_args[param_name] = await self.resolve(dep_type)
            # Try to resolve by type annotation (but skip basic types like str, int, etc.)
            elif (param.annotation and 
                  param.annotation != inspect.Parameter.empty and
                  not self._is_basic_type(param.annotation)):
                constructor_args[param_name] = await self.resolve(param.annotation)
            # Check if parameter has default value
            elif param.default != inspect.Parameter.empty:
                continue  # Use default value
            else:
                raise DependencyResolutionError(
                    f"Cannot resolve parameter '{param_name}' for {constructor.__name__}"
                )
        
        # Create instance
        instance = constructor(**constructor_args)
        
        # Initialize if it's async
        if hasattr(instance, 'initialize') and callable(getattr(instance, 'initialize')):
            if asyncio.iscoroutinefunction(instance.initialize):
                await instance.initialize()
            else:
                instance.initialize()
        
        return instance
    
    def _is_basic_type(self, annotation) -> bool:
        """Check if the annotation is a basic type that shouldn't be resolved as a dependency"""
        basic_types = (str, int, float, bool, bytes, type(None))
        
        # Handle typing annotations
        if hasattr(annotation, '__origin__'):
            return annotation.__origin__ in basic_types
        
        # Handle direct types
        return annotation in basic_types
    
    async def cleanup(self) -> None:
        """Clean up all managed resources"""
        cleanup_errors = []
        
        # Run all cleanup tasks
        for cleanup_task in reversed(self._cleanup_tasks):  # Reverse order
            try:
                if asyncio.iscoroutinefunction(cleanup_task):
                    await cleanup_task()
                else:
                    cleanup_task()
            except Exception as e:
                cleanup_errors.append(e)
        
        # Clear all instances and services
        self._instances.clear()
        self._cleanup_tasks.clear()
        
        # Report cleanup errors
        if cleanup_errors:
            error_messages = [str(e) for e in cleanup_errors]
            raise Exception(f"Cleanup errors occurred: {'; '.join(error_messages)}")
    
    @asynccontextmanager
    async def scope(self):
        """Create a scoped container context"""
        original_instances = self._instances.copy()
        original_cleanup_tasks = self._cleanup_tasks.copy()
        
        try:
            yield self
        finally:
            # Cleanup scope-specific instances
            scope_cleanup_tasks = [
                task for task in self._cleanup_tasks 
                if task not in original_cleanup_tasks
            ]
            
            for cleanup_task in reversed(scope_cleanup_tasks):
                try:
                    if asyncio.iscoroutinefunction(cleanup_task):
                        await cleanup_task()
                    else:
                        cleanup_task()
                except Exception:
                    pass  # Log but don't raise in scope cleanup
            
            # Restore original state
            self._instances = original_instances
            self._cleanup_tasks = original_cleanup_tasks


class ContainerBuilder:
    """Builder for configuring the container"""
    
    def __init__(self):
        self.container = Container()
    
    def add_singleton(self, service_type: Type[T], factory: Optional[Callable[..., T]] = None) -> 'ContainerBuilder':
        """Add a singleton service"""
        self.container.register(service_type, factory, singleton=True)
        return self
    
    def add_transient(self, service_type: Type[T], factory: Optional[Callable[..., T]] = None) -> 'ContainerBuilder':
        """Add a transient service"""
        self.container.register(service_type, factory, singleton=False)
        return self
    
    def add_instance(self, service_type: Type[T], instance: T) -> 'ContainerBuilder':
        """Add a pre-created instance"""
        self.container.register_instance(service_type, instance)
        return self
    
    def build(self) -> Container:
        """Build the configured container"""
        return self.container


# Global container instance for application use
_application_container: Optional[Container] = None


async def get_container() -> Container:
    """Get the application container"""
    global _application_container
    if _application_container is None:
        _application_container = await create_application_container()
    return _application_container


async def create_application_container() -> Container:
    """Create and configure the application container"""
    from src.core.config import Config, load_config
    from src.core.sqlite_storage import SQLiteStorage
    from src.core.qdrant_storage import QdrantStorage
    from src.ingestion.local_embeddings import LocalEmbeddingGenerator
    from src.ingestion.pdf_parser import PDFParser
    from src.ingestion.text_chunker import TextChunker
    from src.search.vector_search import VectorSearchEngine
    from src.search.text_search import TextSearchEngine
    from src.search.hybrid_search import HybridSearch
    
    builder = ContainerBuilder()
    
    # Configuration
    config = load_config()
    builder.add_instance(Config, config)
    
    # Storage
    builder.add_singleton(SQLiteStorage)
    builder.add_singleton(QdrantStorage)
    
    # Embedding
    builder.add_singleton(LocalEmbeddingGenerator)
    
    # Ingestion
    builder.add_singleton(PDFParser)
    builder.add_singleton(TextChunker)
    
    # Search
    builder.add_singleton(VectorSearchEngine)
    builder.add_singleton(TextSearchEngine)
    builder.add_singleton(HybridSearch)
    
    return builder.build()


async def cleanup_application_container():
    """Clean up the application container"""
    global _application_container
    if _application_container:
        await _application_container.cleanup()
        _application_container = None


# Decorator for dependency injection
def inject(**dependencies):
    """
    Decorator for automatic dependency injection
    
    Usage:
    @inject(storage=SQLiteStorage, config=Config)
    async def some_function(storage, config):
        pass
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            container = await get_container()
            
            # Resolve dependencies
            for dep_name, dep_type in dependencies.items():
                if dep_name not in kwargs:
                    kwargs[dep_name] = await container.resolve(dep_type)
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator