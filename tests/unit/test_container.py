"""
Tests for the Dependency Injection Container

Following London School TDD principles with comprehensive test coverage
for the container's dependency resolution and lifecycle management.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import Protocol, runtime_checkable

from src.core.container import (
    Container, ContainerBuilder, IContainer,
    DependencyResolutionError, CircularDependencyError,
    ServiceDefinition, inject, get_container, cleanup_application_container
)


# Test interfaces and implementations
@runtime_checkable
class ITestService(Protocol):
    """Test service interface"""
    
    def get_value(self) -> str:
        ...


@runtime_checkable
class ITestRepository(Protocol):
    """Test repository interface"""
    
    async def get_data(self) -> str:
        ...


class TestService:
    """Simple test service implementation"""
    
    def __init__(self, value: str = "default"):
        self.value = value
        self.initialized = False
        self.cleanup_called = False
    
    def initialize(self):
        """Initialize the service"""
        self.initialized = True
    
    def cleanup(self):
        """Cleanup the service"""
        self.cleanup_called = True
    
    def get_value(self) -> str:
        return self.value


class AsyncTestService:
    """Async test service implementation"""
    
    def __init__(self, dependency: TestService):
        self.dependency = dependency
        self.initialized = False
        self.cleanup_called = False
    
    async def initialize(self):
        """Async initialize"""
        self.initialized = True
    
    async def cleanup(self):
        """Async cleanup"""
        self.cleanup_called = True
    
    def get_value(self) -> str:
        return f"async_{self.dependency.get_value()}"


class TestRepository:
    """Test repository implementation"""
    
    def __init__(self, service: TestService):
        self.service = service
        self.initialized = False
    
    async def initialize(self):
        self.initialized = True
    
    async def get_data(self) -> str:
        return f"data_from_{self.service.get_value()}"


class CircularDependencyA:
    """Test class for circular dependency detection"""
    
    def __init__(self, b: 'CircularDependencyB'):
        self.b = b


class CircularDependencyB:
    """Test class for circular dependency detection"""
    
    def __init__(self, a: CircularDependencyA):
        self.a = a


class TestContainer:
    """Test the Container implementation"""
    
    @pytest.fixture
    def container(self):
        """Provide a fresh container for each test"""
        return Container()
    
    def test_container_implements_interface(self, container):
        """Test that Container implements IContainer"""
        assert isinstance(container, IContainer)
    
    def test_register_service_simple(self, container):
        """Test registering a simple service"""
        container.register(TestService)
        
        assert TestService in container._services
        service_def = container._services[TestService]
        assert service_def.service_type == TestService
        assert service_def.singleton is True
        assert service_def.factory is None
    
    def test_register_service_with_factory(self, container):
        """Test registering a service with factory function"""
        def test_factory():
            return TestService("factory_created")
        
        container.register(TestService, factory=test_factory)
        
        service_def = container._services[TestService]
        assert service_def.factory == test_factory
    
    def test_register_instance(self, container):
        """Test registering a pre-created instance"""
        instance = TestService("instance")
        container.register_instance(TestService, instance)
        
        assert TestService in container._instances
        assert container._instances[TestService] is instance
    
    @pytest.mark.asyncio
    async def test_resolve_simple_service(self, container):
        """Test resolving a simple service"""
        container.register(TestService)
        
        instance = await container.resolve(TestService)
        
        assert isinstance(instance, TestService)
        assert instance.value == "default"
        assert instance.initialized is True  # Should be initialized
    
    @pytest.mark.asyncio
    async def test_resolve_singleton_behavior(self, container):
        """Test that singleton services return the same instance"""
        container.register(TestService)
        
        instance1 = await container.resolve(TestService)
        instance2 = await container.resolve(TestService)
        
        assert instance1 is instance2
    
    @pytest.mark.asyncio
    async def test_resolve_transient_behavior(self, container):
        """Test that transient services return different instances"""
        container.register(TestService, singleton=False)
        
        instance1 = await container.resolve(TestService)
        instance2 = await container.resolve(TestService)
        
        assert instance1 is not instance2
        assert isinstance(instance1, TestService)
        assert isinstance(instance2, TestService)
    
    @pytest.mark.asyncio
    async def test_resolve_with_dependencies(self, container):
        """Test resolving services with dependencies"""
        container.register(TestService)
        container.register(TestRepository)
        
        repository = await container.resolve(TestRepository)
        
        assert isinstance(repository, TestRepository)
        assert isinstance(repository.service, TestService)
        assert repository.initialized is True
    
    @pytest.mark.asyncio
    async def test_resolve_with_factory(self, container):
        """Test resolving service created by factory"""
        def test_factory(service: TestService):
            return TestRepository(service)
        
        container.register(TestService)
        container.register(TestRepository, factory=test_factory, service=TestService)
        
        repository = await container.resolve(TestRepository)
        
        assert isinstance(repository, TestRepository)
        assert isinstance(repository.service, TestService)
    
    @pytest.mark.asyncio
    async def test_resolve_async_service(self, container):
        """Test resolving async service"""
        container.register(TestService)
        container.register(AsyncTestService)
        
        async_service = await container.resolve(AsyncTestService)
        
        assert isinstance(async_service, AsyncTestService)
        assert async_service.initialized is True
        assert async_service.get_value() == "async_default"
    
    @pytest.mark.asyncio
    async def test_resolve_unregistered_service_raises_error(self, container):
        """Test that resolving unregistered service raises error"""
        with pytest.raises(DependencyResolutionError, match="Service TestService is not registered"):
            await container.resolve(TestService)
    
    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, container):
        """Test circular dependency detection"""
        container.register(CircularDependencyA)
        container.register(CircularDependencyB)
        
        with pytest.raises(CircularDependencyError, match="Circular dependency detected"):
            await container.resolve(CircularDependencyA)
    
    @pytest.mark.asyncio
    async def test_resolve_all_implementations(self, container):
        """Test resolving all implementations of a type"""
        container.register(TestService)
        container.register_instance(ITestService, TestService("interface_impl"))
        
        implementations = await container.resolve_all(ITestService)
        
        assert len(implementations) >= 1
        assert any(isinstance(impl, TestService) for impl in implementations)
    
    @pytest.mark.asyncio
    async def test_cleanup_calls_service_cleanup(self, container):
        """Test that cleanup calls cleanup on managed services"""
        container.register(TestService)
        container.register(AsyncTestService)
        
        service = await container.resolve(TestService)
        async_service = await container.resolve(AsyncTestService)
        
        assert service.cleanup_called is False
        assert async_service.cleanup_called is False
        
        await container.cleanup()
        
        assert service.cleanup_called is True
        assert async_service.cleanup_called is True
    
    @pytest.mark.asyncio
    async def test_scoped_container(self, container):
        """Test scoped container behavior"""
        container.register(TestService)
        
        # Create instance in outer scope
        outer_instance = await container.resolve(TestService)
        
        async with container.scope() as scoped_container:
            # Register additional service in scope
            scoped_container.register(AsyncTestService)
            
            # Resolve in scope - should get same outer instance
            scoped_instance = await scoped_container.resolve(TestService)
            async_instance = await scoped_container.resolve(AsyncTestService)
            
            assert scoped_instance is outer_instance
            assert isinstance(async_instance, AsyncTestService)
        
        # After scope, scoped service should not be available
        with pytest.raises(DependencyResolutionError):
            await container.resolve(AsyncTestService)


class TestContainerBuilder:
    """Test the ContainerBuilder"""
    
    def test_builder_pattern(self):
        """Test fluent builder pattern"""
        builder = ContainerBuilder()
        
        container = (builder
                    .add_singleton(TestService)
                    .add_transient(TestRepository)
                    .add_instance(ITestService, TestService("instance"))
                    .build())
        
        assert isinstance(container, Container)
        assert TestService in container._services
        assert TestRepository in container._services
        assert ITestService in container._instances
        
        # Verify configurations
        assert container._services[TestService].singleton is True
        assert container._services[TestRepository].singleton is False
    
    @pytest.mark.asyncio
    async def test_builder_creates_working_container(self):
        """Test that builder creates a working container"""
        builder = ContainerBuilder()
        container = (builder
                    .add_singleton(TestService)
                    .add_singleton(TestRepository)
                    .build())
        
        repository = await container.resolve(TestRepository)
        
        assert isinstance(repository, TestRepository)
        assert isinstance(repository.service, TestService)


class TestDependencyInjection:
    """Test dependency injection features"""
    
    @pytest.mark.asyncio
    async def test_inject_decorator(self):
        """Test the inject decorator"""
        container = Container()
        container.register_instance(TestService, TestService("injected"))
        
        # Mock the global container
        import src.core.container as container_module
        original_container = container_module._application_container
        container_module._application_container = container
        
        try:
            @inject(service=TestService)
            async def test_function(service):
                return service.get_value()
            
            result = await test_function()
            assert result == "injected"
            
        finally:
            container_module._application_container = original_container
    
    @pytest.mark.asyncio
    async def test_inject_decorator_with_explicit_args(self):
        """Test inject decorator doesn't override explicit arguments"""
        container = Container()
        container.register_instance(TestService, TestService("injected"))
        
        import src.core.container as container_module
        original_container = container_module._application_container
        container_module._application_container = container
        
        try:
            @inject(service=TestService)
            async def test_function(service):
                return service.get_value()
            
            explicit_service = TestService("explicit")
            result = await test_function(service=explicit_service)
            assert result == "explicit"
            
        finally:
            container_module._application_container = original_container


class TestServiceDefinition:
    """Test ServiceDefinition dataclass"""
    
    def test_service_definition_creation(self):
        """Test creating ServiceDefinition"""
        def test_factory():
            return TestService()
        
        service_def = ServiceDefinition(
            service_type=TestService,
            factory=test_factory,
            singleton=False,
            async_factory=True,
            dependencies={'dep': str}
        )
        
        assert service_def.service_type == TestService
        assert service_def.factory == test_factory
        assert service_def.singleton is False
        assert service_def.async_factory is True
        assert service_def.dependencies == {'dep': str}


class TestContainerErrorHandling:
    """Test container error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_cleanup_continues_on_error(self):
        """Test that cleanup continues even if some services fail"""
        container = Container()
        
        # Create services where one fails during cleanup
        good_service = MagicMock()
        good_service.cleanup = MagicMock()
        
        bad_service = MagicMock()
        bad_service.cleanup = MagicMock(side_effect=Exception("Cleanup failed"))
        
        container.register_instance(TestService, good_service)
        container.register_instance(AsyncTestService, bad_service)
        
        # Resolve to add to cleanup tasks
        await container.resolve(TestService)
        await container.resolve(AsyncTestService)
        
        # Cleanup should raise exception but call all cleanups
        with pytest.raises(Exception, match="Cleanup errors occurred"):
            await container.cleanup()
        
        good_service.cleanup.assert_called_once()
        bad_service.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_dependency_resolution_error_details(self):
        """Test that dependency resolution errors provide helpful details"""
        container = Container()
        
        class ServiceWithUnresolvableDependency:
            def __init__(self, missing_dep: str):
                self.missing_dep = missing_dep
        
        container.register(ServiceWithUnresolvableDependency)
        
        with pytest.raises(
            DependencyResolutionError,
            match="Cannot resolve parameter 'missing_dep' for ServiceWithUnresolvableDependency"
        ):
            await container.resolve(ServiceWithUnresolvableDependency)


class TestContainerIntegration:
    """Integration tests for container with real components"""
    
    @pytest.mark.asyncio
    async def test_application_container_creation(self):
        """Test creating the application container"""
        # This is an integration test that would use real components
        # For now, we'll test the structure
        from src.core.container import create_application_container
        
        # Mock the imports to avoid dependency issues in tests
        with pytest.mock.patch.multiple(
            'src.core.container',
            Config=MagicMock,
            load_config=MagicMock(return_value=MagicMock()),
            SQLiteStorage=MagicMock,
            QdrantStorage=MagicMock,
            LocalEmbeddingGenerator=MagicMock,
            PDFParser=MagicMock,
            TextChunker=MagicMock,
            VectorSearchEngine=MagicMock,
            TextSearchEngine=MagicMock,
            HybridSearch=MagicMock
        ):
            container = await create_application_container()
            assert isinstance(container, Container)
    
    @pytest.mark.asyncio
    async def test_global_container_management(self):
        """Test global container get and cleanup"""
        from src.core.container import get_container, cleanup_application_container
        
        # Mock to avoid real dependencies
        with pytest.mock.patch('src.core.container.create_application_container') as mock_create:
            mock_container = MagicMock()
            mock_container.cleanup = AsyncMock()
            mock_create.return_value = mock_container
            
            # Get container
            container = await get_container()
            assert container is mock_container
            
            # Second call should return same instance
            container2 = await get_container()
            assert container2 is mock_container
            
            # Cleanup
            await cleanup_application_container()
            mock_container.cleanup.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])