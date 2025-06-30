import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.schemas.user import UserCreate
from app.models.user import User


class TestAuthService:
    
    def test_password_hashing(self):
        password = "testpassword123"
        hashed = AuthService.get_password_hash(password)
        
        assert hashed != password
        assert AuthService.verify_password(password, hashed) is True
        assert AuthService.verify_password("wrongpassword", hashed) is False

    def test_create_access_token(self):
        data = {"sub": "test@example.com"}
        token = AuthService.create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_token_valid(self):
        data = {"sub": "test@example.com"}
        token = AuthService.create_access_token(data)
        
        email = AuthService.verify_token(token)
        assert email == "test@example.com"

    def test_verify_token_invalid(self):
        email = AuthService.verify_token("invalid.token.here")
        assert email is None


class TestUserService:
    
    @pytest.mark.asyncio
    async def test_create_user_success(self, db_session: AsyncSession):
        user_service = UserService(db_session)
        user_data = UserCreate(
            email="newuser@example.com",
            password="password123",
            first_name="New",
            last_name="User"
        )
        
        user = await user_service.create_user(user_data)
        
        assert user is not None
        assert user.email == user_data.email
        assert user.first_name == user_data.first_name
        assert user.last_name == user_data.last_name
        assert user.is_active is True
        assert user.password_hash != user_data.password

    @pytest.mark.asyncio
    async def test_create_user_duplicate_email(self, db_session: AsyncSession):
        user_service = UserService(db_session)
        user_data = UserCreate(
            email="duplicate@example.com",
            password="password123"
        )
        
        # Create first user
        await user_service.create_user(user_data)
        
        # Try to create duplicate
        duplicate_user = await user_service.create_user(user_data)
        assert duplicate_user is None

    @pytest.mark.asyncio
    async def test_get_user_by_email(self, db_session: AsyncSession, test_user: User):
        user_service = UserService(db_session)
        
        found_user = await user_service.get_user_by_email(test_user.email)
        
        assert found_user is not None
        assert found_user.id == test_user.id
        assert found_user.email == test_user.email

    @pytest.mark.asyncio
    async def test_get_user_by_email_not_found(self, db_session: AsyncSession):
        user_service = UserService(db_session)
        
        found_user = await user_service.get_user_by_email("nonexistent@example.com")
        
        assert found_user is None

    @pytest.mark.asyncio
    async def test_get_user_by_id(self, db_session: AsyncSession, test_user: User):
        user_service = UserService(db_session)
        
        found_user = await user_service.get_user_by_id(test_user.id)
        
        assert found_user is not None
        assert found_user.id == test_user.id
        assert found_user.email == test_user.email

    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, db_session: AsyncSession, test_user: User):
        user_service = UserService(db_session)
        
        authenticated_user = await user_service.authenticate_user(
            test_user.email, "testpassword"
        )
        
        assert authenticated_user is not None
        assert authenticated_user.id == test_user.id

    @pytest.mark.asyncio
    async def test_authenticate_user_wrong_password(self, db_session: AsyncSession, test_user: User):
        user_service = UserService(db_session)
        
        authenticated_user = await user_service.authenticate_user(
            test_user.email, "wrongpassword"
        )
        
        assert authenticated_user is None

    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(self, db_session: AsyncSession):
        user_service = UserService(db_session)
        
        authenticated_user = await user_service.authenticate_user(
            "nonexistent@example.com", "password"
        )
        
        assert authenticated_user is None