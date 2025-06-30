import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.services.auth_service import AuthService


class TestUserEndpoints:
    
    @pytest.mark.asyncio
    async def test_get_users_as_superuser(self, client: AsyncClient, db_session: AsyncSession):
        # Create superuser
        superuser = User(
            email="admin@example.com",
            password_hash=AuthService.get_password_hash("adminpass"),
            first_name="Admin",
            last_name="User",
            is_active=True,
            is_superuser=True
        )
        db_session.add(superuser)
        await db_session.commit()
        await db_session.refresh(superuser)
        
        # Create auth token for superuser
        token = AuthService.create_access_token(data={"sub": superuser.email})
        headers = {"Authorization": f"Bearer {token}"}
        
        response = await client.get("/users/", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    @pytest.mark.asyncio
    async def test_get_users_as_regular_user(self, client: AsyncClient, auth_headers):
        response = await client.get("/users/", headers=auth_headers)
        
        assert response.status_code == 403
        assert "Not enough permissions" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_users_no_auth(self, client: AsyncClient):
        response = await client.get("/users/")
        
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_get_user_by_id_own_profile(self, client: AsyncClient, test_user, auth_headers):
        response = await client.get(f"/users/{test_user.id}", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_user.id
        assert data["email"] == test_user.email

    @pytest.mark.asyncio
    async def test_get_user_by_id_other_user(self, client: AsyncClient, auth_headers, db_session: AsyncSession):
        # Create another user
        other_user = User(
            email="other@example.com",
            password_hash=AuthService.get_password_hash("otherpass"),
            first_name="Other",
            last_name="User",
            is_active=True
        )
        db_session.add(other_user)
        await db_session.commit()
        await db_session.refresh(other_user)
        
        response = await client.get(f"/users/{other_user.id}", headers=auth_headers)
        
        assert response.status_code == 403
        assert "Not enough permissions" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_user_by_id_as_superuser(self, client: AsyncClient, db_session: AsyncSession):
        # Create superuser
        superuser = User(
            email="admin2@example.com",
            password_hash=AuthService.get_password_hash("adminpass"),
            first_name="Admin",
            last_name="User",
            is_active=True,
            is_superuser=True
        )
        db_session.add(superuser)
        
        # Create regular user
        regular_user = User(
            email="regular@example.com",
            password_hash=AuthService.get_password_hash("regularpass"),
            first_name="Regular",
            last_name="User",
            is_active=True
        )
        db_session.add(regular_user)
        await db_session.commit()
        await db_session.refresh(superuser)
        await db_session.refresh(regular_user)
        
        # Auth as superuser
        token = AuthService.create_access_token(data={"sub": superuser.email})
        headers = {"Authorization": f"Bearer {token}"}
        
        response = await client.get(f"/users/{regular_user.id}", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == regular_user.id
        assert data["email"] == regular_user.email

    @pytest.mark.asyncio
    async def test_get_nonexistent_user(self, client: AsyncClient, auth_headers):
        response = await client.get("/users/99999", headers=auth_headers)
        
        assert response.status_code == 404
        assert "User not found" in response.json()["detail"]