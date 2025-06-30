import pytest
from httpx import AsyncClient


class TestAuthEndpoints:
    
    @pytest.mark.asyncio
    async def test_register_user_success(self, client: AsyncClient):
        user_data = {
            "email": "newuser@example.com",
            "password": "strongpassword123",
            "first_name": "New",
            "last_name": "User"
        }
        
        response = await client.post("/auth/register", json=user_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["first_name"] == user_data["first_name"]
        assert data["last_name"] == user_data["last_name"]
        assert data["is_active"] is True
        assert "password" not in data
        assert "password_hash" not in data

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, client: AsyncClient, test_user):
        user_data = {
            "email": test_user.email,
            "password": "anotherpassword123"
        }
        
        response = await client.post("/auth/register", json=user_data)
        
        assert response.status_code == 400
        assert "Email already registered" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_register_invalid_email(self, client: AsyncClient):
        user_data = {
            "email": "not-an-email",
            "password": "password123"
        }
        
        response = await client.post("/auth/register", json=user_data)
        
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_login_success(self, client: AsyncClient, test_user):
        login_data = {
            "email": test_user.email,
            "password": "testpassword"
        }
        
        response = await client.post("/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client: AsyncClient, test_user):
        login_data = {
            "email": test_user.email,
            "password": "wrongpassword"
        }
        
        response = await client.post("/auth/login", json=login_data)
        
        assert response.status_code == 401
        assert "Incorrect email or password" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, client: AsyncClient):
        login_data = {
            "email": "nonexistent@example.com",
            "password": "password123"
        }
        
        response = await client.post("/auth/login", json=login_data)
        
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_success(self, client: AsyncClient, auth_headers):
        response = await client.get("/auth/me", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["first_name"] == "Test"
        assert data["last_name"] == "User"

    @pytest.mark.asyncio
    async def test_get_current_user_no_token(self, client: AsyncClient):
        response = await client.get("/auth/me")
        
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, client: AsyncClient):
        headers = {"Authorization": "Bearer invalid-token"}
        response = await client.get("/auth/me", headers=headers)
        
        assert response.status_code == 401