from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError
from typing import Optional, List
from app.models.user import User
from app.schemas.user import UserCreate
from app.services.auth_service import AuthService


class UserService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_email(self, email: str) -> Optional[User]:
        result = await self.db.execute(select(User).filter(User.email == email))
        return result.scalars().first()

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        result = await self.db.execute(select(User).filter(User.id == user_id))
        return result.scalars().first()

    async def get_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        result = await self.db.execute(select(User).offset(skip).limit(limit))
        return result.scalars().all()

    async def create_user(self, user_data: UserCreate) -> Optional[User]:
        try:
            hashed_password = AuthService.get_password_hash(user_data.password)
            db_user = User(
                email=user_data.email,
                password_hash=hashed_password,
                first_name=user_data.first_name,
                last_name=user_data.last_name
            )
            self.db.add(db_user)
            await self.db.commit()
            await self.db.refresh(db_user)
            return db_user
        except IntegrityError:
            await self.db.rollback()
            return None

    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        user = await self.get_user_by_email(email)
        if not user:
            return None
        if not AuthService.verify_password(password, user.password_hash):
            return None
        return user