"""
User management system for authentication and authorization
"""

import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import bcrypt
from pydantic import BaseModel, Field
import aiosqlite
from pathlib import Path

from ..core.config import get_config
from .models import User


class UserCreate(BaseModel):
    """User creation model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=100)
    password: str = Field(..., min_length=8)
    role: str = Field(default="user")
    full_name: Optional[str] = None


class UserUpdate(BaseModel):
    """User update model"""
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class LoginAttempt(BaseModel):
    """Login attempt tracking"""
    username: str
    ip_address: str
    success: bool
    timestamp: datetime
    user_agent: Optional[str] = None


class UserManager:
    """User management with database integration"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.config = get_config()
        self.db_path = db_path or str(Path(self.config.database.sqlite.path).parent / "users.db")
        self._initialized = False
    
    async def initialize(self):
        """Initialize user database tables"""
        if self._initialized:
            return
            
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(self.db_path) as db:
            # Create users table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    full_name TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login TEXT,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TEXT
                )
            """)
            
            # Create login attempts table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS login_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_agent TEXT
                )
            """)
            
            # Create indexes separately
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_username_timestamp 
                ON login_attempts (username, timestamp)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_ip_timestamp 
                ON login_attempts (ip_address, timestamp)
            """)
            
            # Create API keys table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    scopes TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    last_used TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            
            await db.commit()
            
        self._initialized = True
    
    def _hash_password(self, password: str) -> tuple[str, str]:
        """Hash password with salt"""
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8'), salt.decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID"""
        return secrets.token_urlsafe(16)
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user"""
        await self.initialize()
        
        # Validate password strength
        if len(user_data.password) < self.config.api.auth.min_password_length:
            raise ValueError(f"Password must be at least {self.config.api.auth.min_password_length} characters")
        
        user_id = self._generate_user_id()
        password_hash, salt = self._hash_password(user_data.password)
        now = datetime.utcnow().isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.execute("""
                    INSERT INTO users (
                        id, username, email, password_hash, salt, role,
                        full_name, is_active, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, user_data.username, user_data.email,
                    password_hash, salt, user_data.role,
                    user_data.full_name, True, now, now
                ))
                await db.commit()
                
                return User(
                    id=user_id,
                    username=user_data.username,
                    email=user_data.email,
                    role=user_data.role,
                    full_name=user_data.full_name,
                    is_active=True,
                    created_at=now
                )
                
            except aiosqlite.IntegrityError as e:
                if "username" in str(e):
                    raise ValueError("Username already exists")
                elif "email" in str(e):
                    raise ValueError("Email already exists")
                else:
                    raise ValueError("User creation failed")
    
    async def authenticate_user(self, username: str, password: str, ip_address: str = "unknown") -> Optional[User]:
        """Authenticate user with login attempt tracking"""
        await self.initialize()
        
        # Check if user is locked
        if await self._is_user_locked(username):
            await self._log_login_attempt(username, ip_address, False)
            return None
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT id, username, email, password_hash, role, full_name, 
                       is_active, created_at, failed_login_attempts
                FROM users WHERE username = ? AND is_active = 1
            """, (username,)) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    await self._log_login_attempt(username, ip_address, False)
                    return None
                
                user_id, username, email, password_hash, role, full_name, is_active, created_at, failed_attempts = row
                
                if self._verify_password(password, password_hash):
                    # Successful login - reset failed attempts and update last login
                    await db.execute("""
                        UPDATE users SET failed_login_attempts = 0, last_login = ?, locked_until = NULL
                        WHERE id = ?
                    """, (datetime.utcnow().isoformat(), user_id))
                    await db.commit()
                    
                    await self._log_login_attempt(username, ip_address, True)
                    
                    return User(
                        id=user_id,
                        username=username,
                        email=email,
                        role=role,
                        full_name=full_name,
                        is_active=bool(is_active),
                        created_at=created_at
                    )
                else:
                    # Failed login - increment attempts and potentially lock
                    new_attempts = failed_attempts + 1
                    locked_until = None
                    
                    if new_attempts >= self.config.api.auth.max_login_attempts:
                        lockout_time = datetime.utcnow() + timedelta(minutes=self.config.api.auth.lockout_duration_minutes)
                        locked_until = lockout_time.isoformat()
                    
                    await db.execute("""
                        UPDATE users SET failed_login_attempts = ?, locked_until = ?
                        WHERE id = ?
                    """, (new_attempts, locked_until, user_id))
                    await db.commit()
                    
                    await self._log_login_attempt(username, ip_address, False)
                    return None
    
    async def _is_user_locked(self, username: str) -> bool:
        """Check if user is currently locked"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT locked_until FROM users WHERE username = ?
            """, (username,)) as cursor:
                row = await cursor.fetchone()
                
                if not row or not row[0]:
                    return False
                
                locked_until = datetime.fromisoformat(row[0])
                return datetime.utcnow() < locked_until
    
    async def _log_login_attempt(self, username: str, ip_address: str, success: bool, user_agent: Optional[str] = None):
        """Log login attempt"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO login_attempts (username, ip_address, success, timestamp, user_agent)
                VALUES (?, ?, ?, ?, ?)
            """, (username, ip_address, success, datetime.utcnow().isoformat(), user_agent))
            await db.commit()
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT id, username, email, role, full_name, is_active, created_at, last_login
                FROM users WHERE id = ? AND is_active = 1
            """, (user_id,)) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                return User(
                    id=row[0],
                    username=row[1],
                    email=row[2],
                    role=row[3],
                    full_name=row[4],
                    is_active=bool(row[5]),
                    created_at=row[6],
                    last_login=row[7]
                )
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT id, username, email, role, full_name, is_active, created_at, last_login
                FROM users WHERE username = ? AND is_active = 1
            """, (username,)) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                return User(
                    id=row[0],
                    username=row[1],
                    email=row[2],
                    role=row[3],
                    full_name=row[4],
                    is_active=bool(row[5]),
                    created_at=row[6],
                    last_login=row[7]
                )
    
    async def update_user(self, user_id: str, user_data: UserUpdate) -> Optional[User]:
        """Update user information"""
        await self.initialize()
        
        updates = []
        values = []
        
        if user_data.email is not None:
            updates.append("email = ?")
            values.append(user_data.email)
        
        if user_data.full_name is not None:
            updates.append("full_name = ?")
            values.append(user_data.full_name)
        
        if user_data.role is not None:
            updates.append("role = ?")
            values.append(user_data.role)
        
        if user_data.is_active is not None:
            updates.append("is_active = ?")
            values.append(user_data.is_active)
        
        if not updates:
            return await self.get_user_by_id(user_id)
        
        updates.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        values.append(user_id)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"""
                UPDATE users SET {', '.join(updates)} WHERE id = ?
            """, values)
            await db.commit()
        
        return await self.get_user_by_id(user_id)
    
    async def list_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """List all users"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT id, username, email, role, full_name, is_active, created_at, last_login
                FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?
            """, (limit, offset)) as cursor:
                rows = await cursor.fetchall()
                
                return [
                    User(
                        id=row[0],
                        username=row[1],
                        email=row[2],
                        role=row[3],
                        full_name=row[4],
                        is_active=bool(row[5]),
                        created_at=row[6],
                        last_login=row[7]
                    )
                    for row in rows
                ]
    
    async def delete_user(self, user_id: str) -> bool:
        """Soft delete user (deactivate)"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE users SET is_active = 0, updated_at = ? WHERE id = ?
            """, (datetime.utcnow().isoformat(), user_id))
            
            rows_affected = db.total_changes
            await db.commit()
            
            return rows_affected > 0
    
    async def get_login_attempts(self, username: Optional[str] = None, limit: int = 100) -> List[LoginAttempt]:
        """Get recent login attempts"""
        await self.initialize()
        
        query = """
            SELECT username, ip_address, success, timestamp, user_agent
            FROM login_attempts 
        """
        params = []
        
        if username:
            query += "WHERE username = ? "
            params.append(username)
        
        query += "ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                
                return [
                    LoginAttempt(
                        username=row[0],
                        ip_address=row[1],
                        success=bool(row[2]),
                        timestamp=datetime.fromisoformat(row[3]),
                        user_agent=row[4]
                    )
                    for row in rows
                ]
    
    async def cleanup_old_login_attempts(self, days_old: int = 30):
        """Clean up old login attempts"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                DELETE FROM login_attempts WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))
            await db.commit()


# Singleton instance
_user_manager: Optional[UserManager] = None

def get_user_manager() -> UserManager:
    """Get user manager singleton"""
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
    return _user_manager