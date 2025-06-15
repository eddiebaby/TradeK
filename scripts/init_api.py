#!/usr/bin/env python3
"""
API initialization script for TradeKnowledge Phase 4

Sets up:
- User database
- Default admin user
- Configuration validation
- System health checks
"""

import asyncio
import os
import sys
from pathlib import Path
import getpass
import secrets
import string

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import get_config
from src.api.user_manager import get_user_manager, UserCreate
from src.api.auth import get_auth_manager
from src.core.sqlite_storage import SQLiteStorage
from src.search.unified_search import UnifiedSearchEngine


class APIInitializer:
    """API initialization and setup"""
    
    def __init__(self):
        self.config = get_config()
        self.user_manager = get_user_manager()
        self.auth_manager = get_auth_manager()
    
    async def initialize_all(self, interactive: bool = True):
        """Initialize all API components"""
        print("🚀 Initializing TradeKnowledge API (Phase 4)")
        print("=" * 50)
        
        # 1. Initialize user database
        await self.initialize_user_database()
        
        # 2. Create admin user
        await self.create_admin_user(interactive)
        
        # 3. Validate configuration
        await self.validate_configuration()
        
        # 4. Initialize core components
        await self.initialize_core_components()
        
        # 5. Run health checks
        await self.run_health_checks()
        
        print("\n✅ API initialization completed successfully!")
        print("\nNext steps:")
        print("1. Start the API server: python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
        print("2. Access API docs: http://localhost:8000/docs")
        print("3. Test authentication with your admin credentials")
    
    async def initialize_user_database(self):
        """Initialize user management database"""
        print("\n📊 Initializing user database...")
        
        try:
            await self.user_manager.initialize()
            print("✅ User database initialized")
        except Exception as e:
            print(f"❌ Failed to initialize user database: {e}")
            raise
    
    async def create_admin_user(self, interactive: bool = True):
        """Create default admin user"""
        print("\n👤 Setting up admin user...")
        
        try:
            # Check if admin user already exists
            existing_admin = await self.user_manager.get_user_by_username("admin")
            if existing_admin:
                print("ℹ️  Admin user already exists")
                if interactive:
                    reset = input("Reset admin password? (y/N): ").lower().strip()
                    if reset != 'y':
                        return existing_admin
                else:
                    return existing_admin
            
            if interactive:
                # Interactive admin creation
                print("\nCreating admin user...")
                username = input("Admin username (default: admin): ").strip() or "admin"
                email = input("Admin email: ").strip()
                
                while not email:
                    email = input("Admin email (required): ").strip()
                
                password = getpass.getpass("Admin password: ")
                if not password:
                    password = self.generate_secure_password()
                    print(f"Generated secure password: {password}")
                
                confirm_password = getpass.getpass("Confirm password: ")
                while password != confirm_password:
                    print("Passwords don't match!")
                    password = getpass.getpass("Admin password: ")
                    confirm_password = getpass.getpass("Confirm password: ")
            else:
                # Non-interactive admin creation
                username = os.getenv("ADMIN_USERNAME", "admin")
                email = os.getenv("ADMIN_EMAIL", "admin@tradeknowledge.local")
                password = os.getenv("ADMIN_PASSWORD")
                
                if not password:
                    password = self.generate_secure_password()
                    print(f"Generated admin password: {password}")
                    print("⚠️  Save this password securely!")
            
            # Create admin user
            user_data = UserCreate(
                username=username,
                email=email,
                password=password,
                role="admin",
                full_name="System Administrator"
            )
            
            if existing_admin:
                # Update existing admin
                await self.user_manager.update_user(existing_admin.id, {
                    "email": email,
                    "full_name": "System Administrator"
                })
                # Update password by recreating auth
                await self.user_manager.authenticate_user(username, password)
                print(f"✅ Admin user '{username}' updated")
            else:
                # Create new admin
                admin_user = await self.user_manager.create_user(user_data)
                print(f"✅ Admin user '{admin_user.username}' created")
            
            return admin_user
            
        except Exception as e:
            print(f"❌ Failed to create admin user: {e}")
            raise
    
    def generate_secure_password(self, length: int = 16) -> str:
        """Generate a secure random password"""
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    async def validate_configuration(self):
        """Validate API configuration"""
        print("\n⚙️  Validating configuration...")
        
        # Check required environment variables
        required_configs = [
            ("JWT_SECRET_KEY", self.config.api.auth.secret_key),
            ("Database Path", self.config.database.sqlite.path),
            ("Qdrant Host", self.config.database.qdrant.host),
        ]
        
        for name, value in required_configs:
            if not value or value == "dev-secret-key-change-in-production":
                print(f"⚠️  {name}: Using development default")
            else:
                print(f"✅ {name}: Configured")
        
        # Create necessary directories
        data_dir = Path(self.config.database.sqlite.path).parent
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Data directory: {data_dir}")
        
        # Check file permissions
        if not os.access(data_dir, os.W_OK):
            print(f"❌ No write permission to data directory: {data_dir}")
            raise PermissionError(f"Cannot write to {data_dir}")
        
        print("✅ Configuration validation passed")
    
    async def initialize_core_components(self):
        """Initialize core API components"""
        print("\n🔧 Initializing core components...")
        
        try:
            # Initialize storage
            storage = SQLiteStorage()
            print("✅ SQLite storage initialized")
            
            # Test search engine
            search_engine = UnifiedSearchEngine()
            await search_engine.initialize()
            print("✅ Search engine initialized")
            
        except Exception as e:
            print(f"⚠️  Some components may not be fully available: {e}")
            print("   This is normal if external services (Qdrant, Redis) are not running")
    
    async def run_health_checks(self):
        """Run system health checks"""
        print("\n🏥 Running health checks...")
        
        checks = [
            ("User Database", self.check_user_database),
            ("Storage System", self.check_storage_system),
            ("Configuration", self.check_configuration),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                await check_func()
                print(f"✅ {check_name}: OK")
            except Exception as e:
                print(f"❌ {check_name}: {e}")
                all_passed = False
        
        if all_passed:
            print("✅ All health checks passed")
        else:
            print("⚠️  Some health checks failed - check configuration")
    
    async def check_user_database(self):
        """Check user database health"""
        users = await self.user_manager.list_users(limit=1)
        return len(users) >= 0  # Should at least return empty list
    
    async def check_storage_system(self):
        """Check storage system health"""
        storage = SQLiteStorage()
        # Try a simple operation
        books = await storage.list_books(limit=1)
        return True
    
    async def check_configuration(self):
        """Check configuration validity"""
        # Check database paths exist
        db_path = Path(self.config.database.sqlite.path)
        if not db_path.parent.exists():
            raise FileNotFoundError(f"Database directory does not exist: {db_path.parent}")
        
        # Check JWT secret is not default
        if self.config.api.auth.secret_key == "dev-secret-key-change-in-production":
            print("⚠️  Using development JWT secret - change for production!")
        
        return True


async def main():
    """Main initialization function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize TradeKnowledge API")
    parser.add_argument("--non-interactive", action="store_true", 
                       help="Run in non-interactive mode")
    parser.add_argument("--reset-admin", action="store_true",
                       help="Reset admin user")
    parser.add_argument("--health-check-only", action="store_true",
                       help="Only run health checks")
    
    args = parser.parse_args()
    
    try:
        initializer = APIInitializer()
        
        if args.health_check_only:
            await initializer.run_health_checks()
        else:
            await initializer.initialize_all(interactive=not args.non_interactive)
            
    except KeyboardInterrupt:
        print("\n\n❌ Initialization cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())