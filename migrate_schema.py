#!/usr/bin/env python3
"""
Database schema migration for TradeKnowledge
Adds missing columns to match the Book model
"""

import sqlite3
import sys
from pathlib import Path

def migrate_database(db_path="data/knowledge.db"):
    """Migrate database schema to match current models"""
    
    print(f"🔄 Migrating database schema: {db_path}")
    
    # Ensure database exists
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check current schema
            cursor.execute("PRAGMA table_info(books)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            print(f"📋 Current columns: {list(columns.keys())}")
            
            # Add missing columns to books table
            migrations = []
            
            if 'total_pages' not in columns:
                migrations.append("ALTER TABLE books ADD COLUMN total_pages INTEGER")
                print("➕ Adding total_pages column to books")
            
            if 'categories' not in columns:
                migrations.append("ALTER TABLE books ADD COLUMN categories TEXT")
                print("➕ Adding categories column to books")
            
            # Execute book migrations
            for migration in migrations:
                cursor.execute(migration)
                print(f"✅ Executed: {migration}")
            
            # Check chunks table schema
            cursor.execute("PRAGMA table_info(chunks)")
            chunk_columns = {row[1]: row[2] for row in cursor.fetchall()}
            print(f"📋 Current chunk columns: {list(chunk_columns.keys())}")
            
            # Add missing columns to chunks table
            chunk_migrations = []
            
            if 'chunk_type' not in chunk_columns:
                chunk_migrations.append("ALTER TABLE chunks ADD COLUMN chunk_type TEXT DEFAULT 'text'")
                print("➕ Adding chunk_type column to chunks")
            
            if 'chapter' not in chunk_columns:
                chunk_migrations.append("ALTER TABLE chunks ADD COLUMN chapter TEXT")
                print("➕ Adding chapter column to chunks")
            
            if 'section' not in chunk_columns:
                chunk_migrations.append("ALTER TABLE chunks ADD COLUMN section TEXT")
                print("➕ Adding section column to chunks")
                
            if 'page_start' not in chunk_columns:
                chunk_migrations.append("ALTER TABLE chunks ADD COLUMN page_start INTEGER")
                print("➕ Adding page_start column to chunks")
                
            if 'page_end' not in chunk_columns:
                chunk_migrations.append("ALTER TABLE chunks ADD COLUMN page_end INTEGER")
                print("➕ Adding page_end column to chunks")
                
            if 'previous_chunk_id' not in chunk_columns:
                chunk_migrations.append("ALTER TABLE chunks ADD COLUMN previous_chunk_id TEXT")
                print("➕ Adding previous_chunk_id column to chunks")
                
            if 'next_chunk_id' not in chunk_columns:
                chunk_migrations.append("ALTER TABLE chunks ADD COLUMN next_chunk_id TEXT")
                print("➕ Adding next_chunk_id column to chunks")
            
            # Execute chunk migrations
            for migration in chunk_migrations:
                cursor.execute(migration)
                print(f"✅ Executed: {migration}")
            
            # Verify new schemas
            cursor.execute("PRAGMA table_info(books)")
            new_book_columns = {row[1]: row[2] for row in cursor.fetchall()}
            print(f"📋 Updated book columns: {list(new_book_columns.keys())}")
            
            cursor.execute("PRAGMA table_info(chunks)")
            new_chunk_columns = {row[1]: row[2] for row in cursor.fetchall()}
            print(f"📋 Updated chunk columns: {list(new_chunk_columns.keys())}")
            
            conn.commit()
            print("✅ Migration completed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/knowledge.db"
    success = migrate_database(db_path)
    sys.exit(0 if success else 1)