#!/usr/bin/env python3
"""
Generate embeddings for the Neural SDE paper chunks.

This script generates embeddings for the already integrated Neural SDE paper
and stores them in the vector database for semantic search.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List
import sqlite3

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.models import Chunk, ChunkType
from src.core.config import get_config
from src.ingestion.local_embeddings import LocalEmbeddingGenerator
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuralSDEEmbeddingGenerator:
    """Generate embeddings for Neural SDE paper chunks"""
    
    def __init__(self):
        self.config = get_config()
        self.db_path = self.config.database.sqlite.path
        self.embedding_generator = None
    
    async def initialize(self):
        """Initialize embedding generator"""
        logger.info("Initializing embedding generator...")
        
        try:
            self.embedding_generator = LocalEmbeddingGenerator(self.config)
            logger.info("Local embedding generator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embedding generator: {e}")
            raise
    
    def load_neural_sde_chunks(self) -> List[Chunk]:
        """Load Neural SDE paper chunks from database"""
        logger.info("Loading Neural SDE paper chunks...")
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Get Neural SDE book
            cursor.execute("SELECT * FROM books WHERE id = ?", ("neural_sde_bayesian_calibration",))
            book_row = cursor.fetchone()
            
            if not book_row:
                logger.error("Neural SDE book not found in database")
                return []
            
            logger.info(f"Found book: {book_row['title']}")
            
            # Get chunks
            cursor.execute("""
                SELECT * FROM chunks 
                WHERE book_id = ? 
                ORDER BY chunk_index
            """, ("neural_sde_bayesian_calibration",))
            
            chunk_rows = cursor.fetchall()
            logger.info(f"Found {len(chunk_rows)} chunks")
            
            # Convert to Chunk objects
            chunks = []
            for row in chunk_rows:
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                
                chunk = Chunk(
                    id=row['id'],
                    book_id=row['book_id'],
                    chunk_index=row['chunk_index'],
                    text=row['text'],
                    chunk_type=ChunkType(row['chunk_type']) if row['chunk_type'] else ChunkType.TEXT,
                    embedding_id=row['embedding_id'],
                    chapter=row['chapter'],
                    section=row['section'],
                    page_start=row['page_start'],
                    page_end=row['page_end'],
                    previous_chunk_id=row['previous_chunk_id'],
                    next_chunk_id=row['next_chunk_id'],
                    metadata=metadata,
                    created_at=datetime.fromisoformat(row['created_at'])
                )
                chunks.append(chunk)
            
            return chunks
            
        finally:
            conn.close()
    
    async def generate_embeddings_for_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """Generate embeddings for chunks"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        if not self.embedding_generator:
            raise ValueError("Embedding generator not initialized")
        
        # Check if Ollama is available
        try:
            await self.embedding_generator._verify_ollama()
            logger.info("Ollama is available for embedding generation")
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            logger.info("Please ensure Ollama is running: ollama serve")
            logger.info(f"And that the model is installed: ollama pull {self.config.embedding.model}")
            raise
        
        # Generate embeddings in batches
        batch_size = self.config.embedding.batch_size
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            try:
                batch_embeddings = await self.embedding_generator.generate_embeddings(batch_chunks)
                all_embeddings.extend(batch_embeddings)
                
                # Update embedding IDs
                for j, chunk in enumerate(batch_chunks):
                    chunk.embedding_id = chunk.id
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                raise
        
        logger.info(f"Generated {len(all_embeddings)} embeddings total")
        return all_embeddings
    
    def save_embeddings_to_file(self, chunks: List[Chunk], embeddings: List[List[float]], output_file: str = "neural_sde_embeddings.json"):
        """Save embeddings to a JSON file for manual inspection or backup"""
        logger.info(f"Saving embeddings to {output_file}...")
        
        embeddings_data = {
            "book_id": "neural_sde_bayesian_calibration",
            "title": "Robust financial calibration: a Bayesian approach for neural SDEs",
            "generated_at": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "chunks": []
        }
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk_data = {
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "section": chunk.chapter,
                "metadata": chunk.metadata,
                "embedding": embedding
            }
            embeddings_data["chunks"].append(chunk_data)
        
        with open(output_file, 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        logger.info(f"Saved embeddings to {output_file}")
    
    def update_chunk_embedding_ids(self, chunks: List[Chunk]):
        """Update the chunk embedding IDs in the database"""
        logger.info("Updating chunk embedding IDs in database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            updates = [(chunk.embedding_id, chunk.id) for chunk in chunks if chunk.embedding_id]
            
            cursor.executemany(
                "UPDATE chunks SET embedding_id = ? WHERE id = ?",
                updates
            )
            
            conn.commit()
            logger.info(f"Updated {len(updates)} chunk embedding IDs")
            
        finally:
            conn.close()

async def main():
    """Main function"""
    generator = NeuralSDEEmbeddingGenerator()
    
    try:
        # Initialize
        await generator.initialize()
        
        # Load chunks
        chunks = generator.load_neural_sde_chunks()
        
        if not chunks:
            logger.error("No Neural SDE chunks found. Please run the integration script first.")
            return
        
        logger.info(f"Loaded {len(chunks)} chunks for embedding generation")
        
        # Generate embeddings
        embeddings = await generator.generate_embeddings_for_chunks(chunks)
        
        # Save embeddings to file for inspection
        generator.save_embeddings_to_file(chunks, embeddings)
        
        # Update database with embedding IDs
        generator.update_chunk_embedding_ids(chunks)
        
        logger.info("✅ Embedding generation complete!")
        logger.info(f"Generated {len(embeddings)} embeddings")
        logger.info(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
        logger.info(f"Results saved to: neural_sde_embeddings.json")
        
        # Show some sample chunks and their first few embedding values
        logger.info("\nSample chunks with embeddings:")
        for i, (chunk, embedding) in enumerate(zip(chunks[:3], embeddings[:3])):
            logger.info(f"Chunk {i+1}: {chunk.text[:100]}...")
            logger.info(f"  Embedding (first 5 values): {embedding[:5]}")
        
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())