"""
Book management API endpoints

Handles book upload, processing, status tracking, and metadata management.
"""

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import structlog

from ..models import *
from ..main import get_book_processor, get_current_user
from ...ingestion.enhanced_book_processor import EnhancedBookProcessor

logger = structlog.get_logger(__name__)

router = APIRouter()

@router.post("/upload", response_model=BookUploadResponse)
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Book file to upload"),
    title: str = Form(None, description="Book title"),
    author: str = Form(None, description="Book author"),
    description: str = Form(None, description="Book description"),
    tags: str = Form("", description="Comma-separated tags"),
    category: str = Form(None, description="Book category"),
    language: str = Form("en", description="Book language"),
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """
    Upload a new book for processing
    
    Supported formats:
    - PDF (including scanned)
    - EPUB
    - Jupyter Notebooks (.ipynb)
    
    The file will be processed asynchronously and indexed for search.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size (100MB limit)
        if file.size and file.size > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 100MB)")
        
        # Check file type
        supported_extensions = ['.pdf', '.epub', '.ipynb']
        file_path = Path(file.filename)
        if file_path.suffix.lower() not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported: {', '.join(supported_extensions)}"
            )
        
        # Generate unique book ID
        book_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_path.suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = Path(tmp_file.name)
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        
        # Create book metadata
        book_metadata = BookUploadRequest(
            title=title or file_path.stem,
            author=author,
            description=description,
            tags=tag_list,
            category=category,
            language=language
        )
        
        logger.info(
            "Book upload started",
            user_id=user.id,
            book_id=book_id,
            filename=file.filename,
            file_size=file.size
        )
        
        # Start processing in background
        job_id = str(uuid.uuid4())
        background_tasks.add_task(
            process_book_background,
            book_processor,
            book_id,
            tmp_file_path,
            book_metadata,
            user.id,
            job_id
        )
        
        return BookUploadResponse(
            book_id=book_id,
            status=BookStatus.UPLOADED,
            processing_job_id=job_id,
            message=f"Book '{book_metadata.title}' uploaded successfully and is being processed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Book upload failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/list", response_model=BookListResponse)
async def list_books(
    pagination: PaginationParams = Depends(),
    category: str = None,
    author: str = None,
    tags: str = None,
    status: BookStatus = None,
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """
    List books with filtering and pagination
    
    Supports filtering by:
    - Category
    - Author
    - Tags (comma-separated)
    - Processing status
    """
    try:
        # Build filters
        filters = {}
        if category:
            filters['category'] = category
        if author:
            filters['author'] = author
        if tags:
            filters['tags'] = [tag.strip() for tag in tags.split(",")]
        if status:
            filters['status'] = status.value
        
        # Get books
        books_data = await book_processor.list_books(
            offset=pagination.offset,
            limit=pagination.size,
            filters=filters
        )
        
        # Convert to API format
        books = []
        for book_data in books_data['books']:
            books.append(BookInfo(
                id=book_data['id'],
                title=book_data['title'],
                author=book_data.get('author'),
                file_path=book_data['file_path'],
                file_size=book_data['file_size'],
                total_pages=book_data['total_pages'],
                total_chunks=book_data['total_chunks'],
                status=BookStatus(book_data['status']),
                upload_date=book_data['upload_date'],
                last_updated=book_data['last_updated'],
                metadata=book_data.get('metadata', {}),
                tags=book_data.get('tags', [])
            ))
        
        return BookListResponse.create(
            items=books,
            pagination=pagination,
            total=books_data['total']
        )
        
    except Exception as e:
        logger.error("List books failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list books: {str(e)}")

@router.get("/{book_id}", response_model=BookInfo)
async def get_book_details(
    book_id: str,
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """Get detailed information about a specific book"""
    try:
        book_data = await book_processor.get_book(book_id)
        
        if not book_data:
            raise HTTPException(status_code=404, detail="Book not found")
        
        return BookInfo(
            id=book_data['id'],
            title=book_data['title'],
            author=book_data.get('author'),
            file_path=book_data['file_path'],
            file_size=book_data['file_size'],
            total_pages=book_data['total_pages'],
            total_chunks=book_data['total_chunks'],
            status=BookStatus(book_data['status']),
            upload_date=book_data['upload_date'],
            last_updated=book_data['last_updated'],
            metadata=book_data.get('metadata', {}),
            tags=book_data.get('tags', [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get book failed", book_id=book_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get book: {str(e)}")

@router.get("/{book_id}/status", response_model=BookProcessingStatus)
async def get_processing_status(
    book_id: str,
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """Get processing status for a book"""
    try:
        status_data = await book_processor.get_processing_status(book_id)
        
        if not status_data:
            raise HTTPException(status_code=404, detail="Book not found")
        
        return BookProcessingStatus(
            book_id=book_id,
            status=BookStatus(status_data['status']),
            progress=status_data.get('progress', 0.0),
            current_step=status_data.get('current_step', 'Unknown'),
            estimated_completion=status_data.get('estimated_completion'),
            error_message=status_data.get('error_message'),
            chunks_processed=status_data.get('chunks_processed', 0),
            total_chunks=status_data.get('total_chunks', 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get processing status failed", book_id=book_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.put("/{book_id}/metadata")
async def update_book_metadata(
    book_id: str,
    metadata: BookUploadRequest,
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """Update book metadata"""
    try:
        await book_processor.update_book_metadata(book_id, metadata.dict())
        
        logger.info("Book metadata updated", book_id=book_id, user_id=user.id)
        
        return {"success": True, "message": "Metadata updated successfully"}
        
    except Exception as e:
        logger.error("Update metadata failed", book_id=book_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update metadata: {str(e)}")

@router.post("/{book_id}/reprocess")
async def reprocess_book(
    book_id: str,
    background_tasks: BackgroundTasks,
    force: bool = False,
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """Reprocess a book (useful after algorithm updates)"""
    try:
        # Check if book exists
        book_data = await book_processor.get_book(book_id)
        if not book_data:
            raise HTTPException(status_code=404, detail="Book not found")
        
        # Check if already processing
        if book_data['status'] in ['processing', 'indexing'] and not force:
            raise HTTPException(
                status_code=409, 
                detail="Book is already being processed. Use force=true to override."
            )
        
        # Start reprocessing
        job_id = str(uuid.uuid4())
        background_tasks.add_task(
            reprocess_book_background,
            book_processor,
            book_id,
            user.id,
            job_id
        )
        
        logger.info("Book reprocessing started", book_id=book_id, user_id=user.id)
        
        return {
            "success": True,
            "message": "Book reprocessing started",
            "job_id": job_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Reprocess failed", book_id=book_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to reprocess: {str(e)}")

@router.delete("/{book_id}")
async def delete_book(
    book_id: str,
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """Delete a book and all its data"""
    try:
        await book_processor.delete_book(book_id)
        
        logger.info("Book deleted", book_id=book_id, user_id=user.id)
        
        return {"success": True, "message": "Book deleted successfully"}
        
    except Exception as e:
        logger.error("Delete book failed", book_id=book_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete book: {str(e)}")

@router.get("/{book_id}/chunks")
async def get_book_chunks(
    book_id: str,
    pagination: PaginationParams = Depends(),
    chunk_type: str = None,
    book_processor: EnhancedBookProcessor = Depends(get_book_processor),
    user = Depends(get_current_user)
):
    """Get chunks for a specific book"""
    try:
        chunks_data = await book_processor.get_book_chunks(
            book_id=book_id,
            offset=pagination.offset,
            limit=pagination.size,
            chunk_type=chunk_type
        )
        
        return {
            "book_id": book_id,
            "chunks": chunks_data['chunks'],
            "total": chunks_data['total'],
            "page": pagination.page,
            "size": pagination.size
        }
        
    except Exception as e:
        logger.error("Get book chunks failed", book_id=book_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get chunks: {str(e)}")

# Background task functions
async def process_book_background(
    book_processor: EnhancedBookProcessor,
    book_id: str,
    file_path: Path,
    metadata: BookUploadRequest,
    user_id: str,
    job_id: str
):
    """Process book in background"""
    try:
        logger.info("Starting book processing", book_id=book_id, job_id=job_id)
        
        # Process the book
        result = await book_processor.add_book(
            file_path=file_path,
            book_id=book_id,
            metadata=metadata.dict(),
            user_id=user_id
        )
        
        # Clean up temporary file
        file_path.unlink(missing_ok=True)
        
        logger.info("Book processing completed", book_id=book_id, job_id=job_id)
        
    except Exception as e:
        logger.error("Book processing failed", book_id=book_id, job_id=job_id, error=str(e))
        
        # Update status to failed
        await book_processor.update_book_status(
            book_id=book_id,
            status="failed",
            error_message=str(e)
        )
        
        # Clean up temporary file
        file_path.unlink(missing_ok=True)

async def reprocess_book_background(
    book_processor: EnhancedBookProcessor,
    book_id: str,
    user_id: str,
    job_id: str
):
    """Reprocess book in background"""
    try:
        logger.info("Starting book reprocessing", book_id=book_id, job_id=job_id)
        
        await book_processor.reprocess_book(book_id)
        
        logger.info("Book reprocessing completed", book_id=book_id, job_id=job_id)
        
    except Exception as e:
        logger.error("Book reprocessing failed", book_id=book_id, job_id=job_id, error=str(e))
        
        await book_processor.update_book_status(
            book_id=book_id,
            status="failed",
            error_message=str(e)
        )