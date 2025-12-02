from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from typing import Optional
import time
from .schemas import SearchRequest, SearchResponse, SearchResultItem, UploadResponse
from core.logger import get_logger
from service.rag_service import RAGService

logger = get_logger(__name__)

router = APIRouter()

def get_service():
    return RAGService()

@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Поиск документов
    """
    rag_service = get_service()
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    start_time = time.time()
    
    try:
        result = await rag_service.search(
            query=request.query,
            use_hybrid=request.use_hybrid,
            use_rrf=request.use_rrf,
            use_reranking=request.use_reranking,
            use_cache=request.use_cache,
            top_k=request.top_k,
            collection_name=request.collection_name
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))
        
        process_time = (time.time() - start_time) * 1000
        
        # Приводим к модели ответа
        response_items = []
        for item in result.get("results", []):
            response_items.append(SearchResultItem(
                chunk_id=item.get("chunk_id"),
                text=item.get("text"),
                score=item.get("score", 0.0),
                milvus_id=item.get("milvus_id"),
                distance=item.get("distance")
            ))
            
        return SearchResponse(
            status="success",
            count=result.get("count", 0),
            results=response_items,
            search_type=result.get("search_type", "unknown"),
            fusion_method=result.get("fusion_method", "unknown"),
            reranking_applied=result.get("reranking_applied", False),
            from_cache=result.get("from_cache", False),
            process_time_ms=round(process_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Ошибка при поиске: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    source: str = Form("upload"),
    collection_name: Optional[str] = Form(None)
):
    """
    Загрузка документа (PDF, DOCX, TXT, etc.)
    """
    rag_service = get_service()
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        content = await file.read()
        filename = file.filename
        doc_title = title or filename
        
        logger.info(f"Получен файл: {filename} ({len(content)} байт)")
        
        result = await rag_service.save_document(
            content=content,
            title=doc_title,
            source=source,
            filename=filename,
            collection_name=collection_name
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))
            
        return UploadResponse(
            status="success",
            message=result.get("message", "Document saved"),
            document_hash=result.get("document_hash"),
            chunks_count=result.get("chunks_count")
        )
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_hash}")
async def delete_document(document_hash: str):
    """Удаление документа по хешу"""
    rag_service = get_service()
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    result = await rag_service.delete_document(document_hash)
    return result


# --- Collection Management Endpoints ---

@router.get("/collections")
async def list_collections():
    """Список всех коллекций и их статистика"""
    rag_service = get_service()
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        milvus = rag_service.vector_search.milvus_service
        names = milvus.list_collections()
        
        collections = []
        for name in names:
            stats = milvus.get_collection_stats(name)
            collections.append(stats)
            
        return {"count": len(collections), "collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections/{name}")
async def create_collection(name: str):
    """Создание новой коллекции"""
    rag_service = get_service()
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        milvus = rag_service.vector_search.milvus_service
        actual_name = milvus.ensure_collection(name)
        return {"status": "success", "message": f"Collection '{actual_name}' created/ensured"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    """Удаление коллекции (ВНИМАНИЕ: удаляет все данные!)"""
    rag_service = get_service()
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        milvus = rag_service.vector_search.milvus_service
        milvus.drop_collection(name)
        return {"status": "success", "message": f"Collection '{name}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))