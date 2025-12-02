from pydantic import BaseModel, Field
from typing import Optional, List


class SearchRequest(BaseModel):
    query: str = Field(..., description="Поисковый запрос")
    top_k: int = Field(5, description="Количество результатов")
    use_hybrid: bool = Field(True, description="Использовать гибридный поиск (Dense + Sparse)")
    use_rrf: bool = Field(True, description="Использовать RRF Fusion")
    use_reranking: Optional[bool] = Field(None, description="Использовать Cross-Encoder Reranking (None = авто)")
    use_cache: bool = Field(True, description="Использовать Redis кэш")
    collection_name: Optional[str] = Field(None, description="Имя коллекции (опционально)")


class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    milvus_id: Optional[int] = None
    distance: Optional[float] = None


class SearchResponse(BaseModel):
    status: str
    count: int
    results: List[SearchResultItem]
    search_type: str
    fusion_method: str
    reranking_applied: Optional[bool] = False
    from_cache: bool = False
    process_time_ms: float


class UploadResponse(BaseModel):
    status: str
    message: str
    document_hash: Optional[str] = None
    chunks_count: Optional[int] = None
