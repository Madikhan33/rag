"""
Конфигурация для RAG микросервиса
Только Milvus, без PostgreSQL
"""

from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import Optional
from functools import lru_cache


class MilvusSettings(BaseModel):
    """Настройки Milvus (векторная БД)"""
    uri: str
    collection_name: str = "rag_documents"
    embedding_dim: int = 1024  # BGE-M3 dense = 1024D
    # HNSW индекс параметры
    index_type: str = "HNSW"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 256
    hnsw_ef_search: int = 256


class EmbeddingSettings(BaseModel):
    """Настройки для эмбеддингов (BGE-M3)"""
    model: str = "BAAI/bge-m3"  # BGE-M3 для dense (1024D) + sparse embeddings
    embedding_dim: int = 1024   # BGE-M3 dense embedding dimension (1024D)
    device: str = "cuda"


class ChunkingSettings(BaseModel):
    """Настройки для чанкирования документов"""
    chunk_size: int = 2500
    chunk_overlap: int = 200


class RedisSettings(BaseModel):
    """Настройки Redis кэша"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    enabled: bool = True
    ttl: int = 3600  # 1 час


class RAGSettings(BaseModel):
    """Настройки RAG системы"""
    top_k_vector: int = 5
    top_k_bm25: int = 5
    min_similarity: float = 0.3
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    use_reranking: bool = True  # Использовать reranking


class Settings(BaseSettings):
    """Основные настройки микросервиса"""
    # Milvus
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_COLLECTION_NAME: str = "rag_documents"
    
    # Embedding (BGE-M3: 1024D dense + sparse)
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_DEVICE: str = "cuda"
    
    # Chunking
    CHUNKING_CHUNK_SIZE: int = 2500
    CHUNKING_CHUNK_OVERLAP: int = 200
    
    # RAG
    RAG_TOP_K_VECTOR: int = 5
    RAG_TOP_K_BM25: int = 5
    RAG_MIN_SIMILARITY: float = 0.5
    RAG_USE_RERANKING: bool = True
    
    # Redis Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_ENABLED: bool = True
    REDIS_TTL: int = 3600
    
    # Debug режим
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        env_file_encoding = 'utf-8'
    
    @property
    def milvus(self) -> MilvusSettings:
        return MilvusSettings(
            uri=self.MILVUS_URI,
            collection_name=self.MILVUS_COLLECTION_NAME,
            embedding_dim=self.EMBEDDING_DIM
        )
    
    @property
    def embedding(self) -> EmbeddingSettings:
        return EmbeddingSettings(
            model=self.EMBEDDING_MODEL,
            embedding_dim=self.EMBEDDING_DIM,
            device=self.EMBEDDING_DEVICE
        )
    
    @property
    def chunking(self) -> ChunkingSettings:
        return ChunkingSettings(
            chunk_size=self.CHUNKING_CHUNK_SIZE,
            chunk_overlap=self.CHUNKING_CHUNK_OVERLAP
        )
    
    @property
    def rag(self) -> RAGSettings:
        return RAGSettings(
            top_k_vector=self.RAG_TOP_K_VECTOR,
            top_k_bm25=self.RAG_TOP_K_BM25,
            min_similarity=self.RAG_MIN_SIMILARITY,
            use_reranking=self.RAG_USE_RERANKING
        )
    
    @property
    def redis(self) -> RedisSettings:
        return RedisSettings(
            host=self.REDIS_HOST,
            port=self.REDIS_PORT,
            db=self.REDIS_DB,
            password=self.REDIS_PASSWORD,
            enabled=self.REDIS_ENABLED,
            ttl=self.REDIS_TTL
        )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
