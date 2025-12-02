"""
Vector Search Service - поиск в Milvus
Векторный и sparse поиск
"""

import threading
from typing import List, Dict, Any, Optional

from .milvus_service import MilvusService
from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


class VectorSearchService:
    """
    Сервис для поиска в Milvus
    Занимается только поиском векторов
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(VectorSearchService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self.settings = get_settings()
        self.collection_name = self.settings.milvus.collection_name
        
        # Инициализируем MilvusService
        self.milvus_service = MilvusService()
        self.client = self.milvus_service.client
    
    def _resolve_collection(self, collection_name: Optional[str]) -> str:
        """Определяет коллекцию для операции"""
        if collection_name:
            return self.milvus_service.ensure_collection(collection_name)
        return self.collection_name
    
    async def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск похожих документов по вектору (COSINE similarity с HNSW)
        
        Args:
            query_vector: Embedding запроса (1024 чисел для BGE-M3)
            top_k: Количество результатов
            collection_name: Опциональное имя коллекции
        
        Returns:
            Список результатов с chunk_id, text и distance
        """
        try:
            target_collection = self._resolve_collection(collection_name)
            logger.debug(f"Поиск в коллекции: {target_collection}, top_k={top_k}")
            
            if not query_vector or len(query_vector) == 0:
                logger.warning("Пустой вектор запроса")
                return []
            
            logger.debug(f"Размер вектора: {len(query_vector)}")
            
            # Загружаем коллекцию в память
            try:
                self.client.load_collection(collection_name=target_collection)
                logger.debug(f"Коллекция '{target_collection}' загружена")
            except Exception as e:
                logger.debug(f"Коллекция уже загружена: {e}")
            
            # Параметры HNSW
            ef_value = max(64, top_k * 4)
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": ef_value}
            }
            
            logger.debug(f"HNSW: ef={ef_value}, metric=COSINE")
            
            results = self.client.search(
                collection_name=target_collection,
                data=[query_vector],
                anns_field="vector",
                limit=top_k,
                search_params=search_params,
                output_fields=["chunk_id", "text"]
            )
            
            logger.info(f"HNSW поиск вернул: {len(results)} batch(es)")
            
            formatted_results = []
            for batch_idx, hits in enumerate(results):
                logger.info(f"Batch {batch_idx}: {len(hits)} результатов")
                for hit_idx, hit in enumerate(hits):
                    entity = hit.get("entity", {})
                    chunk_id = entity.get("chunk_id")
                    text = entity.get("text", "")
                    distance = hit.get("distance", 0)
                    
                    logger.info(f"  Hit {hit_idx}: chunk_id={chunk_id}, distance={distance:.4f}")
                    
                    if chunk_id is None:
                        logger.warning(f"Результат без chunk_id: {hit}")
                        continue
                    
                    formatted_results.append({
                        "milvus_id": hit.get("id"),
                        "chunk_id": chunk_id,
                        "text": text,
                        "distance": distance
                    })
            
            logger.info(f"Векторный поиск: {len(formatted_results)} результатов")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Ошибка при векторном поиске: {str(e)}", exc_info=True)
            return []
    
    async def bm25_search(
        self,
        query: str,
        top_k: int = 5,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Sparse поиск в Milvus с использованием BGE-M3 sparse vectors
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            collection_name: Опциональное имя коллекции
        
        Returns:
            Список результатов с chunk_id и distance (IP score)
        """
        try:
            target_collection = self._resolve_collection(collection_name)
            logger.debug(f"Sparse поиск в '{target_collection}': '{query}' (top_k={top_k})")
            
            if not query or not query.strip():
                logger.warning("Пустой запрос для Sparse поиска")
                return []
            
            # Генерируем sparse vector через BGE-M3
            from app1.service.sparse_embedding_service import SparseEmbeddingService
            sparse_service = SparseEmbeddingService()
            
            query_embedding = await sparse_service.embed_query(query)
            query_sparse_vec = query_embedding.get('sparse_vec', {})
            
            if not query_sparse_vec:
                logger.warning("Не удалось сгенерировать sparse vector")
                return []
                
            logger.debug(f"Sparse vector: {len(query_sparse_vec)} токенов")
            
            # Параметры поиска
            search_params = {
                "metric_type": "IP",
                "params": {}
            }
            
            # Поиск
            results = self.client.search(
                collection_name=target_collection,
                data=[query_sparse_vec],
                anns_field="sparse_vector",
                limit=top_k,
                search_params=search_params,
                output_fields=["chunk_id", "text"]
            )
            
            formatted_results = []
            for hits in results:
                logger.debug(f"Sparse search: {len(hits)} результатов")
                for hit in hits:
                    entity = hit.get("entity", {})
                    chunk_id = entity.get("chunk_id")
                    text = entity.get("text", "")
                    distance = hit.get("distance", 0)
                    
                    if chunk_id is None:
                        logger.warning(f"Sparse результат без chunk_id: {hit}")
                        continue
                    
                    formatted_results.append({
                        "milvus_id": hit.get("id"),
                        "chunk_id": chunk_id,
                        "text": text,
                        "distance": distance
                    })
            
            logger.debug(f"Sparse поиск: {len(formatted_results)} результатов")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Ошибка при Sparse поиске: {str(e)}", exc_info=True)
            return []
    
    async def add_vectors(
        self,
        chunk_ids: List[str],
        vectors: List[List[float]],
        chunks_data: List[tuple],
        sparse_vectors: Optional[List[Dict[int, float]]] = None,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Добавление векторов чанков в коллекцию
        
        Args:
            chunk_ids: Идентификаторы чанков
            vectors: Векторные представления (плотные)
            chunks_data: Содержимое чанков [(text, metadata), ...]
            sparse_vectors: Разреженные векторы от BGE-M3
        """
        try:
            if len(chunk_ids) != len(vectors) or len(chunk_ids) != len(chunks_data):
                raise ValueError("Несоответствие количества chunk_ids, векторов и содержимого")
            
            logger.info(f"Сохраняю {len(chunk_ids)} чанков в Milvus...")
            
            data = []
            for idx, (chunk_id, vector, (chunk_text, chunk_meta)) in enumerate(
                zip(chunk_ids, vectors, chunks_data)
            ):
                data_item = {
                    "chunk_id": str(chunk_id),
                    "text": chunk_text,
                    "vector": vector
                }
                
                # Добавляем sparse_vector если есть
                if sparse_vectors and idx < len(sparse_vectors):
                    data_item["sparse_vector"] = sparse_vectors[idx]
                    logger.debug(f"  Чанк {idx}: chunk_id={chunk_id}, sparse_terms={len(sparse_vectors[idx]) if sparse_vectors[idx] else 0}")
                else:
                    data_item["sparse_vector"] = {}
                    logger.debug(f"  Чанк {idx}: chunk_id={chunk_id} (без sparse)")
                
                data.append(data_item)
            
            target_collection = self._resolve_collection(collection_name)
            logger.info(f"Вставляю в '{target_collection}' с HNSW + SPARSE_INVERTED_INDEX...")
            
            result = self.client.insert(
                collection_name=target_collection,
                data=data
            )
            
            logger.info(f"✓ Успешно вставлено {len(data)} чанков")
            if sparse_vectors:
                logger.info(f"  ✓ Sparse векторы: сохранены")
            else:
                logger.info(f"  ⚠ Sparse векторы: не передали")
            
            return {
                "status": "success",
                "inserted_count": len(data)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def delete_by_chunk_id(self, chunk_id: str, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Удаление вектора по chunk_id из Milvus"""
        try:
            target_collection = self._resolve_collection(collection_name)
            self.client.delete(
                collection_name=target_collection,
                filter=f"chunk_id == '{chunk_id}'"
            )
            return {"status": "success", "chunk_id": chunk_id}
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "chunk_id": chunk_id
            }
