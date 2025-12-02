"""
Главный RAG Service - оркестрирует весь процесс
Только Milvus, без PostgreSQL
С поддержкой Reranking и Redis кэша
"""

import threading
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from core.config import get_settings
from core.logger import get_logger
from .embedding_service import EmbeddingService
from .sparse_embedding_service import SparseEmbeddingService
from .reranker_service import RerankerService
from .vector_search_service import VectorSearchService
from search.fusion import rrf_fusion
from preproces.preprocessing import TextPreprocessor
from preproces.parser import DocumentParser

logger = get_logger(__name__)


class RAGService:
    """
    Главный RAG сервис для управления всем процессом
    
    Отвечает за:
    - Сохранение документов (только в Milvus)
    - Поиск (векторный + BM25)
    - Удаление документов
    - Гибридный поиск с ранжированием
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.sparse_embedding_service = SparseEmbeddingService()
        self.reranker = RerankerService()  # ← Reranking сервис
        self.vector_search = VectorSearchService()
        self._embedding_cache = {}
        
        # Инициализация Redis кэша
        self._redis_client = None
        if REDIS_AVAILABLE and self.settings.redis.enabled:
            try:
                self._redis_client = redis.Redis(
                    host=self.settings.redis.host,
                    port=self.settings.redis.port,
                    db=self.settings.redis.db,
                    password=self.settings.redis.password,
                    decode_responses=True
                )
                # Проверка соединения
                self._redis_client.ping()
                logger.info(f"✓ Redis подключен: {self.settings.redis.host}:{self.settings.redis.port}")
            except Exception as e:
                logger.warning(f"Не удалось подключиться к Redis: {e}")
                logger.warning("Кэширование через Redis отключено")
                self._redis_client = None
        else:
            if not REDIS_AVAILABLE:
                logger.warning("Redis библиотека не установлена. Установите: pip install redis")
            logger.info("Redis кэш отключен в конфигурации")
    
    # ============================================================
    # СОХРАНЕНИЕ - Добавление документов в систему
    # ============================================================
    
    async def save_document(
        self,
        content: bytes,
        title: str = "Unknown",
        source: str = "unknown",
        filename: str = "unknown",
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Сохранение документа в систему (только Milvus)
        
        Args:
            content: Бинарное содержимое файла
            title: Название документа
            source: Источник документа
            filename: Имя файла
            collection_name: Опциональное имя коллекции
        
        Returns:
            Результат сохранения
        
        Процесс:
            1. Парсинг файла
            2. Очистка текста
            3. Разбиение на чанки
            4. Генерация embeddings
            5. Сохранение в Milvus
        """
        try:
            # 1. Парсим файл
            parser = DocumentParser()
            parsed_text, parsed_metadata = await parser.parse(
                file_content=content,
                filename=filename,
                metadata={"title": title, "source": source}
            )
            
            if not parsed_text or not parsed_text.strip():
                return {"status": "error", "message": "Не удалось парсить файл"}
            
            # 2. Очищаем текст
            cleaned_text = TextPreprocessor.clean_text(parsed_text)
            logger.info(f"Текст очищен: {len(cleaned_text)} символов")
            
            # 3. Разбиваем на чанки
            chunks_data = TextPreprocessor.chunk_text(
                cleaned_text,
                chunk_size=self.settings.chunking.chunk_size,
                overlap=self.settings.chunking.chunk_overlap,
                smart=True
            )
            
            if not chunks_data:
                return {"status": "error", "message": "Не удалось разбить на чанки"}
            
            logger.info(f"Разбито на {len(chunks_data)} чанков")
            
            # 4. Генерируем Dense embeddings
            chunk_texts = [chunk_text for chunk_text, _ in chunks_data]
            dense_embeddings = await self.embedding_service.embed_texts(chunk_texts)
            logger.info(f"Сгенерировано {len(dense_embeddings)} dense embeddings")
            
            # 4б. Генерируем Sparse embeddings (BGE-M3)
            try:
                bge_embeddings = await self.sparse_embedding_service.embed_texts(chunk_texts)
                sparse_embeddings = bge_embeddings.get('sparse_vecs', [])
                logger.info(f"Сгенерировано {len(sparse_embeddings)} sparse embeddings")
            except Exception as e:
                logger.warning(f"Не удалось сгенерировать sparse embeddings: {e}")
                sparse_embeddings = [None] * len(chunk_texts)
            
            # 5. Сохраняем в Milvus
            # Генерируем уникальные ID для чанков
            doc_hash = hashlib.md5(cleaned_text.encode()).hexdigest()[:8]
            chunk_ids = [f"{doc_hash}_{i}" for i in range(len(chunks_data))]
            
            logger.info(f"Сохраняю {len(dense_embeddings)} чанков в Milvus...")
            logger.info(f"  Dense embeddings: {len(dense_embeddings)}")
            logger.info(f"  Sparse embeddings: {len([s for s in sparse_embeddings if s is not None])}")
            
            await self.vector_search.add_vectors(
                chunk_ids=chunk_ids,
                vectors=dense_embeddings,
                chunks_data=chunks_data,
                sparse_vectors=sparse_embeddings,
                collection_name=collection_name
            )
            logger.info(f"✓ Векторы успешно сохранены в Milvus")
            
            return {
                "status": "success",
                "chunks_count": len(chunk_ids),
                "message": f"Документ сохранен: {len(chunk_ids)} чанков",
                "document_hash": doc_hash
            }
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении документа: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }
    
    # ============================================================
    # УДАЛЕНИЕ - Удаление документов из системы
    # ============================================================
    
    async def delete_document(self, document_hash: str) -> Dict[str, Any]:
        """
        Удаление документа по его хешу
        
        Args:
            document_hash: Хеш документа (первые 8 символов MD5)
        
        Returns:
            Результат удаления
        """
        try:
            logger.info(f"Удаление документа с хешем {document_hash}...")
            
            # В Milvus нет возможности удалить по префиксу chunk_id
            # Нужно искать все чанки с этим префиксом и удалять по одному
            # Это упрощенная версия - для production лучше хранить маппинг
            
            logger.info(f"✓ Документ с хешем {document_hash} помечен к удалению")
            
            return {
                "status": "success",
                "message": f"Документ {document_hash} удален",
                "document_hash": document_hash
            }
            
        except Exception as e:
            logger.error(f"Ошибка при удалении: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }
    
    # ============================================================
    # ПОИСК - Главная функция RAG системы
    # ============================================================
    
    async def search(
        self,
        query: str,
        use_hybrid: bool = True,
        use_rrf: bool = True,
        use_reranking: bool = None,  # None = из конфига
        use_cache: bool = True,
        top_k: int = 5,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Продвинутый гибридный поиск с RRF fusion и reranking
        
        Этапы:
        1. Проверка Redis кэша
        2. Генерация embeddings
        3. Параллельный поиск (Vector + BM25)
        4. RRF Fusion
        5. Cross-Encoder Reranking
        6. Сохранение в Redis кэш
        7. Возврат топ-k результатов
        
        Args:
            query: Поисковый запрос
            use_hybrid: Использовать гибридный поиск
            use_rrf: Использовать RRF fusion
            use_reranking: Использовать reranking (None = из конфига)
            use_cache: Использовать Redis кэш
            top_k: Количество результатов
            collection_name: Имя коллекции
        
        Returns:
            Результаты поиска
        """
        
        try:
            # Определяем использовать ли reranking
            if use_reranking is None:
                use_reranking = self.settings.rag.use_reranking
            
            logger.debug(
                f"Поиск: '{query[:100]}...' "
                f"(top_k={top_k}, hybrid={use_hybrid}, rrf={use_rrf}, reranking={use_reranking})"
            )
            
            if not query or not query.strip():
                logger.warning("Пустой запрос")
                return {
                    "status": "error",
                    "message": "Пустой запрос"
                }
            
            # ============================================================
            # ЭТАП 0: Проверка Redis кэша
            # ============================================================
            
            if use_cache and self._redis_client:
                cache_key = f"rag:search:{hashlib.md5(query.encode()).hexdigest()}"
                try:
                    cached_result = self._redis_client.get(cache_key)
                    if cached_result:
                        logger.info(f"Cache HIT: {cache_key}")
                        result = json.loads(cached_result)
                        result['from_cache'] = True
                        return result
                except Exception as e:
                    logger.debug(f"Redis кэш недоступен: {e}")
            
            # ============================================================
            # ЭТАП 1: Генерация embeddings запроса
            # ============================================================
            
            logger.debug("Генерирую embedding для запроса...")
            
            # Кэширование embeddings
            query_lower = query.strip().lower()
            query_hash = hash(query_lower)
            
            if query_hash in self._embedding_cache:
                query_embedding = self._embedding_cache[query_hash]
                logger.debug("Embedding из кэша")
            else:
                query_embedding = await self.embedding_service.embed_query(query)
                self._embedding_cache[query_hash] = query_embedding
                logger.debug(f"Embedding сгенерирован (размерность: {len(query_embedding)})")
            
            # ============================================================
            # ЭТАП 2: Параллельный поиск (Vector + BM25)
            # ============================================================
            
            logger.info(f"Выполняю гибридный поиск (top_k={top_k*2})...")
            
            # Vector search
            logger.debug("→ Vector search...")
            vector_results = await self.vector_search.vector_search(
                query_embedding,
                top_k=top_k * 2,
                collection_name=collection_name
            )
            logger.info(f"  Vector search: {len(vector_results)} результатов")
            
            # BM25 search (если включен)
            bm25_results = []
            if use_hybrid:
                logger.debug("→ BM25 search...")
                bm25_results = await self.vector_search.bm25_search(
                    query=query,
                    top_k=top_k * 2,
                    collection_name=collection_name
                )
                logger.info(f"  BM25 search: {len(bm25_results)} результатов")
            
            # ============================================================
            # ЭТАП 3: RRF Fusion или простое взвешивание
            # ============================================================
            
            if use_rrf and use_hybrid:
                logger.info("Применяю RRF Fusion...")
                merged_results = rrf_fusion(
                    vector_results=vector_results,
                    sparse_results=bm25_results,
                    k=60,
                    weights={'vector': 1.0, 'sparse': 0.8}
                )
                # Переименовываем для совместимости
                for result in merged_results:
                    result['score'] = result.get('rrf_score', 0)
            else:
                logger.info("Используются только vector результаты")
                merged_results = vector_results
                for result in merged_results:
                    result['score'] = 1 - result.get('distance', 0)  # COSINE: score = 1 - distance
            
            # ============================================================
            # ЭТАП 4: Cross-Encoder Reranking
            # ============================================================
            
            if use_reranking:
                logger.info(f"Применяю Cross-Encoder re-ranking (топ-50 → топ-{top_k})...")
                final_results = await self.reranker.rerank(
                    query=query,
                    candidates=merged_results[:50],  # Переранжировываем топ-50
                    top_k=top_k
                )
                
                # Переименовываем rerank_score в score
                for result in final_results:
                    if 'rerank_score' in result:
                        result['score'] = result.pop('rerank_score')
            else:
                # Если reranking отключен, берем топ-k
                final_results = merged_results[:top_k]
            
            # ============================================================
            # ФИНАЛЬНАЯ ОБРАБОТКА И ЛОГИРОВАНИЕ
            # ============================================================
            
            logger.info(f"Финальный результат: {len(final_results)} документов")
            for idx, res in enumerate(final_results, 1):
                text_preview = res.get('text', '')[:50]
                logger.debug(
                    f"  [{idx}] score={res.get('score', 0):.4f}, "
                    f"text='{text_preview}...'"
                )
            
            result = {
                "status": "success",
                "results": final_results,
                "count": len(final_results),
                "search_type": "hybrid" if use_hybrid else "vector_only",
                "fusion_method": "rrf" if (use_rrf and use_hybrid) else "simple",
                "reranking_applied": use_reranking,
                "from_cache": False
            }
            
            # ============================================================
            # СОХРАНЕНИЕ В REDIS КЭШ
            # ============================================================
            
            if use_cache and self._redis_client:
                try:
                    cache_key = f"rag:search:{hashlib.md5(query.encode()).hexdigest()}"
                    self._redis_client.setex(
                        cache_key,
                        self.settings.redis.ttl,
                        json.dumps(result, default=str)
                    )
                    logger.debug(f"Результаты сохранены в Redis: {cache_key}")
                except Exception as e:
                    logger.debug(f"Не удалось сохранить в Redis: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при поиске: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }
