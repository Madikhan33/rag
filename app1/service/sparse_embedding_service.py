"""
Sparse Embedding Service - генерация разреженных векторов с BGE-M3
Используется для гибридного поиска (BM25-подобные sparse vectors)
"""

import threading
from typing import List, Dict, Any
import numpy as np

try:
    from FlagEmbedding import BGEM3FlagModel as _BGEM3FlagModel
    BGEM3FlagModel = _BGEM3FlagModel
except (ImportError, ModuleNotFoundError):
    BGEM3FlagModel = None

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


class SparseEmbeddingError(Exception):
    """Ошибка при генерации sparse embeddings"""
    pass


class SparseEmbeddingService:
    """
    Сервис для генерации Sparse Vectors через BGE-M3
    
    BGE-M3 генерирует:
    - Dense: плотный вектор (1024D) для семантического поиска
    - Sparse: разреженный вектор для поиска по ключевым словам
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SparseEmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self.settings = get_settings()
        
        if BGEM3FlagModel is None:
            logger.warning("FlagEmbedding не установлен. Sparse embeddings отключены")
            self.model = None
            self.model_loaded = False
            return
        
        try:
            model_name = self.settings.embedding.model
            logger.info(f"Загружаю BGE-M3 для sparse embeddings: {model_name}")
            
            self.model = BGEM3FlagModel(
                model_name,
                use_fp16=self.settings.embedding.device == 'cuda'
            )
            logger.info(f"✓ BGE-M3 загружена для sparse")
            logger.info(f"  ✓ Dense dim: 1024")
            logger.info(f"  ✓ Sparse: да")
            self.model_loaded = True
            
        except Exception as e:
            logger.warning(f"Не удалось загрузить BGE-M3: {e}")
            logger.warning("Sparse embeddings отключены")
            self.model = None
            self.model_loaded = False
    
    async def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64
    ) -> Dict[str, Any]:
        """
        Генерирует Dense + Sparse embeddings для текстов
        
        Args:
            texts: Список текстов
            batch_size: Размер батча
        
        Returns:
            Словарь:
            - 'dense_vecs': np.array (len(texts), 1024)
            - 'sparse_vecs': List[Dict[int, float]] — token_id: weight
        """
        
        if not self.model_loaded:
            logger.warning("BGE-M3 не загружена, возвращаю пустой результат")
            return {
                'dense_vecs': np.array([]),
                'sparse_vecs': [{}] * len(texts)
            }
        
        if not texts:
            raise ValueError("Список текстов пуст")
        
        try:
            logger.debug(f"BGE-M3: генерирую embeddings для {len(texts)} текстов...")
            
            # Кодируем через BGE-M3
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                max_length=8192,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False
            )
            
            logger.debug(f"BGE-M3 вернул ключи: {embeddings.keys() if isinstance(embeddings, dict) else type(embeddings)}")
            
            if isinstance(embeddings, dict):
                dense_vecs = embeddings.get('dense_vecs')
                # Ключ 'lexical_weights', а не 'sparse_vecs'
                lexical_weights = embeddings.get('lexical_weights')
                
                if lexical_weights is not None and isinstance(lexical_weights, list):
                    sparse_vecs = lexical_weights
                    logger.debug(f"  Получены lexical_weights: {len(sparse_vecs)} элементов")
                    if len(sparse_vecs) > 0:
                        first_sparse = sparse_vecs[0] if sparse_vecs[0] else {}
                        logger.debug(f"  Первый sparse vector: {len(first_sparse)} токенов")
                else:
                    logger.warning(f"  lexical_weights пустой или None")
                    sparse_vecs = [{}] * len(texts)
            else:
                dense_vecs = embeddings
                sparse_vecs = [{}] * len(texts)
            
            result = {
                'dense_vecs': dense_vecs,
                'sparse_vecs': sparse_vecs
            }
            
            logger.debug(f"  ✓ Dense: {len(dense_vecs) if hasattr(dense_vecs, '__len__') else 1} векторов")
            logger.debug(f"  ✓ Sparse: {len(sparse_vecs) if hasattr(sparse_vecs, '__len__') else 1} векторов")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при генерации BGE-M3 embeddings: {e}", exc_info=True)
            raise SparseEmbeddingError(f"Ошибка при генерации embeddings: {e}")
    
    async def embed_query(self, query: str) -> Dict[str, Any]:
        """
        Генерирует embeddings для одного запроса
        
        Args:
            query: Поисковый запрос
        
        Returns:
            Словарь с 'dense_vec' и 'sparse_vec'
        """
        embeddings = await self.embed_texts([query], batch_size=1)
        
        result = {
            'dense_vec': embeddings['dense_vecs'][0],
            'sparse_vec': embeddings['sparse_vecs'][0]
        }
        
        return result
    
    def is_loaded(self) -> bool:
        """Проверить загружена ли модель"""
        return self.model_loaded
