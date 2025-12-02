"""
Embedding Service - генерация плотных векторов с BGE-M3
"""

import os
import threading
import torch
from typing import List

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


class EmbeddingError(Exception):
    """Ошибка при генерации эмбеддингов"""
    pass


class EmbeddingService:
    """Сервис для генерации dense embeddings с BGE-M3 (синглтон)"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Реализация потокобезопасного синглтона"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Инициализация сервиса с BGE-M3 (выполняется только один раз)"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self.settings = get_settings()
        
        # Определяем device (GPU или CPU)
        logger.info(f"Проверяю доступность CUDA...")
        logger.info(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"  Устройство CUDA: {torch.cuda.get_device_name(0)}")
            logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        if self.settings.embedding.device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            logger.info("✓ Используется CUDA (GPU)")
        else:
            self.device = "cpu"
            logger.warning("⚠️ Используется CPU")
        
        # Загружаем BGE-M3
        self.model_name = "BAAI/bge-m3"
        
        # Устанавливаем кэш HuggingFace
        if "HF_HOME" not in os.environ:
            os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
        
        cache_dir = os.environ["HF_HOME"]
        
        try:
            logger.info(f"Загружаю BGE-M3 модель для dense embeddings")
            logger.info(f"  Model: {self.model_name}")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Кэш: {cache_dir}")
            
            from FlagEmbedding import BGEM3FlagModel
            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=True if self.device == "cuda" else False
            )
            
            self.embedding_dim = self.settings.embedding.embedding_dim  # 1024
            
            logger.info(f"✓ BGE-M3 модель загружена")
            logger.info(f"  ✓ Dense размерность: {self.embedding_dim}D")
            logger.info(f"  ✓ Device: {self.device}")
        except Exception as e:
            logger.error(f"Не удалось загрузить BGE-M3: {e}", exc_info=True)
            raise EmbeddingError(f"Не удалось загрузить BGE-M3: {e}")
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Генерация DENSE эмбеддингов для текстов
        
        Args:
            texts: Список текстов
        
        Returns:
            Список векторов (1024D dense vectors от BGE-M3)
        """
        try:
            if not texts:
                raise ValueError("Список текстов пуст")
            
            logger.debug(f"Генерирую dense embeddings для {len(texts)} текстов через BGE-M3...")
            
            # BGE-M3 encode() возвращает словарь
            result = self.model.encode(
                texts,
                batch_size=64 if self.device == "cuda" else 32,
                max_length=8192,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            
            # Извлекаем dense векторы
            if isinstance(result, dict) and 'dense_vecs' in result:
                vectors = result['dense_vecs']
            else:
                vectors = result
            
            # Преобразуем в список
            if hasattr(vectors, 'tolist'):
                vectors = vectors.tolist()
            
            logger.debug(f"✓ Dense embeddings сгенерированы ({len(vectors)} векторов, {len(vectors[0]) if vectors else 0}D)")
            return vectors
            
        except Exception as e:
            logger.error(f"Ошибка при генерации embeddings: {e}", exc_info=True)
            raise EmbeddingError(f"Ошибка при генерации embeddings: {e}")
    
    async def embed_single(self, text: str) -> List[float]:
        """
        Генерация dense эмбеддинга для одного текста
        
        Args:
            text: Текст
        
        Returns:
            Вектор (1024D dense vector)
        """
        try:
            result = self.model.encode(
                [text],
                batch_size=1,
                max_length=8192,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            
            # Извлекаем dense вектор
            if isinstance(result, dict) and 'dense_vecs' in result:
                vectors = result['dense_vecs']
                vector = vectors[0] if len(vectors) > 0 else vectors
            else:
                vector = result[0] if isinstance(result, list) else result
            
            if hasattr(vector, 'tolist'):
                vector = vector.tolist()
            
            return vector
            
        except Exception as e:
            raise EmbeddingError(f"Ошибка при генерации эмбеддинга: {e}")
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Генерация dense эмбеддинга для поискового запроса
        
        Args:
            query: Поисковый запрос
        
        Returns:
            Вектор (1024D dense vector)
        """
        return await self.embed_single(query)
    
    def get_embedding_dim(self) -> int:
        """Получить размерность (BGE-M3: 1024D)"""
        return self.embedding_dim
    
    def get_model_name(self) -> str:
        """Получить имя модели"""
        return self.model_name
    
    def get_device(self) -> str:
        """Получить используемое устройство"""
        return self.device
