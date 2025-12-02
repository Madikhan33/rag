"""
Milvus Service для управления векторной базой данных
Создание коллекций с гибридным поиском (Dense + Sparse)
"""

import time
import threading
from typing import List, Optional, Dict, Any
from pymilvus import (
    MilvusClient, 
    DataType
)

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


class MilvusConnectionError(Exception):
    """Ошибка подключения к Milvus"""
    pass


class MilvusService:
    """
    Сервис для управления Milvus
    
    Отвечает за:
    - Подключение к Milvus
    - Создание коллекций с HNSW индексами
    - Предоставление клиента для поиска
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Реализация потокобезопасного синглтона"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MilvusService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Инициализация сервиса (выполняется только один раз)"""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self.settings = get_settings()
        self.uri = self.settings.milvus.uri
        self.collection_name = self.settings.milvus.collection_name
        self.embedding_dim = self.settings.milvus.embedding_dim
        
        self.client = None
        self._connect()
        self.ensure_collection(self.collection_name)
    
    def _connect(self) -> None:
        """Подключение к Milvus с повторными попытками"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.client = MilvusClient(uri=self.uri)
                logger.info(f"✓ Подключился к Milvus на {self.uri}")
                return
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Попытка {retry_count}/{max_retries} не удалась, повтор через 3 сек...")
                    time.sleep(3)
                else:
                    logger.error(f"Не удалось подключиться к Milvus после {max_retries} попыток")
                    raise MilvusConnectionError(
                        f"Не удалось подключиться к Milvus на {self.uri}"
                    )
    
    def ensure_collection(self, collection_name: Optional[str] = None) -> str:
        """
        Создает коллекцию с поддержкой гибридного поиска
        
        Поля:
        - chunk_id: идентификатор чанка (varchar)
        - text: текст чанка (varchar, max 65535)
        - vector: плотный вектор (1024D от BGE-M3)
        - sparse_vector: разреженный вектор (от BGE-M3)
        
        Индексы:
        - HNSW на vector для COSINE similarity
        - SPARSE_INVERTED_INDEX на sparse_vector
        """
        target_name = collection_name or self.collection_name
        
        # Валидация имени
        if not target_name:
            raise ValueError("Имя коллекции не может быть пусто")
        
        if not (target_name[0].isalpha() or target_name[0] == '_'):
            logger.warning(f"Имя '{target_name}' начинается с цифры. Добавляю префикс 'col_'")
            target_name = f"col_{target_name}"
        
        # Только буквы, цифры и подчеркивание
        if not all(c.isalnum() or c == '_' for c in target_name):
            logger.warning(f"Имя '{target_name}' содержит недопустимые символы")
            target_name = "col_" + ''.join(c if (c.isalnum() or c == '_') else '_' for c in target_name)
        
        try:
            collections = self.client.list_collections()
            if target_name in collections:
                logger.info(f"✓ Коллекция '{target_name}' уже существует")
                return target_name
            
            logger.info(f"Создаю коллекцию '{target_name}' с гибридным поиском...")
            
            # Создаем схему
            schema = MilvusClient.create_schema(
                auto_id=True,
                enable_dynamic_field=False
            )
            
            # Поля
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            
            # Индексы
            index_params = self.client.prepare_index_params()
            
            # HNSW для плотных векторов
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 32, "efConstruction": 256}
            )
            
            # SPARSE_INVERTED_INDEX для разреженных векторов
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP"
            )
            
            # Создаем коллекцию
            self.client.create_collection(
                collection_name=target_name,
                schema=schema,
                index_params=index_params
            )
            
            logger.info(f"✓ Коллекция '{target_name}' создана:")
            logger.info(f"  ✓ Dense: HNSW ({self.embedding_dim}D, COSINE)")
            logger.info(f"  ✓ Sparse: SPARSE_INVERTED_INDEX (IP)")
            logger.info(f"  ✓ Поля: id, chunk_id, text, vector, sparse_vector")
            return target_name
            
        except Exception as e:
            logger.error(f"Ошибка при создании коллекции '{target_name}': {str(e)}")
            raise
    
    def list_collections(self) -> List[str]:
        """Список коллекций"""
        return self.client.list_collections()
    
    def drop_collection(self, collection_name: str) -> None:
        """Удаляет коллекцию"""
        self.client.drop_collection(collection_name=collection_name)
        logger.info(f"✓ Коллекция '{collection_name}' удалена")

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Получить статистику коллекции"""
        try:
            res = self.client.get_collection_stats(collection_name)
            count = self.client.query(
                collection_name=collection_name,
                filter="",
                output_fields=["count(*)"]
            )
            # В Milvus 2.3+ query count(*) может работать иначе, используем num_entities из stats если доступно
            # Но самый надежный способ для count - query с count(*)
            
            # Альтернатива: client.query(collection_name, "count(*)") возвращает [{'count(*)': N}]
            
            total_rows = 0
            if count and isinstance(count, list) and 'count(*)' in count[0]:
                total_rows = count[0]['count(*)']
            
            return {
                "name": collection_name,
                "row_count": total_rows,
                "stats": res
            }
        except Exception as e:
            logger.error(f"Ошибка получения статистики для {collection_name}: {e}")
            return {"name": collection_name, "error": str(e)}
