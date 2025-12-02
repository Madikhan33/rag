"""
Cross-Encoder Re-ranking Service
Переранжирование результатов поиска по релевантности
"""

import threading
from typing import List, Dict, Any

try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    RERANKER_AVAILABLE = False

from core.logger import get_logger
from core.config import get_settings

logger = get_logger(__name__)


class RerankerService:
    """
    Cross-Encoder для переранжирования результатов поиска
    
    Преимущества:
    - Учитывает контекст пары [query, document]
    - Более точное ранжирование чем простое сравнение сходства
    - Работает даже с короткими запросами
    - Улучшает качество на +20-30%
    
    Использование:
        reranker = RerankerService()
        reranked = await reranker.rerank(query, candidates, top_k=5)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RerankerService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self.settings = get_settings()
        
        if not RERANKER_AVAILABLE:
            logger.warning("FlagEmbedding не установлен. Reranking отключен")
            self.model = None
            self.model_loaded = False
            return
        
        try:
            # bge-reranker-v2-m3 — лучший по качеству/скорости
            model_name = 'BAAI/bge-reranker-v2-m3'
            logger.info(f"Загружаю Reranker модель: {model_name}")
            
            self.model = FlagReranker(
                model_name,
                use_fp16=True,
            )
            logger.info(f"✓ Reranker загружен: {model_name}")
            self.model_loaded = True
            
        except Exception as e:
            logger.warning(f"Не удалось загрузить Reranker: {e}")
            logger.warning("Re-ranking будет отключен")
            self.model = None
            self.model_loaded = False
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Переранжирует кандидатов по релевантности к запросу
        
        Процесс:
        1. Для каждого кандидата создаем пару [query, candidate_text]
        2. Прогоняем через Cross-Encoder модель
        3. Получаем score релевантности для каждой пары
        4. Сортируем по score в убывающем порядке
        5. Возвращаем топ-k результатов
        
        Args:
            query: Поисковый запрос
            candidates: Список документов/чанков (должны иметь поле 'text')
            top_k: Сколько вернуть лучших результатов
        
        Returns:
            Список кандидатов, отсортированный по rerank_score
        """
        if not candidates:
            logger.warning("Пустой список кандидатов для переранжирования")
            return []
        
        # Если модель не загружена, вернуть как есть
        if not self.model_loaded:
            logger.debug("Re-ranking отключен, возвращаю кандидатов как есть")
            return candidates[:top_k]
        
        try:
            logger.debug(f"Re-ranking: переранжирую {len(candidates)} кандидатов по запросу '{query[:50]}...'")
            
            # Извлечь тексты из кандидатов
            texts = []
            for candidate in candidates:
                text = candidate.get('text', '')
                if not text:
                    logger.warning(f"Кандидат не имеет поля 'text': {candidate.keys()}")
                    text = str(candidate)
                texts.append(text)
            
            # Создать пары [query, text] для Reranker
            pairs = [[query, text] for text in texts]
            
            # Прогнать через модель и получить scores
            logger.debug(f"Прогоняю {len(pairs)} пар через Reranker...")
            scores = self.model.compute_score(pairs, normalize=True)
            
            # Добавить score к каждому кандидату
            for candidate, score in zip(candidates, scores):
                candidate['rerank_score'] = float(score) if isinstance(score, (int, float)) else 0.0
            
            # Отсортировать по rerank_score
            reranked = sorted(
                candidates,
                key=lambda x: x.get('rerank_score', 0),
                reverse=True
            )
            
            # Взять топ-k
            final_results = reranked[:top_k]
            
            # Логирование
            logger.debug(f"Re-ranking завершен:")
            for idx, result in enumerate(final_results, 1):
                score = result.get('rerank_score', 0)
                text_preview = result.get('text', '')[:50]
                logger.debug(f"  [{idx}] score={score:.4f}, text='{text_preview}...'")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Ошибка при переранжировании: {str(e)}", exc_info=True)
            # Fallback: вернуть как есть
            return candidates[:top_k]
    
    def is_loaded(self) -> bool:
        """Проверить загружена ли модель"""
        return self.model_loaded
