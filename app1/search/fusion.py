"""
RRF (Reciprocal Rank Fusion) для объединения результатов поиска
Объединяет vector и sparse результаты без нормализации оценок
"""

from typing import List, Dict, Any
from core.logger import get_logger

logger = get_logger(__name__)


def rrf_fusion(
    vector_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
    weights: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """
    RRF (Reciprocal Rank Fusion) объединяет результаты
    
    RRF Score = Sum(1 / (k + rank))
    
    Args:
        vector_results: Результаты векторного поиска
        sparse_results: Результаты sparse/BM25 поиска
        k: Параметр RRF (обычно 60)
        weights: Веса для каждого типа поиска
    
    Returns:
        Объединённые результаты, отсортированные по RRF score
    """
    
    if weights is None:
        weights = {'vector': 1.0, 'sparse': 1.0}
    
    logger.debug(f"RRF Fusion: vector={len(vector_results)}, sparse={len(sparse_results)}, k={k}, weights={weights}")
    
    rrf_scores = {}
    result_map = {}
    
    # Векторные результаты
    vector_weight = weights.get('vector', 1.0)
    for rank, result in enumerate(vector_results, 1):
        doc_id = result.get('id') or result.get('chunk_id')
        
        if not doc_id:
            logger.warning(f"Результат векторного поиска без 'id' или 'chunk_id': {result.keys()}")
            continue
        
        rrf_score = (vector_weight / (k + rank))
        
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
            result_map[doc_id] = result.copy()
            result_map[doc_id]['vector_rank'] = rank
        
        rrf_scores[doc_id] += rrf_score
    
    # Sparse результаты
    sparse_weight = weights.get('sparse', 1.0)
    for rank, result in enumerate(sparse_results, 1):
        doc_id = result.get('id') or result.get('chunk_id')
        
        if not doc_id:
            logger.warning(f"Sparse результат без 'id' или 'chunk_id': {result.keys()}")
            continue
        
        rrf_score = (sparse_weight / (k + rank))
        
        if doc_id in rrf_scores:
            result_map[doc_id]['sparse_rank'] = rank
            rrf_scores[doc_id] += rrf_score
        else:
            rrf_scores[doc_id] = rrf_score
            result_map[doc_id] = result.copy()
            result_map[doc_id]['sparse_rank'] = rank
    
    # Сортировка по RRF score
    sorted_items = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Финальный список
    fused = []
    for doc_id, score in sorted_items:
        result = result_map[doc_id]
        result['rrf_score'] = score
        fused.append(result)
    
    # Логирование топ результатов
    logger.debug(f"RRF Fusion результаты (топ-5):")
    for idx, result in enumerate(fused[:5], 1):
        rrf_score = result.get('rrf_score', 0)
        vector_rank = result.get('vector_rank', '-')
        sparse_rank = result.get('sparse_rank', '-')
        text_preview = result.get('text', '')[:40]
        logger.debug(
            f"  [{idx}] id={result.get('id')}, "
            f"rrf_score={rrf_score:.4f}, "
            f"vector_rank={vector_rank}, sparse_rank={sparse_rank}, "
            f"text='{text_preview}...'"
        )
    
    return fused


def weighted_fusion(
    vector_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    vector_weight: float = 0.6,
    sparse_weight: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Простое взвешенное объединение (fallback если RRF отключен)
    
    Формула: score = vector_score * vector_weight + sparse_score * sparse_weight
    
    Args:
        vector_results: Результаты векторного поиска
        sparse_results: Результаты BM25 поиска
        vector_weight: Вес для vector (обычно 0.6)
        sparse_weight: Вес для sparse (обычно 0.4)
    
    Returns:
        Объединённые результаты
    """
    
    logger.debug(
        f"Weighted Fusion: vector_weight={vector_weight}, "
        f"sparse_weight={sparse_weight}"
    )
    
    combined = {}
    
    # Vector результаты
    for result in vector_results:
        doc_id = result.get('id') or result.get('chunk_id')
        vector_score = result.get('score', result.get('vector_score', 0))
        
        combined[doc_id] = {
            'data': result,
            'vector_score': float(vector_score),
            'sparse_score': 0.0
        }
    
    # Sparse результаты
    for result in sparse_results:
        doc_id = result.get('id') or result.get('chunk_id')
        sparse_score = result.get('score', result.get('bm25_score', 0))
        
        if doc_id in combined:
            combined[doc_id]['sparse_score'] = float(sparse_score)
        else:
            combined[doc_id] = {
                'data': result,
                'vector_score': 0.0,
                'sparse_score': float(sparse_score)
            }
    
    # Финальный score
    for doc_id, scores in combined.items():
        final_score = (
            scores['vector_score'] * vector_weight +
            scores['sparse_score'] * sparse_weight
        )
        scores['data']['combined_score'] = final_score
    
    # Сортировка
    sorted_results = sorted(
        (item['data'] for item in combined.values()),
        key=lambda x: x.get('combined_score', 0),
        reverse=True
    )
    
    return sorted_results
