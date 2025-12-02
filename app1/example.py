"""
Пример использования RAG микросервиса app1
"""

import asyncio
from app1.service.rag_service import RAGService
from app1.core.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Пример использования RAG системы"""
    
    # Инициализируем RAG сервис
    rag = RAGService()
    
    # 1. Сохранение документа
    logger.info("=" * 50)
    logger.info("1. СОХРАНЕНИЕ ДОКУМЕНТА")
    logger.info("=" * 50)
    
    # Пример текстового документа
    sample_text = """
    Искусственный интеллект (ИИ) - это область компьютерных наук, 
    которая занимается созданием интеллектуальных машин.
    
    Машинное обучение - это подмножество ИИ, которое позволяет
    компьютерам учиться без явного программирования.
    
    Глубокое обучение использует нейронные сети с множеством слоев
    для решения сложных задач.
    """
    
    save_result = await rag.save_document(
        content=sample_text.encode('utf-8'),
        title="Введение в ИИ",
        source="example",
        filename="intro_ai.txt"
    )
    
    logger.info(f"Результат сохранения: {save_result}")
    
    # 2. Поиск документов
    logger.info("\n" + "=" * 50)
    logger.info("2. ПОИСК ДОКУМЕНТОВ")
    logger.info("=" * 50)
    
    queries = [
        "Что такое машинное обучение?",
        "Как работают нейронные сети?",
        "Что такое искусственный интеллект?"
    ]
    
    for query in queries:
        logger.info(f"\nЗапрос: '{query}'")
        search_result = await rag.search(
            query=query,
            use_hybrid=True,
            use_rrf=True,
            use_reranking=True,  # ← Включаем reranking
            use_cache=True,       # ← Используем Redis кэш
            top_k=3
        )
        
        if search_result['status'] == 'success':
            logger.info(f"Найдено результатов: {search_result['count']}")
            logger.info(f"Из кэша: {search_result.get('from_cache', False)}")
            logger.info(f"Reranking применен: {search_result.get('reranking_applied', False)}")
            for idx, result in enumerate(search_result['results'], 1):
                score = result.get('score', 0)
                text_preview = result.get('text', '')[:100]
                logger.info(f"  [{idx}] Score: {score:.4f}")
                logger.info(f"       Text: {text_preview}...")
        else:
            logger.error(f"Ошибка поиска: {search_result.get('message')}")


if __name__ == "__main__":
    asyncio.run(main())
