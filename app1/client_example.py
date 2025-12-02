"""
Пример клиента для взаимодействия с RAG микросервисом (app1)
"""

import asyncio
import httpx
import os
from typing import Optional


BASE_URL = "http://localhost:8000"


async def check_health():
    """Проверка здоровья сервиса"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/health")
            print(f"Health Check: {response.status_code}")
            print(response.json())
            return response.status_code == 200
        except Exception as e:
            print(f"Сервис недоступен: {e}")
            return False


async def upload_file(filepath: str, title: Optional[str] = None, collection_name: Optional[str] = None):
    """Загрузка файла"""
    if not os.path.exists(filepath):
        print(f"Файл не найден: {filepath}")
        return None

    filename = os.path.basename(filepath)
    print(f"\nЗагрузка файла: {filename}...")
    if collection_name:
        print(f"  Коллекция: {collection_name}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(filepath, "rb") as f:
            files = {"file": (filename, f)}
            data = {"source": "client_example"}
            if title:
                data["title"] = title
            if collection_name:
                data["collection_name"] = collection_name
            
            response = await client.post(f"{BASE_URL}/upload", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Успешно: {result['message']}")
                print(f"  Hash: {result.get('document_hash')}")
                print(f"  Chunks: {result.get('chunks_count')}")
                return result
            else:
                print(f"❌ Ошибка загрузки: {response.text}")
                return None


async def search(query: str, use_reranking: bool = True, collection_name: Optional[str] = None):
    """Поиск документов"""
    print(f"\nПоиск: '{query}' (reranking={use_reranking})...")
    if collection_name:
        print(f"  Коллекция: {collection_name}")
    
    payload = {
        "query": query,
        "top_k": 3,
        "use_hybrid": True,
        "use_rrf": True,
        "use_reranking": use_reranking,
        "use_cache": True
    }
    
    if collection_name:
        payload["collection_name"] = collection_name
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{BASE_URL}/search", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Найдено: {data['count']} (Time: {data['process_time_ms']}ms)")
            print(f"Cache: {data['from_cache']}, Reranked: {data['reranking_applied']}")
            
            for i, item in enumerate(data['results'], 1):
                print(f"\n  [{i}] Score: {item['score']:.4f}")
                print(f"      Text: {item['text'][:150]}...")
        else:
            print(f"❌ Ошибка поиска: {response.text}")


async def main():
    # 1. Проверяем доступность
    if not await check_health():
        print("Запустите сервер перед запуском клиента!")
        print("cd app1 && python main.py")
        return

    # 2. Создаем тестовый файл если нет
    test_file = "test_doc.txt"
    if not os.path.exists(test_file):
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("""
            Python — высокоуровневый язык программирования общего назначения.
            FastAPI — это современный веб-фреймворк для создания API.
            Milvus — это векторная база данных для приложений ИИ.
            """)
        print(f"Создан тестовый файл: {test_file}")

    # 3. Загрузка в стандартную коллекцию
    print("\n=== ТЕСТ 1: Стандартная коллекция ===")
    await upload_file(test_file, title="General Doc")
    await search("Что такое FastAPI?")

    # 4. Загрузка в СПЕЦИАЛЬНУЮ коллекцию
    print("\n=== ТЕСТ 2: Специальная коллекция 'finance_docs' ===")
    await upload_file(test_file, title="Finance Doc", collection_name="finance_docs")
    
    # Поиск в специальной коллекции
    await search("Что такое Milvus?", collection_name="finance_docs")
    
    # Поиск в несуществующей/пустой коллекции (должен вернуть 0)
    print("\n=== ТЕСТ 3: Пустая коллекция 'empty_col' ===")
    await search("Что такое Python?", collection_name="empty_col")

    # Удаляем тестовый файл
    if os.path.exists(test_file):
        os.remove(test_file)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
