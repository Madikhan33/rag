"""
FastAPI приложение для RAG микросервиса (app1)
Запуск через Granian
"""

import time

from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from service.rag_service import RAGService
from core.config import get_settings
from core.logger import get_logger
from api.router import router

# Настройка логгера
logger = get_logger(__name__)
settings = get_settings()

# Глобальный экземпляр сервиса
rag_service: Optional[RAGService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global rag_service
    logger.info("🚀 Запуск RAG микросервиса...")
    
    # Инициализация сервисов
    try:
        rag_service = RAGService()
        logger.info("✓ RAG сервис инициализирован")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации RAG сервиса: {e}")
        raise
    
    yield
    
    logger.info("🛑 Остановка RAG микросервиса...")
    # Здесь можно добавить закрытие соединений (Milvus, Redis) если нужно



app = FastAPI(
    title="RAG Microservice API",
    description="API для векторного поиска и RAG системы (Milvus only)",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "service": "app1-rag",
        "milvus": settings.milvus.uri,
        "redis_enabled": settings.redis.enabled
    }




def main():
    """Запуск сервера через Granian"""
    from granian import Granian
    
    logger.info("Запуск Granian сервера...")
    
    Granian(
        "main:app",
        address="127.0.0.1",
        port=8000,
        interface="asgi",
        workers=1,
        reload=True
    ).serve()


if __name__ == "__main__":
    main()
