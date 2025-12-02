"""
Логирование для RAG микросервиса
"""

import logging
import sys
from functools import lru_cache


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Настройка логгера для модуля
    
    Args:
        name: Имя logger'а (обычно __name__ модуля)
        level: Уровень logging (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Настроенный logger
    """
    logger = logging.getLogger(name)
    
    # Если logger уже настроен, возвращаем его
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Форматирование
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler для вывода в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


@lru_cache()
def get_logger(name: str) -> logging.Logger:
    """
    Получить logger с кэшированием
    
    Args:
        name: Имя logger'а
    
    Returns:
        Logger instance
    """
    return setup_logger(name, level=logging.DEBUG)
