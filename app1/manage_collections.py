"""
Скрипт управления коллекциями Milvus (CLI)
Использование: python manage_collections.py [command] [args]
"""

import argparse
import sys
from typing import List

from service.milvus_service import MilvusService
from core.logger import get_logger

logger = get_logger(__name__)


def list_collections(service: MilvusService):
    """Список всех коллекций"""
    collections = service.list_collections()
    print(f"\nНайдено коллекций: {len(collections)}")
    for name in collections:
        try:
            stats = service.get_collection_stats(name)
            count = stats.get('row_count', 'N/A')
            print(f"  - {name} (Docs: {count})")
        except:
            print(f"  - {name}")


def create_collection(service: MilvusService, name: str):
    """Создание коллекции"""
    print(f"\nСоздание коллекции '{name}'...")
    try:
        service.ensure_collection(name)
        print(f"✓ Коллекция '{name}' успешно создана/проверена")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


def delete_collection(service: MilvusService, name: str):
    """Удаление коллекции"""
    print(f"\nВНИМАНИЕ: Вы собираетесь удалить коллекцию '{name}'!")
    confirm = input("Вы уверены? Это удалит ВСЕ данные. (y/n): ")
    
    if confirm.lower() == 'y':
        try:
            service.drop_collection(name)
            print(f"✓ Коллекция '{name}' удалена")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    else:
        print("Отмена операции")


def recreate_collection(service: MilvusService, name: str):
    """Пересоздание коллекции (очистка)"""
    print(f"\nВНИМАНИЕ: Вы собираетесь ПЕРЕСОЗДАТЬ коллекцию '{name}'!")
    confirm = input("Вы уверены? Это удалит ВСЕ данные. (y/n): ")
    
    if confirm.lower() == 'y':
        try:
            # Проверяем есть ли она
            cols = service.list_collections()
            if name in cols:
                service.drop_collection(name)
                print(f"✓ Старая коллекция удалена")
            
            service.ensure_collection(name)
            print(f"✓ Новая коллекция '{name}' создана")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    else:
        print("Отмена операции")


def main():
    parser = argparse.ArgumentParser(description="Управление коллекциями Milvus")
    subparsers = parser.add_subparsers(dest="command", help="Команды")
    
    # list
    subparsers.add_parser("list", help="Показать все коллекции")
    
    # create
    create_parser = subparsers.add_parser("create", help="Создать коллекцию")
    create_parser.add_argument("name", help="Имя коллекции")
    
    # delete
    delete_parser = subparsers.add_parser("delete", help="Удалить коллекцию")
    delete_parser.add_argument("name", help="Имя коллекции")
    
    # recreate
    recreate_parser = subparsers.add_parser("recreate", help="Пересоздать (очистить) коллекцию")
    recreate_parser.add_argument("name", help="Имя коллекции")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return

    try:
        service = MilvusService()
        
        if args.command == "list":
            list_collections(service)
        elif args.command == "create":
            create_collection(service, args.name)
        elif args.command == "delete":
            delete_collection(service, args.name)
        elif args.command == "recreate":
            recreate_collection(service, args.name)
            
    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
