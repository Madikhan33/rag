"""
Препроцессинг текста для RAG системы
- Очистка текста
- Умное разделение на чанки
"""

import re
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class ChunkMetadata:
    """Метаданные для чанка"""
    chunk_index: int
    original_start_pos: int
    original_end_pos: int
    section: str = ""
    subsection: str = ""


class TextPreprocessor:
    """Препроцессор для подготовки текста к RAG"""
    
    # Регулярные выражения для очистки
    REGEX_PATTERNS = {
        'multiple_spaces': r'\s{2,}',
        'multiple_newlines': r'\n{3,}',
        'extra_punctuation': r'[!?]{2,}',
        'html_tags': r'<[^>]+>',
    }
    
    # Символы для удаления
    CHARS_TO_REMOVE = {
        '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
        '\x08', '\x0b', '\x0c', '\x0e', '\x0f',
        '\x1a', '\x1b', '\x1c', '\x1d', '\x1e', '\x1f',
        '\ufeff', '\u200b', '\u200c', '\u200d',
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Очистка текста от мусора"""
        if not text:
            return ""
        
        # Удаляем контрольные символы
        cleaned = ""
        for char in text:
            if char not in TextPreprocessor.CHARS_TO_REMOVE:
                cleaned += char
        
        # Удаляем HTML теги
        cleaned = re.sub(TextPreprocessor.REGEX_PATTERNS['html_tags'], '', cleaned)
        
        # Удаляем лишние переносы
        cleaned = re.sub(TextPreprocessor.REGEX_PATTERNS['multiple_newlines'], '\n\n', cleaned)
        
        # Нормализуем пробелы
        lines = cleaned.split('\n')
        normalized_lines = []
        
        for line in lines:
            line = line.strip()
            line = re.sub(TextPreprocessor.REGEX_PATTERNS['multiple_spaces'], ' ', line)
            if line:
                normalized_lines.append(line)
        
        cleaned = '\n'.join(normalized_lines)
        
        # Удаляем лишнюю пунктуацию
        cleaned = re.sub(r'!{2,}', '!', cleaned)
        cleaned = re.sub(r'\?{2,}', '?', cleaned)
        cleaned = re.sub(r'\.{4,}', '...', cleaned)
        
        return cleaned.strip()
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 2500,
        overlap: int = 200,
        smart: bool = True
    ) -> List[Tuple[str, ChunkMetadata]]:
        """
        Разделение текста на умные чанки
        
        Args:
            text: Текст для разделения
            chunk_size: Размер чанка в символах
            overlap: Перекрытие между чанками
            smart: Использовать умное разделение
        
        Returns:
            Список (текст_чанка, метаданные)
        """
        if not text:
            return []
        
        text = TextPreprocessor.clean_text(text)
        
        if len(text) <= chunk_size:
            metadata = ChunkMetadata(
                chunk_index=0,
                original_start_pos=0,
                original_end_pos=len(text)
            )
            return [(text, metadata)]
        
        if smart:
            chunks = TextPreprocessor._smart_chunking(text, chunk_size, overlap)
        else:
            chunks = TextPreprocessor._simple_chunking(text, chunk_size, overlap)
        
        return chunks
    
    @staticmethod
    def _simple_chunking(
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[Tuple[str, ChunkMetadata]]:
        """Простое разделение по размеру"""
        chunks = []
        step = chunk_size - overlap
        
        chunk_index = 0
        for i in range(0, len(text), step):
            chunk_text = text[i:i + chunk_size]
            
            if chunk_text.strip():
                metadata = ChunkMetadata(
                    chunk_index=chunk_index,
                    original_start_pos=i,
                    original_end_pos=min(i + chunk_size, len(text))
                )
                chunks.append((chunk_text, metadata))
                chunk_index += 1
        
        return chunks
    
    @staticmethod
    def _smart_chunking(
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[Tuple[str, ChunkMetadata]]:
        """
        Умное разделение с сохранением структуры
        
        Стратегия:
        1. Разбиваем по секциям (1., 2., 3.)
        2. По подсекциям (1.1., 1.2.)
        3. По параграфам
        4. По предложениям
        """
        chunks = []
        lines = text.split('\n')
        
        chunk_index = 0
        current_chunk = []
        current_size = 0
        current_start_pos = 0
        
        for line_idx, line in enumerate(lines):
            line_len = len(line) + 1  # +1 для '\n'
            
            stripped = line.strip()
            is_section = re.match(r'^(\d+)\.\s', stripped) is not None
            is_subsection = re.match(r'^(\d+\.\d+)\.\s', stripped) is not None
            
            # Если новая секция и чанк большой
            if is_section and current_chunk and current_size > chunk_size * 0.8:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    metadata = ChunkMetadata(
                        chunk_index=chunk_index,
                        original_start_pos=current_start_pos,
                        original_end_pos=current_start_pos + current_size
                    )
                    chunks.append((chunk_text, metadata))
                    chunk_index += 1
                
                current_chunk = [line]
                current_size = line_len
                current_start_pos = sum(len(l) + 1 for l in lines[:line_idx])
            
            # Если новая подсекция и текущий очень большой
            elif is_subsection and current_size > chunk_size:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    metadata = ChunkMetadata(
                        chunk_index=chunk_index,
                        original_start_pos=current_start_pos,
                        original_end_pos=current_start_pos + current_size
                    )
                    chunks.append((chunk_text, metadata))
                    chunk_index += 1
                
                current_chunk = [line]
                current_size = line_len
                current_start_pos = sum(len(l) + 1 for l in lines[:line_idx])
            
            # Если превышает лимит
            elif current_size + line_len > chunk_size and current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    metadata = ChunkMetadata(
                        chunk_index=chunk_index,
                        original_start_pos=current_start_pos,
                        original_end_pos=current_start_pos + current_size
                    )
                    chunks.append((chunk_text, metadata))
                    chunk_index += 1
                
                # Новый чанк с перекрытием
                overlap_lines = current_chunk[-max(1, overlap // 50):]
                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) + 1 for l in current_chunk)
                current_start_pos = sum(len(l) + 1 for l in lines[:line_idx]) - sum(len(l) + 1 for l in overlap_lines)
            
            # Обычное добавление
            else:
                if stripped:
                    current_chunk.append(line)
                    current_size += line_len
        
        # Последний чанк
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                metadata = ChunkMetadata(
                    chunk_index=chunk_index,
                    original_start_pos=current_start_pos,
                    original_end_pos=current_start_pos + current_size
                )
                chunks.append((chunk_text, metadata))
        
        return chunks if chunks else TextPreprocessor._simple_chunking(text, chunk_size, overlap)
