"""Сервіс валідації, парсингу, чанкінгу та індексації документів."""

import hashlib
import io
import logging
import os
from typing import List, Tuple, Optional
from pypdf import PdfReader
from app.ai.base import EmbeddingClient
from app.core.config import Settings
from app.models.document import DocumentModel, DocumentStatus, ChunkModel
from app.repositories.document_repo import DocumentRepository
from app.repositories.chunk_repo import ChunkRepository

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def calculate_sha256(content: bytes) -> str:
    """Обчислення SHA-256 хешу файлу."""
    return hashlib.sha256(content).hexdigest()


def _find_last_whitespace(text: str, start: int, end: int) -> int:
    """Знайти позицію останнього пробільного символу на проміжку [start, end)."""
    for idx in range(end - 1, start - 1, -1):
        if text[idx].isspace():
            return idx
    return -1


def _find_first_whitespace(text: str, start: int, end: int) -> int:
    """Знайти позицію першого пробільного символу на проміжку [start, end)."""
    for idx in range(start, end):
        if text[idx].isspace():
            return idx
    return -1


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Розбиття тексту на фрагменти заданого розміру з перекриттям без розриву слів."""
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    chunks = []
    start = 0
    text_len = len(cleaned_text)

    while start < text_len:
        while start < text_len and cleaned_text[start].isspace():
            start += 1
        if start >= text_len:
            break

        end = start + chunk_size
        if end < text_len:
            search_from = start + int(chunk_size * 0.65)
            nl_pos = cleaned_text.rfind("\n", search_from, end)
            if nl_pos > start:
                end = nl_pos
            elif not cleaned_text[end].isspace() and not cleaned_text[end - 1].isspace():
                ws_pos = _find_last_whitespace(cleaned_text, start, end)
                if ws_pos > start:
                    end = ws_pos
                else:
                    next_ws = _find_first_whitespace(cleaned_text, end, text_len)
                    end = next_ws if next_ws != -1 else text_len
        else:
            end = text_len

        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break

        next_start = max(0, end - chunk_overlap) if chunk_overlap > 0 else end

        if next_start > 0 and not cleaned_text[next_start].isspace() and not cleaned_text[next_start - 1].isspace():
            ws_pos = _find_first_whitespace(cleaned_text, next_start, end)
            if ws_pos != -1:
                next_start = ws_pos + 1
            else:
                prev_ws = _find_last_whitespace(cleaned_text, start, next_start)
                if prev_ws != -1 and (end - (prev_ws + 1)) <= chunk_size // 2:
                    next_start = prev_ws + 1
                else:
                    next_start = end

        next_word_start = end
        while next_word_start < text_len and cleaned_text[next_word_start].isspace():
            next_word_start += 1
        if next_word_start < text_len:
            next_word_end = _find_first_whitespace(cleaned_text, next_word_start, text_len)
            if next_word_end == -1:
                next_word_end = text_len
            if next_word_end - next_start > chunk_size:
                min_start = min(next_word_start, next_word_end - chunk_size)
                if min_start > next_start:
                    ws_pos = _find_first_whitespace(cleaned_text, min_start, next_word_start)
                    next_start = (ws_pos + 1) if ws_pos != -1 else next_word_start

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def extract_text_and_pages(filename: str, content: bytes) -> List[Tuple[Optional[int], str]]:
    """Вилучення тексту та номерів сторінок з файлу."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        pages_text = []
        reader = PdfReader(io.BytesIO(content))
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append((page_idx + 1, text))
        return pages_text

    elif ext in {".txt", ".md"}:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("utf-8", errors="replace")
        return [(1, text)] if text.strip() else []

    else:
        raise ValueError(f"Непідтримуваний формат файлу: {ext}")


class IngestionService:
    """Сервіс для збереження та фонової індексації файлів."""

    def __init__(
        self,
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
        embedding_client: EmbeddingClient,
        settings: Settings,
    ):
        self.doc_repo = document_repo
        self.chunk_repo = chunk_repo
        self.embedding_client = embedding_client
        self.settings = settings

    def validate_file(self, filename: str, content_size: int):
        """Перевірка розширення та максимального розміру файлу."""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Непідтримуваний формат: {ext}")
        max_bytes = self.settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if content_size > max_bytes:
            raise ValueError(f"Розмір файлу перевищує ліміт {self.settings.MAX_FILE_SIZE_MB}MB")

    async def create_document(self, filename: str, content: bytes) -> DocumentModel:
        """Попередня реєстрація документа та перевірка на дублікат."""
        self.validate_file(filename, len(content))
        content_hash = calculate_sha256(content)

        existing = await self.doc_repo.get_by_hash(content_hash)
        if existing:
            raise FileExistsError(f"Файл із хешем {content_hash} уже існує")

        doc = DocumentModel(
            filename=filename,
            content_hash=content_hash,
            file_size=len(content),
            status=DocumentStatus.PROCESSING,
        )
        return await self.doc_repo.create(doc)

    async def process_document_background(self, document_id: str, filename: str, content: bytes):
        """Фонова задача: розбиття на чанки, отримання ембедингів і запис."""
        try:
            pages = extract_text_and_pages(filename, content)
            chunks: List[ChunkModel] = []
            chunk_index = 0

            for page_num, page_text in pages:
                text_splits = split_text_into_chunks(
                    page_text,
                    chunk_size=self.settings.CHUNK_SIZE,
                    chunk_overlap=self.settings.CHUNK_OVERLAP,
                )
                for split in text_splits:
                    chunk = ChunkModel(
                        document_id=document_id,
                        filename=filename,
                        chunk_index=chunk_index,
                        page=page_num,
                        text=split,
                        embedding=[],
                    )
                    chunks.append(chunk)
                    chunk_index += 1

            if chunks:
                texts = [c.text for c in chunks]
                embeddings = await self.embedding_client.get_embeddings(texts)
                for chunk, emb in zip(chunks, embeddings):
                    chunk.embedding = emb

                await self.chunk_repo.insert_many(chunks)

            await self.doc_repo.update_status(
                doc_id=document_id,
                status=DocumentStatus.READY,
                chunks_count=len(chunks),
            )
            logger.info("Документ %s успішно індексовано (%d чанків)", document_id, len(chunks))

        except Exception as e:
            logger.error("Помилка індексації документа %s: %s", document_id, e)
            await self.doc_repo.update_status(
                doc_id=document_id,
                status=DocumentStatus.FAILED,
                error_message=str(e),
            )
