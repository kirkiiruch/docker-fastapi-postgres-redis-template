import pytest
from app.core.config import Settings, get_settings


def test_settings_default_values():
    settings = Settings()
    assert settings.MONGODB_URL is not None
    assert settings.MONGODB_DB_NAME == "knowledge_base"
    assert settings.LLM_MODEL == "llama3.2:3b"
    assert settings.EMBEDDING_MODEL == "nomic-embed-text"
    assert settings.CHUNK_SIZE in (500, 700)
    assert settings.CHUNK_OVERLAP in (50, 150)
    assert settings.TOP_K in (3, 5)
    assert 0.5 <= settings.SCORE_THRESHOLD <= 0.9
    assert settings.MAX_FILE_SIZE_MB == 10


def test_get_settings_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_document_and_chunk_models():
    from app.models.document import DocumentModel, ChunkModel, DocumentStatus

    doc = DocumentModel(
        filename="test.txt",
        content_hash="abc123hash",
        file_size=100
    )
    assert doc.status == DocumentStatus.PROCESSING
    assert doc.chunks_count == 0
    assert doc.error_message is None
    assert doc.id is not None

    chunk = ChunkModel(
        document_id=doc.id,
        filename=doc.filename,
        chunk_index=0,
        page=1,
        text="Sample chunk content",
        embedding=[0.1, 0.2, 0.3]
    )
    assert chunk.document_id == doc.id
    assert chunk.page == 1
    assert len(chunk.embedding) == 3


def test_document_schemas():
    from datetime import datetime, timezone
    from app.schemas.document import DocumentUploadResponse, DocumentListItem, DocumentDetailResponse
    from app.models.document import DocumentStatus

    upload_res = DocumentUploadResponse(id="doc-1", status=DocumentStatus.PROCESSING)
    assert upload_res.id == "doc-1"
    assert upload_res.status == DocumentStatus.PROCESSING

    now = datetime.now(timezone.utc)
    item = DocumentListItem(
        id="doc-1",
        filename="notes.md",
        status=DocumentStatus.READY,
        chunks_count=5,
        created_at=now
    )
    assert item.chunks_count == 5

    detail = DocumentDetailResponse(
        id="doc-1",
        filename="notes.md",
        status=DocumentStatus.READY,
        chunks_count=5,
        file_size=1024,
        created_at=now,
        updated_at=now
    )
    assert detail.file_size == 1024


def test_rag_schemas_validation():
    from app.schemas.rag import AskRequest, AskResponse, SourceItem
    from pydantic import ValidationError

    # Valid request
    req = AskRequest(question="What is the onboarding policy?", top_k=5)
    assert req.question == "What is the onboarding policy?"
    assert req.top_k == 5
    assert req.document_ids is None

    # Invalid question - too short (<3)
    with pytest.raises(ValidationError):
        AskRequest(question="hi")

    # Invalid question - too long (>1000)
    with pytest.raises(ValidationError):
        AskRequest(question="a" * 1001)

    # Invalid top_k (<1 or >10)
    with pytest.raises(ValidationError):
        AskRequest(question="Valid question", top_k=0)

    with pytest.raises(ValidationError):
        AskRequest(question="Valid question", top_k=11)

    # Valid response
    source = SourceItem(
        document_id="doc-123",
        filename="policy.pdf",
        page=2,
        score=0.88,
        snippet="Snippet text"
    )
    res = AskResponse(answer="Here is your answer", sources=[source])
    assert len(res.sources) == 1
    assert res.sources[0].page == 2


@pytest.mark.asyncio
async def test_database_init_indexes_mocked(mocker):
    from app.core.database import init_indexes

    mock_db = mocker.AsyncMock()
    mock_docs_col = mocker.AsyncMock()
    mock_chunks_col = mocker.AsyncMock()

    class EmptyAsyncIter:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise StopAsyncIteration

    mock_chunks_col.list_search_indexes = mocker.MagicMock(return_value=EmptyAsyncIter())

    def getitem_side_effect(name):
        if name == "documents":
            return mock_docs_col
        elif name == "chunks":
            return mock_chunks_col
        return mocker.AsyncMock()

    mock_db.__getitem__.side_effect = getitem_side_effect

    await init_indexes(mock_db)

    # Verify regular index creation calls
    assert mock_docs_col.create_index.call_count >= 2
    assert mock_chunks_col.create_index.call_count >= 2


@pytest.mark.asyncio
async def test_document_repository(mocker):
    from app.repositories.document_repo import DocumentRepository
    from app.models.document import DocumentModel, DocumentStatus

    mock_db = mocker.MagicMock()
    mock_col = mocker.AsyncMock()
    mock_db.__getitem__.return_value = mock_col

    repo = DocumentRepository(mock_db)

    # Test create
    doc = DocumentModel(filename="rules.txt", content_hash="hash123")
    created = await repo.create(doc)
    assert created.id == doc.id
    mock_col.insert_one.assert_awaited_once()

    # Test get_by_id found
    mock_col.find_one.return_value = doc.model_dump()
    found = await repo.get_by_id(doc.id)
    assert found is not None
    assert found.filename == "rules.txt"

    # Test get_by_hash found
    by_hash = await repo.get_by_hash("hash123")
    assert by_hash is not None
    assert by_hash.content_hash == "hash123"

    # Test update_status
    mock_update_result = mocker.MagicMock()
    mock_update_result.modified_count = 1
    mock_col.update_one.return_value = mock_update_result
    ok = await repo.update_status(doc.id, DocumentStatus.READY, chunks_count=10)
    assert ok is True

    # Test delete
    mock_del_result = mocker.MagicMock()
    mock_del_result.deleted_count = 1
    mock_col.delete_one.return_value = mock_del_result
    deleted = await repo.delete(doc.id)
    assert deleted is True


@pytest.mark.asyncio
async def test_chunk_repository(mocker):
    from app.repositories.chunk_repo import ChunkRepository, cosine_similarity
    from app.models.document import ChunkModel

    # Test cosine similarity
    v1 = [1.0, 0.0]
    v2 = [1.0, 0.0]
    assert cosine_similarity(v1, v2) == 1.0  # (1 + 1)/2 = 1.0

    v3 = [0.0, 1.0]
    assert cosine_similarity(v1, v3) == 0.5  # (0 + 1)/2 = 0.5

    mock_db = mocker.MagicMock()
    mock_col = mocker.MagicMock()
    mock_db.__getitem__.return_value = mock_col

    repo = ChunkRepository(mock_db)

    # Test insert_many
    chunks = [
        ChunkModel(document_id="doc1", filename="file.txt", chunk_index=0, text="first", embedding=[1.0, 0.0]),
        ChunkModel(document_id="doc1", filename="file.txt", chunk_index=1, text="second", embedding=[0.0, 1.0]),
    ]
    mock_insert_res = mocker.MagicMock()
    mock_insert_res.inserted_ids = ["id1", "id2"]
    mock_col.insert_many = mocker.AsyncMock(return_value=mock_insert_res)

    count = await repo.insert_many(chunks)
    assert count == 2

    # Test fallback vector search
    mock_col.aggregate.side_effect = Exception("No atlas search")
    
    class AsyncIter:
        def __init__(self, items):
            self.items = items
        def __aiter__(self):
            self._iter = iter(self.items)
            return self
        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

    mock_col.find.return_value = AsyncIter([c.model_dump() for c in chunks])

    results = await repo.vector_search(query_embedding=[1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_index"] == 0
    assert results[0]["score"] == 1.0


def test_ai_interfaces():
    from app.ai.base import LLMClient, EmbeddingClient
    import inspect

    assert inspect.isabstract(LLMClient)
    assert inspect.isabstract(EmbeddingClient)
    assert "generate" in LLMClient.__abstractmethods__
    assert "generate_stream" in LLMClient.__abstractmethods__
    assert "get_embedding" in EmbeddingClient.__abstractmethods__
    assert "get_embeddings" in EmbeddingClient.__abstractmethods__


@pytest.mark.asyncio
async def test_ollama_clients(mocker):
    from app.ai.ollama import OllamaLLMClient, OllamaEmbeddingClient
    import httpx

    llm = OllamaLLMClient(base_url="http://mock-ollama:11434", model="llama3.2:3b")
    embedder = OllamaEmbeddingClient(base_url="http://mock-ollama:11434", model="nomic-embed-text")

    # Mock non-streaming LLM generate
    mock_post_res = mocker.MagicMock()
    mock_post_res.status_code = 200
    mock_post_res.raise_for_status = mocker.MagicMock()
    mock_post_res.json.return_value = {"response": "Paris is the capital of France."}

    mocker.patch.object(httpx.AsyncClient, "post", return_value=mock_post_res)

    answer = await llm.generate("What is the capital of France?")
    assert answer == "Paris is the capital of France."

    # Mock embeddings batch
    mock_embed_res = mocker.MagicMock()
    mock_embed_res.status_code = 200
    mock_embed_res.raise_for_status = mocker.MagicMock()
    mock_embed_res.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    mocker.patch.object(httpx.AsyncClient, "post", return_value=mock_embed_res)

    vec = await embedder.get_embedding("Hello world")
    assert vec == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_ingestion_service(mocker):
    from app.services.ingestion import (
        IngestionService,
        calculate_sha256,
        split_text_into_chunks,
        extract_text_and_pages,
    )
    from app.core.config import Settings
    from app.models.document import DocumentStatus

    # Test chunking
    text = "Hello world! This is a test for chunking mechanism."
    chunks = split_text_into_chunks(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 1

    # Test chunking with long tokens (>150 chars: URLs, hashes, base64)
    long_url = "https://internal.company.com/api/v2/resource?token=" + ("a" * 180)
    long_b64 = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9." + ("B" * 220)
    doc_with_long_tokens = (
        ("Корпоративний текст для перевірки чанкера. " * 10)
        + f"\n{long_url}\n"
        + ("Додатковий опис процедури авторизації. " * 10)
        + f" {long_b64} "
        + ("Завершальний абзац документації. " * 10)
    )
    long_token_chunks = split_text_into_chunks(doc_with_long_tokens, chunk_size=700, chunk_overlap=150)
    assert any(long_url in c for c in long_token_chunks)
    assert any(long_b64 in c for c in long_token_chunks)
    for c in long_token_chunks:
        if "a" * 20 in c:
            assert long_url in c
        if "B" * 20 in c:
            assert long_b64 in c

    # Test text extraction txt
    content = b"Welcome to the knowledge base."
    pages = extract_text_and_pages("guide.txt", content)
    assert len(pages) == 1
    assert pages[0][0] == 1
    assert pages[0][1] == "Welcome to the knowledge base."

    # Test sha256
    hash_val = calculate_sha256(content)
    assert len(hash_val) == 64

    # Setup mocks
    mock_doc_repo = mocker.AsyncMock()
    mock_chunk_repo = mocker.AsyncMock()
    mock_embed_client = mocker.AsyncMock()
    mock_embed_client.get_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]
    settings = Settings(CHUNK_SIZE=20, CHUNK_OVERLAP=5, MAX_FILE_SIZE_MB=10)

    service = IngestionService(
        document_repo=mock_doc_repo,
        chunk_repo=mock_chunk_repo,
        embedding_client=mock_embed_client,
        settings=settings,
    )

    # Test duplicate check
    mock_doc_repo.get_by_hash.return_value = mocker.MagicMock()
    with pytest.raises(FileExistsError):
        await service.create_document("guide.txt", content)

    # Test create_document non-duplicate
    mock_doc_repo.get_by_hash.return_value = None
    mock_doc_repo.create.side_effect = lambda d: d
    new_doc = await service.create_document("guide.txt", content)
    assert new_doc.status == DocumentStatus.PROCESSING
    assert new_doc.filename == "guide.txt"

    # Test background process
    await service.process_document_background(new_doc.id, "guide.txt", content)
    mock_chunk_repo.insert_many.assert_awaited_once()
    mock_doc_repo.update_status.assert_awaited_with(
        doc_id=new_doc.id,
        status=DocumentStatus.READY,
        chunks_count=mocker.ANY,
    )


@pytest.mark.asyncio
async def test_retrieval_service(mocker):
    from app.services.retrieval import RetrievalService
    from app.core.config import Settings

    mock_chunk_repo = mocker.AsyncMock()
    mock_embed_client = mocker.AsyncMock()
    mock_embed_client.get_embedding.return_value = [0.1, 0.2, 0.3]

    settings = Settings(TOP_K=3, SCORE_THRESHOLD=0.6)

    service = RetrievalService(
        chunk_repo=mock_chunk_repo,
        embedding_client=mock_embed_client,
        settings=settings,
    )

    # 1 high score, 1 low score (below threshold 0.6)
    mock_chunk_repo.vector_search.return_value = [
        {
            "id": "c1",
            "document_id": "doc1",
            "filename": "guide.pdf",
            "page": 3,
            "text": "This is a relevant policy excerpt.",
            "score": 0.85,
        },
        {
            "id": "c2",
            "document_id": "doc1",
            "filename": "guide.pdf",
            "page": 1,
            "text": "Irrelevant text.",
            "score": 0.45,
        },
    ]

    chunks, sources = await service.retrieve("How to apply for vacation?")
    assert len(chunks) == 1
    assert len(sources) == 1
    assert sources[0].document_id == "doc1"
    assert sources[0].page == 3
    assert sources[0].score == 0.85
    assert "relevant policy" in sources[0].snippet


@pytest.mark.asyncio
async def test_rag_service_not_found(mocker):
    from app.services.rag import RAGService, NOT_FOUND_MESSAGE

    mock_retrieval = mocker.AsyncMock()
    mock_retrieval.retrieve.return_value = ([], [])
    mock_llm = mocker.AsyncMock()

    rag = RAGService(retrieval_service=mock_retrieval, llm_client=mock_llm)

    # Non-streaming below threshold
    res = await rag.ask("Some obscure question")
    assert res.answer == NOT_FOUND_MESSAGE
    assert res.sources == []
    # LLM should not be called
    mock_llm.generate.assert_not_called()

    # Streaming below threshold
    events = []
    async for event in rag.ask_stream("Some obscure question"):
        events.append(event)

    full_stream = "".join(events)
    assert "event: sources" in full_stream
    assert "event: token" in full_stream
    assert NOT_FOUND_MESSAGE in full_stream
    assert "event: done" in full_stream


@pytest.mark.asyncio
async def test_rag_service_with_context(mocker):
    from app.services.rag import RAGService
    from app.schemas.rag import SourceItem

    mock_retrieval = mocker.AsyncMock()
    source = SourceItem(document_id="d1", filename="onboarding.md", page=1, score=0.9, snippet="Working hours: 9-18")
    chunk = {"document_id": "d1", "filename": "onboarding.md", "page": 1, "text": "Working hours: 9-18", "score": 0.9}
    mock_retrieval.retrieve.return_value = ([chunk], [source])

    mock_llm = mocker.AsyncMock()
    mock_llm.generate.return_value = "Робочі години компанії: 9-18."

    async def mock_stream(prompt, system_prompt=None):
        yield "Робочі "
        yield "години: "
        yield "9-18."

    mock_llm.generate_stream = mock_stream

    rag = RAGService(retrieval_service=mock_retrieval, llm_client=mock_llm)

    # Test non-streaming
    res = await rag.ask("Які робочі години?")
    assert res.answer == "Робочі години компанії: 9-18."
    assert len(res.sources) == 1

    # Test streaming
    events = []
    async for event in rag.ask_stream("Які робочі години?"):
        events.append(event)

    full_stream = "".join(events)
    assert "event: sources" in full_stream
    assert "onboarding.md" in full_stream
    assert "event: token" in full_stream
    assert "Робочі " in full_stream
    assert "event: done" in full_stream


def test_dependency_providers(mocker):
    from app.api.dependencies import (
        get_app_settings,
        get_database,
        get_document_repository,
        get_chunk_repository,
        get_embedding_client,
        get_llm_client,
        get_ingestion_service,
        get_retrieval_service,
        get_rag_service,
    )
    from app.core.config import Settings
    from app.services.ingestion import IngestionService
    from app.services.retrieval import RetrievalService
    from app.services.rag import RAGService

    cfg = get_app_settings()
    assert isinstance(cfg, Settings)

    mock_db = mocker.MagicMock()
    doc_repo = get_document_repository(mock_db)
    chunk_repo = get_chunk_repository(mock_db)
    embed_client = get_embedding_client(cfg)
    llm_client = get_llm_client(cfg)

    ing_service = get_ingestion_service(doc_repo, chunk_repo, embed_client, cfg)
    assert isinstance(ing_service, IngestionService)

    ret_service = get_retrieval_service(chunk_repo, embed_client, cfg)
    assert isinstance(ret_service, RetrievalService)

    rag_service = get_rag_service(ret_service, llm_client)
    assert isinstance(rag_service, RAGService)


@pytest.mark.asyncio
async def test_documents_router(mocker):
    from fastapi import FastAPI
    from httpx import AsyncClient, ASGITransport
    from app.api.routers.documents import router as doc_router
    from app.api.dependencies import (
        get_ingestion_service,
        get_document_repository,
        get_chunk_repository,
        get_app_settings,
    )
    from app.core.config import Settings
    from app.models.document import DocumentModel, DocumentStatus

    test_app = FastAPI()
    test_app.include_router(doc_router)

    mock_ingestion = mocker.AsyncMock()
    mock_doc_repo = mocker.AsyncMock()
    mock_chunk_repo = mocker.AsyncMock()
    settings = Settings(MAX_FILE_SIZE_MB=10)

    test_app.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion
    test_app.dependency_overrides[get_document_repository] = lambda: mock_doc_repo
    test_app.dependency_overrides[get_chunk_repository] = lambda: mock_chunk_repo
    test_app.dependency_overrides[get_app_settings] = lambda: settings

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Unsupported extension -> 415
        files = {"file": ("script.py", b"print(1)", "text/plain")}
        res = await client.post("/api/documents", files=files)
        assert res.status_code == 415

        # 2. Duplicate upload -> 409
        mock_ingestion.create_document.side_effect = FileExistsError("Duplicate")
        files = {"file": ("doc.txt", b"Duplicate content", "text/plain")}
        res = await client.post("/api/documents", files=files)
        assert res.status_code == 409

        # 3. Successful upload -> 202
        doc = DocumentModel(id="doc-123", filename="doc.txt", content_hash="hash123", status=DocumentStatus.PROCESSING)
        mock_ingestion.create_document.side_effect = None
        mock_ingestion.create_document.return_value = doc
        files = {"file": ("doc.txt", b"Unique content", "text/plain")}
        res = await client.post("/api/documents", files=files)
        assert res.status_code == 202
        data = res.json()
        assert data["id"] == "doc-123"
        assert data["status"] == "processing"

        # 4. List documents -> 200
        mock_doc_repo.list_all.return_value = [doc]
        res = await client.get("/api/documents")
        assert res.status_code == 200
        docs_list = res.json()
        assert len(docs_list) == 1
        assert docs_list[0]["id"] == "doc-123"

        # 5. Get document by id -> 200
        mock_doc_repo.get_by_id.return_value = doc
        res = await client.get("/api/documents/doc-123")
        assert res.status_code == 200
        assert res.json()["id"] == "doc-123"

        # 6. Delete document -> 200
        mock_doc_repo.get_by_id.return_value = doc
        res = await client.delete("/api/documents/doc-123")
        assert res.status_code == 200
        assert "deleted successfully" in res.json()["message"]


@pytest.mark.asyncio
async def test_ask_router(mocker):
    from fastapi import FastAPI
    from httpx import AsyncClient, ASGITransport
    from app.api.routers.ask import router as ask_router
    from app.api.dependencies import get_rag_service
    from app.schemas.rag import AskResponse, SourceItem

    test_app = FastAPI()
    test_app.include_router(ask_router)

    mock_rag_service = mocker.AsyncMock()
    test_app.dependency_overrides[get_rag_service] = lambda: mock_rag_service

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Validation error: question too short (<3 chars) -> 422
        res = await client.post("/api/ask", json={"question": "hi"})
        assert res.status_code == 422

        # 2. Validation error: top_k > 10 -> 422
        res = await client.post("/api/ask", json={"question": "Valid question?", "top_k": 20})
        assert res.status_code == 422

        # 3. Successful POST /api/ask
        source = SourceItem(document_id="doc-1", filename="policy.txt", score=0.95, snippet="Vacation is 24 days")
        mock_rag_service.ask.return_value = AskResponse(answer="Відпустка 24 дні.", sources=[source])

        res = await client.post("/api/ask", json={"question": "Скільки днів відпустки?", "top_k": 3})
        assert res.status_code == 200
        data = res.json()
        assert data["answer"] == "Відпустка 24 дні."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["document_id"] == "doc-1"

        # 4. Stream GET /api/ask/stream
        async def mock_stream_gen(**kwargs):
            yield 'event: sources\ndata: []\n\n'
            yield 'event: token\ndata: {"text": "Тест"}\n\n'
            yield 'event: done\ndata: {}\n\n'

        mock_rag_service.ask_stream = mock_stream_gen

        res = await client.get("/api/ask/stream?question=Скільки%20днів%20відпустки?")
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        assert "event: sources" in res.text
        assert "event: token" in res.text
        assert "event: done" in res.text

        # 5. Stream invalid question (<3 chars) -> 422
        res = await client.get("/api/ask/stream?question=ab")
        assert res.status_code == 422


def test_static_files():
    import os
    from pathlib import Path

    static_dir = Path("app/static")
    index_html = static_dir / "index.html"
    style_css = static_dir / "style.css"
    app_js = static_dir / "app.js"

    assert index_html.exists()
    assert style_css.exists()
    assert app_js.exists()

    html_content = index_html.read_text(encoding="utf-8")
    assert "Корпоративна база знань" in html_content
    assert "id=\"upload-form\"" in html_content
    assert "id=\"ask-form\"" in html_content

    js_content = app_js.read_text(encoding="utf-8")
    assert "EventSource" in js_content
    assert "/api/documents" in js_content
    assert "/api/ask/stream" in js_content


@pytest.mark.asyncio
async def test_main_app_and_endpoints(mocker):
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    # Mock MongoDB command for health check
    mock_db = mocker.AsyncMock()
    mock_db.command.return_value = {"ok": 1}
    mocker.patch("app.main.get_db", return_value=mock_db)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Test root / serves static HTML
        res = await client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers.get("content-type", "")
        assert "Корпоративна база знань" in res.text

        # 2. Test health check
        health_res = await client.get("/health")
        assert health_res.status_code == 200
        data = health_res.json()
        assert data["database"] == "connected"
        assert data["status"] == "healthy"


def test_docker_and_env_configuration():
    from pathlib import Path
    import yaml

    compose_file = Path("docker-compose.yml")
    assert compose_file.exists()

    with open(compose_file, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data.get("services", {})
    assert "api" in services
    assert "mongodb" in services
    assert "ollama" in services
    assert "ollama-init" in services

    # Verify models pulled in ollama-init
    init_cmd = services["ollama-init"]["command"]
    assert "llama3.2:3b" in init_cmd
    assert "nomic-embed-text" in init_cmd

    # Verify Dockerfile
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "EXPOSE 8000" in dockerfile
    assert "app.main:app" in dockerfile

    # Verify .env.example
    env_example = Path(".env.example").read_text(encoding="utf-8")
    for key in [
        "MONGODB_URL",
        "MONGODB_DB_NAME",
        "OLLAMA_URL",
        "LLM_MODEL",
        "EMBEDDING_MODEL",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
        "SCORE_THRESHOLD",
    ]:
        assert key in env_example


@pytest.mark.asyncio
async def test_stress_scenarios(mocker):
    import asyncio
    from app.services.ingestion import split_text_into_chunks
    from app.services.retrieval import RetrievalService
    from app.services.rag import RAGService, NOT_FOUND_MESSAGE
    from app.core.config import Settings

    # 1. Stress test chunking on a large text (100k characters)
    large_text = "Corporate knowledge base test paragraph with various keywords. " * 1500
    chunks = split_text_into_chunks(large_text, chunk_size=300, chunk_overlap=30)
    assert len(chunks) > 300
    for c in chunks:
        assert len(c) <= 300

    # 2. Borderline threshold testing in retrieval service
    mock_chunk_repo = mocker.AsyncMock()
    mock_embed = mocker.AsyncMock()
    mock_embed.get_embedding.return_value = [0.1] * 768
    cfg = Settings(SCORE_THRESHOLD=0.5, TOP_K=5)

    mock_chunk_repo.vector_search.return_value = [
        {"id": "1", "filename": "doc.txt", "score": 0.501, "text": "pass"},
        {"id": "2", "filename": "doc.txt", "score": 0.499, "text": "fail"},
        {"id": "3", "filename": "doc.txt", "score": 0.500, "text": "pass"},
    ]

    retrieval = RetrievalService(mock_chunk_repo, mock_embed, cfg)
    filtered_chunks, sources = await retrieval.retrieve("Test question?")
    assert len(filtered_chunks) == 2
    assert sources[0].snippet == "pass"
    assert sources[1].snippet == "pass"

    # 3. Concurrent ask stress test (50 parallel requests)
    mock_llm = mocker.AsyncMock()
    mock_llm.generate.return_value = "Verified answer from document context."
    rag = RAGService(retrieval, mock_llm)

    tasks = [rag.ask(f"Question number {i}?") for i in range(50)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 50
    for r in results:
        assert r.answer == "Verified answer from document context."
        assert len(r.sources) == 2


















