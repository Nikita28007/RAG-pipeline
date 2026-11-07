from pro_implementation import ingest
import pickle
from types import SimpleNamespace

import litellm
import pytest
from tenacity import wait_none

def test_fetch_documents_reads_every_markdown_file():
    documents = ingest.fetch_documents()
    assert documents
    assert all({"type", "source", "text"} <= d.keys() and d["text"] for d in documents)


def test_make_prompt_contains_document_metadata():
    document = {"type": "company", "source": "kb/company/about.md", "text": "x" * 450}
    prompt = ingest.make_prompt(document)
    assert "company" in prompt and "kb/company/about.md" in prompt
    assert "at least 5 chunks" in prompt  # 450 // 100 + 1


DOCUMENT = {"type": "company", "source": "kb/company/about.md", "text": "text"}


def test_chunk_as_result_embeds_headline_summary_and_text():
    chunk = ingest.Chunk(headline="Head", summary="Sum", original_text="Text")
    result = chunk.as_result({"type": "company", "source": "kb/a.md"})
    assert result.page_content == "Head\n\nSum\n\nText"
    assert result.metadata == {"source": "kb/a.md", "type": "company"}


def not_found():
    return litellm.NotFoundError(message="model gone", model="m", llm_provider="openai")


def test_worker_error_is_picklable_and_keeps_the_original_message(monkeypatch):
    def boom(document):
        raise not_found()

    monkeypatch.setattr(ingest, "process_document", boom)
    with pytest.raises(ingest.IngestError) as info:
        ingest.process_document_safe(DOCUMENT)
    restored = pickle.loads(pickle.dumps(info.value))
    assert "model gone" in str(restored)
    assert "NotFoundError" in str(restored)
    assert "kb/company/about.md" in str(restored)


def test_litellm_errors_cannot_cross_a_process_boundary_unwrapped():
    with pytest.raises(TypeError):
        pickle.loads(pickle.dumps(not_found()))


def test_permanent_errors_are_not_retried(monkeypatch):
    calls = []

    def failing(**kwargs):
        calls.append(1)
        raise not_found()

    monkeypatch.setattr(ingest, "completion", failing)
    with pytest.raises(litellm.NotFoundError):
        ingest.process_document.retry_with(wait=wait_none())(DOCUMENT)
    assert len(calls) == 1


def test_transient_errors_are_retried_up_to_the_limit(monkeypatch):
    calls = []

    def failing(**kwargs):
        calls.append(1)
        raise ConnectionError("network")

    monkeypatch.setattr(ingest, "completion", failing)
    with pytest.raises(ConnectionError):
        ingest.process_document.retry_with(wait=wait_none())(DOCUMENT)
    assert len(calls) == 5


def test_preflight_reports_an_unusable_chunking_model(monkeypatch):
    def failing(**kwargs):
        raise not_found()

    monkeypatch.setattr(ingest, "completion", failing)
    with pytest.raises(SystemExit) as info:
        ingest.preflight()
    message = str(info.value)
    assert ingest.MODEL in message and "NotFoundError" in message and "INGEST_MODEL" in message


def test_preflight_reports_an_unusable_embedding_model(monkeypatch):
    def failing(model, input):
        raise RuntimeError("embedding model not found")

    monkeypatch.setattr(ingest, "completion", lambda **kwargs: None)
    monkeypatch.setattr(ingest, "openai", SimpleNamespace(embeddings=SimpleNamespace(create=failing)))
    with pytest.raises(SystemExit) as info:
        ingest.preflight()
    assert ingest.embedding_model in str(info.value)