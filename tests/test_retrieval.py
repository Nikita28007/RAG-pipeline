from types import SimpleNamespace

from pro_implementation import answer as ans


def result(text, source="kb/a.md"):
    return ans.Result(page_content=text, metadata={"source": source})


def test_merge_chunks_deduplicates_by_content():
    first = [result("a"), result("b")]
    second = [result("b"), result("c")]
    merged = ans.merge_chunks(first, second)
    assert [c.page_content for c in merged] == ["a", "b", "c"]


def test_make_rag_messages_includes_sources_and_history():
    chunks = [result("Alice founded it.", "kb/company/about.md")]
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    messages = ans.make_rag_messages("Who founded it?", history, chunks)
    assert messages[0]["role"] == "system"
    assert "kb/company/about.md" in messages[0]["content"]
    assert "Alice founded it." in messages[0]["content"]
    assert messages[1:3] == history
    assert messages[-1] == {"role": "user", "content": "Who founded it?"}


def test_rerank_applies_model_order(monkeypatch):
    reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"order": [2, 1]}'))]
    )
    monkeypatch.setattr(ans, "completion", lambda **kwargs: reply)
    ranked = ans.rerank("q", [result("first"), result("second")])
    assert [c.page_content for c in ranked] == ["second", "first"]


def test_fetch_context_unranked_returns_results_from_vector_store(monkeypatch):
    embeddings = SimpleNamespace(
        create=lambda model, input: SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])
    )
    monkeypatch.setattr(ans, "openai", SimpleNamespace(embeddings=embeddings))
    calls = {}

    def query(query_embeddings, n_results):
        calls["n_results"] = n_results
        return {"documents": [["doc1", "doc2"]], "metadatas": [[{"source": "a"}, {"source": "b"}]]}

    monkeypatch.setattr(ans, "collection", SimpleNamespace(query=query))
    chunks = ans.fetch_context_unranked("question")
    assert [c.page_content for c in chunks] == ["doc1", "doc2"]
    assert calls["n_results"] == ans.RETRIEVAL_K


def test_fetch_context_searches_original_and_rewritten_query(monkeypatch):
    searched = []

    def fake_unranked(question):
        searched.append(question)
        return [result(f"chunk for {question}")]

    monkeypatch.setattr(ans, "rewrite_query", lambda q, history=None: "rewritten")
    monkeypatch.setattr(ans, "fetch_context_unranked", fake_unranked)
    monkeypatch.setattr(ans, "rerank", lambda q, chunks: list(reversed(chunks)))

    context = ans.fetch_context("original")
    assert searched == ["original", "rewritten"]
    assert [c.page_content for c in context] == ["chunk for rewritten", "chunk for original"]


def test_fetch_context_limits_result_to_final_k(monkeypatch):
    many = [result(f"c{i}") for i in range(ans.FINAL_K + 5)]
    monkeypatch.setattr(ans, "rewrite_query", lambda q, history=None: q)
    monkeypatch.setattr(ans, "fetch_context_unranked", lambda q: many)
    monkeypatch.setattr(ans, "rerank", lambda q, chunks: chunks)
    assert len(ans.fetch_context("q")) == ans.FINAL_K