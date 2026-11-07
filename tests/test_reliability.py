from types import SimpleNamespace

import pytest
from tenacity import wait_none

from pro_implementation import answer as ans


def result(text):
    return ans.Result(page_content=text, metadata={"source": "kb/a.md"})


def reply(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_apply_order_follows_model_order():
    chunks = [result("a"), result("b"), result("c")]
    assert [c.page_content for c in ans.apply_order(chunks, [3, 1, 2])] == ["c", "a", "b"]


def test_apply_order_ignores_out_of_range_and_repeated_ids():
    chunks = [result("a"), result("b")]
    ranked = ans.apply_order(chunks, [0, 5, 2, 2, -1])
    assert [c.page_content for c in ranked] == ["b", "a"]


def test_apply_order_appends_chunks_the_model_left_out():
    chunks = [result("a"), result("b"), result("c")]
    ranked = ans.apply_order(chunks, [2])
    assert [c.page_content for c in ranked] == ["b", "a", "c"]


def test_apply_order_with_empty_order_keeps_original_order():
    chunks = [result("a"), result("b")]
    assert ans.apply_order(chunks, []) == chunks


def test_rerank_never_loses_chunks_on_incomplete_answer(monkeypatch):
    monkeypatch.setattr(ans, "completion", lambda **kwargs: reply('{"order": [2]}'))
    ranked = ans.rerank.retry_with(wait=wait_none())("q", [result("a"), result("b"), result("c")])
    assert [c.page_content for c in ranked] == ["b", "a", "c"]


def test_retry_is_bounded_and_reraises_original_error(monkeypatch):
    calls = []

    def failing(**kwargs):
        calls.append(1)
        raise ConnectionError("api down")

    monkeypatch.setattr(ans, "completion", failing)
    with pytest.raises(ConnectionError):
        ans.rewrite_query.retry_with(wait=wait_none())("question")
    assert len(calls) == 4


def test_fetch_context_falls_back_to_retrieval_order_when_rerank_fails(monkeypatch):
    monkeypatch.setattr(ans, "HYBRID", False)
    monkeypatch.setattr(ans, "rewrite_query", lambda q, history=None: q)
    monkeypatch.setattr(ans, "fetch_context_unranked", lambda q: [result("a"), result("b")])

    def broken_rerank(question, chunks):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(ans, "rerank", broken_rerank)
    assert [c.page_content for c in ans.fetch_context("q")] == ["a", "b"]