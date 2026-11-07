from pro_implementation import answer as ans
from pro_implementation.hybrid import Bm25Index, reciprocal_rank_fusion, tokenize


def result(text):
    return ans.Result(page_content=text, metadata={"source": "kb/a.md"})


def test_tokenize_lowercases_and_keeps_unicode_words():
    assert tokenize("Maxine Thompson, IIOTY-2023 año") == ["maxine", "thompson", "iioty", "2023", "año"]


def test_rrf_prefers_documents_ranked_high_in_several_lists():
    a, b, c = result("a"), result("b"), result("c")
    fused = reciprocal_rank_fusion([[a, b, c], [c, b], [b, c]])
    assert [x.page_content for x in fused][0] == "b"


def test_rrf_score_formula():
    a, b = result("a"), result("b")
    fused = reciprocal_rank_fusion([[a, b], [b]], k=60)
    # a: 1/61 = 0.01639 ; b: 1/62 + 1/61 = 0.03252
    assert [x.page_content for x in fused] == ["b", "a"]


def test_rrf_deduplicates_by_content():
    fused = reciprocal_rank_fusion([[result("a"), result("b")], [result("b"), result("a")]])
    assert sorted(x.page_content for x in fused) == ["a", "b"]


def test_rrf_ties_keep_first_seen_order():
    fused = reciprocal_rank_fusion([[result("x")], [result("y")]])
    assert [x.page_content for x in fused] == ["x", "y"]


def test_rrf_handles_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_bm25_finds_exact_name_match():
    chunks = [
        result("Quarterly revenue grew in the insurance sector"),
        result("Maxine Thompson won the IIOTY award in 2023"),
        result("Employee benefits and holiday policy"),
    ]
    found = Bm25Index(chunks).search("Maxine Thompson award", k=2)
    assert found[0].page_content.startswith("Maxine Thompson")


def test_bm25_returns_nothing_without_overlap_or_query():
    index = Bm25Index([result("alpha beta"), result("gamma delta")])
    assert index.search("zzz", k=3) == []
    assert index.search("   ", k=3) == []
    assert Bm25Index([]).search("alpha", k=3) == []


def test_hybrid_fetch_context_fuses_vector_and_keyword_results(monkeypatch):
    monkeypatch.setattr(ans, "HYBRID", True)
    monkeypatch.setattr(ans, "rewrite_query", lambda q, history=None: "rewritten")
    monkeypatch.setattr(ans, "fetch_context_unranked", lambda q: [result("vector hit")])
    monkeypatch.setattr(ans, "fetch_context_keyword", lambda q: [result("keyword hit")])
    monkeypatch.setattr(ans, "rerank", lambda q, chunks: chunks)
    texts = [c.page_content for c in ans.fetch_context("original")]
    assert texts == ["vector hit", "keyword hit"]


def test_fetch_context_keyword_uses_stored_chunks(monkeypatch):
    class FakeCollection:
        def get(self, include):
            return {
                "documents": [
                    "Avery Lancaster founded Insurellm",
                    "Holiday policy",
                    "Contract renewal terms",
                    "Office locations and travel rules",
                ],
                "metadatas": [{"source": s} for s in ("a", "b", "c", "d")],
            }

    monkeypatch.setattr(ans, "collection", FakeCollection())
    monkeypatch.setattr(ans, "_bm25_index", None)
    found = ans.fetch_context_keyword("who founded Insurellm Lancaster")
    assert found[0].metadata["source"] == "a"