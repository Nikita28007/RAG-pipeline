"""Keyword (BM25) search and Reciprocal Rank Fusion."""

import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(rankings, k=60, key=lambda item: item.page_content):
    """Merge several ranked lists. score(d) = sum over lists of 1 / (k + rank)."""
    scores, first_seen = {}, {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            ident = key(item)
            scores[ident] = scores.get(ident, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(ident, item)
    ordered = sorted(scores, key=lambda ident: scores[ident], reverse=True)
    return [first_seen[ident] for ident in ordered]


class Bm25Index:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        corpus = [tokenize(chunk.page_content) for chunk in self.chunks]
        self._bm25 = BM25Okapi(corpus) if self.chunks else None

    def search(self, query: str, k: int):
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in ranked if scores[i] > 0]