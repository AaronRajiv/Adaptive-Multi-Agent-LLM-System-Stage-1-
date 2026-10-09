"""
BM25 Keyword Search Engine (Okapi BM25).
Pure-Python, zero-dependency implementation compatible with rank-bm25.
"""

import math
import re
from collections import Counter
from typing import Any, List, Sequence, Tuple


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return []
    return re.findall(r"\b[a-z0-9]+(?:\.[a-z0-9]+)*\b", text.lower())


class BM25Okapi:
    """
    Standard Okapi BM25 implementation for lexical keyword search.
    Scores documents based on query term frequency, inverse document frequency,
    and document length normalization.
    """

    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0.0

        # Term counts per document and document frequency across corpus
        self.doc_freqs: List[Counter] = [Counter(doc) for doc in corpus]
        df: Counter = Counter()
        for doc_counter in self.doc_freqs:
            for term in doc_counter.keys():
                df[term] += 1

        # Calculate standard smoothed IDF for each term in corpus
        self.idf: dict[str, float] = {}
        for term, freq in df.items():
            # Standard Okapi IDF with +1 smoothing to avoid negative weights
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query_tokens: Sequence[str]) -> List[float]:
        """Compute BM25 scores for all corpus documents against query tokens."""
        scores = [0.0] * self.corpus_size
        if not self.corpus_size or not query_tokens or self.avgdl == 0:
            return scores

        for q in query_tokens:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            for doc_idx, doc_counter in enumerate(self.doc_freqs):
                if q not in doc_counter:
                    continue
                q_freq = doc_counter[q]
                numerator = q_freq * (self.k1 + 1.0)
                denominator = q_freq + self.k1 * (1.0 - self.b + self.b * (self.doc_len[doc_idx] / self.avgdl))
                scores[doc_idx] += q_idf * (numerator / denominator)

        return scores

    def search(
        self,
        query: str,
        items: Sequence[Any],
        top_k: int = 10,
    ) -> List[Tuple[Any, float]]:
        """
        Search corpus items using query string and return top_k (item, score) pairs.
        Only items with positive scores are returned.
        """
        if not items or top_k < 1:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.get_scores(tokens)
        scored_pairs = [(items[i], scores[i]) for i in range(len(items)) if scores[i] > 0.0]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]
