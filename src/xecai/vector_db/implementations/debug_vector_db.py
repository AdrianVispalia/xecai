import logging
import math
import re
from collections import Counter
from typing import List
from xecai.embeddings.embedding_interface import EmbeddingInterface
from xecai.models import Chunk, SearchType
from xecai.vector_db.vector_db_interface import VectorDBInterface


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer."""
    return re.findall(r"\w+", text.lower())


def compute_bm25_scores(
    query: str, documents: List[str], k1: float = 1.5, b: float = 0.75
) -> List[float]:
    """
    Computes BM25 scores for a query against a list of documents.
    """
    query_terms = tokenize(query)
    if not query_terms or not documents:
        return [0.0] * len(documents)

    doc_tokens = [tokenize(doc) for doc in documents]
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    avgdl = sum(doc_lengths) / len(documents) if documents else 1.0
    N = len(documents)

    # Calculate document frequencies for query terms
    df = Counter()
    for tokens in doc_tokens:
        unique_tokens = set(tokens)
        for term in query_terms:
            if term in unique_tokens:
                df[term] += 1

    # Calculate IDF for query terms
    idf = {}
    for term in query_terms:
        # Standard BM25 IDF formula
        n_q = df[term]
        idf[term] = math.log(((N - n_q + 0.5) / (n_q + 0.5)) + 1.0)

    # Calculate scores
    scores = []
    for i, tokens in enumerate(doc_tokens):
        score = 0.0
        term_counts = Counter(tokens)
        doc_len = doc_lengths[i]

        for term in query_terms:
            if term not in term_counts:
                continue
            tf = term_counts[term]
            # BM25 term frequency normalization
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
            score += idf[term] * (numerator / denominator)

        scores.append(score)

    return scores


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


class DebugVectorDB(VectorDBInterface):
    """
    Debug implementation of VectorDBInterface that stores chunks in a list and queries them in-memory.
    Supports basic semantic search using Cosine Similarity and Hybrid Search combining it with BM25 keyword search.
    """

    def __init__(
        self,
        embedding_interface: EmbeddingInterface | None = None,
        embedding_model: str | None = None,
    ):
        if not embedding_interface or not embedding_model:
            raise ValueError(
                "embedding_interface and embedding_model must be provided."
            )

        logging.warning(
            "DebugVectorDB is for debugging purposes only and is not recommended for production usecases."
        )
        self.embedding_interface = embedding_interface
        self.embedding_model = embedding_model
        self._chunks: List[Chunk] = []
        self._embeddings: List[List[float]] = []

    def _search(
        self, query: str, query_embedding: List[float], k: int, search_type: SearchType
    ) -> List[Chunk]:
        if not self._chunks:
            return []

        # Calculate semantic scores
        semantic_scores = [
            cosine_similarity(query_embedding, chunk_emb)
            for chunk_emb in self._embeddings
        ]

        if search_type == SearchType.SEMANTIC:
            final_scores = semantic_scores
        else:
            # Hybrid search: Combine semantic score with BM25 keyword score
            documents = [chunk.content for chunk in self._chunks]
            bm25_scores = compute_bm25_scores(query, documents)

            # Normalize BM25 scores (min-max normalization) to combine with cosine similarity (0 to 1)
            max_bm25 = max(bm25_scores) if bm25_scores else 0
            min_bm25 = min(bm25_scores) if bm25_scores else 0

            normalized_bm25 = []
            for score in bm25_scores:
                if max_bm25 == min_bm25:
                    normalized_bm25.append(0.0 if max_bm25 == 0 else 1.0)
                else:
                    normalized_bm25.append((score - min_bm25) / (max_bm25 - min_bm25))

            # Simple average for hybrid
            final_scores = [
                (sem_score + kw_score) / 2.0
                for sem_score, kw_score in zip(semantic_scores, normalized_bm25)
            ]

        # Apply scores and sort
        scored_chunks = []
        for i, chunk in enumerate(self._chunks):
            result_chunk = chunk.model_copy()
            result_chunk.score = final_scores[i]
            scored_chunks.append(result_chunk)

        # Sort descending by score
        scored_chunks.sort(key=lambda x: x.score or 0.0, reverse=True)
        return scored_chunks[:k]

    def sync_retrieve(
        self, query: str, k: int, search_type: SearchType = SearchType.SEMANTIC
    ) -> List[Chunk]:
        query_embedding = self.embedding_interface.sync_get_embeddings(
            query, self.embedding_model
        )
        return self._search(query, query_embedding, k, search_type)

    async def async_retrieve(
        self, query: str, k: int, search_type: SearchType = SearchType.SEMANTIC
    ) -> List[Chunk]:
        query_embedding = await self.embedding_interface.async_get_embeddings(
            query, self.embedding_model
        )
        return self._search(query, query_embedding, k, search_type)

    def sync_insert(self, chunks: List[Chunk]) -> None:
        for chunk in chunks:
            embedding = self.embedding_interface.sync_get_embeddings(
                chunk.content, self.embedding_model
            )
            self._chunks.append(chunk)
            self._embeddings.append(embedding)

    async def async_insert(self, chunks: List[Chunk]) -> None:
        for chunk in chunks:
            embedding = await self.embedding_interface.async_get_embeddings(
                chunk.content, self.embedding_model
            )
            self._chunks.append(chunk)
            self._embeddings.append(embedding)

    def sync_get_num_documents(self) -> int:
        return len(self._chunks)

    async def async_get_num_documents(self) -> int:
        return len(self._chunks)
