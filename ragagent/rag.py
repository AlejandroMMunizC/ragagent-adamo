"""Chunking, embedding and FAISS-based retrieval.

The :class:`RagIndex` class encapsulates the whole pipeline:

    1.  Chunk every document into overlapping windows.
    2.  Encode each chunk with a SentenceTransformer model.
    3.  Build a FAISS L2 index from those embeddings.
    4.  Expose a ``retrieve(question, k)`` method that returns the *k* most
        relevant chunks for a natural-language question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from . import config
from .documents import load_corpus


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split *text* into fixed-size character windows with *overlap*."""
    chunks: List[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start : start + size])
        start += step
    return chunks


@dataclass
class RagIndex:
    """In-memory FAISS index built from a folder of documents."""

    docs_dir: Path = field(default_factory=lambda: config.DOCS_DIR)
    chunk_size: int = config.CHUNK_SIZE
    chunk_overlap: int = config.CHUNK_OVERLAP
    embedding_model_name: str = config.EMBEDDING_MODEL

    # populated by build()
    _model: SentenceTransformer = field(init=False, default=None)
    _index: "faiss.Index" = field(init=False, default=None)
    _chunks_by_doc: Dict[str, List[str]] = field(init=False, default_factory=dict)
    _mapping: List[Tuple[str, int]] = field(init=False, default_factory=list)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self) -> "RagIndex":
        corpus = load_corpus(self.docs_dir)

        self._chunks_by_doc = {
            name: chunk_text(text, self.chunk_size, self.chunk_overlap)
            for name, text in corpus.items()
        }

        self._model = SentenceTransformer(self.embedding_model_name)
        dimension = self._model.get_sentence_embedding_dimension()
        self._index = faiss.IndexFlatL2(dimension)
        self._mapping = []

        for doc_name, chunks in self._chunks_by_doc.items():
            for idx, chunk in enumerate(chunks):
                vector = self._model.encode(chunk).reshape(1, -1)
                self._index.add(vector)
                self._mapping.append((doc_name, idx))

        return self

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not self._mapping

    def retrieve(self, question: str, k: int = config.DEFAULT_K) -> List[str]:
        """Return the *k* chunks most similar to *question*."""
        if self.is_empty:
            return []
        query_vec = self._model.encode(question).reshape(1, -1)
        _, indices = self._index.search(query_vec, k)
        results: List[str] = []
        for idx in indices[0]:
            if idx == -1:
                continue
            doc_name, chunk_idx = self._mapping[idx]
            results.append(self._chunks_by_doc[doc_name][chunk_idx])
        return results

    def stats(self) -> Dict[str, int]:
        return {
            "documents": len(self._chunks_by_doc),
            "chunks": len(self._mapping),
        }
