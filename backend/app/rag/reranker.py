from __future__ import annotations

import logging
import threading

from app.config import Settings
from app.rag.vector_store import ScoredChunk

log = logging.getLogger(__name__)


class Reranker:
    """bge-reranker-base cross-encoder. Disabled -> simple truncation; load failure -> degrade gracefully."""

    def __init__(self, settings: Settings):
        self.enabled = settings.reranker_enabled
        self.model_name = settings.reranker_model
        self.device = settings.embedding_device
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None and not self._failed:
            with self._lock:
                if self._model is None and not self._failed:
                    try:
                        from sentence_transformers import CrossEncoder

                        self._model = CrossEncoder(self.model_name, device=self.device)
                    except Exception as exc:
                        log.warning("reranker unavailable (%s); continuing without it", exc)
                        self._failed = True
        return self._model

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not self.enabled or len(chunks) <= 1:
            return chunks[:top_n]
        model = self._load()
        if model is None:
            return chunks[:top_n]
        scores = model.predict([(query, c.text) for c in chunks])
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
        return sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_n]
