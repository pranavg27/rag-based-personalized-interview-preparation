"""BGE embeddings. Queries get the BGE instruction prefix; passages do not."""
from __future__ import annotations

import threading
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import Settings, get_settings


class BGEEmbeddings(Embeddings):
    def __init__(self, model_name: str, device: str = "cpu", query_prefix: str = ""):
        self.model_name, self.device, self.query_prefix = model_name, device, query_prefix
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer  # heavy import, lazy

                    self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0].tolist()


@lru_cache
def _cached(model: str, device: str, prefix: str) -> BGEEmbeddings:
    return BGEEmbeddings(model, device, prefix)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    return _cached(s.embedding_model, s.embedding_device, s.embedding_query_prefix)
