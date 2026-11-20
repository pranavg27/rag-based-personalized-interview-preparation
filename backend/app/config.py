"""Centralised configuration. Every tunable lives here and is read from environment / .env."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent

DEFAULT_RUBRIC: dict[str, float] = {
    "technical_correctness": 0.30,
    "completeness": 0.20,
    "relevance": 0.15,
    "depth_of_understanding": 0.15,
    "clarity": 0.10,
    "use_of_examples": 0.10,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(REPO_DIR / ".env"), ".env"), extra="ignore", case_sensitive=False
    )

    # app
    app_env: str = "development"
    log_level: str = "INFO"
    backend_url: str = "http://localhost:8000"
    cors_origins: str = "http://localhost:8501"

    # auth
    jwt_secret_key: str = "change-me-with-openssl-rand-hex-32"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # mongo
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "interview_prep"
    checkpointer: str = "mongo"  # mongo | memory
    checkpoint_collection: str = "langgraph_checkpoints"
    checkpoint_writes_collection: str = "langgraph_checkpoint_writes"

    # llm
    hf_token: str = ""
    llm_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    hf_provider: str = "auto"
    llm_temperature: float = 0.3
    llm_max_new_tokens: int = 1024
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 60

    # embeddings / rerank
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_device: str = "cpu"
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"

    # vector store
    vector_store: str = "chroma"  # chroma | faiss
    chroma_mode: str = "persistent"  # persistent | http
    chroma_persist_dir: str = "./data/chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "study_materials"
    faiss_index_dir: str = "./data/faiss"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 8
    rerank_top_n: int = 4
    similarity_threshold: float = 0.35
    rag_allow_general_fallback: bool = True

    # uploads
    upload_dir: str = "./data/uploads"
    max_upload_mb: int = 10
    allowed_resume_types: str = "pdf"
    allowed_doc_types: str = "pdf,txt,docx"

    # adaptive difficulty / memory
    difficulty_up_threshold: float = 8.0
    difficulty_down_threshold: float = 5.0
    difficulty_window: int = 3
    weak_topic_threshold: float = 6.0
    strong_topic_threshold: float = 8.0
    min_attempts_for_topic: int = 2
    memory_progress_keep: int = 20

    # evaluation rubric (JSON object: criterion -> weight)
    eval_rubric_json: str = ""

    # rate limiting
    rate_limit_enabled: bool = True
    rate_limit_default: str = "300/minute"
    rate_limit_auth: str = "10/minute"

    # MCP
    mcp_mode: str = "embedded"  # embedded | external | disabled
    mcp_fallback_local: bool = True
    mcp_resume_url: str = "http://localhost:8101/mcp"
    mcp_questions_url: str = "http://localhost:8102/mcp"
    mcp_analytics_url: str = "http://localhost:8103/mcp"
    mcp_jd_url: str = "http://localhost:8104/mcp"
    mcp_internal_token: str = "change-me-shared-secret"
    mcp_timeout_seconds: int = 90

    # langsmith
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-interview-prep"

    # ---- derived helpers -------------------------------------------------
    @staticmethod
    def _abs(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (REPO_DIR / path).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def upload_path(self) -> Path:
        return self._abs(self.upload_dir)

    @property
    def chroma_path(self) -> Path:
        return self._abs(self.chroma_persist_dir)

    @property
    def faiss_path(self) -> Path:
        return self._abs(self.faiss_index_dir)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def resume_exts(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_resume_types.split(",") if e.strip()}

    @property
    def doc_exts(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_doc_types.split(",") if e.strip()}

    @property
    def eval_rubric(self) -> dict[str, float]:
        if self.eval_rubric_json.strip():
            try:
                data: Any = json.loads(self.eval_rubric_json)
                rubric = {str(k): float(v) for k, v in data.items() if float(v) > 0}
                if rubric:
                    return rubric
            except (ValueError, TypeError, AttributeError):
                pass
        return dict(DEFAULT_RUBRIC)

    @property
    def mcp_urls(self) -> dict[str, str]:
        if self.mcp_mode == "embedded":
            base = self.backend_url.rstrip("/")
            return {n: f"{base}/mcp-servers/{n}/mcp" for n in ("resume", "questions", "analytics", "jd")}
        return {
            "resume": self.mcp_resume_url,
            "questions": self.mcp_questions_url,
            "analytics": self.mcp_analytics_url,
            "jd": self.mcp_jd_url,
        }

    def validate_for_runtime(self) -> None:
        if self.app_env == "production" and self.jwt_secret_key.startswith("change-me"):
            raise RuntimeError("Set a strong JWT_SECRET_KEY before running in production.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
