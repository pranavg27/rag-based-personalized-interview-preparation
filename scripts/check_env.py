"""Verify your setup: dependency versions, MongoDB, ChromaDB, embeddings, Hugging Face model access.

    python scripts/check_env.py            # everything
    python scripts/check_env.py --skip-llm # without the (billable) LLM test call
"""
import asyncio
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.config import get_settings  # noqa: E402

S = get_settings()
PACKAGES = ["fastapi", "uvicorn", "pydantic", "pymongo", "langchain", "langchain-core", "langgraph",
            "langgraph-checkpoint-mongodb", "langsmith", "langchain-huggingface", "huggingface-hub", "chromadb",
            "sentence-transformers", "fastmcp", "mcp", "langchain-mcp-adapters", "PyJWT", "pwdlib", "streamlit"]
bad = 0


def ok(m):
    print(f"  [OK]   {m}")


def fail(m):
    global bad
    bad += 1
    print(f"  [FAIL] {m}")


def packages():
    print("Packages")
    for p in PACKAGES:
        try:
            ok(f"{p}=={version(p)}")
        except PackageNotFoundError:
            fail(f"{p} not installed")


async def mongo():
    print("MongoDB")
    from pymongo import AsyncMongoClient
    try:
        c = AsyncMongoClient(S.mongodb_uri, serverSelectionTimeoutMS=3000)
        await c.admin.command("ping")
        ok(f"ping {S.mongodb_uri}")
        await c.close()
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")


def embeddings():
    print("Embeddings / reranker")
    try:
        from app.rag.embeddings import get_embeddings
        ok(f"{S.embedding_model} dim={len(get_embeddings(S).embed_query('hello'))}")
        if S.reranker_enabled:
            from sentence_transformers import CrossEncoder
            CrossEncoder(S.reranker_model, device=S.embedding_device)
            ok(f"{S.reranker_model} loaded")
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")


def chroma():
    print("Vector store")
    try:
        import chromadb
        chromadb.PersistentClient(path=str(S.chroma_path)).heartbeat()
        ok(f"ChromaDB persistent dir {S.chroma_path}")
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")


async def llm():
    print(f"LLM {S.llm_model_id} (provider={S.hf_provider})")
    try:
        from app.services.llm_service import LLMService
        out = await LLMService(S).ainvoke_text([("human", "Reply with the single word: ready")])
        ok(f"model replied: {out.strip()[:40]!r}")
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
        print("         -> accept the Llama license on the model page, check token permissions / HF_PROVIDER.")


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]}", "OK" if sys.version_info >= (3, 11) else "-> need 3.11+")
    packages()
    asyncio.run(mongo())
    chroma()
    embeddings()
    if "--skip-llm" not in sys.argv:
        asyncio.run(llm())
    print("\nAll checks passed." if not bad else f"\n{bad} check(s) failed.")
    sys.exit(1 if bad else 0)
