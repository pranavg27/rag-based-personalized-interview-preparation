"""Test doubles: async in-memory Mongo, scripted chat model, deterministic hash embeddings."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


# ----------------------------------------------------------------------------- Mongo
def _get(doc: dict, path: str):
    cur: Any = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _match(doc: dict, flt: dict) -> bool:
    for k, cond in flt.items():
        if k == "$or":
            if not any(_match(doc, c) for c in cond):
                return False
            continue
        if k == "$and":
            if not all(_match(doc, c) for c in cond):
                return False
            continue
        val = _get(doc, k)
        if isinstance(cond, dict) and cond and all(str(x).startswith("$") for x in cond):
            for op, arg in cond.items():
                if op == "$in" and val not in arg:
                    return False
                if op == "$ne" and val == arg:
                    return False
                if op == "$gt" and not (val is not None and val > arg):
                    return False
                if op == "$gte" and not (val is not None and val >= arg):
                    return False
                if op == "$lt" and not (val is not None and val < arg):
                    return False
                if op == "$lte" and not (val is not None and val <= arg):
                    return False
                if op == "$exists" and (val is not None) != arg:
                    return False
        elif val != cond:
            return False
    return True


class FakeCursor:
    def __init__(self, docs: list[dict]):
        self.docs = docs

    def sort(self, key, direction=1):
        keys = key if isinstance(key, list) else [(key, direction)]
        for k, d in reversed(keys):
            self.docs.sort(key=lambda x: (_get(x, k) is None, _get(x, k)), reverse=d < 0)
        return self

    def skip(self, n):
        self.docs = self.docs[n:]
        return self

    def limit(self, n):
        self.docs = self.docs[:n] if n else self.docs
        return self

    async def to_list(self, length=None):
        return copy.deepcopy(self.docs[:length] if length else self.docs)


class FakeCollection:
    def __init__(self):
        self.docs: list[dict] = []

    async def create_index(self, *a, **k):
        return "idx"

    async def insert_one(self, doc):
        self.docs.append(copy.deepcopy(doc))
        return SimpleNamespace(inserted_id=doc.get("_id"))

    async def find_one(self, flt=None, sort=None, **_):
        rows = [d for d in self.docs if _match(d, flt or {})]
        if sort:
            rows = FakeCursor(rows).sort(sort).docs
        return copy.deepcopy(rows[0]) if rows else None

    def find(self, flt=None, projection=None):
        return FakeCursor([d for d in self.docs if _match(d, flt or {})])

    def _apply(self, doc: dict, upd: dict, inserting: bool = False):
        for k, v in upd.get("$set", {}).items():
            doc[k] = copy.deepcopy(v)
        if inserting:
            for k, v in upd.get("$setOnInsert", {}).items():
                doc[k] = copy.deepcopy(v)

    async def update_one(self, flt, upd, upsert=False):
        for d in self.docs:
            if _match(d, flt):
                self._apply(d, upd)
                return SimpleNamespace(matched_count=1, modified_count=1, upserted_id=None)
        if upsert:
            new = {k: v for k, v in flt.items() if not k.startswith("$") and not isinstance(v, dict)}
            self._apply(new, upd, inserting=True)
            self.docs.append(new)
            return SimpleNamespace(matched_count=0, modified_count=0, upserted_id=new.get("_id"))
        return SimpleNamespace(matched_count=0, modified_count=0, upserted_id=None)

    async def find_one_and_update(self, flt, upd, **_):
        for d in self.docs:
            if _match(d, flt):
                before = copy.deepcopy(d)
                self._apply(d, upd)
                return before
        return None

    async def delete_one(self, flt):
        for i, d in enumerate(self.docs):
            if _match(d, flt):
                del self.docs[i]
                return SimpleNamespace(deleted_count=1)
        return SimpleNamespace(deleted_count=0)

    async def delete_many(self, flt):
        keep = [d for d in self.docs if not _match(d, flt)]
        n = len(self.docs) - len(keep)
        self.docs = keep
        return SimpleNamespace(deleted_count=n)

    async def count_documents(self, flt):
        return sum(1 for d in self.docs if _match(d, flt))


class FakeDB:
    def __init__(self):
        self._c: dict[str, FakeCollection] = {}

    def __getitem__(self, name):
        return self._c.setdefault(name, FakeCollection())

    __getattr__ = __getitem__

    async def command(self, *_a, **_k):
        return {"ok": 1}


# ----------------------------------------------------------------------------- embeddings
class HashEmbeddings(Embeddings):
    """Bag-of-words hashing embeddings: deterministic, no downloads, similar text -> similar vectors."""

    DIM = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


# ----------------------------------------------------------------------------- chat model
EVAL_JSON = {
    "score": 7, "max_score": 10, "correctness": "Mostly correct.",
    "missing_points": ["mention indexes"], "strengths": ["clear structure"], "improvements": ["add an example"],
    "ideal_answer": "A model answer.", "follow_up_question": "How would this scale?",
    "rubric": {"technical_correctness": 8, "completeness": 6, "relevance": 8, "depth_of_understanding": 6,
               "clarity": 8, "use_of_examples": 5},
}


def default_responder(system: str, user: str) -> str:
    s = system.lower()
    if "interview question" in s and "generate one" in s:
        n = len(re.findall(r"question", user.lower()))
        topic = re.search(r"Topic: (.+)", user)
        return f"What is {topic.group(1) if topic else 'X'}? (variant {n})"
    if "interview evaluator" in s:
        return json.dumps(EVAL_JSON)
    if "personalized feedback" in s:
        return "Nice work. Add a concrete example next time."
    if "interview summary" in s:
        return "You did well overall. Keep practising weak topics."
    if "resume parser" in s:
        return json.dumps({"name": "Test User", "summary": "Backend dev", "skills": ["Python", "FastAPI"],
                           "projects": [{"name": "Chatbot", "description": "RAG bot", "technologies": ["LangChain"]}],
                           "experience": [], "education": []})
    if "job description analyzer" in s:
        return json.dumps({"role": "Backend", "required_skills": ["Python", "Docker"], "responsibilities": []})
    if "list of interview questions" in s:
        return json.dumps({"questions": [{"question": "Q1?", "topic": "T", "difficulty": "easy", "expected_points": []}]})
    if "progress report" in s:
        return "Progress is steady."
    if "numbered context" in s or "general-knowledge" in s or "nothing relevant" in s:
        return "Answer with citation [1]."
    return "ok"


class FakeChatModel(BaseChatModel):
    responder: Any = None
    calls: list = []
    fail_times: int = 0
    fail_status: int = 503

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _text(self, messages) -> str:
        system = next((m.content for m in messages if m.type == "system"), "")
        user = "\n".join(m.content for m in messages if m.type == "human")
        self.calls.append((system, user))
        if self.fail_times > 0:
            self.fail_times -= 1
            err = RuntimeError(f"{self.fail_status} Server Error: fake")
            err.status_code = self.fail_status  # type: ignore[attr-defined]
            raise err
        return (self.responder or default_responder)(system, user)

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._text(messages)))])

    def _stream(self, messages, stop=None, run_manager=None, **kw) -> Iterator[ChatGenerationChunk]:
        text = self._text(messages)
        for tok in re.findall(r"\S+\s*", text):
            yield ChatGenerationChunk(message=AIMessageChunk(content=tok))


def make_pdf(text: str) -> bytes:
    """Minimal valid single-page PDF containing `text` (extractable by pypdf)."""
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    lines = safe.split("\n")
    stream = "BT /F1 11 Tf 40 780 Td 14 TL " + " T* ".join(f"({l}) Tj" for l in lines) + " ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out
