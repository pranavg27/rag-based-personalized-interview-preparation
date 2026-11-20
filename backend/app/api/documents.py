from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from app.database import mongo as m
from app.dependencies import CurrentUser, Svc
from app.errors import NotFoundError, ValidationFailed
from app.models.enums import SKILL_OPTIONS
from app.rag.ingestion import build_chunks, ingest_file
from app.schemas.content import DocumentOut, JobDescriptionIn
from app.utils.files import remove_file, save_upload
from app.utils.ids import new_id, serialize, utcnow

router = APIRouter(prefix="/documents", tags=["documents"])


def _out(d: dict) -> DocumentOut:
    return DocumentOut(**serialize({"id": d["_id"], "name": d["name"], "doc_type": d["doc_type"],
                                    "skill_category": d["skill_category"], "chunk_count": d["chunk_count"],
                                    "size_bytes": d["size_bytes"], "uploaded_at": d["uploaded_at"]}))


async def _store(svc, user_id: str, *, name: str, ext: str, path: Path | None, size: int, doc_type: str,
                 skill_category: str | None, text_override: str | None = None) -> dict:
    doc_id = new_id()
    s = svc.settings
    if text_override is not None:
        from langchain_core.documents import Document
        res = build_chunks([Document(page_content=text_override, metadata={})], settings=s, user_id=user_id,
                           document_id=doc_id, document_name=name, skill_category=skill_category, doc_type=doc_type)
    else:
        res = await asyncio.to_thread(ingest_file, path, ext, settings=s, user_id=user_id, document_id=doc_id,
                                      document_name=name, skill_category=skill_category, doc_type=doc_type)
    await asyncio.to_thread(svc.vector_store.add_chunks, res.chunks)
    doc = {"_id": doc_id, "user_id": user_id, "name": name, "doc_type": doc_type, "ext": ext,
           "skill_category": res.skill_category, "chunk_count": len(res.chunks), "size_bytes": size,
           "stored_path": str(path) if path else None, "uploaded_at": utcnow()}
    if doc_type == "job_description":
        doc["text"] = "\n".join(c.page_content for c in res.chunks)[:30000] if text_override is None else text_override
    await m.get_db()[m.DOCUMENTS].insert_one(doc)
    return doc


@router.post("/upload", response_model=DocumentOut, status_code=201)
async def upload_document(user: CurrentUser, svc: Svc, file: UploadFile = File(...),
                          skill_category: str | None = Form(default=None),
                          doc_type: str = Form(default="study")):
    if doc_type not in ("study", "job_description"):
        raise ValidationFailed("doc_type must be 'study' or 'job_description'.")
    if skill_category and skill_category not in SKILL_OPTIONS + ["General"]:
        raise ValidationFailed(f"skill_category must be one of {SKILL_OPTIONS + ['General']}")
    s = svc.settings
    saved = await save_upload(file, base_dir=s.upload_path / "documents", user_id=user["_id"],
                              allowed=s.doc_exts, max_bytes=s.max_upload_bytes)
    try:
        doc = await _store(svc, user["_id"], name=saved.original_name, ext=saved.ext, path=saved.path,
                           size=saved.size, doc_type=doc_type, skill_category=skill_category)
    except Exception:
        remove_file(saved.path)
        raise
    return _out(doc)


@router.post("/job-description", response_model=DocumentOut, status_code=201)
async def add_job_description(body: JobDescriptionIn, user: CurrentUser, svc: Svc):
    doc = await _store(svc, user["_id"], name=body.title, ext="txt", path=None, size=len(body.text),
                       doc_type="job_description", skill_category=None, text_override=body.text)
    return _out(doc)


@router.get("", response_model=list[DocumentOut])
async def list_documents(user: CurrentUser, doc_type: str | None = None):
    flt = {"user_id": user["_id"]}
    if doc_type:
        flt["doc_type"] = doc_type
    rows = await m.get_db()[m.DOCUMENTS].find(flt).sort("uploaded_at", -1).to_list(200)
    return [_out(r) for r in rows]


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str, user: CurrentUser, svc: Svc):
    col = m.get_db()[m.DOCUMENTS]
    doc = await col.find_one({"_id": document_id, "user_id": user["_id"]})
    if not doc:
        raise NotFoundError("Document not found.")
    await asyncio.to_thread(svc.vector_store.delete_document, user["_id"], document_id)
    await col.delete_one({"_id": document_id, "user_id": user["_id"]})
    remove_file(doc.get("stored_path"))
