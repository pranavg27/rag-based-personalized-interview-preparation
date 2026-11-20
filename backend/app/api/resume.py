from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, Form, UploadFile

from app.dependencies import CurrentUser, Svc
from app.errors import ValidationFailed
from app.models.enums import JOB_ROLES
from app.schemas.content import ResumeOut
from app.services.resume_service import extract_text_from_pdf
from app.utils.files import remove_file, save_upload
from app.utils.ids import serialize

router = APIRouter(prefix="/resume", tags=["resume"])


def _out(r: dict) -> ResumeOut:
    return ResumeOut(**serialize({"id": r["_id"], "filename": r["filename"], "size_bytes": r["size_bytes"],
                                  "target_role": r.get("target_role"), "extraction": r["extraction"],
                                  "profile": r["profile"], "uploaded_at": r["uploaded_at"]}))


@router.post("/upload", response_model=ResumeOut, status_code=201)
async def upload_resume(user: CurrentUser, svc: Svc, file: UploadFile = File(...),
                        target_role: str | None = Form(default=None)):
    if target_role and target_role not in JOB_ROLES:
        raise ValidationFailed(f"target_role must be one of {JOB_ROLES}")
    s = svc.settings
    saved = await save_upload(file, base_dir=s.upload_path / "resumes", user_id=user["_id"],
                              allowed=s.resume_exts, max_bytes=s.max_upload_bytes)
    try:
        text = await asyncio.to_thread(extract_text_from_pdf, saved.path)
        profile, how = await svc.resumes.extract_profile(text)
        doc = await svc.resumes.create(user_id=user["_id"], filename=saved.original_name,
                                       stored_path=str(saved.path), size=saved.size, text=text,
                                       profile=profile, extraction=how, target_role=target_role)
    except Exception:
        remove_file(saved.path)
        raise
    await svc.memory.remember_skills(user["_id"], profile.skills)
    return _out(doc)


@router.get("", response_model=list[ResumeOut])
async def list_resumes(user: CurrentUser, svc: Svc):
    return [_out(r) for r in await svc.resumes.list(user["_id"])]


@router.delete("/{resume_id}", status_code=204)
async def delete_resume(resume_id: str, user: CurrentUser, svc: Svc):
    doc = await svc.resumes.delete(user["_id"], resume_id)
    remove_file(doc.get("stored_path"))
