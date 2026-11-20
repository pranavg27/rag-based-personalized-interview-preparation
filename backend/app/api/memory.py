from __future__ import annotations

from fastapi import APIRouter

from app.dependencies import CurrentUser, Svc
from app.errors import NotFoundError
from app.schemas.memory import PreferencesIn
from app.utils.ids import serialize

router = APIRouter(tags=["memory"])


@router.get("/memory")
async def list_memory(user: CurrentUser, svc: Svc, kind: str | None = None):
    rows = await svc.memory.list(user["_id"], [kind] if kind else None)
    return serialize([{k: v for k, v in r.items() if k != "user_id"} for r in rows])


@router.get("/memory/summary")
async def memory_summary(user: CurrentUser, svc: Svc):
    return {"summary": await svc.memory.summarize(user["_id"]),
            "snapshot": serialize(await svc.memory.profile_snapshot(user["_id"]))}


@router.delete("/memory/{memory_id}", status_code=204)
async def delete_memory(memory_id: str, user: CurrentUser, svc: Svc):
    if not await svc.memory.delete(user["_id"], memory_id):
        raise NotFoundError("Memory not found.")


@router.get("/preferences")
async def get_preferences(user: CurrentUser, svc: Svc):
    return await svc.memory.get_preferences(user["_id"])


@router.put("/preferences")
async def put_preferences(body: PreferencesIn, user: CurrentUser, svc: Svc):
    return await svc.memory.set_preferences(user["_id"], body.model_dump(mode="json"))
