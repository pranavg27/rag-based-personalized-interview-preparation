from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token
from app.database import mongo as m
from app.errors import UnauthorizedError
from app.services.container import Services, get_services

bearer = HTTPBearer(auto_error=False)


async def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict[str, Any]:
    if creds is None:
        raise UnauthorizedError("Not authenticated.")
    user_id = decode_access_token(creds.credentials)
    user = await m.get_db()[m.USERS].find_one({"_id": user_id})
    if not user or not user.get("is_active", True):
        raise UnauthorizedError("User not found or inactive.")
    return user


CurrentUser = Annotated[dict[str, Any], Depends(get_current_user)]
Svc = Annotated[Services, Depends(get_services)]
