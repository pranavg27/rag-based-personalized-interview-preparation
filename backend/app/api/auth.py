from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import get_settings
from app.core.rate_limit import limiter
from app.core.security import create_access_token, hash_password, verify_password
from app.database import mongo as m
from app.dependencies import CurrentUser
from app.errors import ConflictError, UnauthorizedError
from app.models.documents import user_doc
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])
_AUTH_LIMIT = get_settings().rate_limit_auth


def _out(u: dict) -> UserOut:
    return UserOut(id=u["_id"], email=u["email"], full_name=u["full_name"])


@router.post("/register", response_model=UserOut, status_code=201)
@limiter.limit(_AUTH_LIMIT)
async def register(request: Request, body: RegisterRequest):
    col = m.get_db()[m.USERS]
    if await col.find_one({"email": body.email}):
        raise ConflictError("An account with this email already exists.")
    doc = user_doc(body.email, body.full_name, hash_password(body.password))
    await col.insert_one(doc)
    return _out(doc)


@router.post("/login", response_model=TokenResponse)
@limiter.limit(_AUTH_LIMIT)
async def login(request: Request, body: LoginRequest):
    user = await m.get_db()[m.USERS].find_one({"email": body.email})
    if not verify_password(body.password, user["password_hash"] if user else None) or not user:
        raise UnauthorizedError("Incorrect email or password.")
    token, ttl = create_access_token(user["_id"])
    return TokenResponse(access_token=token, expires_in=ttl, user=_out(user))


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser):
    return _out(user)
