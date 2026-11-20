from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegisterRequest(BaseModel):
    email: str = Field(max_length=254)
    full_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL.match(v):
            raise ValueError("invalid email address")
        return v

    @field_validator("password")
    @classmethod
    def _pw(cls, v: str) -> str:
        if not (re.search(r"[A-Za-z]", v) and re.search(r"\d", v)):
            raise ValueError("password must contain letters and digits")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return v.strip().lower()


class UserOut(BaseModel):
    id: str
    email: str
    full_name: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut
