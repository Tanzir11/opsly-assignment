from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from fastapi import Header, HTTPException, Request, status


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(f"{data}{padding}")


def _sign(signing_input: str, secret: str) -> str:
    digest = hmac.new(
        secret.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(digest)


def create_jwt_token(
    user_id: str,
    secret: str,
    expires_minutes: int = 60,
    extra_claims: dict[str, Any] | None = None,
) -> tuple[str, int]:
    now = int(time.time())
    exp = now + (expires_minutes * 60)

    header = {"alg": "HS256", "typ": "JWT"}
    payload: dict[str, Any] = {
        "sub": user_id,
        "iat": now,
        "exp": exp,
    }
    if extra_claims:
        payload.update(extra_claims)

    header_segment = _b64url_encode(
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    payload_segment = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signing_input = f"{header_segment}.{payload_segment}"
    signature = _sign(signing_input, secret)
    return f"{signing_input}.{signature}", exp


def verify_jwt_token(token: str, secret: str) -> dict[str, Any]:
    try:
        header_segment, payload_segment, signature = token.split(".")
    except ValueError as exc:
        raise ValueError("Malformed JWT token") from exc

    signing_input = f"{header_segment}.{payload_segment}"
    expected_signature = _sign(signing_input, secret)
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("Invalid JWT signature")

    try:
        payload_bytes = _b64url_decode(payload_segment)
        payload = json.loads(payload_bytes)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid JWT payload") from exc

    now = int(time.time())
    exp = payload.get("exp")
    if not isinstance(exp, int) or exp < now:
        raise ValueError("JWT token has expired")

    return payload


def decode_basic_credentials(authorization: str) -> tuple[str, str]:
    scheme, _, encoded_credentials = authorization.partition(" ")
    if scheme.lower() != "basic" or not encoded_credentials:
        raise ValueError("Authorization header must use Basic scheme")

    try:
        decoded = base64.b64decode(encoded_credentials).decode("utf-8")
        user_id, password = decoded.split(":", 1)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid Basic authorization header") from exc

    if not user_id or not password:
        raise ValueError("Basic authorization must include user_id and password")

    return user_id, password


def get_jwt_subject_from_authorization(authorization: str | None, secret: str) -> str:
    if not authorization:
        raise ValueError("Missing Authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ValueError("Authorization header must use Bearer token")

    payload = verify_jwt_token(token, secret)
    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise ValueError("JWT token missing subject")

    return subject


async def require_jwt_user(
    request: Request,
    authorization: str | None = Header(default=None),
) -> str:
    secret = request.app.state.config.auth_jwt_secret
    try:
        return get_jwt_subject_from_authorization(authorization, secret)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
