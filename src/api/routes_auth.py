from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request, status

from src.core.auth import create_jwt_token, decode_basic_credentials
from src.core.models import AuthRequest, AuthTokenResponse

router = APIRouter(prefix="/api", tags=["auth"])


@router.post("/auth/token", response_model=AuthTokenResponse)
async def issue_auth_token(
    request: Request,
    body: AuthRequest | None = None,
    authorization: str | None = Header(default=None),
) -> AuthTokenResponse:
    if authorization:
        try:
            user_id, password = decode_basic_credentials(authorization)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
            ) from exc
    elif body is not None:
        user_id = body.user_id
        password = body.password
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide Basic Authorization or JSON body with user_id/password",
        )

    config = request.app.state.config
    if user_id != config.auth_user_id or password != config.auth_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user_id or password",
        )

    token, exp_timestamp = create_jwt_token(
        user_id=user_id,
        secret=config.auth_jwt_secret,
        expires_minutes=config.auth_jwt_exp_minutes,
    )
    expires_at = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
    return AuthTokenResponse(access_token=token, expires_at=expires_at)
