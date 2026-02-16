from __future__ import annotations

from src.core.auth import (
    create_jwt_token,
    decode_basic_credentials,
    get_jwt_subject_from_authorization,
    verify_jwt_token,
)


def test_jwt_create_and_verify_roundtrip() -> None:
    token, _ = create_jwt_token(
        user_id="tanzir",
        secret="dev-secret",
        expires_minutes=5,
    )
    payload = verify_jwt_token(token, "dev-secret")
    assert payload["sub"] == "tanzir"


def test_jwt_expired_token_rejected() -> None:
    token, _ = create_jwt_token(
        user_id="tanzir",
        secret="dev-secret",
        expires_minutes=-1,
    )
    try:
        verify_jwt_token(token, "dev-secret")
        assert False, "Expected expired token validation to fail"
    except ValueError as exc:
        assert "expired" in str(exc)


def test_decode_basic_credentials() -> None:
    user_id, password = decode_basic_credentials("Basic dGFuemlyOmtoYW4=")
    assert user_id == "tanzir"
    assert password == "khan"


def test_get_jwt_subject_from_authorization() -> None:
    token, _ = create_jwt_token(
        user_id="tanzir",
        secret="dev-secret",
        expires_minutes=5,
    )
    subject = get_jwt_subject_from_authorization(
        f"Bearer {token}",
        "dev-secret",
    )
    assert subject == "tanzir"
