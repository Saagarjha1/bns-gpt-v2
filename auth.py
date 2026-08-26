import os
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-in-production")

# Serializer uses the SECRET_KEY for HMAC signing
_serializer = URLSafeTimedSerializer(SECRET_KEY, salt="bns-gpt-v2-auth")


def create_session_token(user_id: str) -> str:
    """
    Creates a cryptographically signed, time-stamped session token
    containing the user's Supabase UUID.
    """
    return _serializer.dumps(user_id)


def decode_session_token(token: str, max_age_days: int = 30) -> str | None:
    """
    Validates and decodes a session token.
    Returns the user_id string, or None if the token is invalid or expired.
    """
    if not token:
        return None
    try:
        user_id = _serializer.loads(token, max_age=max_age_days * 86400)
        return user_id
    except (BadSignature, SignatureExpired):
        return None
