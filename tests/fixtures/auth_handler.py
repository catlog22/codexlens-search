"""Authentication handler for user login and session management."""
import hashlib
import secrets
from datetime import datetime, timedelta


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class SessionExpiredError(AuthenticationError):
    """Raised when a session token has expired."""
    pass


def hash_password(password: str, salt: str = "") -> str:
    """Hash a password using SHA-256 with optional salt."""
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash."""
    salt, expected = stored_hash.split(":", 1)
    actual = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return actual == expected


class SessionManager:
    """Manages user sessions with expiration tracking."""

    def __init__(self, timeout_minutes: int = 30):
        self._sessions = {}
        self._timeout = timedelta(minutes=timeout_minutes)

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        self._sessions[token] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + self._timeout,
        }
        return token

    def validate_session(self, token: str) -> str:
        session = self._sessions.get(token)
        if not session:
            raise AuthenticationError("Invalid session token")
        if datetime.utcnow() > session["expires_at"]:
            del self._sessions[token]
            raise SessionExpiredError("Session has expired")
        return session["user_id"]

    def revoke_session(self, token: str) -> bool:
        return self._sessions.pop(token, None) is not None
