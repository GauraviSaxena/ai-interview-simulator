"""SQLite + SQLAlchemy ORM database layer for user authentication and interview sessions."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    desc,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Database file location: data/interview_history.db relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "interview_history.db"


class Base(DeclarativeBase):
    pass


class User(Base):
    """User account model for authentication."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    password_salt = Column(String(64), nullable=False)
    full_name = Column(String(120), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }


class InterviewSession(Base):
    """Stores full details and metrics of an interview session."""

    __tablename__ = "interview_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_email = Column(String(120), nullable=True, index=True)
    candidate_name = Column(String(120), nullable=False, index=True)
    target_role = Column(String(160), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # High-level and section-wise scores (0 - 100)
    overall_score = Column(Float, nullable=False, default=0.0)
    self_intro_score = Column(Float, nullable=True, default=0.0)
    technical_score = Column(Float, nullable=True, default=0.0)
    dsa_score = Column(Float, nullable=True, default=0.0)
    project_score = Column(Float, nullable=True, default=0.0)
    aptitude_score = Column(Float, nullable=True, default=0.0)
    personality_score = Column(Float, nullable=True, default=0.0)
    audio_recording_score = Column(Float, nullable=True, default=0.0)

    # Legacy score aliases for backward compatibility
    resume_score = Column(Float, nullable=True, default=0.0)
    voice_drill_score = Column(Float, nullable=True, default=0.0)

    # Vision & Body Language metrics
    eye_contact_pct = Column(Float, nullable=True, default=0.0)
    posture_score_pct = Column(Float, nullable=True, default=0.0)
    total_posture_errors = Column(Integer, nullable=False, default=0)

    # Speech metrics
    wpm = Column(Float, nullable=True, default=0.0)
    filler_words_total = Column(Integer, nullable=False, default=0)
    voice_tone = Column(String(80), nullable=True, default="Confident & Expressive")

    # Serialized JSON containing detailed question answers, test case results, section states
    details_json = Column(Text, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert session model to dictionary."""
        details = {}
        if self.details_json:
            try:
                details = json.loads(self.details_json)
            except Exception:
                details = {}
        return {
            "id": self.id,
            "user_email": self.user_email or "",
            "candidate_name": self.candidate_name,
            "target_role": self.target_role,
            "timestamp": self.timestamp.isoformat() if self.timestamp else "",
            "overall_score": float(self.overall_score or 0.0),
            "self_intro_score": float(self.self_intro_score or 0.0),
            "technical_score": float(self.technical_score or 0.0),
            "dsa_score": float(self.dsa_score or 0.0) if self.dsa_score is not None else None,
            "project_score": float(self.project_score or 0.0),
            "aptitude_score": float(self.aptitude_score or 0.0),
            "personality_score": float(self.personality_score or 0.0),
            "audio_recording_score": float(self.audio_recording_score or 0.0),
            "resume_score": float(self.resume_score or 0.0),
            "voice_drill_score": float(self.voice_drill_score or 0.0),
            "eye_contact_pct": float(self.eye_contact_pct or 0.0),
            "posture_score_pct": float(self.posture_score_pct or 0.0),
            "total_posture_errors": int(self.total_posture_errors or 0),
            "wpm": float(self.wpm or 0.0),
            "filler_words_total": int(self.filler_words_total or 0),
            "voice_tone": self.voice_tone or "Confident & Expressive",
            "details": details,
        }


_engine = None
_SessionFactory = None


def get_engine(db_path: Optional[Path | str] = None):
    """Obtain or initialize the SQLite SQLAlchemy engine."""
    global _engine, _SessionFactory
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    db_uri = f"sqlite:///{path.as_posix()}"
    if _engine is None or str(_engine.url) != db_uri:
        _engine = create_engine(db_uri, echo=False, connect_args={"check_same_thread": False})
        Base.metadata.create_all(_engine)
        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session(db_path: Optional[Path | str] = None) -> Session:
    """Provide a thread-safe Session object."""
    get_engine(db_path)
    return _SessionFactory()


def init_db(db_path: Optional[Path | str] = None) -> None:
    """Ensure database directories and tables are provisioned."""
    get_engine(db_path)


import re

# ==============================================================================
# AUTHENTICATION HELPERS
# ==============================================================================

def validate_strong_password(password: str) -> tuple[bool, str]:
    """Verify that password meets strict enterprise security complexity requirements."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least 1 uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least 1 lowercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least 1 number."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>\-_=+[\]\\/`~;']", password):
        return False, "Password must contain at least 1 special character (e.g. !@#$%^&*)."
    return True, ""


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Generate SHA256 PBKDF2 hash with unique salt."""
    pwd_clean = str(password or "").strip()
    if not salt:
        salt = os.urandom(16).hex()
    else:
        salt = str(salt).strip()
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        pwd_clean.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100000,
    ).hex()
    return hashed, salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    """Verify password against stored hash with consistent utf-8 encoding and whitespace trimming."""
    import hmac
    pwd_clean = str(password or "").strip()
    salt_clean = str(salt or "").strip()
    hash_clean = str(hashed or "").strip()
    calc_hash, _ = hash_password(pwd_clean, salt_clean)
    return hmac.compare_digest(calc_hash, hash_clean)


GMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._+-]+@gmail\.com$")


def is_valid_gmail(email_candidate: str) -> bool:
    """Validate that email matches standard Gmail address specification (letters, numbers, dots, underscores, plus, hyphen)."""
    if not email_candidate:
        return False
    return bool(GMAIL_REGEX.match(str(email_candidate).strip().lower()))


def create_user(
    *args: Any,
    full_name: Optional[str] = None,
    email: Optional[str] = None,
    password: Optional[str] = None,
    db_path: Optional[Path | str] = None,
    **kwargs: Any,
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """Register a new user after verifying @gmail.com domain and strong password.

    Defensively handles any argument ordering:
    - create_user(full_name, email, password)
    - create_user(email, password, full_name)
    - create_user(full_name=..., email=..., password=...)
    """
    # Parse positional arguments defensively
    str_args = [str(a).strip() for a in args if isinstance(a, str) and str(a).strip()]

    # 1. Detect email among positional arguments if not supplied as kwarg
    if not email:
        for a in str_args:
            if "@" in a:
                email = a
                break

    # 2. Extract remaining positional arguments for full_name and password
    rem = [a for a in str_args if a != email]
    if len(rem) == 2:
        # Determine based on position of email in str_args
        try:
            email_idx = str_args.index(email)
        except ValueError:
            email_idx = -1

        if email_idx == 0:
            # create_user(email, password, full_name)
            p_cand, n_cand = rem[0], rem[1]
        elif email_idx == 1:
            # create_user(full_name, email, password)
            n_cand, p_cand = rem[0], rem[1]
        else:
            # Heuristic check with validate_strong_password
            if validate_strong_password(rem[0])[0] and not validate_strong_password(rem[1])[0]:
                p_cand, n_cand = rem[0], rem[1]
            else:
                n_cand, p_cand = rem[0], rem[1]

        if not full_name:
            full_name = n_cand
        if not password:
            password = p_cand
    elif len(rem) == 1:
        if not password:
            password = rem[0]

    for a in args:
        if isinstance(a, (Path, str)) and str(a).endswith((".db", ".sqlite")):
            db_path = a

    clean_email = str(email or kwargs.get("email", "")).strip().lower()
    clean_pwd = str(password or kwargs.get("password", "")).strip()
    clean_name = str(full_name or kwargs.get("full_name", "")).strip()

    if not clean_email or not is_valid_gmail(clean_email):
        return False, "Error Detected: Please enter a valid @gmail.com address", None

    is_strong, pwd_err = validate_strong_password(clean_pwd)
    if not is_strong:
        return False, pwd_err, None

    init_db(db_path)
    with get_session(db_path) as db:
        existing = db.scalar(select(User).where(User.email == clean_email))
        if existing:
            return False, "Account already exists with this email address. Please log in.", None

        pwd_hash, pwd_salt = hash_password(clean_pwd)
        resolved_name = clean_name or clean_email.split("@")[0].replace(".", " ").title()
        new_user = User(
            email=clean_email,
            password_hash=pwd_hash,
            password_salt=pwd_salt,
            full_name=resolved_name,
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return True, "Account created successfully", new_user.to_dict()


def authenticate_user(
    *args: Any,
    email: Optional[str] = None,
    password: Optional[str] = None,
    db_path: Optional[Path | str] = None,
    **kwargs: Any,
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """Authenticate user with email and password."""
    # Defensively extract email and password from args
    str_args = [str(a).strip() for a in args if isinstance(a, str) and str(a).strip()]
    if not email:
        for a in str_args:
            if "@" in a:
                email = a
                break
    rem = [a for a in str_args if a != email]
    if not password and rem:
        password = rem[0]

    for a in args:
        if isinstance(a, (Path, str)) and str(a).endswith((".db", ".sqlite")):
            db_path = a

    clean_email = str(email or kwargs.get("email", "")).strip().lower()
    clean_pwd = str(password or kwargs.get("password", "")).strip()

    if not is_valid_gmail(clean_email):
        return False, "Error Detected: Please enter a valid @gmail.com address", None

    if not clean_pwd:
        return False, "Please enter your password.", None

    init_db(db_path)
    with get_session(db_path) as db:
        user = db.scalar(select(User).where(User.email == clean_email))
        if not user:
            return False, "Account not found. Please Sign Up first.", None

        if not verify_password(clean_pwd, user.password_hash, user.password_salt):
            return False, "Incorrect password. Please try again.", None

        return True, "Login successful", user.to_dict()


register_user = create_user


def get_user_by_email(email: str, db_path: Optional[Path | str] = None) -> Optional[dict[str, Any]]:
    """Retrieve user dictionary by email."""
    init_db(db_path)
    clean_email = email.strip().lower()
    with get_session(db_path) as db:
        user = db.scalar(select(User).where(User.email == clean_email))
        return user.to_dict() if user else None


# ==============================================================================
# SESSION HELPERS
# ==============================================================================

def save_session(session_data: dict[str, Any], db_path: Optional[Path | str] = None) -> int:
    """Persist an interview session and return its database primary key ID."""
    init_db(db_path)
    with get_session(db_path) as db:
        ts = session_data.get("timestamp")
        if isinstance(ts, str):
            try:
                parsed_ts = datetime.fromisoformat(ts)
            except Exception:
                parsed_ts = datetime.utcnow()
        elif isinstance(ts, datetime):
            parsed_ts = ts
        else:
            parsed_ts = datetime.utcnow()

        details = session_data.get("details", {})
        if not isinstance(details, str):
            try:
                details_json = json.dumps(details)
            except Exception:
                details_json = "{}"
        else:
            details_json = details

        def _flt(k, default=0.0):
            val = session_data.get(k)
            if val is None:
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        def _int(k, default=0):
            val = session_data.get(k)
            if val is None:
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        record = InterviewSession(
            user_email=str(session_data.get("user_email") or "").strip().lower(),
            candidate_name=str(session_data.get("candidate_name") or "Anonymous").strip(),
            target_role=str(session_data.get("target_role") or "Software Engineer").strip(),
            timestamp=parsed_ts,
            overall_score=_flt("overall_score", 0.0),
            self_intro_score=_flt("self_intro_score", 0.0),
            technical_score=_flt("technical_score", 0.0),
            dsa_score=float(session_data["dsa_score"]) if session_data.get("dsa_score") is not None else None,
            project_score=_flt("project_score", 0.0),
            aptitude_score=_flt("aptitude_score", 0.0),
            personality_score=_flt("personality_score", 0.0),
            audio_recording_score=_flt("audio_recording_score", 0.0),
            resume_score=_flt("resume_score", 0.0),
            voice_drill_score=_flt("voice_drill_score", 0.0),
            eye_contact_pct=_flt("eye_contact_pct", 0.0),
            posture_score_pct=_flt("posture_score_pct", 0.0),
            total_posture_errors=_int("total_posture_errors", 0),
            wpm=_flt("wpm", 0.0),
            filler_words_total=_int("filler_words_total", 0),
            voice_tone=str(session_data.get("voice_tone") or "Confident & Expressive"),
            details_json=details_json,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id


def get_candidate_history(
    candidate_identifier: str,
    db_path: Optional[Path | str] = None,
) -> list[dict[str, Any]]:
    """Retrieve all historical sessions for a given email or candidate name, ordered chronologically."""
    init_db(db_path)
    clean_id = candidate_identifier.strip()
    with get_session(db_path) as db:
        stmt = (
            select(InterviewSession)
            .where(
                (InterviewSession.user_email == clean_id.lower()) |
                (InterviewSession.candidate_name.ilike(clean_id))
            )
            .order_by(InterviewSession.timestamp.asc())
        )
        results = db.scalars(stmt).all()
        return [r.to_dict() for r in results]


def get_all_candidates(db_path: Optional[Path | str] = None) -> list[str]:
    """Retrieve distinct candidate names from the database."""
    init_db(db_path)
    with get_session(db_path) as db:
        stmt = select(InterviewSession.candidate_name).distinct().order_by(InterviewSession.candidate_name)
        names = db.scalars(stmt).all()
        return [str(name) for name in names if name]


def get_session_by_id(session_id: int, db_path: Optional[Path | str] = None) -> Optional[dict[str, Any]]:
    """Fetch a single session record by primary key ID."""
    init_db(db_path)
    with get_session(db_path) as db:
        record = db.get(InterviewSession, session_id)
        return record.to_dict() if record else None


def delete_session(session_id: int, db_path: Optional[Path | str] = None) -> bool:
    """Delete a session by ID."""
    init_db(db_path)
    with get_session(db_path) as db:
        record = db.get(InterviewSession, session_id)
        if record:
            db.delete(record)
            db.commit()
            return True
        return False
