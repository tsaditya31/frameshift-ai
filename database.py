import logging
import os
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey,
    Float, create_engine
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_db_url() -> str:
    # Check Railway's variable names: DATABASE_PUBLIC_URL, DATABASE_URL, or PGHOST-style
    url = (
        os.environ.get("DATABASE_PUBLIC_URL")
        or os.environ.get("DATABASE_URL")
        or settings.database_url
    )
    # Railway provides postgres:// but SQLAlchemy needs postgresql+asyncpg://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    logger.info(f"Database URL scheme: {url.split('@')[0].split('://')[0] if '://' in url else 'unknown'}")
    return url


engine = create_async_engine(_get_db_url(), echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(256), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=True)
    google_id = Column(String(256), unique=True, nullable=True)
    name = Column(String(256), default="")
    reset_token = Column(String(128), nullable=True)
    reset_token_expires = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    storyboards = relationship("Storyboard", back_populates="user")


class Storyboard(Base):
    __tablename__ = "storyboards"

    id = Column(Integer, primary_key=True)
    title = Column(String(256), nullable=False)
    description = Column(Text, default="")
    total_episodes = Column(Integer, default=0)
    # draft | generating | ready | in_production | complete
    status = Column(String(32), default="draft")
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    user = relationship("User", back_populates="storyboards")
    episodes = relationship("Episode", back_populates="storyboard", cascade="all, delete-orphan")
    characters = relationship("Character", back_populates="storyboard", cascade="all, delete-orphan")
    knowledge_chunks = relationship("KnowledgeChunk", back_populates="storyboard", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="storyboard", cascade="all, delete-orphan")
    upload_schedule = relationship("UploadSchedule", back_populates="storyboard", uselist=False, cascade="all, delete-orphan")


class Episode(Base):
    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True)
    storyboard_id = Column(Integer, ForeignKey("storyboards.id"), nullable=False)
    episode_num = Column(Integer, nullable=False)
    title = Column(String(512), default="")
    storyline = Column(Text, default="")
    script_json = Column(Text, default="")           # JSON: list of scenes
    narration_translations = Column(Text, default="")  # JSON: {lang: translated_text}
    # draft | approved | queued | producing | done | published
    production_status = Column(String(32), default="draft")
    youtube_urls_json = Column(Text, default="")     # JSON: {lang: url}
    scheduled_upload_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    storyboard = relationship("Storyboard", back_populates="episodes")
    jobs = relationship("Job", back_populates="episode", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="episode", cascade="all, delete-orphan")


class Character(Base):
    __tablename__ = "characters"

    id = Column(Integer, primary_key=True)
    storyboard_id = Column(Integer, ForeignKey("storyboards.id"), nullable=False)
    name = Column(String(256), nullable=False)
    description = Column(Text, default="")
    ref_images_dir = Column(String(512), default="")
    higgsfield_soul_id = Column(String(256), default="")
    # needs_images | ready_to_train | training | trained | failed
    training_status = Column(String(32), default="needs_images")
    hero_image_path = Column(String(512), default="")
    profile_prompt = Column(Text, default="")
    locked = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    storyboard = relationship("Storyboard", back_populates="characters")


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True)
    storyboard_id = Column(Integer, ForeignKey("storyboards.id"), nullable=False)
    source_file = Column(String(512), default="")
    content_text = Column(Text, default="")
    # text | image | url
    chunk_type = Column(String(32), default="text")
    created_at = Column(DateTime, default=datetime.utcnow)

    storyboard = relationship("Storyboard", back_populates="knowledge_chunks")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    storyboard_id = Column(Integer, ForeignKey("storyboards.id"), nullable=False)
    episode_id = Column(Integer, ForeignKey("episodes.id"), nullable=True)
    role = Column(String(16), nullable=False)   # user | assistant
    content = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    storyboard = relationship("Storyboard", back_populates="chat_messages")
    episode = relationship("Episode", back_populates="chat_messages")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer, ForeignKey("episodes.id"), nullable=False)
    # scene_image | ken_burns | tts | stitch | upload
    job_type = Column(String(32), nullable=False)
    language = Column(String(8), default="en")
    higgsfield_job_id = Column(String(256), default="")
    # pending | running | done | failed
    status = Column(String(16), default="pending")
    output_path = Column(String(512), default="")
    error_msg = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    episode = relationship("Episode", back_populates="jobs")


class UploadSchedule(Base):
    __tablename__ = "upload_schedules"

    id = Column(Integer, primary_key=True)
    storyboard_id = Column(Integer, ForeignKey("storyboards.id"), nullable=False)
    first_upload_at = Column(DateTime, nullable=True)
    interval_days = Column(Integer, default=7)
    timezone = Column(String(64), default="UTC")
    active = Column(Integer, default=1)  # 0=paused, 1=active

    storyboard = relationship("Storyboard", back_populates="upload_schedule")


async def init_db():
    logger.info(f"init_db: creating tables with engine {engine.url.get_backend_name()}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("init_db: tables created successfully")


async def get_db():
    async with async_session() as session:
        yield session
