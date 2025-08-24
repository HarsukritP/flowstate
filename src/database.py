"""
FlowState Database Configuration
SQLAlchemy setup for user data and song metadata
"""

import asyncio
from typing import AsyncGenerator
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import logging

from .config import settings

logger = logging.getLogger(__name__)

# Database models base
Base = declarative_base()

class SongMetadata(Base):
    """Store analyzed song metadata and features"""
    __tablename__ = "song_metadata"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    artist = Column(String, nullable=False)
    album = Column(String)
    duration_ms = Column(Integer)
    spotify_id = Column(String, unique=True)
    
    # Audio features
    tempo = Column(Float)
    key = Column(Integer)
    mode = Column(Integer)
    energy = Column(Float)
    valence = Column(Float)
    danceability = Column(Float)
    acousticness = Column(Float)
    instrumentalness = Column(Float)
    loudness = Column(Float)
    speechiness = Column(Float)
    liveness = Column(Float)
    
    # FlowState features
    emotional_arousal = Column(Float)
    emotional_dominance = Column(Float)
    flow_compatibility = Column(Float)
    primary_emotion = Column(String)
    emotional_tags = Column(Text)  # JSON string
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    analysis_version = Column(String, default="1.0")

class UserSession(Base):
    """Store user session data and preferences"""
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True)
    session_token = Column(String, unique=True)
    
    # Preferences
    preferred_journey = Column(String, default="gradual_flow")
    queue_length = Column(Integer, default=10)
    flow_sensitivity = Column(Float, default=0.6)
    
    # Usage stats
    queues_generated = Column(Integer, default=0)
    total_songs_played = Column(Integer, default=0)
    avg_flow_score = Column(Float, default=0.0)
    
    created_at = Column(DateTime, server_default=func.now())
    last_active = Column(DateTime, server_default=func.now())

class QueueHistory(Base):
    """Store generated queue history for analysis"""
    __tablename__ = "queue_history"
    
    id = Column(String, primary_key=True)
    session_id = Column(String)
    
    # Queue data
    seed_song_id = Column(String)
    emotional_journey = Column(String)
    queue_length = Column(Integer)
    songs_json = Column(Text)  # JSON string of song IDs
    
    # Performance metrics
    flow_score = Column(Float)
    generation_time_ms = Column(Float)
    completion_rate = Column(Float)  # How much of queue was played
    skip_rate = Column(Float)
    
    created_at = Column(DateTime, server_default=func.now())

# Database connection setup
def get_sync_engine():
    """Get synchronous database engine"""
    return create_engine(
        settings.get_database_url(),
        echo=settings.is_development,
        pool_pre_ping=True,
        pool_recycle=3600,
    )

def get_async_engine():
    """Get asynchronous database engine"""
    # Convert sync URL to async URL
    async_url = settings.get_database_url()
    if async_url.startswith("sqlite"):
        async_url = async_url.replace("sqlite:///", "sqlite+aiosqlite:///")
    elif async_url.startswith("postgresql"):
        async_url = async_url.replace("postgresql://", "postgresql+asyncpg://")
    
    return create_async_engine(
        async_url,
        echo=settings.is_development,
        pool_pre_ping=True,
        pool_recycle=3600,
    )

# Session makers
sync_engine = get_sync_engine()
async_engine = get_async_engine()

SyncSessionLocal = sessionmaker(bind=sync_engine)
AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

def create_tables():
    """Create all database tables"""
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=sync_engine)
    logger.info("✅ Database tables created")

async def create_tables_async():
    """Create all database tables asynchronously"""
    logger.info("Creating database tables asynchronously...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables created")

# Dependency for FastAPI
def get_sync_db() -> Session:
    """Get synchronous database session"""
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get asynchronous database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Database initialization
async def init_database():
    """Initialize database on startup"""
    try:
        if settings.database_url.startswith("sqlite"):
            # For SQLite, create tables synchronously
            create_tables()
        else:
            # For PostgreSQL, create tables asynchronously
            await create_tables_async()
        
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        raise

# Utility functions
async def save_song_metadata(song_data: dict, features: dict, emotional_profile: dict):
    """Save analyzed song metadata to database"""
    async with AsyncSessionLocal() as session:
        try:
            song_metadata = SongMetadata(
                id=song_data["id"],
                title=song_data["title"],
                artist=song_data["artist"],
                album=song_data.get("album"),
                duration_ms=song_data.get("duration_ms"),
                spotify_id=song_data.get("spotify_id"),
                
                # Audio features
                tempo=features["tempo"],
                key=features["key"],
                mode=features["mode"],
                energy=features["energy"],
                valence=features["valence"],
                danceability=features["danceability"],
                acousticness=features["acousticness"],
                instrumentalness=features["instrumentalness"],
                loudness=features["loudness"],
                speechiness=features["speechiness"],
                liveness=features["liveness"],
                
                # FlowState features
                emotional_arousal=features["emotional_arousal"],
                emotional_dominance=features["emotional_dominance"],
                flow_compatibility=features["flow_compatibility"],
                primary_emotion=emotional_profile["primary_emotion"],
                emotional_tags=",".join(emotional_profile["emotional_tags"]),
            )
            
            session.add(song_metadata)
            await session.commit()
            
            logger.debug(f"✅ Saved metadata for song: {song_data['title']}")
            
        except Exception as e:
            await session.rollback()
            logger.error(f"❌ Failed to save song metadata: {str(e)}")
            raise

async def get_song_metadata(song_id: str) -> dict:
    """Retrieve song metadata from database"""
    async with AsyncSessionLocal() as session:
        try:
            song = await session.get(SongMetadata, song_id)
            if song:
                return {
                    "id": song.id,
                    "title": song.title,
                    "artist": song.artist,
                    "features": {
                        "tempo": song.tempo,
                        "key": song.key,
                        "energy": song.energy,
                        "valence": song.valence,
                        # ... other features
                    },
                    "emotional_profile": {
                        "primary_emotion": song.primary_emotion,
                        "emotional_tags": song.emotional_tags.split(",") if song.emotional_tags else [],
                    }
                }
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve song metadata: {str(e)}")
            return None
