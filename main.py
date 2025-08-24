"""
FlowState - AI-Powered Emotional Music Queue Optimization
Main FastAPI application entry point
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

from src.config import settings, get_cors_origins
from src.database import init_database
from src.audio_analyzer import AudioAnalyzer
from src.queue_optimizer import QueueOptimizer
from src.models import Song, QueueRequest, QueueResponse, AnalysisResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
audio_analyzer = None
queue_optimizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources"""
    global audio_analyzer, queue_optimizer
    
    logger.info("🌊 Initializing FlowState backend...")
    
    # Initialize database
    await init_database()
    
    # Initialize components
    audio_analyzer = AudioAnalyzer()
    queue_optimizer = QueueOptimizer()
    
    logger.info("✅ FlowState backend ready!")
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down FlowState backend...")

# Create FastAPI app
app = FastAPI(
    title="FlowState API",
    description="AI-Powered Emotional Music Queue Optimization",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "🌊 FlowState API is running",
        "version": "0.1.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "audio_analyzer": audio_analyzer is not None,
        "queue_optimizer": queue_optimizer is not None,
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.post("/analyze-song", response_model=AnalysisResponse)
async def analyze_song(song: Song):
    """
    Analyze a song's emotional and musical characteristics
    """
    try:
        if not audio_analyzer:
            raise HTTPException(status_code=503, detail="Audio analyzer not initialized")
        
        logger.info(f"🎵 Analyzing song: {song.title} by {song.artist}")
        
        # Extract audio features
        features = await audio_analyzer.extract_features(song)
        
        # Calculate emotional profile
        emotional_profile = await audio_analyzer.analyze_emotion(features)
        
        return AnalysisResponse(
            song_id=song.id,
            features=features,
            emotional_profile=emotional_profile,
            processing_time_ms=features.get("processing_time_ms", 0)
        )
        
    except Exception as e:
        logger.error(f"❌ Error analyzing song {song.id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/generate-queue", response_model=QueueResponse)
async def generate_queue(request: QueueRequest):
    """
    Generate an emotionally optimized queue from a seed song
    """
    try:
        if not queue_optimizer:
            raise HTTPException(status_code=503, detail="Queue optimizer not initialized")
        
        logger.info(f"🎯 Generating queue from seed: {request.seed_song.title}")
        
        # Generate optimized queue
        queue = await queue_optimizer.generate_queue(
            seed_song=request.seed_song,
            queue_length=request.queue_length,
            emotional_journey=request.emotional_journey,
            available_songs=request.available_songs
        )
        
        return QueueResponse(
            queue=queue.songs,
            emotional_journey=queue.emotional_journey,
            flow_score=queue.flow_score,
            generation_time_ms=queue.generation_time_ms,
            metadata=queue.metadata
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating queue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Queue generation failed: {str(e)}")

@app.post("/reoptimize-queue", response_model=QueueResponse)
async def reoptimize_queue(request: QueueRequest):
    """
    Re-optimize an existing queue with new song insertion
    """
    try:
        if not queue_optimizer:
            raise HTTPException(status_code=503, detail="Queue optimizer not initialized")
        
        logger.info(f"🔄 Re-optimizing queue with {len(request.current_queue)} songs")
        
        # Efficiently re-optimize the queue
        optimized_queue = await queue_optimizer.reoptimize_queue(
            current_queue=request.current_queue,
            new_song=request.new_song,
            insertion_point=request.insertion_point
        )
        
        return QueueResponse(
            queue=optimized_queue.songs,
            emotional_journey=optimized_queue.emotional_journey,
            flow_score=optimized_queue.flow_score,
            generation_time_ms=optimized_queue.generation_time_ms,
            metadata=optimized_queue.metadata
        )
        
    except Exception as e:
        logger.error(f"❌ Error re-optimizing queue: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Queue re-optimization failed: {str(e)}")

@app.get("/song-compatibility/{song_id1}/{song_id2}")
async def calculate_compatibility(song_id1: str, song_id2: str):
    """
    Calculate compatibility score between two songs
    """
    try:
        if not queue_optimizer:
            raise HTTPException(status_code=503, detail="Queue optimizer not initialized")
        
        compatibility = await queue_optimizer.calculate_compatibility(song_id1, song_id2)
        
        return {
            "song_1": song_id1,
            "song_2": song_id2,
            "compatibility_score": compatibility.score,
            "factors": compatibility.factors,
            "transition_quality": compatibility.transition_quality
        }
        
    except Exception as e:
        logger.error(f"❌ Error calculating compatibility: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Compatibility calculation failed: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": "2024-01-01T00:00:00Z",
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENVIRONMENT", "development") == "development"
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info"
    )
