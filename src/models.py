"""
FlowState Data Models
Pydantic models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum

class EmotionalJourney(str, Enum):
    """Types of emotional journeys for queue generation"""
    GRADUAL_FLOW = "gradual_flow"
    MAINTAIN_VIBE = "maintain_vibe"
    ADVENTURE_MODE = "adventure_mode"
    WIND_DOWN = "wind_down"
    PUMP_UP = "pump_up"
    MEDITATIVE = "meditative"

class Song(BaseModel):
    """Song model with metadata and Spotify integration"""
    id: str = Field(..., description="Unique song identifier")
    title: str = Field(..., description="Song title")
    artist: str = Field(..., description="Primary artist name")
    album: Optional[str] = Field(None, description="Album name")
    duration_ms: Optional[int] = Field(None, description="Song duration in milliseconds")
    spotify_id: Optional[str] = Field(None, description="Spotify track ID")
    preview_url: Optional[str] = Field(None, description="30-second preview URL")
    external_urls: Optional[Dict[str, str]] = Field(None, description="External URLs")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "song_001",
                "title": "Breathe Me",
                "artist": "Sia",
                "album": "Colour the Small One",
                "duration_ms": 257000,
                "spotify_id": "3pHfMKRUqi7YM7b2MuNB1e",
                "preview_url": "https://p.scdn.co/mp3-preview/..."
            }
        }

class AudioFeatures(BaseModel):
    """Audio analysis features extracted from songs"""
    tempo: float = Field(..., description="Beats per minute")
    key: int = Field(..., description="Musical key (0-11, C=0)")
    mode: int = Field(..., description="Major (1) or minor (0)")
    energy: float = Field(..., ge=0, le=1, description="Energy level 0-1")
    valence: float = Field(..., ge=0, le=1, description="Musical positivity 0-1")
    danceability: float = Field(..., ge=0, le=1, description="Danceability 0-1")
    acousticness: float = Field(..., ge=0, le=1, description="Acoustic content 0-1")
    instrumentalness: float = Field(..., ge=0, le=1, description="Instrumental content 0-1")
    loudness: float = Field(..., description="Loudness in dB")
    speechiness: float = Field(..., ge=0, le=1, description="Speech content 0-1")
    liveness: float = Field(..., ge=0, le=1, description="Live recording probability 0-1")
    
    # Custom FlowState features
    emotional_arousal: float = Field(..., ge=0, le=1, description="Emotional intensity 0-1")
    emotional_dominance: float = Field(..., ge=0, le=1, description="Emotional control 0-1")
    flow_compatibility: float = Field(..., ge=0, le=1, description="General flow compatibility 0-1")
    processing_time_ms: Optional[float] = Field(None, description="Analysis processing time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tempo": 120.0,
                "key": 4,
                "mode": 0,
                "energy": 0.7,
                "valence": 0.3,
                "danceability": 0.5,
                "acousticness": 0.8,
                "instrumentalness": 0.1,
                "loudness": -8.5,
                "speechiness": 0.04,
                "liveness": 0.2,
                "emotional_arousal": 0.6,
                "emotional_dominance": 0.4,
                "flow_compatibility": 0.8
            }
        }

class EmotionalProfile(BaseModel):
    """Emotional characteristics of a song"""
    primary_emotion: str = Field(..., description="Primary emotional category")
    secondary_emotion: Optional[str] = Field(None, description="Secondary emotional category")
    arousal: float = Field(..., ge=0, le=1, description="Emotional intensity")
    valence: float = Field(..., ge=0, le=1, description="Emotional positivity")
    dominance: float = Field(..., ge=0, le=1, description="Emotional control/power")
    emotional_tags: List[str] = Field(default=[], description="Emotional descriptors")
    mood_vector: List[float] = Field(..., description="Multi-dimensional mood representation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "primary_emotion": "melancholic",
                "secondary_emotion": "contemplative",
                "arousal": 0.4,
                "valence": 0.3,
                "dominance": 0.6,
                "emotional_tags": ["sad", "reflective", "intimate"],
                "mood_vector": [0.3, 0.4, 0.6, 0.2, 0.8]
            }
        }

class CompatibilityScore(BaseModel):
    """Compatibility analysis between two songs"""
    score: float = Field(..., ge=0, le=1, description="Overall compatibility 0-1")
    tempo_compatibility: float = Field(..., ge=0, le=1, description="Tempo transition quality")
    key_compatibility: float = Field(..., ge=0, le=1, description="Musical key compatibility")
    energy_compatibility: float = Field(..., ge=0, le=1, description="Energy level transition")
    emotional_compatibility: float = Field(..., ge=0, le=1, description="Emotional flow quality")
    transition_quality: str = Field(..., description="Qualitative transition assessment")
    factors: Dict[str, float] = Field(..., description="Detailed compatibility factors")

class QueueMetadata(BaseModel):
    """Metadata about generated queue"""
    total_duration_ms: int = Field(..., description="Total queue duration")
    average_energy: float = Field(..., description="Average energy level")
    emotional_arc: List[str] = Field(..., description="Emotional progression")
    genre_diversity: float = Field(..., description="Genre variety score")
    tempo_variance: float = Field(..., description="Tempo variation measure")
    optimization_algorithm: str = Field(..., description="Algorithm used")

class OptimizedQueue(BaseModel):
    """Generated queue with optimization data"""
    songs: List[Song] = Field(..., description="Ordered list of songs")
    emotional_journey: List[EmotionalProfile] = Field(..., description="Emotional progression")
    flow_score: float = Field(..., ge=0, le=1, description="Overall flow quality")
    generation_time_ms: float = Field(..., description="Generation processing time")
    metadata: QueueMetadata = Field(..., description="Queue metadata")

class QueueRequest(BaseModel):
    """Request for queue generation or optimization"""
    seed_song: Song = Field(..., description="Starting song for queue generation")
    queue_length: int = Field(default=10, ge=5, le=50, description="Desired queue length")
    emotional_journey: EmotionalJourney = Field(default=EmotionalJourney.GRADUAL_FLOW)
    available_songs: Optional[List[Song]] = Field(None, description="Available song pool")
    
    # For re-optimization
    current_queue: Optional[List[Song]] = Field(None, description="Current queue for re-optimization")
    new_song: Optional[Song] = Field(None, description="Song to insert")
    insertion_point: Optional[int] = Field(None, description="Where to insert new song")
    
    # User preferences
    avoid_explicit: bool = Field(default=False, description="Avoid explicit content")
    preferred_genres: Optional[List[str]] = Field(None, description="Preferred genres")
    energy_preference: Optional[str] = Field(None, description="Energy level preference")
    
    class Config:
        json_schema_extra = {
            "example": {
                "seed_song": {
                    "id": "song_001",
                    "title": "Breathe Me",
                    "artist": "Sia",
                    "spotify_id": "3pHfMKRUqi7YM7b2MuNB1e"
                },
                "queue_length": 10,
                "emotional_journey": "gradual_flow",
                "avoid_explicit": False
            }
        }

class QueueResponse(BaseModel):
    """Response containing optimized queue"""
    queue: List[Song] = Field(..., description="Optimized song queue")
    emotional_journey: List[EmotionalProfile] = Field(..., description="Emotional progression")
    flow_score: float = Field(..., ge=0, le=1, description="Overall flow quality score")
    generation_time_ms: float = Field(..., description="Processing time in milliseconds")
    metadata: QueueMetadata = Field(..., description="Queue analysis metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "queue": [{"id": "song_001", "title": "Breathe Me", "artist": "Sia"}],
                "flow_score": 0.89,
                "generation_time_ms": 75.5,
                "metadata": {
                    "total_duration_ms": 2400000,
                    "average_energy": 0.7,
                    "emotional_arc": ["melancholic", "contemplative", "uplifting"],
                    "optimization_algorithm": "greedy_flow_v1"
                }
            }
        }

class AnalysisResponse(BaseModel):
    """Response for song analysis"""
    song_id: str = Field(..., description="Analyzed song ID")
    features: AudioFeatures = Field(..., description="Extracted audio features")
    emotional_profile: EmotionalProfile = Field(..., description="Emotional analysis")
    processing_time_ms: float = Field(..., description="Analysis time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "song_id": "song_001",
                "processing_time_ms": 450.2,
                "features": {"tempo": 120.0, "energy": 0.7},
                "emotional_profile": {"primary_emotion": "melancholic", "arousal": 0.4}
            }
        }
