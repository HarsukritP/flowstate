"""
FlowState Audio Analyzer
Extracts musical and emotional features from audio using librosa and ML models
"""

import numpy as np
import asyncio
import time
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

# Optional librosa import for production deployment
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("Librosa not available - using mock audio analysis only")

from .models import Song, AudioFeatures, EmotionalProfile

logger = logging.getLogger(__name__)

@dataclass
class CircleOfFifths:
    """Musical key relationships for compatibility analysis"""
    # Key mappings: 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
    MAJOR_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    MINOR_KEYS = ["Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm"]
    
    # Circle of fifths compatibility matrix (0=incompatible, 1=perfect)
    COMPATIBILITY_MATRIX = np.array([
        [1.0, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 1.0, 0.2, 0.8, 0.2, 0.8],  # C
        [0.2, 1.0, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 1.0, 0.2, 0.8, 0.2],  # C#
        [0.8, 0.2, 1.0, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 1.0, 0.2, 0.8],  # D
        [0.2, 0.8, 0.2, 1.0, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 1.0, 0.2],  # D#
        [0.8, 0.2, 0.8, 0.2, 1.0, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 1.0],  # E
        [0.8, 0.8, 0.2, 0.8, 0.2, 1.0, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2],  # F
        [0.2, 0.8, 0.8, 0.2, 0.8, 0.2, 1.0, 0.2, 0.8, 0.2, 0.8, 0.8],  # F#
        [1.0, 0.2, 0.8, 0.8, 0.2, 0.8, 0.2, 1.0, 0.2, 0.8, 0.2, 0.8],  # G
        [0.2, 1.0, 0.2, 0.8, 0.8, 0.2, 0.8, 0.2, 1.0, 0.2, 0.8, 0.2],  # G#
        [0.8, 0.2, 1.0, 0.2, 0.8, 0.8, 0.2, 0.8, 0.2, 1.0, 0.2, 0.8],  # A
        [0.2, 0.8, 0.2, 1.0, 0.2, 0.8, 0.8, 0.2, 0.8, 0.2, 1.0, 0.2],  # A#
        [0.8, 0.2, 0.8, 0.2, 1.0, 0.2, 0.8, 0.8, 0.2, 0.8, 0.2, 1.0],  # B
    ])

class AudioAnalyzer:
    """
    Advanced audio analysis system for musical and emotional feature extraction
    """
    
    def __init__(self):
        """Initialize the audio analyzer with ML models and feature extractors"""
        self.sample_rate = 22050
        self.hop_length = 512
        self.n_fft = 2048
        
        # Emotional mapping models (simplified for MVP)
        self.emotion_keywords = {
            'happy': ['upbeat', 'joyful', 'celebration', 'party', 'dance'],
            'sad': ['melancholic', 'lonely', 'heartbreak', 'cry', 'loss'],
            'energetic': ['pump', 'workout', 'rock', 'metal', 'electronic'],
            'calm': ['peaceful', 'ambient', 'meditation', 'chill', 'relaxing'],
            'angry': ['aggressive', 'rage', 'metal', 'punk', 'intense'],
            'romantic': ['love', 'romantic', 'valentine', 'kiss', 'heart'],
            'nostalgic': ['memory', 'old', 'vintage', 'classic', 'reminisce']
        }
        
        if LIBROSA_AVAILABLE:
            logger.info("🎵 AudioAnalyzer initialized with librosa")
        else:
            logger.info("🎵 AudioAnalyzer initialized in mock mode (librosa not available)")

    async def extract_features(self, song: Song) -> AudioFeatures:
        """
        Extract comprehensive audio features from a song
        For MVP: Using mock data since we need actual audio files
        In production: Would analyze audio from Spotify preview URLs
        """
        start_time = time.time()
        
        try:
            # In production, this would download and analyze the preview_url
            # For MVP, we'll generate realistic mock features based on song metadata
            features = await self._extract_mock_features(song)
            
            processing_time = (time.time() - start_time) * 1000
            features.processing_time_ms = processing_time
            
            logger.info(f"✅ Extracted features for '{song.title}' in {processing_time:.1f}ms")
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed for '{song.title}': {str(e)}")
            raise

    async def _extract_mock_features(self, song: Song) -> AudioFeatures:
        """
        Generate realistic mock features for MVP development
        In production, replace with actual librosa analysis
        """
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Generate features based on song metadata and artist style
        artist_lower = song.artist.lower()
        title_lower = song.title.lower()
        
        # Base features influenced by artist/genre patterns
        if any(genre in artist_lower for genre in ['electronic', 'daft', 'deadmau5', 'skrillex']):
            # Electronic music characteristics
            tempo = np.random.uniform(120, 140)
            energy = np.random.uniform(0.7, 0.95)
            danceability = np.random.uniform(0.8, 0.95)
            valence = np.random.uniform(0.6, 0.9)
            acousticness = np.random.uniform(0.0, 0.1)
            instrumentalness = np.random.uniform(0.3, 0.8)
            
        elif any(genre in artist_lower for genre in ['classical', 'mozart', 'bach', 'beethoven']):
            # Classical music characteristics
            tempo = np.random.uniform(60, 120)
            energy = np.random.uniform(0.3, 0.7)
            danceability = np.random.uniform(0.1, 0.4)
            valence = np.random.uniform(0.4, 0.8)
            acousticness = np.random.uniform(0.8, 0.99)
            instrumentalness = np.random.uniform(0.8, 0.99)
            
        elif any(genre in artist_lower for genre in ['rock', 'metal', 'punk']):
            # Rock/Metal characteristics
            tempo = np.random.uniform(100, 180)
            energy = np.random.uniform(0.8, 0.99)
            danceability = np.random.uniform(0.4, 0.7)
            valence = np.random.uniform(0.3, 0.8)
            acousticness = np.random.uniform(0.0, 0.2)
            instrumentalness = np.random.uniform(0.0, 0.3)
            
        elif any(genre in artist_lower for genre in ['jazz', 'blues']):
            # Jazz/Blues characteristics
            tempo = np.random.uniform(80, 140)
            energy = np.random.uniform(0.4, 0.8)
            danceability = np.random.uniform(0.5, 0.8)
            valence = np.random.uniform(0.3, 0.7)
            acousticness = np.random.uniform(0.3, 0.8)
            instrumentalness = np.random.uniform(0.2, 0.7)
            
        else:
            # Pop/General characteristics
            tempo = np.random.uniform(90, 130)
            energy = np.random.uniform(0.5, 0.8)
            danceability = np.random.uniform(0.6, 0.9)
            valence = np.random.uniform(0.4, 0.8)
            acousticness = np.random.uniform(0.1, 0.5)
            instrumentalness = np.random.uniform(0.0, 0.2)
        
        # Emotional adjustments based on song title
        valence_modifier = 0
        energy_modifier = 0
        
        for emotion, keywords in self.emotion_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                if emotion == 'sad':
                    valence_modifier -= 0.3
                    energy_modifier -= 0.2
                elif emotion == 'happy':
                    valence_modifier += 0.3
                    energy_modifier += 0.1
                elif emotion == 'energetic':
                    energy_modifier += 0.3
                elif emotion == 'calm':
                    energy_modifier -= 0.3
                    valence_modifier += 0.1
        
        valence = np.clip(valence + valence_modifier, 0, 1)
        energy = np.clip(energy + energy_modifier, 0, 1)
        
        # Calculate derived emotional features
        emotional_arousal = (energy * 0.7 + (1 - acousticness) * 0.3)
        emotional_dominance = (energy * 0.5 + valence * 0.3 + (1 - acousticness) * 0.2)
        flow_compatibility = self._calculate_flow_compatibility(tempo, energy, valence)
        
        return AudioFeatures(
            tempo=round(tempo, 1),
            key=np.random.randint(0, 12),
            mode=np.random.choice([0, 1]),  # 0=minor, 1=major
            energy=round(energy, 3),
            valence=round(valence, 3),
            danceability=round(danceability, 3),
            acousticness=round(acousticness, 3),
            instrumentalness=round(instrumentalness, 3),
            loudness=round(np.random.uniform(-20, -5), 1),
            speechiness=round(np.random.uniform(0.02, 0.2), 3),
            liveness=round(np.random.uniform(0.05, 0.4), 3),
            emotional_arousal=round(emotional_arousal, 3),
            emotional_dominance=round(emotional_dominance, 3),
            flow_compatibility=round(flow_compatibility, 3)
        )

    def _calculate_flow_compatibility(self, tempo: float, energy: float, valence: float) -> float:
        """Calculate general flow compatibility based on musical characteristics"""
        # Songs with moderate tempo and energy tend to flow better
        tempo_score = 1.0 - abs(tempo - 120) / 120  # Optimal around 120 BPM
        energy_score = 1.0 - abs(energy - 0.6) / 0.6  # Optimal around moderate energy
        valence_score = 1.0 - abs(valence - 0.5) / 0.5  # Moderate valence flows well
        
        return np.clip((tempo_score * 0.4 + energy_score * 0.4 + valence_score * 0.2), 0, 1)

    async def analyze_emotion(self, features: AudioFeatures) -> EmotionalProfile:
        """
        Analyze emotional content from audio features
        """
        try:
            # Map audio features to emotional categories
            primary_emotion = self._classify_primary_emotion(features)
            secondary_emotion = self._classify_secondary_emotion(features, primary_emotion)
            
            # Emotional tags based on features
            emotional_tags = self._generate_emotional_tags(features)
            
            # Multi-dimensional mood vector (5D: Joy, Sadness, Energy, Calm, Tension)
            mood_vector = self._calculate_mood_vector(features)
            
            return EmotionalProfile(
                primary_emotion=primary_emotion,
                secondary_emotion=secondary_emotion,
                arousal=features.emotional_arousal,
                valence=features.valence,
                dominance=features.emotional_dominance,
                emotional_tags=emotional_tags,
                mood_vector=mood_vector
            )
            
        except Exception as e:
            logger.error(f"❌ Emotional analysis failed: {str(e)}")
            raise

    def _classify_primary_emotion(self, features: AudioFeatures) -> str:
        """Classify primary emotion based on valence and arousal"""
        if features.valence > 0.6 and features.emotional_arousal > 0.6:
            return "joyful"
        elif features.valence > 0.6 and features.emotional_arousal < 0.4:
            return "peaceful"
        elif features.valence < 0.4 and features.emotional_arousal > 0.6:
            return "angry"
        elif features.valence < 0.4 and features.emotional_arousal < 0.4:
            return "melancholic"
        elif features.energy > 0.8:
            return "energetic"
        elif features.acousticness > 0.7:
            return "contemplative"
        else:
            return "neutral"

    def _classify_secondary_emotion(self, features: AudioFeatures, primary: str) -> Optional[str]:
        """Classify secondary emotion nuance"""
        if primary == "joyful" and features.danceability > 0.8:
            return "exuberant"
        elif primary == "melancholic" and features.acousticness > 0.6:
            return "introspective"
        elif primary == "energetic" and features.tempo > 140:
            return "intense"
        elif primary == "peaceful" and features.instrumentalness > 0.5:
            return "meditative"
        return None

    def _generate_emotional_tags(self, features: AudioFeatures) -> List[str]:
        """Generate descriptive emotional tags"""
        tags = []
        
        if features.valence > 0.7:
            tags.append("uplifting")
        elif features.valence < 0.3:
            tags.append("somber")
        
        if features.energy > 0.8:
            tags.append("high-energy")
        elif features.energy < 0.3:
            tags.append("low-energy")
        
        if features.danceability > 0.8:
            tags.append("danceable")
        
        if features.acousticness > 0.7:
            tags.append("acoustic")
        
        if features.instrumentalness > 0.7:
            tags.append("instrumental")
        
        if features.tempo > 140:
            tags.append("fast")
        elif features.tempo < 80:
            tags.append("slow")
        
        return tags

    def _calculate_mood_vector(self, features: AudioFeatures) -> List[float]:
        """Calculate 5-dimensional mood vector"""
        joy = features.valence * features.danceability
        sadness = (1 - features.valence) * features.acousticness
        energy = features.energy
        calm = features.acousticness * (1 - features.energy)
        tension = features.energy * (1 - features.valence)
        
        return [
            round(joy, 3),
            round(sadness, 3),
            round(energy, 3),
            round(calm, 3),
            round(tension, 3)
        ]

    def calculate_key_compatibility(self, key1: int, key2: int) -> float:
        """Calculate musical key compatibility using circle of fifths"""
        return CircleOfFifths.COMPATIBILITY_MATRIX[key1][key2]

    def calculate_tempo_compatibility(self, tempo1: float, tempo2: float) -> float:
        """Calculate tempo transition compatibility"""
        tempo_diff = abs(tempo1 - tempo2)
        
        # Perfect match
        if tempo_diff == 0:
            return 1.0
        
        # Good transitions (within 10 BPM)
        elif tempo_diff <= 10:
            return 1.0 - (tempo_diff / 20)
        
        # Acceptable transitions (within 20 BPM)
        elif tempo_diff <= 20:
            return 0.7 - (tempo_diff - 10) / 40
        
        # Poor transitions
        else:
            return max(0.1, 0.5 - (tempo_diff - 20) / 100)

    async def calculate_transition_quality(self, song1: Song, song2: Song) -> Dict[str, float]:
        """
        Calculate comprehensive transition quality between two songs
        """
        try:
            features1 = await self.extract_features(song1)
            features2 = await self.extract_features(song2)
            
            # Musical compatibility
            key_compat = self.calculate_key_compatibility(features1.key, features2.key)
            tempo_compat = self.calculate_tempo_compatibility(features1.tempo, features2.tempo)
            
            # Energy transition smoothness
            energy_diff = abs(features1.energy - features2.energy)
            energy_compat = max(0, 1.0 - energy_diff)
            
            # Emotional flow compatibility
            valence_diff = abs(features1.valence - features2.valence)
            emotional_compat = max(0, 1.0 - valence_diff)
            
            # Overall transition score
            overall_score = (
                key_compat * 0.25 +
                tempo_compat * 0.35 +
                energy_compat * 0.25 +
                emotional_compat * 0.15
            )
            
            return {
                "overall": round(overall_score, 3),
                "key_compatibility": round(key_compat, 3),
                "tempo_compatibility": round(tempo_compat, 3),
                "energy_compatibility": round(energy_compat, 3),
                "emotional_compatibility": round(emotional_compat, 3)
            }
            
        except Exception as e:
            logger.error(f"❌ Transition quality calculation failed: {str(e)}")
            raise
