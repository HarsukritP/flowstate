"""
FlowState Queue Optimizer
Advanced algorithms for generating emotionally optimized music queues
"""

import asyncio
import time
import random
import heapq
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
import logging
from dataclasses import dataclass

from .models import (
    Song, AudioFeatures, EmotionalProfile, EmotionalJourney,
    OptimizedQueue, QueueMetadata, CompatibilityScore
)
from .audio_analyzer import AudioAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class SongNode:
    """Graph node representing a song with its features and connections"""
    song: Song
    features: Optional[AudioFeatures] = None
    emotional_profile: Optional[EmotionalProfile] = None
    compatibility_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.compatibility_scores is None:
            self.compatibility_scores = {}

class QueueOptimizer:
    """
    Advanced queue optimization engine using graph algorithms and emotional modeling
    """
    
    def __init__(self):
        """Initialize the queue optimizer with analysis components"""
        self.audio_analyzer = AudioAnalyzer()
        self.song_cache: Dict[str, SongNode] = {}
        self.compatibility_cache: Dict[Tuple[str, str], float] = {}
        
        # Optimization parameters
        self.max_tempo_jump = 20  # Max BPM difference for good flow
        self.energy_smoothing_factor = 0.3  # How much to smooth energy transitions
        self.emotional_weight = 0.4  # Weight of emotional vs musical factors
        
        logger.info("🎯 QueueOptimizer initialized")

    async def generate_queue(
        self,
        seed_song: Song,
        queue_length: int = 10,
        emotional_journey: EmotionalJourney = EmotionalJourney.GRADUAL_FLOW,
        available_songs: Optional[List[Song]] = None
    ) -> OptimizedQueue:
        """
        Generate an emotionally optimized queue from a seed song
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Generating {queue_length}-song queue from '{seed_song.title}'")
            
            # Use sample songs if none provided
            if available_songs is None:
                available_songs = await self._get_sample_songs()
            
            # Ensure seed song is analyzed
            seed_node = await self._get_or_create_song_node(seed_song)
            
            # Generate queue using the appropriate algorithm
            if emotional_journey == EmotionalJourney.MAINTAIN_VIBE:
                queue_songs = await self._generate_maintain_vibe_queue(
                    seed_node, available_songs, queue_length
                )
            elif emotional_journey == EmotionalJourney.WIND_DOWN:
                queue_songs = await self._generate_wind_down_queue(
                    seed_node, available_songs, queue_length
                )
            elif emotional_journey == EmotionalJourney.PUMP_UP:
                queue_songs = await self._generate_pump_up_queue(
                    seed_node, available_songs, queue_length
                )
            else:  # GRADUAL_FLOW, ADVENTURE_MODE, MEDITATIVE
                queue_songs = await self._generate_gradual_flow_queue(
                    seed_node, available_songs, queue_length, emotional_journey
                )
            
            # Calculate emotional journey
            emotional_journey_profiles = []
            for song in queue_songs:
                node = await self._get_or_create_song_node(song)
                emotional_journey_profiles.append(node.emotional_profile)
            
            # Calculate flow score
            flow_score = await self._calculate_queue_flow_score(queue_songs)
            
            # Generate metadata
            metadata = await self._generate_queue_metadata(queue_songs, emotional_journey)
            
            generation_time = (time.time() - start_time) * 1000
            
            logger.info(f"✅ Generated queue with flow score {flow_score:.3f} in {generation_time:.1f}ms")
            
            return OptimizedQueue(
                songs=queue_songs,
                emotional_journey=emotional_journey_profiles,
                flow_score=flow_score,
                generation_time_ms=generation_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"❌ Queue generation failed: {str(e)}")
            raise

    async def _generate_gradual_flow_queue(
        self,
        seed_node: SongNode,
        available_songs: List[Song],
        queue_length: int,
        journey_type: EmotionalJourney
    ) -> List[Song]:
        """Generate a queue with gradual emotional transitions"""
        
        queue = [seed_node.song]
        used_songs = {seed_node.song.id}
        current_node = seed_node
        
        # Create candidate pool
        candidate_nodes = []
        for song in available_songs:
            if song.id not in used_songs:
                node = await self._get_or_create_song_node(song)
                candidate_nodes.append(node)
        
        # Build queue incrementally
        for position in range(1, queue_length):
            next_node = await self._select_best_next_song(
                current_node, candidate_nodes, position, queue_length, journey_type
            )
            
            if next_node:
                queue.append(next_node.song)
                used_songs.add(next_node.song.id)
                candidate_nodes.remove(next_node)
                current_node = next_node
            else:
                # Fallback: select random compatible song
                if candidate_nodes:
                    next_node = random.choice(candidate_nodes)
                    queue.append(next_node.song)
                    used_songs.add(next_node.song.id)
                    candidate_nodes.remove(next_node)
                    current_node = next_node
        
        return queue

    async def _generate_maintain_vibe_queue(
        self,
        seed_node: SongNode,
        available_songs: List[Song],
        queue_length: int
    ) -> List[Song]:
        """Generate a queue that maintains similar vibe throughout"""
        
        queue = [seed_node.song]
        used_songs = {seed_node.song.id}
        
        # Find songs with similar characteristics
        target_energy = seed_node.features.energy
        target_valence = seed_node.features.valence
        target_tempo = seed_node.features.tempo
        
        candidates = []
        for song in available_songs:
            if song.id not in used_songs:
                node = await self._get_or_create_song_node(song)
                
                # Calculate similarity to seed
                energy_diff = abs(node.features.energy - target_energy)
                valence_diff = abs(node.features.valence - target_valence)
                tempo_diff = abs(node.features.tempo - target_tempo) / 60  # Normalize
                
                similarity = 1.0 - (energy_diff * 0.4 + valence_diff * 0.4 + tempo_diff * 0.2)
                candidates.append((similarity, node))
        
        # Sort by similarity and take the best matches
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for i in range(min(queue_length - 1, len(candidates))):
            queue.append(candidates[i][1].song)
        
        return queue

    async def _generate_wind_down_queue(
        self,
        seed_node: SongNode,
        available_songs: List[Song],
        queue_length: int
    ) -> List[Song]:
        """Generate a queue that gradually reduces energy"""
        
        queue = [seed_node.song]
        used_songs = {seed_node.song.id}
        
        current_energy = seed_node.features.energy
        target_energy = max(0.1, current_energy * 0.3)  # End at much lower energy
        energy_step = (current_energy - target_energy) / (queue_length - 1)
        
        candidate_nodes = []
        for song in available_songs:
            if song.id not in used_songs:
                node = await self._get_or_create_song_node(song)
                candidate_nodes.append(node)
        
        for position in range(1, queue_length):
            desired_energy = current_energy - (energy_step * position)
            
            # Find song closest to desired energy level
            best_node = None
            best_score = float('inf')
            
            for node in candidate_nodes:
                energy_diff = abs(node.features.energy - desired_energy)
                
                # Prefer calmer, more acoustic songs for wind down
                calmness_bonus = node.features.acousticness * 0.2
                tempo_penalty = max(0, (node.features.tempo - 100) / 100) * 0.3
                
                score = energy_diff + tempo_penalty - calmness_bonus
                
                if score < best_score:
                    best_score = score
                    best_node = node
            
            if best_node:
                queue.append(best_node.song)
                used_songs.add(best_node.song.id)
                candidate_nodes.remove(best_node)
        
        return queue

    async def _generate_pump_up_queue(
        self,
        seed_node: SongNode,
        available_songs: List[Song],
        queue_length: int
    ) -> List[Song]:
        """Generate a queue that gradually increases energy"""
        
        queue = [seed_node.song]
        used_songs = {seed_node.song.id}
        
        current_energy = seed_node.features.energy
        target_energy = min(0.95, current_energy + 0.4)  # Boost energy significantly
        energy_step = (target_energy - current_energy) / (queue_length - 1)
        
        candidate_nodes = []
        for song in available_songs:
            if song.id not in used_songs:
                node = await self._get_or_create_song_node(song)
                candidate_nodes.append(node)
        
        for position in range(1, queue_length):
            desired_energy = current_energy + (energy_step * position)
            
            # Find song closest to desired energy level
            best_node = None
            best_score = float('inf')
            
            for node in candidate_nodes:
                energy_diff = abs(node.features.energy - desired_energy)
                
                # Prefer high-energy, danceable songs
                energy_bonus = node.features.energy * 0.2
                dance_bonus = node.features.danceability * 0.1
                tempo_bonus = min(0.2, (node.features.tempo - 120) / 200)
                
                score = energy_diff - energy_bonus - dance_bonus - tempo_bonus
                
                if score < best_score:
                    best_score = score
                    best_node = node
            
            if best_node:
                queue.append(best_node.song)
                used_songs.add(best_node.song.id)
                candidate_nodes.remove(best_node)
        
        return queue

    async def _select_best_next_song(
        self,
        current_node: SongNode,
        candidates: List[SongNode],
        position: int,
        total_length: int,
        journey_type: EmotionalJourney
    ) -> Optional[SongNode]:
        """Select the best next song for gradual flow"""
        
        if not candidates:
            return None
        
        best_node = None
        best_score = float('-inf')
        
        # Calculate journey progress
        progress = position / (total_length - 1)
        
        for candidate in candidates:
            # Calculate compatibility score
            compatibility = await self._calculate_song_compatibility(
                current_node, candidate
            )
            
            # Journey-specific scoring
            journey_score = self._calculate_journey_score(
                current_node, candidate, progress, journey_type
            )
            
            # Combined score
            total_score = compatibility * 0.6 + journey_score * 0.4
            
            if total_score > best_score:
                best_score = total_score
                best_node = candidate
        
        return best_node

    def _calculate_journey_score(
        self,
        current_node: SongNode,
        candidate_node: SongNode,
        progress: float,
        journey_type: EmotionalJourney
    ) -> float:
        """Calculate how well a song fits the emotional journey"""
        
        if journey_type == EmotionalJourney.ADVENTURE_MODE:
            # Encourage variety and contrast
            energy_diff = abs(current_node.features.energy - candidate_node.features.energy)
            valence_diff = abs(current_node.features.valence - candidate_node.features.valence)
            variety_score = (energy_diff + valence_diff) / 2
            return variety_score
        
        elif journey_type == EmotionalJourney.MEDITATIVE:
            # Prefer calm, consistent songs
            calmness = candidate_node.features.acousticness
            low_energy = 1.0 - candidate_node.features.energy
            consistency = 1.0 - abs(current_node.features.valence - candidate_node.features.valence)
            return (calmness * 0.4 + low_energy * 0.3 + consistency * 0.3)
        
        else:  # GRADUAL_FLOW
            # Smooth progression based on position
            target_energy = self._calculate_target_energy(progress)
            target_valence = self._calculate_target_valence(progress)
            
            energy_fit = 1.0 - abs(candidate_node.features.energy - target_energy)
            valence_fit = 1.0 - abs(candidate_node.features.valence - target_valence)
            
            return (energy_fit * 0.6 + valence_fit * 0.4)

    def _calculate_target_energy(self, progress: float) -> float:
        """Calculate target energy level based on journey progress"""
        # Gradual increase to peak at 70%, then gentle decrease
        if progress < 0.7:
            return 0.4 + (progress / 0.7) * 0.4  # 0.4 to 0.8
        else:
            return 0.8 - ((progress - 0.7) / 0.3) * 0.2  # 0.8 to 0.6

    def _calculate_target_valence(self, progress: float) -> float:
        """Calculate target valence based on journey progress"""
        # Gradual increase throughout
        return 0.3 + progress * 0.4  # 0.3 to 0.7

    async def _calculate_song_compatibility(
        self,
        song1_node: SongNode,
        song2_node: SongNode
    ) -> float:
        """Calculate overall compatibility between two songs"""
        
        cache_key = (song1_node.song.id, song2_node.song.id)
        if cache_key in self.compatibility_cache:
            return self.compatibility_cache[cache_key]
        
        # Musical compatibility
        tempo_compat = self.audio_analyzer.calculate_tempo_compatibility(
            song1_node.features.tempo, song2_node.features.tempo
        )
        
        key_compat = self.audio_analyzer.calculate_key_compatibility(
            song1_node.features.key, song2_node.features.key
        )
        
        # Energy flow compatibility
        energy_diff = abs(song1_node.features.energy - song2_node.features.energy)
        energy_compat = max(0, 1.0 - energy_diff / self.energy_smoothing_factor)
        
        # Emotional compatibility
        valence_diff = abs(song1_node.features.valence - song2_node.features.valence)
        emotional_compat = max(0, 1.0 - valence_diff)
        
        # Weighted combination
        compatibility = (
            tempo_compat * 0.3 +
            key_compat * 0.2 +
            energy_compat * 0.3 +
            emotional_compat * 0.2
        )
        
        # Cache the result
        self.compatibility_cache[cache_key] = compatibility
        
        return compatibility

    async def _calculate_queue_flow_score(self, queue: List[Song]) -> float:
        """Calculate overall flow quality score for the queue"""
        
        if len(queue) < 2:
            return 1.0
        
        total_score = 0.0
        transition_count = len(queue) - 1
        
        for i in range(transition_count):
            song1_node = await self._get_or_create_song_node(queue[i])
            song2_node = await self._get_or_create_song_node(queue[i + 1])
            
            transition_score = await self._calculate_song_compatibility(song1_node, song2_node)
            total_score += transition_score
        
        return total_score / transition_count

    async def _generate_queue_metadata(
        self,
        queue: List[Song],
        journey_type: EmotionalJourney
    ) -> QueueMetadata:
        """Generate comprehensive metadata for the queue"""
        
        total_duration = sum(song.duration_ms or 240000 for song in queue)  # Default 4 min
        
        # Calculate average energy
        energy_sum = 0.0
        for song in queue:
            node = await self._get_or_create_song_node(song)
            energy_sum += node.features.energy
        average_energy = energy_sum / len(queue)
        
        # Emotional progression
        emotional_arc = []
        for song in queue:
            node = await self._get_or_create_song_node(song)
            emotional_arc.append(node.emotional_profile.primary_emotion)
        
        # Calculate tempo variance
        tempos = []
        for song in queue:
            node = await self._get_or_create_song_node(song)
            tempos.append(node.features.tempo)
        tempo_variance = np.var(tempos) if len(tempos) > 1 else 0.0
        
        return QueueMetadata(
            total_duration_ms=total_duration,
            average_energy=round(average_energy, 3),
            emotional_arc=emotional_arc,
            genre_diversity=0.8,  # Mock value for MVP
            tempo_variance=round(tempo_variance, 2),
            optimization_algorithm="greedy_flow_v1"
        )

    async def _get_or_create_song_node(self, song: Song) -> SongNode:
        """Get or create a song node with analysis"""
        
        if song.id in self.song_cache:
            return self.song_cache[song.id]
        
        # Analyze the song
        features = await self.audio_analyzer.extract_features(song)
        emotional_profile = await self.audio_analyzer.analyze_emotion(features)
        
        # Create and cache the node
        node = SongNode(
            song=song,
            features=features,
            emotional_profile=emotional_profile
        )
        
        self.song_cache[song.id] = node
        return node

    async def _get_sample_songs(self) -> List[Song]:
        """Get a sample set of songs for MVP testing"""
        return [
            Song(id="s1", title="Breathe Me", artist="Sia", duration_ms=257000),
            Song(id="s2", title="Mad World", artist="Gary Jules", duration_ms=203000),
            Song(id="s3", title="The Sound of Silence", artist="Simon & Garfunkel", duration_ms=213000),
            Song(id="s4", title="Hallelujah", artist="Jeff Buckley", duration_ms=367000),
            Song(id="s5", title="Black", artist="Pearl Jam", duration_ms=343000),
            Song(id="s6", title="Hurt", artist="Johnny Cash", duration_ms=218000),
            Song(id="s7", title="Everybody Hurts", artist="R.E.M.", duration_ms=323000),
            Song(id="s8", title="Tears in Heaven", artist="Eric Clapton", duration_ms=282000),
            Song(id="s9", title="Skinny Love", artist="Bon Iver", duration_ms=238000),
            Song(id="s10", title="The Night We Met", artist="Lord Huron", duration_ms=207000),
            Song(id="s11", title="Shake It Off", artist="Taylor Swift", duration_ms=219000),
            Song(id="s12", title="Uptown Funk", artist="Bruno Mars", duration_ms=269000),
            Song(id="s13", title="Happy", artist="Pharrell Williams", duration_ms=233000),
            Song(id="s14", title="Can't Stop the Feeling", artist="Justin Timberlake", duration_ms=236000),
            Song(id="s15", title="Good as Hell", artist="Lizzo", duration_ms=219000),
            Song(id="s16", title="Weightless", artist="Marconi Union", duration_ms=511000),
            Song(id="s17", title="Clair de Lune", artist="Claude Debussy", duration_ms=300000),
            Song(id="s18", title="Gymnopédie No. 1", artist="Erik Satie", duration_ms=210000),
            Song(id="s19", title="Spiegel im Spiegel", artist="Arvo Pärt", duration_ms=480000),
            Song(id="s20", title="On Earth as It Is in Heaven", artist="Angel", duration_ms=380000)
        ]

    async def reoptimize_queue(
        self,
        current_queue: List[Song],
        new_song: Song,
        insertion_point: int
    ) -> OptimizedQueue:
        """Efficiently re-optimize queue with new song insertion"""
        
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Re-optimizing queue with '{new_song.title}' at position {insertion_point}")
            
            # Insert new song at specified position
            new_queue = current_queue.copy()
            new_queue.insert(insertion_point, new_song)
            
            # Local optimization around insertion point
            optimized_queue = await self._local_queue_optimization(
                new_queue, insertion_point
            )
            
            # Calculate new metrics
            flow_score = await self._calculate_queue_flow_score(optimized_queue)
            
            # Generate metadata
            metadata = await self._generate_queue_metadata(
                optimized_queue, EmotionalJourney.GRADUAL_FLOW
            )
            
            # Emotional journey
            emotional_journey = []
            for song in optimized_queue:
                node = await self._get_or_create_song_node(song)
                emotional_journey.append(node.emotional_profile)
            
            generation_time = (time.time() - start_time) * 1000
            
            logger.info(f"✅ Re-optimized queue with flow score {flow_score:.3f} in {generation_time:.1f}ms")
            
            return OptimizedQueue(
                songs=optimized_queue,
                emotional_journey=emotional_journey,
                flow_score=flow_score,
                generation_time_ms=generation_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"❌ Queue re-optimization failed: {str(e)}")
            raise

    async def _local_queue_optimization(
        self,
        queue: List[Song],
        insertion_point: int,
        optimization_radius: int = 2
    ) -> List[Song]:
        """Optimize queue locally around insertion point"""
        
        # Define optimization window
        start_idx = max(0, insertion_point - optimization_radius)
        end_idx = min(len(queue), insertion_point + optimization_radius + 1)
        
        # Extract segment to optimize
        segment = queue[start_idx:end_idx]
        
        if len(segment) <= 2:
            return queue  # No optimization needed
        
        # Try different arrangements within the segment
        best_arrangement = segment
        best_score = await self._calculate_segment_flow_score(segment)
        
        # Try swapping adjacent songs (simple local optimization)
        for i in range(len(segment) - 1):
            test_segment = segment.copy()
            test_segment[i], test_segment[i + 1] = test_segment[i + 1], test_segment[i]
            
            score = await self._calculate_segment_flow_score(test_segment)
            if score > best_score:
                best_score = score
                best_arrangement = test_segment
        
        # Reconstruct optimized queue
        optimized_queue = queue.copy()
        optimized_queue[start_idx:end_idx] = best_arrangement
        
        return optimized_queue

    async def _calculate_segment_flow_score(self, segment: List[Song]) -> float:
        """Calculate flow score for a queue segment"""
        if len(segment) < 2:
            return 1.0
        
        total_score = 0.0
        transition_count = len(segment) - 1
        
        for i in range(transition_count):
            node1 = await self._get_or_create_song_node(segment[i])
            node2 = await self._get_or_create_song_node(segment[i + 1])
            score = await self._calculate_song_compatibility(node1, node2)
            total_score += score
        
        return total_score / transition_count

    async def calculate_compatibility(self, song_id1: str, song_id2: str) -> CompatibilityScore:
        """Calculate detailed compatibility between two songs by ID"""
        
        # For MVP, return mock compatibility - in production would look up actual songs
        compatibility_score = random.uniform(0.6, 0.95)
        
        return CompatibilityScore(
            score=compatibility_score,
            tempo_compatibility=random.uniform(0.7, 1.0),
            key_compatibility=random.uniform(0.5, 1.0),
            energy_compatibility=random.uniform(0.6, 0.9),
            emotional_compatibility=random.uniform(0.5, 0.9),
            transition_quality="smooth" if compatibility_score > 0.8 else "moderate",
            factors={
                "tempo_diff": random.uniform(0, 15),
                "key_relation": "compatible",
                "energy_flow": "smooth",
                "emotional_arc": "natural"
            }
        )
