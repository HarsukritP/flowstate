#!/usr/bin/env python3
"""
FlowState API Test Script
Quick test of the core FlowState functionality
"""

import asyncio
import json
import time
from src.models import Song, QueueRequest, EmotionalJourney
from src.audio_analyzer import AudioAnalyzer
from src.queue_optimizer import QueueOptimizer

async def test_audio_analysis():
    """Test audio feature extraction and emotional analysis"""
    print("🎵 Testing Audio Analysis...")
    
    analyzer = AudioAnalyzer()
    
    test_song = Song(
        id="test_001",
        title="Breathe Me",
        artist="Sia",
        duration_ms=257000
    )
    
    # Extract features
    start_time = time.time()
    features = await analyzer.extract_features(test_song)
    analysis_time = (time.time() - start_time) * 1000
    
    print(f"✅ Features extracted in {analysis_time:.1f}ms")
    print(f"   Tempo: {features.tempo} BPM")
    print(f"   Key: {features.key} ({['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][features.key]})")
    print(f"   Energy: {features.energy:.3f}")
    print(f"   Valence: {features.valence:.3f}")
    print(f"   Emotional Arousal: {features.emotional_arousal:.3f}")
    
    # Analyze emotion
    emotional_profile = await analyzer.analyze_emotion(features)
    print(f"   Primary Emotion: {emotional_profile.primary_emotion}")
    print(f"   Emotional Tags: {', '.join(emotional_profile.emotional_tags)}")
    
    return features, emotional_profile

async def test_queue_generation():
    """Test queue generation with different emotional journeys"""
    print("\n🎯 Testing Queue Generation...")
    
    optimizer = QueueOptimizer()
    
    seed_song = Song(
        id="seed_001",
        title="Mad World",
        artist="Gary Jules",
        duration_ms=203000
    )
    
    # Test different journey types
    journeys = [
        EmotionalJourney.GRADUAL_FLOW,
        EmotionalJourney.MAINTAIN_VIBE,
        EmotionalJourney.WIND_DOWN,
        EmotionalJourney.PUMP_UP
    ]
    
    for journey in journeys:
        print(f"\n  Testing {journey.value}...")
        
        start_time = time.time()
        queue = await optimizer.generate_queue(
            seed_song=seed_song,
            queue_length=8,
            emotional_journey=journey
        )
        generation_time = (time.time() - start_time) * 1000
        
        print(f"  ✅ Generated in {generation_time:.1f}ms")
        print(f"     Flow Score: {queue.flow_score:.3f}")
        print(f"     Songs: {[s.title for s in queue.songs[:3]]}...")
        print(f"     Emotional Arc: {queue.metadata.emotional_arc[:3]}...")

async def test_compatibility_scoring():
    """Test song compatibility calculations"""
    print("\n🔗 Testing Compatibility Scoring...")
    
    analyzer = AudioAnalyzer()
    
    song1 = Song(id="s1", title="Hurt", artist="Johnny Cash")
    song2 = Song(id="s2", title="Mad World", artist="Gary Jules")
    song3 = Song(id="s3", title="Happy", artist="Pharrell Williams")
    
    # Test similar songs (both melancholic)
    transition1 = await analyzer.calculate_transition_quality(song1, song2)
    print(f"  Melancholic → Melancholic: {transition1['overall']:.3f}")
    
    # Test contrasting songs (sad → happy)
    transition2 = await analyzer.calculate_transition_quality(song1, song3)
    print(f"  Melancholic → Happy: {transition2['overall']:.3f}")
    
    print(f"  Breakdown: Tempo={transition1['tempo_compatibility']:.3f}, "
          f"Key={transition1['key_compatibility']:.3f}, "
          f"Energy={transition1['energy_compatibility']:.3f}")

async def test_queue_reoptimization():
    """Test efficient queue re-optimization"""
    print("\n🔄 Testing Queue Re-optimization...")
    
    optimizer = QueueOptimizer()
    
    # Generate initial queue
    seed_song = Song(id="seed", title="Breathe Me", artist="Sia")
    initial_queue = await optimizer.generate_queue(seed_song, queue_length=6)
    
    print(f"  Initial queue flow score: {initial_queue.flow_score:.3f}")
    
    # Add a new song and re-optimize
    new_song = Song(id="new", title="Tears in Heaven", artist="Eric Clapton")
    
    start_time = time.time()
    reoptimized_queue = await optimizer.reoptimize_queue(
        current_queue=initial_queue.songs,
        new_song=new_song,
        insertion_point=3
    )
    reopt_time = (time.time() - start_time) * 1000
    
    print(f"  ✅ Re-optimized in {reopt_time:.1f}ms")
    print(f"  New flow score: {reoptimized_queue.flow_score:.3f}")
    print(f"  Queue length: {len(initial_queue.songs)} → {len(reoptimized_queue.songs)}")

async def test_performance_benchmark():
    """Run performance benchmarks"""
    print("\n⚡ Performance Benchmarks...")
    
    optimizer = QueueOptimizer()
    times = []
    
    # Generate 10 queues and measure performance
    for i in range(10):
        seed_song = Song(id=f"perf_{i}", title=f"Test Song {i}", artist="Test Artist")
        
        start_time = time.time()
        queue = await optimizer.generate_queue(seed_song, queue_length=10)
        generation_time = (time.time() - start_time) * 1000
        times.append(generation_time)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"  Average generation time: {avg_time:.1f}ms")
    print(f"  Range: {min_time:.1f}ms - {max_time:.1f}ms")
    print(f"  Target (<100ms): {'✅ PASS' if avg_time < 100 else '❌ FAIL'}")

async def main():
    """Run all tests"""
    print("🌊 FlowState API Test Suite")
    print("=" * 50)
    
    try:
        await test_audio_analysis()
        await test_queue_generation()
        await test_compatibility_scoring()
        await test_queue_reoptimization()
        await test_performance_benchmark()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
