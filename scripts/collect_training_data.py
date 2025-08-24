#!/usr/bin/env python3
"""
FlowState Training Data Collection
Collects real song data from Spotify API for model training
"""

import asyncio
import json
import os
import time
import logging
from typing import List, Dict, Optional
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import librosa
import numpy as np
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.database import save_song_metadata, AsyncSessionLocal
from src.models import Song

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyDataCollector:
    """Collect training data from Spotify API"""
    
    def __init__(self):
        """Initialize Spotify client"""
        if not settings.spotify_client_id or not settings.spotify_client_secret:
            raise ValueError("Spotify API credentials not found. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
        
        client_credentials_manager = SpotifyClientCredentials(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret
        )
        self.spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        
        # Create data directories
        self.data_dir = Path("training_data")
        self.audio_dir = self.data_dir / "audio"
        self.metadata_dir = self.data_dir / "metadata"
        
        for dir_path in [self.data_dir, self.audio_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info("🎵 Spotify data collector initialized")

    async def collect_featured_playlists(self, limit: int = 20) -> List[Dict]:
        """Collect songs from Spotify's featured playlists"""
        logger.info(f"🎯 Collecting songs from featured playlists (limit: {limit})")
        
        songs = []
        try:
            # Get featured playlists
            playlists = self.spotify.featured_playlists(limit=10)['playlists']['items']
            
            for playlist in playlists:
                logger.info(f"📃 Processing playlist: {playlist['name']}")
                
                # Get tracks from playlist
                tracks = self.spotify.playlist_tracks(playlist['id'], limit=50)['items']
                
                for item in tracks:
                    if not item['track'] or not item['track']['preview_url']:
                        continue
                        
                    track = item['track']
                    
                    song_data = {
                        'id': f"spotify_{track['id']}",
                        'title': track['name'],
                        'artist': track['artists'][0]['name'],
                        'album': track['album']['name'],
                        'duration_ms': track['duration_ms'],
                        'spotify_id': track['id'],
                        'preview_url': track['preview_url'],
                        'popularity': track['popularity']
                    }
                    
                    songs.append(song_data)
                    
                    if len(songs) >= limit:
                        break
                
                if len(songs) >= limit:
                    break
                    
        except Exception as e:
            logger.error(f"❌ Error collecting playlists: {str(e)}")
            
        logger.info(f"✅ Collected {len(songs)} songs with preview URLs")
        return songs[:limit]

    async def get_audio_features(self, spotify_ids: List[str]) -> Dict[str, Dict]:
        """Get Spotify's audio features for tracks"""
        logger.info(f"🎼 Getting audio features for {len(spotify_ids)} tracks")
        
        features_map = {}
        
        # Spotify API allows up to 100 tracks per request
        batch_size = 100
        for i in range(0, len(spotify_ids), batch_size):
            batch = spotify_ids[i:i + batch_size]
            
            try:
                features_batch = self.spotify.audio_features(batch)
                
                for features in features_batch:
                    if features:  # Some tracks might not have features
                        features_map[features['id']] = features
                        
            except Exception as e:
                logger.error(f"❌ Error getting audio features: {str(e)}")
                
        logger.info(f"✅ Retrieved audio features for {len(features_map)} tracks")
        return features_map

    async def download_audio_preview(self, song: Dict) -> Optional[str]:
        """Download 30-second preview audio file"""
        if not song.get('preview_url'):
            return None
            
        try:
            audio_filename = f"{song['spotify_id']}.mp3"
            audio_path = self.audio_dir / audio_filename
            
            if audio_path.exists():
                return str(audio_path)
            
            # Download audio file
            response = requests.get(song['preview_url'], timeout=30)
            response.raise_for_status()
            
            with open(audio_path, 'wb') as f:
                f.write(response.content)
                
            logger.debug(f"📥 Downloaded: {song['title']} by {song['artist']}")
            return str(audio_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to download {song['title']}: {str(e)}")
            return None

    async def extract_librosa_features(self, audio_path: str) -> Optional[Dict]:
        """Extract detailed audio features using librosa"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, duration=30)  # 30 seconds max
            
            # Extract features
            features = {}
            
            # Tempo and beat
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_frames'] = len(beats)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            
            # Zero crossing rate (indicates speech vs music)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zero_crossing_rate_mean'] = float(np.mean(zcr))
            
            # MFCCs (mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            
            # Chroma features (pitch class profiles)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
            
            # RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            
            # Harmonic and percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            features['harmonic_mean'] = float(np.mean(y_harmonic))
            features['percussive_mean'] = float(np.mean(y_percussive))
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Librosa feature extraction failed: {str(e)}")
            return None

    async def process_song(self, song_data: Dict, spotify_features: Dict) -> Dict:
        """Process a single song: download, analyze, and prepare training data"""
        logger.info(f"🎵 Processing: {song_data['title']} by {song_data['artist']}")
        
        # Download audio preview
        audio_path = await self.download_audio_preview(song_data)
        if not audio_path:
            logger.warning(f"⚠️ No audio available for {song_data['title']}")
            return None
        
        # Extract librosa features
        librosa_features = await self.extract_librosa_features(audio_path)
        if not librosa_features:
            logger.warning(f"⚠️ Feature extraction failed for {song_data['title']}")
            return None
        
        # Combine all features
        combined_features = {
            **song_data,
            'spotify_features': spotify_features,
            'librosa_features': librosa_features,
            'audio_path': audio_path,
            'processed_at': time.time()
        }
        
        # Save metadata
        metadata_file = self.metadata_dir / f"{song_data['spotify_id']}.json"
        with open(metadata_file, 'w') as f:
            json.dump(combined_features, f, indent=2)
        
        logger.info(f"✅ Processed: {song_data['title']}")
        return combined_features

    async def collect_training_dataset(self, num_songs: int = 100) -> List[Dict]:
        """Main function to collect complete training dataset"""
        logger.info(f"🚀 Starting training data collection for {num_songs} songs")
        
        # Step 1: Collect song metadata
        songs = await self.collect_featured_playlists(limit=num_songs)
        if not songs:
            logger.error("❌ No songs collected")
            return []
        
        # Step 2: Get Spotify audio features
        spotify_ids = [song['spotify_id'] for song in songs]
        spotify_features_map = await self.get_audio_features(spotify_ids)
        
        # Step 3: Process each song
        processed_songs = []
        for song in songs:
            spotify_id = song['spotify_id']
            
            if spotify_id not in spotify_features_map:
                logger.warning(f"⚠️ No Spotify features for {song['title']}")
                continue
            
            spotify_features = spotify_features_map[spotify_id]
            processed_song = await self.process_song(song, spotify_features)
            
            if processed_song:
                processed_songs.append(processed_song)
            
            # Add delay to avoid rate limiting
            await asyncio.sleep(0.5)
        
        logger.info(f"🎉 Training dataset collection complete: {len(processed_songs)} songs processed")
        
        # Save complete dataset
        dataset_file = self.data_dir / "complete_dataset.json"
        with open(dataset_file, 'w') as f:
            json.dump(processed_songs, f, indent=2)
        
        return processed_songs

async def main():
    """Run training data collection"""
    collector = SpotifyDataCollector()
    
    # Collect training data (start with 50 songs for initial training)
    dataset = await collector.collect_training_dataset(num_songs=50)
    
    print(f"\n🎉 Training data collection complete!")
    print(f"📊 Total songs processed: {len(dataset)}")
    print(f"📁 Data saved in: training_data/")
    print(f"🎵 Audio files: training_data/audio/")
    print(f"📋 Metadata: training_data/metadata/")
    
    if dataset:
        print(f"\n✅ Ready for model training!")
        print(f"Next steps:")
        print(f"1. Run: python scripts/train_emotion_model.py")
        print(f"2. Run: python scripts/deploy_models.py")

if __name__ == "__main__":
    asyncio.run(main())
