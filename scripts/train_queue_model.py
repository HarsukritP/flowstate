#!/usr/bin/env python3
"""
FlowState Queue Optimization Model Training
Trains ML models to learn optimal song sequencing from real playlist data
"""

import asyncio
import json
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueueOptimizationTrainer:
    """Train ML models to predict optimal song transitions from playlist data"""
    
    def __init__(self):
        """Initialize trainer"""
        self.data_dir = Path("training_data")
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
        
        logger.info("🎵 Queue optimization trainer initialized")

    def load_playlist_sequences(self) -> List[Dict]:
        """Load playlist sequences collected from Spotify"""
        sequences_file = self.data_dir / "playlist_sequences.json"
        
        if not sequences_file.exists():
            raise FileNotFoundError(f"Playlist sequences not found at {sequences_file}. Run collect_training_data.py first.")
        
        with open(sequences_file, 'r') as f:
            data = json.load(f)
        
        sequences = data['sequences']
        logger.info(f"📚 Loaded {len(sequences)} playlist sequences")
        
        # Print dataset statistics
        total_tracks = sum(len(seq['tracks']) for seq in sequences)
        categories = list(set(seq['category'] for seq in sequences))
        
        logger.info(f"🎵 Total tracks: {total_tracks}")
        logger.info(f"📂 Categories: {', '.join(categories)}")
        
        return sequences

    def extract_transitions(self, sequences: List[Dict]) -> pd.DataFrame:
        """Extract song-to-song transitions with features and success metrics"""
        logger.info("🔄 Extracting song transitions from playlists...")
        
        transitions = []
        
        for seq in sequences:
            tracks = seq['tracks']
            category = seq['category']
            
            for i in range(len(tracks) - 1):
                current = tracks[i]
                next_track = tracks[i + 1]
                
                # Only process if both tracks have audio features
                if ('audio_features' not in current or 'audio_features' not in next_track):
                    continue
                
                curr_features = current['audio_features']
                next_features = next_track['audio_features']
                
                # Calculate transition features
                transition = {
                    # Sequence context
                    'sequence_id': seq['sequence_id'],
                    'category': category,
                    'position_in_playlist': i,
                    'playlist_length': len(tracks),
                    'follower_count': seq.get('follower_count', 0),
                    
                    # Current song features
                    'curr_tempo': curr_features['tempo'],
                    'curr_energy': curr_features['energy'],
                    'curr_valence': curr_features['valence'],
                    'curr_danceability': curr_features['danceability'],
                    'curr_acousticness': curr_features['acousticness'],
                    'curr_key': curr_features['key'],
                    'curr_mode': curr_features['mode'],
                    'curr_loudness': curr_features['loudness'],
                    
                    # Next song features
                    'next_tempo': next_features['tempo'],
                    'next_energy': next_features['energy'],
                    'next_valence': next_features['valence'],
                    'next_danceability': next_features['danceability'],
                    'next_acousticness': next_features['acousticness'],
                    'next_key': next_features['key'],
                    'next_mode': next_features['mode'],
                    'next_loudness': next_features['loudness'],
                    
                    # Transition deltas
                    'tempo_change': next_features['tempo'] - curr_features['tempo'],
                    'energy_change': next_features['energy'] - curr_features['energy'],
                    'valence_change': next_features['valence'] - curr_features['valence'],
                    'danceability_change': next_features['danceability'] - curr_features['danceability'],
                    'acousticness_change': next_features['acousticness'] - curr_features['acousticness'],
                    'loudness_change': next_features['loudness'] - curr_features['loudness'],
                    
                    # Transition magnitudes
                    'tempo_change_abs': abs(next_features['tempo'] - curr_features['tempo']),
                    'energy_change_abs': abs(next_features['energy'] - curr_features['energy']),
                    'valence_change_abs': abs(next_features['valence'] - curr_features['valence']),
                    
                    # Musical compatibility
                    'key_compatibility': self._calculate_key_compatibility(curr_features['key'], next_features['key']),
                    'mode_match': 1 if curr_features['mode'] == next_features['mode'] else 0,
                    
                    # Success metrics (proxy from playlist curation quality)
                    'playlist_quality_score': self._calculate_playlist_quality_score(seq),
                    'transition_quality_score': self._calculate_transition_quality(curr_features, next_features)
                }
                
                transitions.append(transition)
        
        df = pd.DataFrame(transitions)
        logger.info(f"✅ Extracted {len(df)} transitions")
        
        return df

    def _calculate_key_compatibility(self, key1: int, key2: int) -> float:
        """Calculate musical key compatibility using circle of fifths"""
        # Circle of fifths compatibility matrix (simplified)
        key_distance = min(abs(key1 - key2), 12 - abs(key1 - key2))
        
        if key_distance == 0:
            return 1.0  # Same key
        elif key_distance <= 2:
            return 0.8  # Close keys
        elif key_distance <= 4:
            return 0.6  # Moderate distance
        else:
            return 0.3  # Distant keys

    def _calculate_playlist_quality_score(self, sequence: Dict) -> float:
        """Calculate playlist quality score based on metadata"""
        # Proxy metrics for playlist quality
        follower_count = sequence.get('follower_count', 0)
        track_count = len(sequence['tracks'])
        
        # Normalize follower count (log scale)
        follower_score = min(np.log10(follower_count + 1) / 6, 1.0)  # Cap at 1M followers
        
        # Track count score (sweet spot around 20-50 tracks)
        optimal_length = 30
        length_score = 1.0 - abs(track_count - optimal_length) / optimal_length
        length_score = max(0.2, min(1.0, length_score))
        
        return (follower_score * 0.7 + length_score * 0.3)

    def _calculate_transition_quality(self, curr_features: Dict, next_features: Dict) -> float:
        """Calculate transition quality score based on music theory"""
        # Smooth transitions are generally better
        tempo_score = 1.0 - min(abs(curr_features['tempo'] - next_features['tempo']) / 50, 1.0)
        energy_score = 1.0 - abs(curr_features['energy'] - next_features['energy'])
        valence_score = 1.0 - abs(curr_features['valence'] - next_features['valence'])
        
        # Key compatibility
        key_score = self._calculate_key_compatibility(curr_features['key'], next_features['key'])
        
        # Weighted combination
        return (tempo_score * 0.3 + energy_score * 0.3 + valence_score * 0.2 + key_score * 0.2)

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix for ML training"""
        logger.info("🔧 Preparing features for ML training...")
        
        # Select features for training
        feature_columns = [
            # Current song features
            'curr_tempo', 'curr_energy', 'curr_valence', 'curr_danceability', 
            'curr_acousticness', 'curr_loudness',
            
            # Next song features  
            'next_tempo', 'next_energy', 'next_valence', 'next_danceability',
            'next_acousticness', 'next_loudness',
            
            # Transition features
            'tempo_change', 'energy_change', 'valence_change',
            'tempo_change_abs', 'energy_change_abs', 'valence_change_abs',
            
            # Musical compatibility
            'key_compatibility', 'mode_match',
            
            # Context features
            'position_in_playlist', 'playlist_length'
        ]
        
        # Add category one-hot encoding
        category_dummies = pd.get_dummies(df['category'], prefix='category')
        df = pd.concat([df, category_dummies], axis=1)
        feature_columns.extend(category_dummies.columns.tolist())
        
        # Prepare feature matrix
        X = df[feature_columns].fillna(0)
        
        # Target variable: transition quality score
        y = df['transition_quality_score']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"📊 Prepared {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
        
        return X_scaled, y.values, feature_columns

    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train multiple models to predict transition quality"""
        logger.info("🏋️ Training queue optimization models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Model configurations
        model_configs = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                alpha=0.01
            ),
            'linear_regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in model_configs.items():
            logger.info(f"🤖 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Detailed metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'train_r2': train_score,
                'test_r2': test_score,
                'mse': mse,
                'mae': mae,
                'predictions': y_pred,
                'y_test': y_test
            }
            
            logger.info(f"✅ {name}: R² = {test_score:.3f}, MSE = {mse:.4f}, MAE = {mae:.4f}")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        logger.info(f"🏆 Best model: {best_model_name} (R² = {results[best_model_name]['test_r2']:.3f})")
        
        self.models = results
        return results

    def save_models(self, feature_names: List[str]):
        """Save trained models for production use"""
        logger.info("💾 Saving queue optimization models...")
        
        # Save best model
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['test_r2'])
        best_model = self.models[best_model_name]['model']
        
        model_data = {
            'model': best_model,
            'scaler': self.scaler,
            'feature_names': feature_names,
            'model_type': best_model_name,
            'r2_score': self.models[best_model_name]['test_r2'],
            'mse': self.models[best_model_name]['mse']
        }
        
        # Save to pickle file
        model_file = self.models_dir / "queue_optimizer.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save metadata
        metadata = {
            'model_type': best_model_name,
            'r2_score': self.models[best_model_name]['test_r2'],
            'feature_count': len(feature_names),
            'training_date': pd.Timestamp.now().isoformat(),
            'purpose': 'queue_optimization'
        }
        
        metadata_file = self.models_dir / "queue_model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Model saved to {model_file}")
        return model_file

    def create_evaluation_report(self):
        """Create evaluation visualizations"""
        logger.info("📊 Creating model evaluation report...")
        
        viz_dir = self.models_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Model comparison
        model_scores = {name: results['test_r2'] for name, results in self.models.items()}
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_scores.keys(), model_scores.values())
        plt.title('Queue Optimization Model Performance')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, model_scores.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "model_comparison.png")
        plt.close()
        
        # Prediction vs actual for best model
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['test_r2'])
        best_results = self.models[best_model_name]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(best_results['y_test'], best_results['predictions'], alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
        plt.xlabel('Actual Transition Quality')
        plt.ylabel('Predicted Transition Quality')
        plt.title(f'Prediction Accuracy - {best_model_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / f"{best_model_name}_predictions.png")
        plt.close()
        
        logger.info(f"📈 Evaluation reports saved to {viz_dir}/")

    async def train_complete_pipeline(self):
        """Run complete queue optimization training pipeline"""
        logger.info("🚀 Starting queue optimization model training...")
        
        try:
            # Load playlist sequences
            sequences = self.load_playlist_sequences()
            
            # Extract transitions
            transitions_df = self.extract_transitions(sequences)
            
            if len(transitions_df) < 100:
                logger.warning(f"⚠️ Only {len(transitions_df)} transitions found. Consider collecting more data.")
            
            # Prepare features
            X, y, feature_names = self.prepare_features(transitions_df)
            
            # Train models
            results = self.train_models(X, y)
            
            # Save models
            model_file = self.save_models(feature_names)
            
            # Create evaluation report
            self.create_evaluation_report()
            
            logger.info("🎉 Queue optimization training complete!")
            
            return {
                'model_file': model_file,
                'results': results,
                'feature_count': len(feature_names),
                'transition_count': len(transitions_df)
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            raise

async def main():
    """Run queue optimization model training"""
    trainer = QueueOptimizationTrainer()
    
    try:
        results = await trainer.train_complete_pipeline()
        
        print(f"\n🎉 Queue optimization training complete!")
        print(f"📁 Model saved: {results['model_file']}")
        print(f"📊 Features used: {results['feature_count']}")
        print(f"🔄 Transitions analyzed: {results['transition_count']}")
        
        best_model = max(results['results'].keys(), key=lambda x: results['results'][x]['test_r2'])
        r2_score = results['results'][best_model]['test_r2']
        print(f"🏆 Best model: {best_model} (R² = {r2_score:.3f})")
        
        print(f"\n✅ Ready for deployment!")
        print(f"Next step: python scripts/deploy_models.py")
        
    except FileNotFoundError:
        print(f"\n❌ Training data not found!")
        print(f"First run: python scripts/collect_training_data.py")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
