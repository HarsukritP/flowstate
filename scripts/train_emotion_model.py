#!/usr/bin/env python3
"""
FlowState Emotion Classification Model Training
Trains ML models to classify emotional content from audio features
"""

import asyncio
import json
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Add parent directory to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionModelTrainer:
    """Train emotion classification models from audio features"""
    
    def __init__(self):
        """Initialize trainer"""
        self.data_dir = Path("training_data")
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        
        logger.info("🧠 Emotion model trainer initialized")

    def load_training_data(self) -> pd.DataFrame:
        """Load and prepare training data"""
        logger.info("📊 Loading training data...")
        
        dataset_file = self.data_dir / "complete_dataset.json"
        if not dataset_file.exists():
            raise FileNotFoundError("Training dataset not found. Run collect_training_data.py first.")
        
        with open(dataset_file, 'r') as f:
            raw_data = json.load(f)
        
        # Convert to DataFrame
        data_rows = []
        for song in raw_data:
            row = {
                'song_id': song['spotify_id'],
                'title': song['title'],
                'artist': song['artist'],
                'popularity': song.get('popularity', 0),
            }
            
            # Add Spotify features
            spotify_features = song.get('spotify_features', {})
            for key, value in spotify_features.items():
                if isinstance(value, (int, float)) and key != 'id':
                    row[f'spotify_{key}'] = value
            
            # Add librosa features
            librosa_features = song.get('librosa_features', {})
            for key, value in librosa_features.items():
                if isinstance(value, (int, float)):
                    row[f'librosa_{key}'] = value
            
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        logger.info(f"✅ Loaded {len(df)} songs with {len(df.columns)} features")
        
        return df

    def create_emotion_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create emotion labels based on audio features (supervised approach)"""
        logger.info("🏷️ Creating emotion labels from audio features...")
        
        # Define emotion classification rules based on valence and energy
        def classify_emotion(row):
            valence = row.get('spotify_valence', 0.5)
            energy = row.get('spotify_energy', 0.5)
            danceability = row.get('spotify_danceability', 0.5)
            acousticness = row.get('spotify_acousticness', 0.5)
            tempo = row.get('spotify_tempo', 120)
            
            # High valence, high energy
            if valence > 0.7 and energy > 0.7:
                return 'joyful'
            # High valence, low energy
            elif valence > 0.6 and energy < 0.4:
                return 'peaceful'
            # Low valence, high energy
            elif valence < 0.4 and energy > 0.7:
                return 'angry'
            # Low valence, low energy
            elif valence < 0.4 and energy < 0.4:
                return 'melancholic'
            # High energy, high danceability
            elif energy > 0.8 and danceability > 0.8:
                return 'energetic'
            # High acousticness, low energy
            elif acousticness > 0.7 and energy < 0.5:
                return 'contemplative'
            # Fast tempo, high energy
            elif tempo > 140 and energy > 0.6:
                return 'intense'
            # Slow tempo, high acousticness
            elif tempo < 80 and acousticness > 0.5:
                return 'serene'
            else:
                return 'neutral'
        
        df['emotion_label'] = df.apply(classify_emotion, axis=1)
        
        # Show emotion distribution
        emotion_counts = df['emotion_label'].value_counts()
        logger.info(f"📊 Emotion distribution:\n{emotion_counts}")
        
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and labels for training"""
        logger.info("🔧 Preparing features for training...")
        
        # Select relevant features
        feature_columns = [col for col in df.columns if 
                          col.startswith(('spotify_', 'librosa_')) and 
                          df[col].dtype in ['float64', 'int64']]
        
        # Remove features with too many NaN values
        valid_features = []
        for col in feature_columns:
            if df[col].notna().sum() > len(df) * 0.7:  # At least 70% non-null
                valid_features.append(col)
        
        logger.info(f"📋 Using {len(valid_features)} features for training")
        
        # Create feature matrix
        X = df[valid_features].fillna(df[valid_features].median())
        y = df['emotion_label']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_scaled, y_encoded, valid_features

    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train multiple emotion classification models"""
        logger.info("🏋️ Training emotion classification models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models to train
        model_configs = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                random_state=42,
                probability=True
            )
        }
        
        results = {}
        
        for name, model in model_configs.items():
            logger.info(f"🤖 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Predictions for detailed evaluation
            y_pred = model.predict(X_test)
            
            results[name] = {
                'model': model,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'predictions': y_pred,
                'y_test': y_test
            }
            
            logger.info(f"✅ {name}: Test Accuracy = {test_score:.3f}, CV = {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        best_model = results[best_model_name]['model']
        
        logger.info(f"🏆 Best model: {best_model_name} (Accuracy: {results[best_model_name]['test_accuracy']:.3f})")
        
        self.models = results
        
        return results

    def save_models(self, feature_names: List[str]):
        """Save trained models and preprocessing objects"""
        logger.info("💾 Saving trained models...")
        
        # Save best model
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['test_accuracy'])
        best_model = self.models[best_model_name]['model']
        
        model_data = {
            'model': best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': feature_names,
            'model_type': best_model_name,
            'accuracy': self.models[best_model_name]['test_accuracy'],
            'emotion_classes': list(self.label_encoder.classes_)
        }
        
        # Save to pickle file
        model_file = self.models_dir / "emotion_classifier.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save metadata
        metadata = {
            'model_type': best_model_name,
            'accuracy': self.models[best_model_name]['test_accuracy'],
            'emotion_classes': list(self.label_encoder.classes_),
            'feature_count': len(feature_names),
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = self.models_dir / "emotion_model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Models saved to {self.models_dir}/")
        
        return model_file

    def create_evaluation_report(self):
        """Create detailed evaluation report with visualizations"""
        logger.info("📊 Creating evaluation report...")
        
        # Create visualizations directory
        viz_dir = self.models_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        for model_name, results in self.models.items():
            y_test = results['y_test']
            y_pred = results['predictions']
            
            # Classification report
            report = classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            # Save report
            report_file = viz_dir / f"{model_name}_classification_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d',
                xticklabels=self.label_encoder.classes_,
                yticklabels=self.label_encoder.classes_,
                cmap='Blues'
            )
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(viz_dir / f"{model_name}_confusion_matrix.png")
            plt.close()
        
        logger.info(f"📈 Evaluation reports saved to {viz_dir}/")

    async def train_complete_pipeline(self):
        """Run complete training pipeline"""
        logger.info("🚀 Starting complete emotion model training pipeline...")
        
        try:
            # Load data
            df = self.load_training_data()
            
            # Create emotion labels
            df = self.create_emotion_labels(df)
            
            # Prepare features
            X, y, feature_names = self.prepare_features(df)
            
            # Train models
            results = self.train_models(X, y)
            
            # Save models
            model_file = self.save_models(feature_names)
            
            # Create evaluation report
            self.create_evaluation_report()
            
            logger.info("🎉 Emotion model training complete!")
            
            return {
                'model_file': model_file,
                'results': results,
                'feature_count': len(feature_names),
                'emotion_classes': list(self.label_encoder.classes_)
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            raise

async def main():
    """Run emotion model training"""
    trainer = EmotionModelTrainer()
    
    try:
        results = await trainer.train_complete_pipeline()
        
        print(f"\n🎉 Emotion model training complete!")
        print(f"📁 Model saved: {results['model_file']}")
        print(f"🎯 Emotion classes: {results['emotion_classes']}")
        print(f"📊 Features used: {results['feature_count']}")
        
        best_model = max(results['results'].keys(), key=lambda x: results['results'][x]['test_accuracy'])
        accuracy = results['results'][best_model]['test_accuracy']
        print(f"🏆 Best model: {best_model} (Accuracy: {accuracy:.3f})")
        
        print(f"\n✅ Ready for deployment!")
        print(f"Next step: python scripts/deploy_models.py")
        
    except FileNotFoundError:
        print(f"\n❌ Training data not found!")
        print(f"First run: python scripts/collect_training_data.py")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
