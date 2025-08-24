# 🌊 FlowState - AI-Powered Emotional Music Queue Optimization

FlowState creates emotionally optimized song queues that transport users through seamless musical journeys. When you click any song, FlowState instantly analyzes its emotional DNA and generates a perfect flowing queue.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Development Server
```bash
python start.py
```

### 3. Test the API
Visit: http://localhost:8000/docs

## 📖 API Endpoints

### Core Endpoints

- **POST `/analyze-song`** - Analyze emotional/musical characteristics
- **POST `/generate-queue`** - Generate optimized queue from seed song  
- **POST `/reoptimize-queue`** - Efficiently re-optimize existing queue
- **GET `/song-compatibility/{id1}/{id2}`** - Calculate song compatibility

### Example Usage

```python
import requests

# Analyze a song
response = requests.post("http://localhost:8000/analyze-song", json={
    "id": "song_001",
    "title": "Breathe Me",
    "artist": "Sia"
})

# Generate a queue
response = requests.post("http://localhost:8000/generate-queue", json={
    "seed_song": {
        "id": "song_001", 
        "title": "Breathe Me", 
        "artist": "Sia"
    },
    "queue_length": 10,
    "emotional_journey": "gradual_flow"
})
```

## 🎯 Emotional Journey Types

- **`gradual_flow`** - Smooth emotional progression
- **`maintain_vibe`** - Keep similar energy throughout  
- **`adventure_mode`** - Encourage variety and contrast
- **`wind_down`** - Gradually reduce energy for relaxation
- **`pump_up`** - Build energy for motivation
- **`meditative`** - Calm, consistent, peaceful songs

## 🏗️ Architecture

```
FlowState/
├── main.py              # FastAPI application entry point
├── src/
│   ├── models.py        # Pydantic data models
│   ├── audio_analyzer.py # Musical feature extraction  
│   └── queue_optimizer.py # Queue generation algorithms
├── requirements.txt     # Python dependencies
└── start.py            # Development server launcher
```

## 🧠 How It Works

### 1. Audio Analysis
- Extracts tempo, key, energy, emotional valence
- Calculates emotional arousal and dominance  
- Creates musical compatibility matrices

### 2. Queue Optimization
- Uses graph algorithms for smooth transitions
- Considers emotional journey preferences
- Optimizes for flow while maintaining variety

### 3. Real-time Re-optimization  
- Efficient local optimization when songs added
- Maintains overall queue flow quality
- Sub-100ms response times

## 🎵 Sample Songs Available

The MVP includes 20 sample songs across different emotional categories:

- **Melancholic**: "Breathe Me" (Sia), "Mad World" (Gary Jules)
- **Uplifting**: "Shake It Off" (Taylor Swift), "Happy" (Pharrell)  
- **Meditative**: "Weightless" (Marconi Union), "Clair de Lune" (Debussy)

## 📊 Performance Goals

- ⚡ Queue generation: <100ms
- 🎯 Flow score: >0.85 average
- 🔄 Re-optimization: <50ms
- 📈 Skip rate reduction: 40%+ vs shuffle

## 🔧 Development Status

**Current MVP Phase**: Core audio analysis and queue generation

✅ **Completed**:
- FastAPI backend with async endpoints
- Audio feature extraction pipeline  
- Emotional analysis engine
- Queue optimization algorithms
- Song compatibility scoring

🚧 **In Progress**:
- Chrome extension for Spotify integration
- Real-time WebSocket updates
- Performance optimization

📋 **Next Steps**:
- Spotify Web API integration
- Chrome extension deployment
- User testing and feedback

## 🌐 Future Integrations

- **Chrome Extension** - Universal music platform support
- **Spotify Web API** - Real song analysis and playback
- **Apple Music** - Cross-platform compatibility  
- **YouTube Music** - Additional streaming service

## 📝 License

MIT License - Feel free to use for personal and commercial projects.

---

**Built with ❤️ for music lovers who want their playlists to flow like emotional journeys**
