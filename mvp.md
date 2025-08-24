# FlowState MVP - First Steps Priority

## MVP Goal
Build a functional Chrome extension that generates emotionally optimized queues for Spotify Web Player users.

## MVP Scope (Weeks 1-4)
**Core Feature**: User clicks song → AI generates perfect 10-song queue

## Priority 1: Audio Analysis Foundation (Week 1)
### Immediate Tasks
1. **Set up development environment**
   ```bash
   mkdir flowstate-mvp
   cd flowstate-mvp
   npm init -y
   pip install fastapi librosa numpy pandas scikit-learn
   ```

2. **Create basic audio feature extraction**
   - Extract tempo, key, energy from audio files
   - Calculate emotional valence/arousal scores
   - Build song feature vector representation

3. **Sample data collection**
   - Gather 100-200 songs with manual emotional labels
   - Create training dataset for emotional analysis

### Success Criteria
- Extract meaningful audio features from songs
- Generate consistent emotional scores
- Process songs in <2 seconds each

## Priority 2: Queue Generation Algorithm (Week 2)
### Immediate Tasks
1. **Build compatibility scoring system**
   ```python
   def calculate_song_compatibility(song_a, song_b):
       # Tempo compatibility (max 20 BPM difference)
       # Key compatibility (circle of fifths)
       # Energy level smoothness
       # Emotional flow continuity
   ```

2. **Implement basic queue optimization**
   - Use greedy algorithm for initial MVP
   - Focus on smooth transitions over global optimization
   - Generate 10-song queues from single seed song

3. **Create simple API endpoint**
   ```python
   @app.post("/generate-queue")
   async def generate_queue(seed_song: Song):
       features = extract_features(seed_song)
       queue = optimize_queue(features, available_songs)
       return {"queue": queue}
   ```

### Success Criteria
- Generate reasonable 10-song queues
- Ensure no jarring transitions
- API responds in <2 seconds

## Priority 3: Spotify Integration (Week 3)
### Immediate Tasks
1. **Build Chrome extension skeleton**
   ```javascript
   // manifest.json for Spotify integration
   // content script for Spotify Web Player
   // popup UI for FlowState controls
   ```

2. **Spotify Web Player integration**
   - Detect current song playing
   - Extract song metadata (title, artist, Spotify ID)
   - Inject FlowState button into Spotify UI

3. **Basic queue application**
   - Create Spotify playlist with generated queue
   - Replace current queue with optimized songs
   - Handle basic error cases

### Success Criteria
- Extension loads on Spotify Web Player
- Can detect currently playing song
- Successfully creates optimized playlists

## Priority 4: Dynamic Re-optimization (Week 4)
### Immediate Tasks
1. **Efficient queue modification**
   ```python
   def reoptimize_queue(current_queue, new_song, insertion_point):
       # Smart insertion without full rebuild
       # Local optimization around insertion point
       # Maintain overall flow quality
   ```

2. **Real-time queue updates**
   - WebSocket connection for live updates
   - Handle user queue modifications
   - Efficiently recalculate transitions

3. **Performance optimization**
   - Cache common song combinations
   - Pre-compute compatibility matrices
   - Optimize for <100ms response times

### Success Criteria
- Queue re-optimization in <100ms
- Smooth integration of new songs
- No noticeable performance degradation

## MVP Non-Goals (Skip for Now)
- ❌ Multiple platform support (focus on Spotify only)
- ❌ Advanced ML models (use simple algorithms first)
- ❌ User accounts/personalization (anonymous usage)
- ❌ Mobile app (web extension only)
- ❌ Social features (individual experience only)

## Technical Stack for MVP
- **Backend**: FastAPI, Python
- **Audio Processing**: Librosa, NumPy
- **Database**: SQLite (simple file-based)
- **Frontend**: Vanilla JavaScript (Chrome Extension)
- **ML**: Scikit-learn (simple models)

## Week 1 Specific Tasks
1. Set up FastAPI server with basic audio analysis endpoint
2. Install Chrome extension development environment
3. Create basic audio feature extraction for 10 test songs
4. Build simple song similarity calculation
5. Test basic queue generation with hardcoded song database

## Success Definition for MVP
"A user can click on any song in Spotify Web Player, click the FlowState button, and get a 10-song playlist that flows smoothly and feels emotionally coherent, with the ability to add songs and have the queue intelligently adjust."
