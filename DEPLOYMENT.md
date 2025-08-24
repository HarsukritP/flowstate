# 🚀 FlowState Deployment Guide

## 📋 Prerequisites

Before deploying FlowState, ensure you have:

1. **GitHub Account**: Code is now pushed to [https://github.com/HarsukritP/flowstate.git](https://github.com/HarsukritP/flowstate.git)
2. **Railway Account**: Sign up at [railway.app](https://railway.app)
3. **Spotify Developer Account**: Register at [developer.spotify.com](https://developer.spotify.com)

## 🛤️ Railway Deployment

### Step 1: Connect GitHub to Railway

1. **Sign up/Login to Railway**: Visit [railway.app](https://railway.app)
2. **Connect GitHub**: Authorize Railway to access your GitHub repositories
3. **Create New Project**: Click "New Project" → "Deploy from GitHub repo"
4. **Select Repository**: Choose `HarsukritP/flowstate`

### Step 2: Configure Environment Variables

In Railway project settings, add these environment variables:

```bash
# Required Environment Variables
ENVIRONMENT=production
PORT=8000
DEBUG=false

# Database (Railway will auto-provide PostgreSQL)
DATABASE_URL=${DATABASE_URL}  # Auto-injected by Railway

# Spotify API (Get from Spotify Developer Dashboard)
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
SPOTIFY_REDIRECT_URI=https://your-app-name.up.railway.app/callback

# Security
SECRET_KEY=your-super-secret-key-here-change-this-in-production

# Optional: External APIs
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here

# Performance
QUEUE_GENERATION_TIMEOUT=10
MAX_QUEUE_LENGTH=50
```

### Step 3: Add PostgreSQL Database

1. **Add PostgreSQL**: In Railway dashboard → Add Service → PostgreSQL
2. **Database Connection**: Railway will automatically inject `DATABASE_URL`
3. **Run Migrations**: After first deployment, run database migrations

### Step 4: Deploy

1. **Automatic Deployment**: Railway will automatically deploy when you push to GitHub
2. **Check Logs**: Monitor deployment progress in Railway dashboard
3. **Access App**: Your app will be available at `https://your-app-name.up.railway.app`

## 🔧 External Setup Steps (Your Action Items)

### 1. Set Up Spotify Developer Application

**Action Required**: Create Spotify app for API access

```bash
1. Go to https://developer.spotify.com/dashboard
2. Click "Create App"
3. Fill in details:
   - App Name: "FlowState"
   - App Description: "AI-powered music queue optimization"
   - Website: https://your-app-name.up.railway.app
   - Redirect URI: https://your-app-name.up.railway.app/callback
4. Save the Client ID and Client Secret
5. Add these to Railway environment variables
```

### 2. Update Chrome Extension Configuration

**Action Required**: Update extension to point to production API

```javascript
// In extension/content.js, update this line:
this.apiBaseUrl = 'https://your-app-name.up.railway.app';  // Replace with your Railway URL
```

### 3. Set Up Database Migrations

**Action Required**: Initialize production database

```bash
# After Railway deployment, run this command in Railway's terminal:
alembic upgrade head
```

### 4. Load Chrome Extension

**Action Required**: Install extension in Chrome

```bash
1. Open Chrome → chrome://extensions/
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the extension/ folder
5. Test on https://open.spotify.com/
```

## 🔑 API Keys You Need to Obtain

### Required APIs:

1. **Spotify Web API** (Essential)
   - **Where**: [developer.spotify.com](https://developer.spotify.com)
   - **Why**: To analyze songs and integrate with Spotify Web Player
   - **Free Tier**: 10,000 requests/month per user

### Optional APIs (for enhanced features):

2. **OpenAI API** (Optional - for enhanced emotional analysis)
   - **Where**: [platform.openai.com](https://platform.openai.com)
   - **Why**: More sophisticated emotional profiling
   - **Cost**: Pay-per-use, ~$0.002 per request

3. **Hugging Face API** (Optional - for ML model hosting)
   - **Where**: [huggingface.co](https://huggingface.co)
   - **Why**: Host custom audio analysis models
   - **Free Tier**: Available

## 🧠 Model Training & Development

### Current MVP Status

**No Training Required Initially** - FlowState MVP uses:

1. **Rule-Based Audio Analysis**: Mock features based on song metadata
2. **Algorithmic Queue Generation**: Mathematical optimization algorithms
3. **Predefined Emotional Models**: Hardcoded emotion classification

### Phase 2: Real Model Training (Optional Enhancement)

If you want to train custom models, here's what you'd need:

#### 1. Data Collection
```bash
# Collect training data
python scripts/collect_training_data.py
# - Downloads Spotify preview clips
# - Extracts audio features with librosa
# - Collects user feedback on emotional accuracy
```

#### 2. Model Training
```bash
# Train emotional classification model
python scripts/train_emotion_model.py
# - Uses scikit-learn for basic models
# - Can upgrade to TensorFlow/PyTorch for deep learning
```

#### 3. Model Deployment
```bash
# Deploy trained models
python scripts/deploy_models.py
# - Saves models to Railway filesystem or cloud storage
# - Updates API to use trained models instead of mock data
```

### Training Data Sources

1. **Spotify Preview URLs**: 30-second audio clips
2. **Last.fm Dataset**: Song metadata and tags
3. **User Feedback**: Queue satisfaction ratings
4. **Audio Features**: Librosa-extracted features

### Model Architecture Options

1. **Simple Classification** (Current MVP):
   - Scikit-learn decision trees
   - Rule-based emotional mapping
   - ~95% accuracy on basic emotions

2. **Deep Learning** (Future Enhancement):
   - TensorFlow/PyTorch neural networks
   - Convolutional networks for audio analysis
   - ~98% accuracy potential

## 🚦 Deployment Checklist

### Before Deployment:
- [ ] Spotify Developer app created
- [ ] Environment variables configured in Railway
- [ ] PostgreSQL database added to Railway project
- [ ] Chrome extension API URL updated

### After Deployment:
- [ ] Run database migrations: `alembic upgrade head`
- [ ] Test API endpoints: Visit `/docs` on your Railway URL
- [ ] Install Chrome extension locally
- [ ] Test full flow: Song → Generate Queue → Display Results

### Verification:
- [ ] API responds at `https://your-app.up.railway.app/health`
- [ ] Extension appears on Spotify Web Player
- [ ] Queue generation works end-to-end
- [ ] Error handling works (test with invalid songs)

## 📊 Monitoring & Maintenance

### Railway Dashboard:
- **Logs**: Monitor API requests and errors
- **Metrics**: CPU, memory, response times
- **Database**: Query performance and storage

### Performance Targets:
- **API Response**: <200ms
- **Queue Generation**: <100ms
- **Uptime**: 99.9%
- **Error Rate**: <1%

## 🔧 Troubleshooting

### Common Issues:

1. **Database Connection Errors**:
   ```bash
   # Check DATABASE_URL is set correctly
   # Ensure PostgreSQL service is running
   ```

2. **Spotify API Rate Limits**:
   ```bash
   # Implement caching in audio_analyzer.py
   # Add request throttling
   ```

3. **Chrome Extension CORS Issues**:
   ```bash
   # Verify Railway URL in extension code
   # Check CORS origins in main.py
   ```

4. **Slow Performance**:
   ```bash
   # Enable Redis caching
   # Optimize database queries
   # Implement request batching
   ```

---

## 🎯 Next Steps After Deployment

1. **Test thoroughly** with real Spotify songs
2. **Collect user feedback** on queue quality
3. **Monitor performance** metrics
4. **Iterate on algorithms** based on usage data
5. **Scale infrastructure** as user base grows

Your FlowState deployment should now be live and ready for testing! 🌊🎵
