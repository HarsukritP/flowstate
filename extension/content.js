/**
 * FlowState Content Script - Bulletproof Implementation
 * Integrates with Spotify Web Player using robust detection methods
 */

// Prevent multiple injections
if (window.flowStateInjected) {
    console.log('🌊 FlowState already injected, skipping...');
} else {
    window.flowStateInjected = true;
    console.log('🌊 FlowState extension loaded - bulletproof version');

    class SpotifyTrackDetector {
        constructor() {
            this.currentTrack = null;
            this.observers = [];
            this.detectionStrategies = [];
            this.initializeStrategies();
        }

        initializeStrategies() {
            // Strategy 1: Spotify Web Playback SDK (if available)
            this.detectionStrategies.push({
                name: 'WebPlaybackSDK',
                detect: () => this.detectViaWebPlaybackSDK(),
                priority: 1
            });

            // Strategy 2: Now Playing Widget Data Attributes
            this.detectionStrategies.push({
                name: 'DataAttributes',
                detect: () => this.detectViaDataAttributes(),
                priority: 2
            });

            // Strategy 3: Document Title Analysis
            this.detectionStrategies.push({
                name: 'DocumentTitle',
                detect: () => this.detectViaDocumentTitle(),
                priority: 3
            });

            // Strategy 4: Structured DOM Analysis
            this.detectionStrategies.push({
                name: 'StructuredDOM',
                detect: () => this.detectViaStructuredDOM(),
                priority: 4
            });

            // Strategy 5: URL Pattern Analysis
            this.detectionStrategies.push({
                name: 'URLPattern',
                detect: () => this.detectViaURLPattern(),
                priority: 5
            });

            // Sort by priority
            this.detectionStrategies.sort((a, b) => a.priority - b.priority);
        }

        async detectCurrentTrack() {
            console.log('🔍 Starting bulletproof track detection...');
            
            for (const strategy of this.detectionStrategies) {
                try {
                    console.log(`🎯 Trying detection strategy: ${strategy.name}`);
                    const result = await strategy.detect();
                    
                    if (this.validateTrackData(result)) {
                        console.log(`✅ Successfully detected track via ${strategy.name}:`, result);
                        this.currentTrack = result;
                        return result;
                    } else {
                        console.log(`❌ Strategy ${strategy.name} returned invalid data:`, result);
                    }
                } catch (error) {
                    console.warn(`⚠️ Strategy ${strategy.name} failed:`, error);
                }
            }

            console.warn('❌ All detection strategies failed');
            return null;
        }

        async detectViaWebPlaybackSDK() {
            // Check if Spotify Web Playback SDK is available
            if (window.Spotify && window.Spotify.Player) {
                console.log('🎵 Spotify Web Playback SDK detected');
                
                // This would require proper SDK integration, which needs user auth
                // For now, we'll check if there's any player state available
                const playerState = window.Spotify.Player?.getCurrentState?.();
                if (playerState && playerState.track_window?.current_track) {
                    const track = playerState.track_window.current_track;
                    return {
                        title: track.name,
                        artist: track.artists.map(a => a.name).join(', '),
                        spotify_id: track.id,
                        source: 'WebPlaybackSDK'
                    };
                }
            }
            return null;
        }

        async detectViaDataAttributes() {
            // Look for elements with data attributes that contain track info
            const selectors = [
                '[data-testid="now-playing-widget"]',
                '[data-testid="track-info"]',
                '[data-track-name]',
                '[data-artist-name]',
                '[aria-label*="now playing"]'
            ];

            for (const selector of selectors) {
                const element = document.querySelector(selector);
                if (element) {
                    console.log(`🔍 Found element with selector: ${selector}`);
                    
                    // Extract from data attributes
                    const trackName = element.dataset.trackName || 
                                    element.getAttribute('data-track-name') ||
                                    element.getAttribute('aria-label');
                    
                    const artistName = element.dataset.artistName || 
                                     element.getAttribute('data-artist-name');

                    if (trackName && artistName) {
                        return {
                            title: trackName,
                            artist: artistName,
                            spotify_id: this.extractSpotifyIdFromElement(element),
                            source: 'DataAttributes'
                        };
                    }

                    // Try to extract from structured content
                    const trackData = this.extractFromStructuredElement(element);
                    if (trackData) return trackData;
                }
            }
            return null;
        }

        async detectViaDocumentTitle() {
            const title = document.title;
            console.log('🔍 Analyzing document title:', title);
            
            // Spotify format is usually "Track Name • Artist Name | Spotify"
            const spotifyPattern = /^(.+?)\s*•\s*(.+?)\s*\|\s*Spotify/;
            const match = title.match(spotifyPattern);
            
            if (match) {
                return {
                    title: match[1].trim(),
                    artist: match[2].trim(),
                    spotify_id: this.extractSpotifyIdFromURL(),
                    source: 'DocumentTitle'
                };
            }
            return null;
        }

        async detectViaStructuredDOM() {
            const nowPlayingWidget = document.querySelector('[data-testid="now-playing-widget"]');
            if (!nowPlayingWidget) {
                console.log('❌ No now-playing widget found');
                return null;
            }

            console.log('🔍 Analyzing now-playing widget structure...');
            
            // Look for track links (these usually have href="/track/...")
            const trackLinks = nowPlayingWidget.querySelectorAll('a[href*="/track/"]');
            
            // Look for artist links (these usually have href="/artist/...")
            const artistLinks = nowPlayingWidget.querySelectorAll('a[href*="/artist/"]');
            
            let trackName = null;
            let artistName = null;
            let spotifyId = null;

            // Extract track name from track links
            if (trackLinks.length > 0) {
                trackName = trackLinks[0].textContent?.trim();
                spotifyId = this.extractIdFromHref(trackLinks[0].href, 'track');
                console.log('🎵 Found track link:', trackName, spotifyId);
            }

            // Extract artist name from artist links
            if (artistLinks.length > 0) {
                artistName = artistLinks[0].textContent?.trim();
                console.log('🎤 Found artist link:', artistName);
            }

            // If we don't have track name from links, try other methods
            if (!trackName) {
                const trackSelectors = [
                    '[data-testid="context-item-link"]',
                    'a[dir="auto"]:first-of-type',
                    '.Type__TypeElement-goli3j-0[variant="forte"]'
                ];

                for (const selector of trackSelectors) {
                    const element = nowPlayingWidget.querySelector(selector);
                    if (element?.textContent?.trim()) {
                        trackName = element.textContent.trim();
                        console.log(`🎵 Found track via selector ${selector}:`, trackName);
                        break;
                    }
                }
            }

            // If we don't have artist name from links, try other methods
            if (!artistName) {
                const artistSelectors = [
                    'a[dir="auto"]:last-of-type',
                    'span[dir="auto"]',
                    '.Type__TypeElement-goli3j-0[variant="mesto"]'
                ];

                for (const selector of artistSelectors) {
                    const element = nowPlayingWidget.querySelector(selector);
                    if (element?.textContent?.trim() && element.textContent !== trackName) {
                        artistName = element.textContent.trim();
                        console.log(`🎤 Found artist via selector ${selector}:`, artistName);
                        break;
                    }
                }
            }

            if (trackName && artistName) {
                return {
                    title: trackName,
                    artist: artistName,
                    spotify_id: spotifyId,
                    source: 'StructuredDOM'
                };
            }

            return null;
        }

        async detectViaURLPattern() {
            const url = window.location.href;
            console.log('🔍 Analyzing URL pattern:', url);
            
            // Extract track ID from URL patterns
            const patterns = [
                /\/track\/([a-zA-Z0-9]+)/,
                /\/album\/[^/]+\?highlight=spotify:track:([a-zA-Z0-9]+)/,
                /spotify:track:([a-zA-Z0-9]+)/
            ];

            let trackId = null;
            for (const pattern of patterns) {
                const match = url.match(pattern);
                if (match) {
                    trackId = match[1];
                    break;
                }
            }

            if (trackId) {
                console.log('🆔 Found track ID in URL:', trackId);
                // We have the track ID, but need to get track name and artist
                // This would typically require a Spotify API call
                // For now, return what we can extract from the page
                return this.enrichTrackDataFromPage(trackId);
            }

            return null;
        }

        async enrichTrackDataFromPage(spotifyId) {
            // Try to extract track and artist name from the page content
            const pageText = document.body.textContent;
            
            // Look for patterns in the page that might indicate track/artist
            const metaTags = document.querySelectorAll('meta[property^="og:"], meta[name^="twitter:"]');
            
            for (const meta of metaTags) {
                const property = meta.getAttribute('property') || meta.getAttribute('name');
                const content = meta.getAttribute('content');
                
                if (property?.includes('title') && content) {
                    const titleMatch = content.match(/^(.+?)\s*-\s*(.+?)$/);
                    if (titleMatch) {
                        return {
                            title: titleMatch[1].trim(),
                            artist: titleMatch[2].trim(),
                            spotify_id: spotifyId,
                            source: 'URLPattern'
                        };
                    }
                }
            }

            return null;
        }

        extractFromStructuredElement(element) {
            // Try to extract structured data from the element
            const links = element.querySelectorAll('a[href]');
            const spans = element.querySelectorAll('span, div');
            
            let trackName = null;
            let artistName = null;
            let spotifyId = null;

            // Analyze links first
            for (const link of links) {
                const href = link.href;
                const text = link.textContent?.trim();
                
                if (href.includes('/track/') && text) {
                    trackName = text;
                    spotifyId = this.extractIdFromHref(href, 'track');
                } else if (href.includes('/artist/') && text && text !== trackName) {
                    artistName = text;
                }
            }

            return trackName && artistName ? {
                title: trackName,
                artist: artistName,
                spotify_id: spotifyId,
                source: 'StructuredElement'
            } : null;
        }

        extractSpotifyIdFromElement(element) {
            // Look for Spotify ID in various attributes
            const attributes = ['data-spotify-id', 'data-track-id', 'href'];
            
            for (const attr of attributes) {
                const value = element.getAttribute(attr);
                if (value) {
                    const id = this.extractIdFromString(value);
                    if (id) return id;
                }
            }

            return this.extractSpotifyIdFromURL();
        }

        extractSpotifyIdFromURL() {
            const url = window.location.href;
            const match = url.match(/\/track\/([a-zA-Z0-9]+)/);
            return match ? match[1] : null;
        }

        extractIdFromHref(href, type) {
            const pattern = new RegExp(`/${type}/([a-zA-Z0-9]+)`);
            const match = href.match(pattern);
            return match ? match[1] : null;
        }

        extractIdFromString(str) {
            const patterns = [
                /spotify:track:([a-zA-Z0-9]+)/,
                /\/track\/([a-zA-Z0-9]+)/,
                /^([a-zA-Z0-9]{22})$/
            ];

            for (const pattern of patterns) {
                const match = str.match(pattern);
                if (match) return match[1];
            }
            return null;
        }

        validateTrackData(trackData) {
            if (!trackData) return false;
            
            const { title, artist } = trackData;
            
            // Basic validation
            if (!title || !artist) return false;
            if (typeof title !== 'string' || typeof artist !== 'string') return false;
            if (title.length === 0 || artist.length === 0) return false;
            if (title === artist) return false;
            
            // Check for common invalid values
            const invalidValues = ['undefined', 'null', '', 'N/A', '...'];
            if (invalidValues.includes(title) || invalidValues.includes(artist)) return false;
            
            // Check minimum length
            if (title.length < 1 || artist.length < 1) return false;
            
            return true;
        }

        startDetection() {
            console.log('🔄 Starting continuous track detection...');
            
            // Initial detection
            this.detectCurrentTrack();
            
            // Set up observers for DOM changes
            this.setupDOMObservers();
            
            // Set up periodic detection as fallback
            this.setupPeriodicDetection();
        }

        setupDOMObservers() {
            const targetSelectors = [
                '[data-testid="now-playing-widget"]',
                'title',
                '[data-testid="track-info"]'
            ];

            targetSelectors.forEach(selector => {
                const element = document.querySelector(selector);
                if (element) {
                    const observer = new MutationObserver(() => {
                        console.log(`🔄 DOM change detected in ${selector}`);
                        setTimeout(() => this.detectCurrentTrack(), 100);
                    });

                    observer.observe(element, {
                        childList: true,
                        subtree: true,
                        attributes: true,
                        attributeFilter: ['href', 'data-testid', 'aria-label']
                    });

                    this.observers.push(observer);
                    console.log(`✅ Observer set up for ${selector}`);
                }
            });

            // Observe title changes
            const titleObserver = new MutationObserver(() => {
                console.log('📄 Document title changed');
                setTimeout(() => this.detectCurrentTrack(), 100);
            });

            titleObserver.observe(document.querySelector('title'), {
                childList: true
            });
            this.observers.push(titleObserver);
        }

        setupPeriodicDetection() {
            // Fallback periodic detection every 5 seconds
            setInterval(() => {
                console.log('⏰ Periodic detection check...');
                this.detectCurrentTrack();
            }, 5000);
        }

        stopDetection() {
            this.observers.forEach(observer => observer.disconnect());
            this.observers = [];
        }

        getCurrentTrack() {
            return this.currentTrack;
        }
    }

    class FlowStateSpotifyIntegration {
        constructor() {
            this.apiBaseUrl = 'https://flowstate.up.railway.app';
            this.detector = new SpotifyTrackDetector();
            this.flowStateActive = false;
            this.currentQueue = [];
            
            this.init();
        }
        
        async init() {
            console.log('🎵 Initializing FlowState Spotify integration...');
            
            // Wait for Spotify interface to load
            await this.waitForSpotifyLoad();
            
            // Start track detection
            this.detector.startDetection();
            
            // Inject FlowState UI components
            this.injectFlowStateButton();
            this.injectQueuePreview();
            
            console.log('✅ FlowState integration ready');
        }
        
        async waitForSpotifyLoad() {
            return new Promise((resolve) => {
                const checkSpotify = () => {
                    const nowPlaying = document.querySelector('[data-testid="now-playing-widget"]');
                    if (nowPlaying) {
                        resolve();
                    } else {
                        setTimeout(checkSpotify, 1000);
                    }
                };
                checkSpotify();
            });
        }
        
        injectFlowStateButton() {
            // Find the now playing controls area
            const controls = document.querySelector('[data-testid="player-controls"]');
            if (!controls) {
                console.warn('❌ Could not find player controls');
                return;
            }
            
            // Create FlowState button
            const flowButton = document.createElement('button');
            flowButton.id = 'flowstate-button';
            flowButton.className = 'flowstate-btn';
            flowButton.innerHTML = '🌊 FlowState';
            flowButton.title = 'Generate AI-optimized queue';
            
            flowButton.addEventListener('click', () => this.generateFlowQueue());
            
            // Insert after the existing controls
            const controlsContainer = controls.parentElement;
            controlsContainer.appendChild(flowButton);
            
            console.log('✅ FlowState button injected');
        }
        
        injectQueuePreview() {
            // Create queue preview container
            const queuePreview = document.createElement('div');
            queuePreview.id = 'flowstate-queue-preview';
            queuePreview.className = 'flowstate-queue-preview hidden';
            queuePreview.innerHTML = `
                <div class="flowstate-header">
                    <h3>🌊 FlowState Queue</h3>
                    <button id="flowstate-close" class="flowstate-close">×</button>
                </div>
                <div class="flowstate-status">
                    <div class="flowstate-journey">Emotional Journey: <span id="journey-type">Analyzing...</span></div>
                    <div class="flowstate-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 0%"></div>
                        </div>
                        <span id="progress-text">0% Complete</span>
                    </div>
                </div>
                <div class="flowstate-queue-list" id="flowstate-queue-list">
                    <div class="loading">🎵 Generating optimal flow...</div>
                </div>
                <div class="flowstate-actions">
                    <button id="add-song-btn" class="flowstate-action-btn">+ Add Song</button>
                    <button id="reoptimize-btn" class="flowstate-action-btn">⚡ Re-optimize</button>
                </div>
            `;
            
            // Insert into main content area
            const mainContent = document.querySelector('[data-testid="main"]') || document.body;
            mainContent.appendChild(queuePreview);
            
            // Set up close button
            document.getElementById('flowstate-close').addEventListener('click', () => {
                this.hideQueuePreview();
            });
            
            document.getElementById('reoptimize-btn').addEventListener('click', () => {
                this.reoptimizeQueue();
            });
            
            console.log('✅ Queue preview injected');
        }
        
        async generateFlowQueue() {
            console.log('🌊 Generating FlowState queue...');
            
            // Get current track from detector
            const currentTrack = this.detector.getCurrentTrack();
            
            if (!currentTrack) {
                // Force detection
                console.log('🔄 No current track, forcing detection...');
                const detectedTrack = await this.detector.detectCurrentTrack();
                
                if (!detectedTrack) {
                    alert('🎵 Please play a song first to generate a FlowState queue');
                    return;
                }
            }
            
            const trackToUse = currentTrack || await this.detector.detectCurrentTrack();
            console.log('🎵 Using track for queue generation:', trackToUse);
            
            this.showQueuePreview();
            this.flowStateActive = true;
            
            try {
                const response = await fetch(`${this.apiBaseUrl}/generate-queue`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        seed_song: {
                            id: trackToUse.spotify_id || `spotify_${Date.now()}`,
                            title: trackToUse.title,
                            artist: trackToUse.artist,
                            spotify_id: trackToUse.spotify_id
                        },
                        queue_length: 10,
                        emotional_journey: 'gradual_flow'
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`API request failed: ${response.status}`);
                }
                
                const queueData = await response.json();
                this.currentQueue = queueData.queue;
                
                this.displayQueue(queueData);
                this.updateJourneyInfo(queueData);
                
                console.log('✅ FlowState queue generated:', queueData);
                
            } catch (error) {
                console.error('❌ Failed to generate queue:', error);
                this.showError('Failed to generate FlowState queue. Make sure the backend is running.');
            }
        }
        
        async reoptimizeQueue() {
            if (!this.currentQueue.length) return;
            
            console.log('🔄 Re-optimizing FlowState queue...');
            
            const currentTrack = this.detector.getCurrentTrack();
            if (!currentTrack) {
                console.warn('⚠️ No current track for re-optimization');
                return;
            }
            
            try {
                const response = await fetch(`${this.apiBaseUrl}/reoptimize-queue`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        seed_song: {
                            id: currentTrack.spotify_id || `spotify_${Date.now()}`,
                            title: currentTrack.title,
                            artist: currentTrack.artist,
                            spotify_id: currentTrack.spotify_id
                        },
                        current_queue: this.currentQueue,
                        queue_length: this.currentQueue.length,
                        emotional_journey: 'gradual_flow'
                    })
                });
                
                const queueData = await response.json();
                this.currentQueue = queueData.queue;
                
                this.displayQueue(queueData);
                console.log('✅ Queue re-optimized');
                
            } catch (error) {
                console.error('❌ Failed to re-optimize queue:', error);
            }
        }
        
        displayQueue(queueData) {
            const queueList = document.getElementById('flowstate-queue-list');
            if (!queueList) return;
            
            queueList.innerHTML = '';
            
            queueData.queue.forEach((song, index) => {
                const songElement = document.createElement('div');
                songElement.className = 'flowstate-song-item';
                songElement.innerHTML = `
                    <div class="song-info">
                        <div class="song-title">${song.title}</div>
                        <div class="song-artist">${song.artist}</div>
                    </div>
                    <div class="emotional-transition">
                        ${this.getEmotionalTransition(queueData, index)}
                    </div>
                `;
                
                if (index === 0) {
                    songElement.classList.add('current-song');
                    songElement.insertAdjacentHTML('afterbegin', '<div class="playing-indicator">▶</div>');
                }
                
                queueList.appendChild(songElement);
            });
        }
        
        getEmotionalTransition(queueData, index) {
            if (index < queueData.emotional_journey.length) {
                const emotion = queueData.emotional_journey[index];
                return `[${emotion.primary_emotion}]`;
            }
            return '[analyzing...]';
        }
        
        updateJourneyInfo(queueData) {
            const journeyType = document.getElementById('journey-type');
            if (journeyType) {
                journeyType.textContent = 'Gradual Flow';
            }
            
            // Update flow score as progress
            const progressFill = document.querySelector('.progress-fill');
            const progressText = document.getElementById('progress-text');
            if (progressFill && progressText) {
                const percentage = Math.round(queueData.flow_score * 100);
                progressFill.style.width = `${percentage}%`;
                progressText.textContent = `Flow Quality: ${percentage}%`;
            }
        }
        
        showQueuePreview() {
            const preview = document.getElementById('flowstate-queue-preview');
            if (preview) {
                preview.classList.remove('hidden');
            }
        }
        
        hideQueuePreview() {
            const preview = document.getElementById('flowstate-queue-preview');
            if (preview) {
                preview.classList.add('hidden');
            }
            this.flowStateActive = false;
        }
        
        showError(message) {
            const queueList = document.getElementById('flowstate-queue-list');
            if (queueList) {
                queueList.innerHTML = `<div class="error">❌ ${message}</div>`;
            }
        }
    }

    // Initialize FlowState integration when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            new FlowStateSpotifyIntegration();
        });
    } else {
        new FlowStateSpotifyIntegration();
    }

} // End of flowStateInjected check
