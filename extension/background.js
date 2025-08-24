/**
 * FlowState Background Service Worker
 * Handles extension lifecycle and communication between components
 */

console.log('🌊 FlowState background service worker loaded');

// Extension installation and updates
chrome.runtime.onInstalled.addListener((details) => {
    if (details.reason === 'install') {
        console.log('🎉 FlowState extension installed');
        
        // Set default settings
        chrome.storage.sync.set({
            flowstate_settings: {
                journey: 'gradual_flow',
                queueLength: 10,
                flowSensitivity: 6,
                autoOptimize: true,
                notificationsEnabled: true
            }
        });
        
        // Open welcome page
        chrome.tabs.create({
            url: 'https://open.spotify.com/browse/featured'
        });
        
    } else if (details.reason === 'update') {
        console.log('🔄 FlowState extension updated');
    }
    
    // Create context menu for Spotify pages (only once on install/update)
    try {
        chrome.contextMenus.create({
            id: 'flowstate-generate',
            title: '🌊 Generate FlowState Queue',
            contexts: ['page'],
            documentUrlPatterns: ['https://open.spotify.com/*']
        });
        
        chrome.contextMenus.create({
            id: 'flowstate-settings',
            title: '⚙️ FlowState Settings',
            contexts: ['page'],
            documentUrlPatterns: ['https://open.spotify.com/*']
        });
    } catch (error) {
        console.log('Context menu creation skipped:', error.message);
    }
});

// Handle messages from content scripts and popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('📨 Background received message:', message);
    
    switch (message.type) {
        case 'QUEUE_GENERATED':
            handleQueueGenerated(message.data);
            break;
            
        case 'SONG_CHANGED':
            handleSongChanged(message.data);
            break;
            
        case 'ERROR_OCCURRED':
            handleError(message.data);
            break;
            
        case 'GET_SETTINGS':
            getSettings(sendResponse);
            return true; // Keep message channel open for async response
            
        case 'UPDATE_STATS':
            updateStats(message.data);
            break;
            
        default:
            console.warn('Unknown message type:', message.type);
    }
});

// Handle tab updates (when user navigates to Spotify)
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url?.includes('open.spotify.com')) {
        console.log('🎵 Spotify tab detected');
        
        // Inject content script if needed
        chrome.scripting.executeScript({
            target: { tabId: tabId },
            files: ['content.js']
        }).catch(err => {
            // Script already injected or failed
            console.log('Content script injection result:', err.message);
        });
    }
});

// Background functions
function handleQueueGenerated(data) {
    console.log('✅ Queue generated:', data);
    
    // Update badge with queue length
    chrome.action.setBadgeText({
        text: data.queue?.length?.toString() || ''
    });
    
    chrome.action.setBadgeBackgroundColor({
        color: '#6366f1'
    });
    
    // Show notification if enabled
    showNotification('FlowState Queue Ready', {
        message: `Generated ${data.queue?.length} songs with ${Math.round(data.flow_score * 100)}% flow quality`,
        iconUrl: 'icons/icon-48.png'
    });
}

function handleSongChanged(data) {
    console.log('🎵 Song changed:', data);
    
    // Update extension title with current song
    chrome.action.setTitle({
        title: `FlowState - Now Playing: ${data.title} by ${data.artist}`
    });
}

function handleError(error) {
    console.error('❌ Error reported:', error);
    
    // Show error notification
    showNotification('FlowState Error', {
        message: error.message || 'An error occurred',
        iconUrl: 'icons/icon-48.png',
        type: 'basic'
    });
    
    // Reset badge
    chrome.action.setBadgeText({ text: '' });
}

function getSettings(sendResponse) {
    chrome.storage.sync.get(['flowstate_settings'], (result) => {
        sendResponse(result.flowstate_settings || {});
    });
}

function updateStats(stats) {
    console.log('📊 Stats updated:', stats);
    
    // Store stats for popup display
    chrome.storage.local.set({
        flowstate_stats: {
            ...stats,
            lastUpdated: Date.now()
        }
    });
}

function showNotification(title, options) {
    // Check if notifications are enabled
    chrome.storage.sync.get(['flowstate_settings'], (result) => {
        const settings = result.flowstate_settings || {};
        
        if (settings.notificationsEnabled !== false) {
            chrome.notifications.create({
                type: 'basic',
                iconUrl: 'icons/icon-48.png',
                title: title,
                ...options
            });
        }
    });
}

// Periodic health check
setInterval(() => {
    // Check if FlowState backend is accessible
    fetch('https://flowstate.up.railway.app/health')
        .then(response => response.json())
        .then(data => {
            console.log('🏥 Health check passed:', data);
            
            // Update connection status
            chrome.storage.local.set({
                flowstate_connection: {
                    status: 'connected',
                    lastCheck: Date.now(),
                    serverInfo: data
                }
            });
        })
        .catch(error => {
            console.warn('⚠️ Health check failed:', error);
            
            chrome.storage.local.set({
                flowstate_connection: {
                    status: 'disconnected',
                    lastCheck: Date.now(),
                    error: error.message
                }
            });
        });
}, 30000); // Check every 30 seconds

// Context menus are created in the main onInstalled listener above

chrome.contextMenus.onClicked.addListener((info, tab) => {
    switch (info.menuItemId) {
        case 'flowstate-generate':
            // Send message to content script to generate queue
            chrome.tabs.sendMessage(tab.id, {
                type: 'GENERATE_QUEUE_FROM_CONTEXT'
            });
            break;
            
        case 'flowstate-settings':
            // Open extension popup
            chrome.action.openPopup();
            break;
    }
});

// Handle keyboard shortcuts
chrome.commands.onCommand.addListener((command) => {
    console.log('⌨️ Keyboard command:', command);
    
    switch (command) {
        case 'generate-queue':
            // Find active Spotify tab and generate queue
            chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                const activeTab = tabs[0];
                if (activeTab.url?.includes('open.spotify.com')) {
                    chrome.tabs.sendMessage(activeTab.id, {
                        type: 'GENERATE_QUEUE_SHORTCUT'
                    });
                }
            });
            break;
            
        case 'toggle-flowstate':
            // Toggle FlowState panel
            chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                const activeTab = tabs[0];
                if (activeTab.url?.includes('open.spotify.com')) {
                    chrome.tabs.sendMessage(activeTab.id, {
                        type: 'TOGGLE_FLOWSTATE_PANEL'
                    });
                }
            });
            break;
    }
});

// Analytics and usage tracking (privacy-friendly)
function trackUsage(event, data = {}) {
    // Store usage data locally for analytics
    chrome.storage.local.get(['flowstate_usage'], (result) => {
        const usage = result.flowstate_usage || [];
        
        usage.push({
            event,
            data,
            timestamp: Date.now()
        });
        
        // Keep only last 100 events
        if (usage.length > 100) {
            usage.splice(0, usage.length - 100);
        }
        
        chrome.storage.local.set({ flowstate_usage: usage });
    });
}

// FlowState background service worker ready
console.log('✅ FlowState background service worker initialized');
