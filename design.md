# FlowState Frontend Design

## Chrome Extension UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    SPOTIFY WEB PLAYER                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Now Playing: Song Title                │   │
│  │                   Artist Name                       │   │
│  │  ◄◄  ▐▐  ►►   ────●──────  ♡  🔀  🔁  💻  📱     │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              FLOWSTATE INJECTION AREA                │   │
│  │  ┌───────────┐  ┌─────────────────────────────────┐ │   │
│  │  │    🌊     │  │        FlowState Active         │ │   │
│  │  │ FlowState │  │     Emotional Journey: Calm     │ │   │
│  │  │  Button   │  │      ▓░░░░░░░░░ 20% Complete   │ │   │
│  │  └───────────┘  └─────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  QUEUE PREVIEW                      │   │
│  │  ┌─── ▶ Current Song ─────────────────────────────┐ │   │
│  │  │  🎵 Song Title - Artist         [Calm → Upbeat]│ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │  ┌─── Next in Flow ──────────────────────────────┐  │   │
│  │  │  🎵 Next Song - Artist          [Upbeat → Joy]│  │   │
│  │  │  🎵 Third Song - Artist         [Joy → Peace] │  │   │
│  │  │  🎵 Fourth Song - Artist      [Peace → Energy]│  │   │
│  │  │  ... 6 more songs in optimal flow ...         │  │   │
│  │  │  ┌─────────────────────────────────────────┐  │  │   │
│  │  │  │  + Add Song to Flow  │  ⚡ Re-optimize   │  │  │   │
│  │  │  └─────────────────────────────────────────┘  │  │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  FLOWSTATE SETTINGS PANEL                   │
├─────────────────────────────────────────────────────────────┤
│  Emotional Journey Type:                                    │
│  ○ Gradual Flow    ● Maintain Vibe    ○ Adventure Mode     │
│                                                             │
│  Queue Length: [████████░░] 10 songs                       │
│                                                             │
│  Flow Sensitivity:                                          │
│  Gentle ────●──────────── Dramatic                         │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              EMOTIONAL VISUALIZATION                 │   │
│  │         Energy ▲                                    │   │
│  │              ░ │ ▓                                   │   │
│  │              ░ │   ▓                                 │   │
│  │              ░ │     ▓                               │   │
│  │              ░ │       ▓●                           │   │
│  │              ░ │         ▓                           │   │
│  │              ░ │           ▓                         │   │
│  │         Calm  ░─┼─────────────▓─────► Time           │   │
│  │                 │               ▓                    │   │
│  │                 │                 ▓                  │   │
│  │                 │                   ▓                │   │
│  │                Now              Future               │   │
│  │                                                      │   │
│  │  🎵 Current emotion: Calm, Contemplative             │   │
│  │  🎯 Journey target: Uplifting, Energetic            │   │
│  │  📈 Transition style: Smooth, Natural               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   QUICK ACTIONS                     │   │
│  │  [🎯 Focus Mode]  [💤 Wind Down]  [⚡ Pump Up]     │   │
│  │  [🧘 Meditate]    [🎉 Celebrate]  [😢 Process]     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ✓ Connected to Spotify                             │   │
│  │  ♪ 15,247 songs analyzed in your library           │   │
│  │  🎯 Queue optimization: 94ms avg response          │   │
│  │  📊 Your flow satisfaction: 4.7/5.0 ★★★★★          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              MOBILE APP DESIGN (FUTURE)                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ◄  FlowState                           ⚙️  👤      │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │                 🌊 ACTIVE FLOW                      │   │
│  │              "Sunset Contemplation"                 │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │            ●●●●●●○○○○                        │   │
│  │  │         6/10 songs completed                │   │
│  │  │                                             │   │
│  │  │     🎵 "Breathe Me" - Sia                   │   │
│  │  │        ●───────────○ 2:14 / 4:17           │   │
│  │  │                                             │   │
│  │  │     Next: "Mad World" - Gary Jules          │   │
│  │  │     Flow: Melancholic → Reflective         │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  ┌─── EMOTIONAL JOURNEY ─────────────────────┐     │   │
│  │  │                                           │     │   │
│  │  │    Energy ▲                               │     │   │
│  │  │          ░│    ●                          │     │   │
│  │  │          ░│  ╱   ╲                        │     │   │
│  │  │          ░│╱       ╲                      │     │   │
│  │  │    Calm  ░●           ╲───────► Time      │     │   │
│  │  │            │             ╲                │     │   │
│  │  │          Start        Current   End       │     │   │
│  │  │                                           │     │   │
│  │  └───────────────────────────────────────────┘     │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  🎯 QUICK FLOW STARTERS                     │   │   │
│  │  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │   │   │
│  │  │  │ 🧘  │ │ ⚡  │ │ 💤  │ │ 🎉  │ │ 😢  │   │   │   │
│  │  │  │Calm │ │ Pump│ │Sleep│ │Party│ │Feel │   │   │   │
│  │  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘   │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  [🔄 Re-optimize Flow] [+ Add Song] [⚙️ Settings] │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Visual Design Principles

### Color Palette
```
Primary:   #6366f1 (Indigo)  - FlowState brand color
Secondary: #8b5cf6 (Purple)  - Emotional visualization
Accent:    #06b6d4 (Cyan)    - Active states
Calm:      #10b981 (Green)   - Peaceful emotions
Energy:    #f59e0b (Amber)   - High energy emotions
Alert:     #ef4444 (Red)     - Attention/warnings
Dark:      #1f2937 (Gray)    - Background
Light:     #f9fafb (Gray)    - Text/contrast
```

### Typography
```
Headers:   Inter, 600 weight
Body:      Inter, 400 weight  
Code:      JetBrains Mono
Icons:     Lucide React icons
```

### Animation Principles
```
- Gentle, organic movements (no harsh transitions)
- Emotional state changes: 300ms ease-out
- Queue updates: Staggered 100ms delays
- Loading states: Pulsing, not spinning
- Flow visualization: Smooth curves, gradual color changes
```

## Component Architecture
```
FlowStateExtension/
├── components/
│   ├── FlowButton.tsx           # Main activation button
│   ├── QueuePreview.tsx         # Show upcoming songs
│   ├── EmotionalViz.tsx         # Emotional journey chart
│   ├── QuickActions.tsx         # Preset emotional flows
│   └── SettingsPanel.tsx        # User preferences
├── services/
│   ├── SpotifyAdapter.ts        # Platform integration
│   ├── AudioAnalyzer.ts         # Client-side analysis
│   └── FlowStateAPI.ts          # Backend communication
└── styles/
    ├── flowstate.css            # Main extension styles
    └── emotional-colors.css     # Emotion-based color system
```

## Interaction Flow
```
1. User plays song on Spotify
2. FlowState detects song change
3. Gentle notification: "🌊 FlowState available"
4. User clicks FlowState button
5. Brief loading animation (wave ripple effect)
6. Queue preview slides in from bottom
7. Emotional visualization appears
8. User can modify queue in real-time
9. Changes trigger smooth re-optimization
10. Continuous emotional feedback throughout session
```
