#!/usr/bin/env python3
"""
Simple script to create basic icon files for FlowState Chrome extension
"""

import base64
import os

# Simple 16x16 blue PNG (base64 encoded)
icon_16_data = """
iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAFYSURBVDiNpZM9SwNBEIafgwQSCwsLwcJCG1sLwcJCsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQsLGwsLBQ
"""

# Create icons directory
icons_dir = "extension/icons"
os.makedirs(icons_dir, exist_ok=True)

def create_simple_png(size, filename):
    """Create a simple colored PNG file"""
    # Create a simple blue square PNG
    from io import BytesIO
    
    # Simple PNG data for a colored square (minimal)
    # This is a 1x1 blue pixel that we'll use for all sizes
    png_data = base64.b64decode("""
iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA4Ie6cwAAAABJRU5ErkJggg==
""".strip())
    
    # Write the PNG data
    filepath = os.path.join(icons_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(png_data)
    
    print(f"Created {filepath}")

# Create all required icon sizes
create_simple_png(16, "icon-16.png")
create_simple_png(32, "icon-32.png")
create_simple_png(48, "icon-48.png")
create_simple_png(128, "icon-128.png")

print("✅ All icon files created successfully!")
