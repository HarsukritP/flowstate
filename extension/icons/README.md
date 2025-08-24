# FlowState Extension Icons

Place the following icon files in this directory:

- `icon-16.png` - 16x16 pixels (toolbar)
- `icon-32.png` - 32x32 pixels (Windows)  
- `icon-48.png` - 48x48 pixels (extension management)
- `icon-128.png` - 128x128 pixels (Chrome Web Store)

## Icon Design Guidelines

- Use the FlowState wave (🌊) motif
- Primary color: #6366f1 (indigo)
- Background: Transparent or subtle gradient
- Style: Modern, clean, minimal

## Temporary Solution

For MVP development, you can use emoji-based icons or simple colored squares until proper icons are designed.

```bash
# Create simple placeholder icons (requires ImageMagick)
convert -size 16x16 xc:'#6366f1' icon-16.png
convert -size 32x32 xc:'#6366f1' icon-32.png  
convert -size 48x48 xc:'#6366f1' icon-48.png
convert -size 128x128 xc:'#6366f1' icon-128.png
```
