#!/bin/bash

# Create simple PWA icons using ImageMagick or fallback to basic files

STATIC_DIR="/opt/praktik/translation-pwa/static"

echo "Creating PWA icons..."

# Check if ImageMagick is available
if command -v convert &> /dev/null; then
    echo "Using ImageMagick to create icons..."
    
    # Create a simple gradient circle icon
    convert -size 512x512 xc:transparent \
        -fill "rgb(59,130,246)" \
        -draw "circle 256,256 256,50" \
        -fill "rgb(255,255,255)" \
        -draw "circle 256,256 256,150" \
        -fill "rgb(59,130,246)" \
        -pointsize 48 -gravity center \
        -annotate +0+50 "🌍" \
        "$STATIC_DIR/icon-512.png"
    
    # Resize to other sizes
    convert "$STATIC_DIR/icon-512.png" -resize 192x192 "$STATIC_DIR/icon-192.png"
    convert "$STATIC_DIR/icon-512.png" -resize 32x32 "$STATIC_DIR/favicon-32x32.png"
    convert "$STATIC_DIR/icon-512.png" -resize 16x16 "$STATIC_DIR/favicon-16x16.png"
    
    echo "Icons created successfully with ImageMagick"
    
elif command -v python3 &> /dev/null; then
    echo "Using Python to create simple icons..."
    
    python3 << 'EOF'
import os

def create_simple_icon(size, filename):
    """Create a simple text-based icon file"""
    static_dir = "/opt/praktik/translation-pwa/static"
    
    # Create a simple HTML file that can be converted to PNG
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; }}
            .icon {{
                width: {size}px;
                height: {size}px;
                background: linear-gradient(135deg, #3b82f6, #1d4ed8);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: {size//3}px;
                font-family: Arial, sans-serif;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="icon">🌍</div>
    </body>
    </html>
    """
    
    # For now, just create a placeholder text file
    with open(os.path.join(static_dir, filename.replace('.png', '.txt')), 'w') as f:
        f.write(f"Icon placeholder for {size}x{size} - Replace with actual PNG file")
    
    print(f"Created placeholder for {filename}")

# Create placeholders for all required sizes
sizes = [(16, 'favicon-16x16.png'), (32, 'favicon-32x32.png'), 
         (192, 'icon-192.png'), (512, 'icon-512.png')]

for size, filename in sizes:
    create_simple_icon(size, filename)

print("Placeholder icons created")
EOF

else
    echo "Creating simple placeholder files..."
    
    # Create simple placeholder files
    for size in 16 32 192 512; do
        if [ $size -eq 16 ]; then
            filename="favicon-16x16.png"
        elif [ $size -eq 32 ]; then
            filename="favicon-32x32.png"
        elif [ $size -eq 192 ]; then
            filename="icon-192.png"
        else
            filename="icon-512.png"
        fi
        
        echo "Placeholder icon for ${size}x${size}" > "$STATIC_DIR/$filename"
    done
    
    echo "Placeholder files created"
fi

echo "Icon creation complete!"
echo "Note: Replace placeholder files with actual PNG icons for production use."