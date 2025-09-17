#!/usr/bin/env python3
"""
Generate PWA icons from SVG source
Requires: pip install Pillow cairosvg
"""

import os
from PIL import Image
import io

def create_icon_from_svg(svg_path, output_path, size):
    """Create PNG icon from SVG"""
    try:
        import cairosvg
        
        # Convert SVG to PNG bytes
        png_bytes = cairosvg.svg2png(url=svg_path, output_width=size, output_height=size)
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(png_bytes)
        
        print(f"Created {output_path} ({size}x{size})")
        return True
        
    except ImportError:
        print("cairosvg not available, creating simple placeholder icons")
        return create_placeholder_icon(output_path, size)
    except Exception as e:
        print(f"Error creating icon: {e}")
        return create_placeholder_icon(output_path, size)

def create_placeholder_icon(output_path, size):
    """Create a simple placeholder icon"""
    try:
        # Create a simple gradient icon
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        
        # Draw a simple circle with gradient effect
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Background circle
        margin = size // 20
        draw.ellipse([margin, margin, size-margin, size-margin], 
                     fill=(59, 130, 246, 255), outline=(255, 255, 255, 255), width=2)
        
        # Inner circle
        inner_margin = size // 4
        draw.ellipse([inner_margin, inner_margin, size-inner_margin, size-inner_margin], 
                     fill=(255, 255, 255, 255))
        
        # Save
        img.save(output_path, 'PNG')
        print(f"Created placeholder {output_path} ({size}x{size})")
        return True
        
    except Exception as e:
        print(f"Error creating placeholder icon: {e}")
        return False

def main():
    """Generate all required PWA icons"""
    svg_path = '/opt/praktik/translation-pwa/static/icon.svg'
    static_dir = '/opt/praktik/translation-pwa/static'
    
    # Required icon sizes
    sizes = [
        (16, 'favicon-16x16.png'),
        (32, 'favicon-32x32.png'),
        (192, 'icon-192.png'),
        (512, 'icon-512.png')
    ]
    
    print("Generating PWA icons...")
    
    for size, filename in sizes:
        output_path = os.path.join(static_dir, filename)
        
        if os.path.exists(svg_path):
            create_icon_from_svg(svg_path, output_path, size)
        else:
            create_placeholder_icon(output_path, size)
    
    print("Icon generation complete!")

if __name__ == '__main__':
    main()