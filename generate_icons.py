#!/usr/bin/env python3
"""Generate PWA icons for Stock Analysis Pro"""

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available, will use SVG data URI fallback")

import os

def generate_png_icons():
    """Generate PNG icons using PIL"""
    sizes = [192, 512]
    
    for size in sizes:
        # Create image with gradient background
        img = Image.new('RGB', (size, size), '#6366f1')
        draw = ImageDraw.Draw(img)
        
        # Try to use a font, fallback to default
        try:
            font_size = size // 3
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Draw text "SP"
        text = "SP"
        
        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        position = ((size - text_width) // 2, (size - text_height) // 2 - size // 20)
        
        # Draw text with shadow
        shadow_offset = size // 40
        draw.text((position[0] + shadow_offset, position[1] + shadow_offset), 
                 text, font=font, fill='#000000')
        draw.text(position, text, font=font, fill='#ffffff')
        
        # Save
        output_path = f'static/icon-{size}.png'
        img.save(output_path, 'PNG')
        print(f"[OK] Generated {output_path}")

def generate_svg_fallback():
    """Generate SVG icons as fallback"""
    svg_template = '''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <rect width="{size}" height="{size}" fill="#6366f1"/>
  <text x="50%" y="50%" font-family="Arial, sans-serif" font-size="{font_size}" font-weight="bold" fill="white" text-anchor="middle" dominant-baseline="central">SP</text>
</svg>'''
    
    sizes = [192, 512]
    for size in sizes:
        font_size = size // 3
        svg_content = svg_template.format(size=size, font_size=font_size)
        
        # For now, just create a simple colored PNG fallback
        print(f"SVG fallback mode for {size}x{size}")
        
        # Since we can't easily convert SVG to PNG without PIL,
        # create a very basic PNG with solid color
        try:
            from PIL import Image
            img = Image.new('RGB', (size, size), '#6366f1')
            output_path = f'static/icon-{size}.png'
            img.save(output_path, 'PNG')
            print(f"[OK] Generated basic {output_path}")
        except ImportError:
            print(f"[ERROR] Cannot generate {size}x{size} icon without PIL")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if PIL_AVAILABLE:
        generate_png_icons()
    else:
        generate_svg_fallback()
    
    print("\n[OK] Icon generation complete!")
