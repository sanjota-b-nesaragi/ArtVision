"""
ARTVISION - Sample Image Generator (Standard Library)
Generates valid standard uncompressed BMP and PPM sample images for:
- Impressionism
- Cubism
- Abstract
- Surrealism
- Room / Interior images
Requires zero external dependencies (pure Python standard library).
"""

import os
import struct
import math
import random


def write_bmp(filepath: str, width: int, height: int, pixels: list):
    """
    Writes a 24-bit uncompressed Windows BMP file.
    pixels: list of rows from bottom to top, or we handle top-to-bottom.
    Each pixel is (R, G, B) tuple.
    """
    row_padding = (4 - (width * 3) % 4) % 4
    image_size = (width * 3 + row_padding) * height
    file_size = 54 + image_size

    # BMP Header (14 bytes)
    header = struct.pack(
        "<2sIHHI",
        b"BM",
        file_size,
        0, 0,
        54  # pixel offset
    )

    # DIB Header (BITMAPINFOHEADER - 40 bytes)
    dib = struct.pack(
        "<IIIHHIIIIII",
        40,          # DIB size
        width,
        height,      # positive = bottom-to-top
        1,           # color planes
        24,          # bits per pixel (RGB)
        0,           # compression (BI_RGB)
        image_size,  # image size
        2835, 2835,  # 72 DPI
        0, 0
    )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(dib)
        # Write pixels bottom-to-top
        for y in reversed(range(height)):
            row = bytearray()
            for x in range(width):
                r, g, b = pixels[y][x]
                # BMP format stores B, G, R
                row.extend([int(b) & 0xFF, int(g) & 0xFF, int(r) & 0xFF])
            row.extend(b"\x00" * row_padding)
            f.write(row)


def generate_impressionism_image(width=224, height=224):
    """Generates impressionist sunlit garden landscape with broken pastel brushstrokes."""
    random.seed(42)
    pixels = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
    
    # Sky: soft sky blue to lilac gradient
    for y in range(height):
        for x in range(width):
            if y < height * 0.45:
                # Sky with clouds
                t = y / (height * 0.45)
                r = int(170 + 60 * t + random.randint(-15, 15))
                g = int(200 + 40 * t + random.randint(-15, 15))
                b = int(235 + 15 * (1 - t) + random.randint(-10, 10))
            elif y < height * 0.65:
                # Distant hills / foliage: lavender, sage, soft emerald
                r = int(120 + random.randint(-25, 25))
                g = int(160 + random.randint(-20, 20))
                b = int(140 + random.randint(-25, 25))
            else:
                # Foreground water and water lilies: cobalt reflections with pink/yellow dabs
                t = (y - height * 0.65) / (height * 0.35)
                if random.random() < 0.25:
                    # Pink / yellow water lily dab
                    r = random.choice([245, 255, 230])
                    g = random.choice([180, 220, 200])
                    b = random.choice([190, 150, 120])
                else:
                    # Water reflection
                    r = int(70 + 40 * (1 - t) + random.randint(-15, 15))
                    g = int(120 + 50 * t + random.randint(-15, 15))
                    b = int(160 + 40 * (1 - t) + random.randint(-15, 15))
            pixels[y][x] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    return pixels


def generate_cubism_image(width=224, height=224):
    """Generates geometric faceted cubist composition with earthy ochres and angular planes."""
    random.seed(101)
    pixels = [[(180, 160, 130) for _ in range(width)] for _ in range(height)]
    
    # Draw intersecting geometric angular facets
    planes = [
        {"center": (60, 70), "size": 55, "color": (160, 110, 60), "angle": 0.4},
        {"center": (140, 60), "size": 65, "color": (90, 85, 80), "angle": -0.6},
        {"center": (110, 120), "size": 75, "color": (210, 180, 120), "angle": 0.2},
        {"center": (60, 160), "size": 60, "color": (70, 70, 75), "angle": 0.8},
        {"center": (165, 150), "size": 70, "color": (140, 100, 70), "angle": -0.3},
        {"center": (112, 112), "size": 40, "color": (230, 210, 160), "angle": 0.0},
    ]
    
    for y in range(height):
        for x in range(width):
            r, g, b = 180, 160, 135
            for p in planes:
                cx, cy = p["center"]
                dx = x - cx
                dy = y - cy
                # Rotated coordinates
                rx = dx * math.cos(p["angle"]) - dy * math.sin(p["angle"])
                ry = dx * math.sin(p["angle"]) + dy * math.cos(p["angle"])
                if abs(rx) + abs(ry) < p["size"]:
                    pr, pg, pb = p["color"]
                    # Add planar shading
                    shade = int((rx - ry) * 0.4)
                    r = int(pr + shade)
                    g = int(pg + shade)
                    b = int(pb + shade)
                    # Edge facet line
                    if abs(abs(rx) + abs(ry) - p["size"]) < 2:
                        r, g, b = 40, 35, 30
            pixels[y][x] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    return pixels


def generate_abstract_image(width=224, height=224):
    """Generates neoplastic / gestural abstract artwork with primary color blocks and dynamic lines."""
    pixels = [[(245, 243, 238) for _ in range(width)] for _ in range(height)]
    
    # Red rectangle
    for y in range(20, 100):
        for x in range(20, 120):
            pixels[y][x] = (220, 40, 35)
            
    # Blue rectangle
    for y in range(120, 204):
        for x in range(140, 204):
            pixels[y][x] = (30, 75, 175)
            
    # Yellow rectangle
    for y in range(130, 190):
        for x in range(20, 70):
            pixels[y][x] = (245, 195, 35)
            
    # Black structural grid lines
    h_lines = [18, 102, 118, 192, 206]
    v_lines = [18, 72, 122, 138, 206]
    
    for y in range(height):
        for x in range(width):
            for hl in h_lines:
                if abs(y - hl) <= 2 and 15 <= x <= 210:
                    pixels[y][x] = (20, 20, 22)
            for vl in v_lines:
                if abs(x - vl) <= 2 and 15 <= y <= 210:
                    pixels[y][x] = (20, 20, 22)
    return pixels


def generate_surrealism_image(width=224, height=224):
    """Generates surrealist dreamscape with desert horizon, glowing floating celestial orb, and eerie shadows."""
    pixels = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
    
    horizon = int(height * 0.58)
    
    for y in range(height):
        for x in range(width):
            if y < horizon:
                # Unearthly sky: deep twilight cobalt to strange amber green at horizon
                t = y / horizon
                r = int(25 + 160 * (t ** 2))
                g = int(35 + 130 * (t ** 2))
                b = int(95 + 40 * (1 - t))
            else:
                # Barren surreal desert plane
                t = (y - horizon) / (height - horizon)
                r = int(185 - 55 * t)
                g = int(115 - 40 * t)
                b = int(60 - 25 * t)
                
            # Floating celestial eye / sphere at (112, 65) with radius 28
            dx = x - 112
            dy = y - 65
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= 28:
                # Sphere shading
                light_dx = dx - 8
                light_dy = dy - 8
                l_dist = math.sqrt(light_dx * light_dx + light_dy * light_dy)
                glow = max(0.0, 1.0 - (l_dist / 38.0))
                r = int(240 * glow + 40 * (1 - glow))
                g = int(210 * glow + 30 * (1 - glow))
                b = int(110 * glow + 50 * (1 - glow))
                
            # Cast shadow on ground from floating object
            shadow_dx = (x - 112) / 35.0
            shadow_dy = (y - (horizon + 35)) / 10.0
            if (shadow_dx ** 2 + shadow_dy ** 2) <= 1.0:
                r = int(r * 0.45)
                g = int(g * 0.45)
                b = int(b * 0.45)

            pixels[y][x] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    return pixels


def generate_room_image(style="warm", width=224, height=224):
    """Generates simulated room/interior spaces with distinct color palettes."""
    pixels = [[(200, 200, 200) for _ in range(width)] for _ in range(height)]
    
    floor_y = int(height * 0.7)
    
    if style == "warm":
        # Warm sunlit living room: creamy beige walls, natural oak floor, terracotta rug
        for y in range(height):
            for x in range(width):
                if y < floor_y:
                    # Wall
                    r, g, b = 238, 230, 218
                    # Window light on left
                    if x < width * 0.35 and 30 < y < floor_y - 20:
                        r, g, b = 250, 246, 235
                else:
                    # Oak floor with terracotta rug in center
                    if width * 0.25 <= x <= width * 0.75:
                        r, g, b = 180, 85, 60  # Terracotta rug
                    else:
                        r, g, b = 195, 145, 95  # Oak wood
                pixels[y][x] = (r, g, b)
    elif style == "minimalist":
        # Cool minimalist loft: slate gray accent wall, white adjacent wall, polished concrete floor
        for y in range(height):
            for x in range(width):
                if y < floor_y:
                    if x < width * 0.5:
                        r, g, b = 80, 88, 98   # Slate gray wall
                    else:
                        r, g, b = 240, 242, 245 # Off-white gallery wall
                else:
                    r, g, b = 135, 138, 142    # Polished concrete floor
                pixels[y][x] = (r, g, b)
    else:  # moody
        # Moody library / dark academia: deep navy wall, dark walnut floor, brass tones
        for y in range(height):
            for x in range(width):
                if y < floor_y:
                    r, g, b = 28, 38, 52   # Deep midnight navy
                else:
                    r, g, b = 55, 38, 28   # Dark walnut
                pixels[y][x] = (r, g, b)
                
    return pixels


def create_all_sample_images(base_dir: str):
    """Create a suite of test artwork and room images."""
    sample_dir = os.path.join(base_dir, "dataset", "sample_data")
    assets_dir = os.path.join(base_dir, "assets", "images")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    # 4 Artwork styles
    generators = {
        "Impressionism": generate_impressionism_image,
        "Cubism": generate_cubism_image,
        "Abstract": generate_abstract_image,
        "Surrealism": generate_surrealism_image
    }
    
    created_files = []
    for style_name, gen_func in generators.items():
        pixels = gen_func(224, 224)
        fname = f"sample_{style_name.lower()}.bmp"
        path = os.path.join(sample_dir, fname)
        write_bmp(path, 224, 224, pixels)
        created_files.append(path)
        # Also copy to assets
        asset_path = os.path.join(assets_dir, fname)
        write_bmp(asset_path, 224, 224, pixels)

    # 3 Room styles
    room_types = [
        ("room_warm_living.bmp", "warm"),
        ("room_minimalist_loft.bmp", "minimalist"),
        ("room_moody_den.bmp", "moody")
    ]
    for fname, room_style in room_types:
        r_pixels = generate_room_image(room_style, 224, 224)
        r_path = os.path.join(sample_dir, fname)
        write_bmp(r_path, 224, 224, r_pixels)
        created_files.append(r_path)
        write_bmp(os.path.join(assets_dir, fname), 224, 224, r_pixels)
        
    print(f"Generated {len(created_files)} sample BMP images in {sample_dir}")
    return created_files


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    create_all_sample_images(project_root)
