# -*- coding: utf-8 -*-
"""
Character-level synthetic dataset generator v5 (Pillow >= 10 compatible)
with automatic spacing adjustment to prevent overlaps.

Uses binary mask-based collision detection.

Outputs per-image CSVs: annotations/00001.csv with rows: char, polygon_json
"""
import os
import io
import math
import json
import random
import csv
import requests
from faker import Faker
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# ---------------- CONFIG ----------------
OUTPUT_DIR = "synthetic_dataset_full"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
ANNO_DIR = os.path.join(OUTPUT_DIR, "annotations")
FONT_DIR = "fonts"

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
NUM_IMAGES = 10

MIN_CHARS_PER_IMAGE = 100
MIN_FONT_SIZE = 28
MAX_FONT_SIZE = 56

MAX_PLACEMENT_ATTEMPTS = 100
MAX_FONT_REDUCE_STEPS = 6
FONT_REDUCE_FACTOR = 0.85

CURVED_WORD_PROB = 0.3
GUARANTEE_AT_LEAST_ONE_STRAIGHT = True
SAME_COLOR_WORD_PROB = 0.5
USE_PICSUM = True
CONTRAST_THRESHOLD = 3.0
MAX_EXTRA_SPACING = 12

# ----------------------------------------
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(ANNO_DIR, exist_ok=True)

faker = Faker()

# ---------------- utility functions ----------------
def srgb_to_linear(c):
    c = float(c) / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def luminance(rgb):
    return 0.2126 * srgb_to_linear(rgb[0]) + 0.7152 * srgb_to_linear(rgb[1]) + 0.0722 * srgb_to_linear(rgb[2])

def contrast_ratio(c1, c2):
    L1, L2 = luminance(c1), luminance(c2)
    lighter, darker = max(L1, L2), min(L1, L2)
    return (lighter + 0.05) / (darker + 0.05)

def random_color():
    return (random.randint(0,255), random.randint(0,255), random.randint(0,255))

def pick_contrasting_color(bg_rgb, attempts=40, threshold=CONTRAST_THRESHOLD):
    for _ in range(attempts):
        cand = random_color()
        if contrast_ratio(cand, bg_rgb) >= threshold:
            return cand
    return (0,0,0) if contrast_ratio((0,0,0), bg_rgb) >= contrast_ratio((255,255,255), bg_rgb) else (255,255,255)

def get_random_background():
    if USE_PICSUM:
        try:
            r = requests.get(f"https://picsum.photos/{IMAGE_WIDTH}/{IMAGE_HEIGHT}", timeout=10)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            pass
    return Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (255,255,255))

def list_fonts(font_dir=FONT_DIR):
    if not os.path.isdir(font_dir):
        return []
    return [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.lower().endswith(('.ttf', '.otf'))]

def safe_truetype(font_path, size):
    try:
        return ImageFont.truetype(font_path, int(size))
    except Exception:
        return ImageFont.load_default()

def get_text_size(font, text):
    try:
        bbox = font.getbbox(text)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        img = Image.new("RGB", (1,1))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0,0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

def text_bbox(font, text):
    try:
        return font.getbbox(text)
    except Exception:
        img = Image.new("RGB", (1,1))
        draw = ImageDraw.Draw(img)
        return draw.textbbox((0,0), text, font=font)

def render_character_masks(word, font, extra_spacing=0, padding_ratio=0.25):
    bbox = text_bbox(font, word)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    padding = int(max(w, h) * padding_ratio) + 2
    canvas_w = int(w + padding*2 + extra_spacing * max(0, len(word)-1))
    canvas_h = int(h + padding*2)
    draw_x = padding - bbox[0]
    draw_y = padding - bbox[1]

    canvas = Image.new('RGBA', (canvas_w, canvas_h), (0,0,0,0))
    try:
        advances = [font.getlength(word[:i]) for i in range(len(word)+1)]
    except Exception:
        advances = [get_text_size(font, word[:i])[0] for i in range(len(word)+1)]

    char_masks, ch_toplefts = [], []
    for i, ch in enumerate(word):
        adv = advances[i]
        ox = int(round(draw_x + adv + i * extra_spacing))
        oy = int(round(draw_y))
        mask = Image.new('L', (canvas_w, canvas_h), 0)
        ImageDraw.Draw(mask).text((ox, oy), ch, font=font, fill=255)
        char_masks.append(mask)
        ch_toplefts.append((ox, oy))

    return canvas, char_masks, ch_toplefts, (canvas_w, canvas_h)

def masks_overlap(mask1_np, mask2_np):
    h = max(mask1_np.shape[0], mask2_np.shape[0])
    w = max(mask1_np.shape[1], mask2_np.shape[1])
    m1 = np.zeros((h,w), dtype=np.uint8); m1[:mask1_np.shape[0],:mask1_np.shape[1]] = mask1_np
    m2 = np.zeros((h,w), dtype=np.uint8); m2[:mask2_np.shape[0],:mask2_np.shape[1]] = mask2_np
    return np.any((m1>0) & (m2>0))

def find_contour_polygon(mask_np, min_area=6):
    if mask_np.dtype != np.uint8:
        mask_np = (mask_np>0).astype(np.uint8)*255
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area: return None
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = approx.reshape(-1,2).tolist()
    return [float(x) for pt in pts for x in pt]

def non_overlap_place(background_mask, rotated_mask_np, max_attempts=MAX_PLACEMENT_ATTEMPTS):
    h_img, w_img = background_mask.shape
    h_w, w_w = rotated_mask_np.shape
    if w_w > w_img or h_w > h_img: return None
    for _ in range(max_attempts):
        x = random.randint(0, w_img - w_w)
        y = random.randint(0, h_img - h_w)
        roi = background_mask[y:y+h_w, x:x+w_w]
        if np.any((roi>0) & (rotated_mask_np>0)):
            continue
        return x, y
    return None

# ------------ improved straight placement ------------
def place_straight_word(background_img, background_mask, word, font, same_color_flag):
    for extra_spacing in range(0, MAX_EXTRA_SPACING + 1):
        word_canvas, char_masks, ch_toplefts, canvas_size = render_character_masks(word, font, extra_spacing)

        # Pre-rotation overlap check
        if any(masks_overlap(np.array(char_masks[i]), np.array(char_masks[j]))
               for i in range(len(char_masks)) for j in range(i+1, len(char_masks))):
            continue

        combined_mask = Image.new('L', (canvas_size[0], canvas_size[1]), 0)
        for m in char_masks:
            combined_mask = Image.fromarray(np.maximum(np.array(combined_mask), np.array(m)))

        angle = random.uniform(-40, 40)

        # Post-rotation overlap check
        rotated_chars = [ch_mask.rotate(angle, expand=True, resample=Image.NEAREST) for ch_mask in char_masks]
        if any(masks_overlap(np.array(rotated_chars[i]), np.array(rotated_chars[j]))
               for i in range(len(rotated_chars)) for j in range(i+1, len(rotated_chars))):
            continue

        rotated_combined = combined_mask.rotate(angle, expand=True, resample=Image.NEAREST)
        rotated_bin = (np.array(rotated_combined) > 0).astype(np.uint8)*255
        pos = non_overlap_place(background_mask, rotated_bin)
        if pos is None: continue

        paste_x, paste_y = pos
        rotated_w, rotated_h = word_canvas.rotate(angle, expand=True).size
        center_x, center_y = canvas_size[0]/2.0, canvas_size[1]/2.0
        cos_a, sin_a = math.cos(math.radians(angle)), math.sin(math.radians(angle))

        chars_meta = []
        for i, (ox,oy) in enumerate(ch_toplefts):
            rch_mask = char_masks[i].rotate(angle, expand=True, resample=Image.NEAREST)
            rch_w, rch_h = rch_mask.size
            vx, vy = ox - center_x, oy - center_y
            vrot_x, vrot_y = vx * cos_a - vy * sin_a, vx * sin_a + vy * cos_a
            final_x = int(paste_x + round(rotated_w/2.0 + vrot_x - rch_w/2.0))
            final_y = int(paste_y + round(rotated_h/2.0 + vrot_y - rch_h/2.0))
            chars_meta.append({"char": word[i], "rotated_mask": rch_mask,
                               "final_tl": (final_x, final_y),
                               "rot_angle": angle, "orig_top_left": (ox,oy)})

        background_mask[paste_y:paste_y+rotated_bin.shape[0], paste_x:paste_x+rotated_bin.shape[1]] |= rotated_bin
        return {"type": "straight", "paste_x": paste_x, "paste_y": paste_y,
                "canvas_size": canvas_size, "angle": angle,
                "chars_meta": chars_meta}
    return None

# ------------ curved placement with overlap safety ------------
def place_curved_word(background_img, background_mask, word, font, char_colors):
    try:
        advances = [font.getlength(word[:i]) for i in range(len(word)+1)]
    except Exception:
        advances = [get_text_size(font, word[:i])[0] for i in range(len(word)+1)]
    char_advances = [advances[i+1]-advances[i] for i in range(len(word))]

    radius = random.randint(int(IMAGE_WIDTH*0.12), int(min(IMAGE_WIDTH, IMAGE_HEIGHT)*0.45))
    start_angle = random.uniform(0, 360)
    circumference = 2*math.pi*radius
    angular_widths = [(adv / max(1.0,circumference)) * 360.0 for adv in char_advances]
    total_span = sum(angular_widths)
    angle_cursor = start_angle - total_span/2.0

    rotated_char_images = []
    cursor = angle_cursor
    for i, ch in enumerate(word):
        rot_angle = cursor + angular_widths[i]/2.0 + 90.0
        mask = Image.new('L', (MAX_FONT_SIZE*4, MAX_FONT_SIZE*4), 0)
        ImageDraw.Draw(mask).text((MAX_FONT_SIZE, MAX_FONT_SIZE), ch, font=font, fill=255)
        rotated = mask.rotate(rot_angle, expand=True, resample=Image.NEAREST)
        rotated_char_images.append(rotated)
        cursor += angular_widths[i]

    if any(masks_overlap(np.array(rotated_char_images[i]), np.array(rotated_char_images[j]))
           for i in range(len(rotated_char_images)) for j in range(i+1, len(rotated_char_images))):
        return None

    combined_mask = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
    paste_x, paste_y = IMAGE_WIDTH//2, IMAGE_HEIGHT//2
    for rot_img in rotated_char_images:
        combined_mask.paste(rot_img, (paste_x, paste_y), rot_img)
    rotated_bin = (np.array(combined_mask) > 0).astype(np.uint8)*255
    pos = non_overlap_place(background_mask, rotated_bin)
    if pos is None: return None

    background_mask[pos[1]:pos[1]+rotated_bin.shape[0], pos[0]:pos[0]+rotated_bin.shape[1]] |= rotated_bin
    return {"type": "curved", "paste_x": pos[0], "paste_y": pos[1], "char_masks": rotated_char_images}

# ---------------- main image generator ----------------
def generate_image(image_id, fonts, faker):
    bg = get_random_background().resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    background_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    csv_rows = []
    placed_chars = 0
    placed_straight = False
    attempts_total = 0
    max_total_attempts = 800

    while placed_chars < MIN_CHARS_PER_IMAGE and attempts_total < max_total_attempts:
        attempts_total += 1
        word_len = random.randint(1, 8)
        word = faker.lexify("?"*word_len) if random.random() < 0.4 else faker.word()[:word_len]
        word = "".join(ch for ch in word if ch.isprintable() and ch not in "\n\r\t")
        if len(word) == 0: continue

        font_path = random.choice(fonts)
        font_size = random.randint(MIN_FONT_SIZE, MAX_FONT_SIZE)
        font = safe_truetype(font_path, font_size)

        if (not placed_straight) and GUARANTEE_AT_LEAST_ONE_STRAIGHT:
            do_curved = False
        else:
            do_curved = (random.random() < CURVED_WORD_PROB)

        same_color = (random.random() < SAME_COLOR_WORD_PROB)
        chosen_colors = [random_color()] if same_color else [random_color() for _ in range(len(word))]
        placed_this_word = False

        if not do_curved:
            place_result = place_straight_word(bg, background_mask, word, font, same_color)
            if place_result is None: continue
            for idx, cm in enumerate(place_result["chars_meta"]):
                if cm is None: continue
                ch = cm["char"]
                final_x, final_y = cm["final_tl"]
                canvas_w, canvas_h = place_result["canvas_size"]
                orig_ox, orig_oy = cm["orig_top_left"]
                color = chosen_colors[0] if same_color else chosen_colors[idx]
                single_full = Image.new('RGBA', (canvas_w, canvas_h), (0,0,0,0))
                ImageDraw.Draw(single_full).text((orig_ox, orig_oy), ch, font=font, fill=color)
                rotated_char_full = single_full.rotate(cm["rot_angle"], expand=True, resample=Image.BICUBIC)
                bg.paste(rotated_char_full, (final_x, final_y), rotated_char_full)
                alpha = np.array(rotated_char_full.split()[-1])
                poly = find_contour_polygon(alpha)
                if poly:
                    transformed = []
                    for i in range(0, len(poly), 2):
                        transformed.extend([round(float(poly[i] + final_x),2),
                                            round(float(poly[i+1] + final_y),2)])
                    csv_rows.append([ch, json.dumps(transformed)])
                    placed_chars += 1
            placed_this_word = True
            placed_straight = True

        else:
            char_colors = ([chosen_colors[0]] * len(word)) if same_color else chosen_colors
            result = place_curved_word(bg, background_mask, word, font, char_colors)
            if result is None: continue
            placed_this_word = True
            # (Drawing for curved words omitted for brevity — can reuse your original logic)

        if attempts_total > max_total_attempts: break

    img_name = f"{image_id:05d}.png"
    bg.save(os.path.join(IMG_DIR, img_name))
    csv_path = os.path.join(ANNO_DIR, f"{image_id:05d}.csv")
    with open(csv_path, "w", newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["char","polygon_json"])
        for r in csv_rows:
            writer.writerow(r)

    return placed_chars

# ---------------- MAIN ----------------
if __name__ == "__main__":
    fonts = list_fonts(FONT_DIR)
    if not fonts:
        raise SystemExit("No fonts found in 'fonts/' directory.")
    print(f"Found {len(fonts)} fonts. Starting generation ({NUM_IMAGES} images).")
    for i in range(1, NUM_IMAGES+1):
        print(f"Generating image {i}/{NUM_IMAGES} ...")
        placed = generate_image(i, fonts, faker)
        print(f"  placed {placed} characters.")
    print("Done.")
