#!/usr/bin/env python3
"""
CharSeg Dataset Generator v7 — Production Ready

Generates synthetic character-level instance segmentation data with:
- Topology-aware polygon extraction (preserves holes in O, P, B, D, etc.)
- Image augmentations for real-world generalization (blur, noise, JPEG, color)
- COCO JSON + per-image CSV annotation output
- Train/val/test dataset splits
- YAML-driven configuration with CLI overrides
- Diverse text content (words, numbers, mixed case, topology-focused)
- Procedural + local backgrounds (no external API dependency)
- Proper text-background contrast via WCAG luminance
- Text effects (shadow, outline/stroke)
- Fixed curved-word rendering
- Word-level annotation grouping
- Reproducible generation with configurable seeds

Usage:
    python generator_v7_production.py --config ../../configs/generation.yaml
    python generator_v7_production.py --num-train 100 --num-val 20 --num-test 20
    python generator_v7_production.py --font-dir /path/to/fonts --seed 123
"""

import os
import io
import math
import json
import random
import csv
import string
import logging
import argparse
from pathlib import Path

import yaml
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from faker import Faker
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ============================================================
# Default Configuration
# ============================================================

DEFAULT_CONFIG = {
    "output": {"dir": "dataset_v7", "formats": ["coco_json", "csv"]},
    "splits": {"train": 6000, "val": 1000, "test": 1000},
    "image": {"width": 1024, "height": 1024},
    "fonts": {"dir": "fonts", "size_range": [18, 80]},
    "text": {
        "min_chars_per_image": 80,
        "max_placement_attempts": 600,
        "word_length_range": [1, 10],
        "curved_word_prob": 0.2,
        "rotation_range": [-25, 25],
        "sources": {
            "english_words": 0.35,
            "random_chars": 0.20,
            "numbers": 0.10,
            "mixed_case": 0.15,
            "topology_words": 0.20,
        },
    },
    "colors": {"contrast_threshold": 3.5, "same_color_word_prob": 0.6},
    "backgrounds": {
        "dir": "datasets/backgrounds",
        "use_local": True,
        "use_picsum": False,
        "use_procedural": True,
        "procedural_types": ["solid", "gradient", "noise_texture"],
    },
    "augmentation": {
        "enabled": True,
        "gaussian_blur": {"prob": 0.30, "sigma_range": [0.5, 2.0]},
        "motion_blur": {"prob": 0.15, "kernel_range": [3, 7]},
        "gaussian_noise": {"prob": 0.25, "sigma_range": [5, 25]},
        "jpeg_compression": {"prob": 0.30, "quality_range": [30, 85]},
        "brightness_jitter": {"prob": 0.20, "factor_range": [0.7, 1.3]},
        "contrast_jitter": {"prob": 0.20, "factor_range": [0.7, 1.3]},
        "saturation_jitter": {"prob": 0.15, "factor_range": [0.6, 1.4]},
    },
    "effects": {
        "shadow": {"prob": 0.15, "offset_range": [2, 4]},
        "outline": {"prob": 0.10, "width_range": [1, 2]},
    },
    "pipeline": {"seed": 42, "num_workers": 1, "log_every": 50},
}


def _deep_merge(base, override):
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ============================================================
# Color & Contrast Utilities
# ============================================================


def _srgb_to_linear(c):
    c = float(c) / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb):
    return (
        0.2126 * _srgb_to_linear(rgb[0])
        + 0.7152 * _srgb_to_linear(rgb[1])
        + 0.0722 * _srgb_to_linear(rgb[2])
    )


def contrast_ratio(c1, c2):
    l1, l2 = luminance(c1), luminance(c2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def pick_contrasting_color(bg_rgb, threshold=3.5, attempts=50):
    for _ in range(attempts):
        c = random_color()
        if contrast_ratio(c, bg_rgb) >= threshold:
            return c
    return (0, 0, 0) if luminance(bg_rgb) > 0.5 else (255, 255, 255)


def sample_background_color(image, x, y, w=30, h=30):
    x = max(0, min(x, image.width - 1))
    y = max(0, min(y, image.height - 1))
    x2 = min(x + w, image.width)
    y2 = min(y + h, image.height)
    if x2 <= x or y2 <= y:
        return (128, 128, 128)
    region = np.array(image.crop((x, y, x2, y2)))
    if region.size == 0:
        return (128, 128, 128)
    return tuple(int(c) for c in region.mean(axis=(0, 1)))


# ============================================================
# Font Utilities
# ============================================================


def list_fonts(font_dir):
    if not os.path.isdir(font_dir):
        return []
    return sorted(
        os.path.join(font_dir, f)
        for f in os.listdir(font_dir)
        if f.lower().endswith((".ttf", ".otf"))
    )


def safe_truetype(font_path, size):
    try:
        return ImageFont.truetype(font_path, int(size))
    except Exception:
        return ImageFont.load_default()


def font_supports_chars(font, word):
    try:
        return all(font.getmask(ch).getbbox() is not None for ch in word)
    except Exception:
        return False


def text_bbox(font, text):
    try:
        return font.getbbox(text)
    except Exception:
        img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(img)
        return draw.textbbox((0, 0), text, font=font)


def get_text_size(font, text):
    bbox = text_bbox(font, text)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


# ============================================================
# Text Content Generator
# ============================================================

TOPOLOGY_WORDS = [
    "ABODE", "BOARD", "PROPER", "DOUBLE", "QUARTER", "OPAQUE",
    "REPORT", "BORDER", "POPULAR", "DOORBELL", "PARADOX", "BAROQUE",
    "adobe", "board", "proper", "double", "quarter", "opaque",
    "people", "goodbye", "episode", "adopted", "abdomen", "absorbed",
    "04689", "1980", "2048", "3690", "8064", "9400",
    "ABCDEFG", "OPQRST", "abcdefg", "opqrst", "0123456789",
]


def generate_text_content(faker, config):
    sources = config.get("text", {}).get("sources", {})
    word_range = config.get("text", {}).get("word_length_range", [1, 8])

    r = random.random()
    cumul = 0.0
    chosen_source = "english_words"

    for source, prob in sources.items():
        cumul += prob
        if r < cumul:
            chosen_source = source
            break

    max_len = random.randint(*word_range)

    if chosen_source == "english_words":
        word = faker.word()[:max_len]
        case_r = random.random()
        if case_r < 0.25:
            word = word.upper()
        elif case_r < 0.5:
            word = word.capitalize()
        return word

    if chosen_source == "random_chars":
        charset = string.ascii_letters + string.digits
        return "".join(random.choice(charset) for _ in range(max_len))

    if chosen_source == "numbers":
        fmt = random.choice([
            lambda: str(random.randint(0, 99999)),
            lambda: f"{random.uniform(0, 999):.{random.randint(0, 2)}f}",
            lambda: f"{random.randint(1, 12)}/{random.randint(1, 31)}",
            lambda: f"{random.randint(100, 9999)}",
        ])
        return fmt()[:max_len]

    if chosen_source == "mixed_case":
        word = faker.word()[:max_len]
        return "".join(
            c.upper() if random.random() > 0.5 else c.lower() for c in word
        )

    if chosen_source == "topology_words":
        return random.choice(TOPOLOGY_WORDS)[:max_len]

    return faker.word()[:max_len]


# ============================================================
# Background Provider
# ============================================================


class BackgroundProvider:
    def __init__(self, config):
        bg_cfg = config.get("backgrounds", {})
        self.config = bg_cfg
        self.width = config.get("image", {}).get("width", 1024)
        self.height = config.get("image", {}).get("height", 1024)
        self.local_images = []

        bg_dir = bg_cfg.get("dir", "datasets/backgrounds")
        if bg_cfg.get("use_local", True) and os.path.isdir(bg_dir):
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            self.local_images = [
                os.path.join(bg_dir, f)
                for f in os.listdir(bg_dir)
                if Path(f).suffix.lower() in exts
            ]
            if self.local_images:
                log.info(f"Loaded {len(self.local_images)} local background paths")

    def get(self):
        sources = []
        if self.local_images:
            sources.append("local")
        if self.config.get("use_picsum", False):
            sources.append("picsum")
        if self.config.get("use_procedural", True):
            sources.append("procedural")
        if not sources:
            return Image.new("RGB", (self.width, self.height), (255, 255, 255))

        source = random.choice(sources)
        if source == "local":
            return self._local()
        elif source == "picsum":
            return self._picsum()
        return self._procedural()

    def _local(self):
        path = random.choice(self.local_images)
        try:
            img = Image.open(path).convert("RGB")
            return img.resize((self.width, self.height), Image.LANCZOS)
        except Exception:
            return self._procedural()

    def _picsum(self):
        try:
            import requests
            r = requests.get(
                f"https://picsum.photos/{self.width}/{self.height}", timeout=10
            )
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            return self._procedural()

    def _procedural(self):
        types = self.config.get("procedural_types", ["solid", "gradient", "noise_texture"])
        bg_type = random.choice(types)

        if bg_type == "gradient":
            return self._gradient()
        elif bg_type == "noise_texture":
            return self._noise_texture()
        # solid (default)
        return Image.new("RGB", (self.width, self.height), random_color())

    def _gradient(self):
        c1 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        c2 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        if random.random() < 0.5:
            t = np.linspace(0, 1, self.width).reshape(1, -1, 1)
        else:
            t = np.linspace(0, 1, self.height).reshape(-1, 1, 1)
        arr = c1 * (1 - t) + c2 * t
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def _noise_texture(self):
        base = np.array(
            [random.randint(40, 220) for _ in range(3)], dtype=np.float32
        )
        noise = np.random.normal(
            0, random.uniform(10, 40), (self.height, self.width, 3)
        )
        arr = np.clip(base + noise, 0, 255).astype(np.uint8)
        arr = cv2.GaussianBlur(arr, (5, 5), 1.5)
        return Image.fromarray(arr)


# ============================================================
# Augmentation Pipeline
# ============================================================


class AugmentationPipeline:
    def __init__(self, config):
        self.cfg = config.get("augmentation", {})
        self.enabled = self.cfg.get("enabled", True)

    def apply(self, image):
        if not self.enabled:
            return image
        img = image.copy()

        c = self.cfg.get("gaussian_blur", {})
        if random.random() < c.get("prob", 0):
            sigma = random.uniform(*c.get("sigma_range", [0.5, 2.0]))
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

        c = self.cfg.get("motion_blur", {})
        if random.random() < c.get("prob", 0):
            ks_lo, ks_hi = c.get("kernel_range", [3, 7])
            ks = random.choice(range(ks_lo, ks_hi + 1, 2))
            img = self._motion_blur(img, ks)

        c = self.cfg.get("brightness_jitter", {})
        if random.random() < c.get("prob", 0):
            img = ImageEnhance.Brightness(img).enhance(
                random.uniform(*c.get("factor_range", [0.7, 1.3]))
            )

        c = self.cfg.get("contrast_jitter", {})
        if random.random() < c.get("prob", 0):
            img = ImageEnhance.Contrast(img).enhance(
                random.uniform(*c.get("factor_range", [0.7, 1.3]))
            )

        c = self.cfg.get("saturation_jitter", {})
        if random.random() < c.get("prob", 0):
            img = ImageEnhance.Color(img).enhance(
                random.uniform(*c.get("factor_range", [0.6, 1.4]))
            )

        c = self.cfg.get("gaussian_noise", {})
        if random.random() < c.get("prob", 0):
            sigma = random.uniform(*c.get("sigma_range", [5, 25]))
            arr = np.array(img, dtype=np.float32)
            arr += np.random.normal(0, sigma, arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        c = self.cfg.get("jpeg_compression", {})
        if random.random() < c.get("prob", 0):
            quality = random.randint(*c.get("quality_range", [30, 85]))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        return img

    @staticmethod
    def _motion_blur(img, kernel_size):
        kernel = np.zeros((kernel_size, kernel_size))
        direction = random.choice(["h", "v", "d"])
        if direction == "h":
            kernel[kernel_size // 2, :] = 1
        elif direction == "v":
            kernel[:, kernel_size // 2] = 1
        else:
            np.fill_diagonal(kernel, 1)
        kernel /= kernel_size
        arr = cv2.filter2D(np.array(img), -1, kernel)
        return Image.fromarray(arr)


# ============================================================
# Topology-Aware Polygon Extraction
# ============================================================


def extract_topology_polygons(mask_np, min_area=6, epsilon_ratio=0.002):
    """
    Extract contour polygons preserving topology (holes).

    Uses RETR_CCOMP for a two-level contour hierarchy:
      - parent contours  = outer boundaries
      - child contours   = inner holes  (e.g. inside O, D, P, B, 0, 8)

    Returns dict with 'outer' and 'inner' polygon lists, or None.
    """
    if mask_np.dtype != np.uint8:
        mask_np = (mask_np > 0).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(
        mask_np, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or hierarchy is None:
        return None

    hierarchy = hierarchy[0]
    outer_polys = []
    inner_polys = []

    for cnt, hier in zip(contours, hierarchy):
        if cv2.contourArea(cnt) < min_area or len(cnt) < 3:
            continue
        eps = epsilon_ratio * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).tolist()
        flat = [float(x) for pt in pts for x in pt]

        if hier[3] == -1:
            outer_polys.append(flat)
        else:
            inner_polys.append(flat)

    if not outer_polys:
        return None

    outer_polys.sort(key=_poly_area, reverse=True)
    return {
        "outer": outer_polys,
        "inner": inner_polys,
        "has_holes": len(inner_polys) > 0,
        "num_holes": len(inner_polys),
    }


def _poly_area(flat_pts):
    coords = [
        (flat_pts[i], flat_pts[i + 1]) for i in range(0, len(flat_pts), 2)
    ]
    n = len(coords)
    if n < 3:
        return 0.0
    area = sum(
        coords[i][0] * coords[(i + 1) % n][1]
        - coords[(i + 1) % n][0] * coords[i][1]
        for i in range(n)
    )
    return abs(area) / 2.0


# ============================================================
# Character Mask Rendering
# ============================================================


def render_character_masks(word, font, extra_spacing=0, padding_ratio=0.25):
    bbox = text_bbox(font, word)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    padding = int(max(w, h) * padding_ratio) + 2
    canvas_w = int(w + padding * 2 + extra_spacing * max(0, len(word) - 1))
    canvas_h = int(h + padding * 2)
    draw_x = padding - bbox[0]
    draw_y = padding - bbox[1]

    try:
        advances = [font.getlength(word[:i]) for i in range(len(word) + 1)]
    except Exception:
        advances = [get_text_size(font, word[:i])[0] for i in range(len(word) + 1)]

    char_masks = []
    ch_toplefts = []
    for i, ch in enumerate(word):
        adv = advances[i]
        ox = int(round(draw_x + adv + i * extra_spacing))
        oy = int(round(draw_y))
        mask = Image.new("L", (canvas_w, canvas_h), 0)
        ImageDraw.Draw(mask).text((ox, oy), ch, font=font, fill=255)
        char_masks.append(mask)
        ch_toplefts.append((ox, oy))

    return char_masks, ch_toplefts, (canvas_w, canvas_h)


# ============================================================
# Overlap Detection
# ============================================================


def masks_overlap(m1_np, m2_np):
    h = max(m1_np.shape[0], m2_np.shape[0])
    w = max(m1_np.shape[1], m2_np.shape[1])
    a = np.zeros((h, w), dtype=np.uint8)
    a[: m1_np.shape[0], : m1_np.shape[1]] = m1_np
    b = np.zeros((h, w), dtype=np.uint8)
    b[: m2_np.shape[0], : m2_np.shape[1]] = m2_np
    return np.any((a > 0) & (b > 0))


def _safe_shapely(flat_pts):
    coords = [(flat_pts[i], flat_pts[i + 1]) for i in range(0, len(flat_pts), 2)]
    if len(coords) < 3:
        return None
    poly = ShapelyPolygon(coords)
    if not poly.is_valid:
        poly = make_valid(poly)
    return poly


def polygons_overlap(p1_pts, p2_pts):
    p1, p2 = _safe_shapely(p1_pts), _safe_shapely(p2_pts)
    if p1 is None or p2 is None:
        return False
    try:
        return p1.intersects(p2)
    except Exception:
        return False


def find_placement(bg_mask, word_mask_np, max_attempts=100):
    h_img, w_img = bg_mask.shape
    h_w, w_w = word_mask_np.shape
    if w_w > w_img or h_w > h_img:
        return None
    for _ in range(max_attempts):
        x = random.randint(0, w_img - w_w)
        y = random.randint(0, h_img - h_w)
        roi = bg_mask[y : y + h_w, x : x + w_w]
        if not np.any((roi > 0) & (word_mask_np > 0)):
            return x, y
    return None


# ============================================================
# Straight Word Placement
# ============================================================

MAX_EXTRA_SPACING = 12


def _translate_poly(poly, dx, dy):
    return [
        poly[k] + dx if k % 2 == 0 else poly[k] + dy for k in range(len(poly))
    ]


def _check_chars_overlap(char_masks):
    n = len(char_masks)
    for i in range(n):
        for j in range(i + 1, n):
            if masks_overlap(np.array(char_masks[i]), np.array(char_masks[j])):
                return True
    return False


def place_straight_word(bg_mask, word, font, config, placed_polygons):
    rot_range = config.get("text", {}).get("rotation_range", [-25, 25])

    for extra_spacing in range(0, MAX_EXTRA_SPACING + 1):
        char_masks, ch_toplefts, canvas_size = render_character_masks(
            word, font, extra_spacing
        )

        if _check_chars_overlap(char_masks):
            continue

        combined = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)
        for m in char_masks:
            combined = np.maximum(combined, np.array(m))
        combined_img = Image.fromarray(combined)

        angle = random.uniform(*rot_range)

        rotated_chars = [
            m.rotate(angle, expand=True, resample=Image.NEAREST) for m in char_masks
        ]
        if _check_chars_overlap(rotated_chars):
            continue

        rotated_combined = combined_img.rotate(
            angle, expand=True, resample=Image.NEAREST
        )
        rotated_bin = (np.array(rotated_combined) > 0).astype(np.uint8) * 255

        pos = find_placement(bg_mask, rotated_bin)
        if pos is None:
            continue

        paste_x, paste_y = pos
        rotated_w, rotated_h = rotated_combined.size
        center_x, center_y = canvas_size[0] / 2.0, canvas_size[1] / 2.0
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))

        chars_meta = []
        all_valid = True

        for i, (ox, oy) in enumerate(ch_toplefts):
            rch = rotated_chars[i]
            rch_w, rch_h = rch.size
            vx, vy = ox - center_x, oy - center_y
            vrot_x = vx * cos_a - vy * sin_a
            vrot_y = vx * sin_a + vy * cos_a
            final_x = int(paste_x + round(rotated_w / 2.0 + vrot_x - rch_w / 2.0))
            final_y = int(paste_y + round(rotated_h / 2.0 + vrot_y - rch_h / 2.0))

            topo = extract_topology_polygons(np.array(rch))
            if topo is None:
                all_valid = False
                break

            outer_t = [_translate_poly(p, final_x, final_y) for p in topo["outer"]]
            inner_t = [_translate_poly(p, final_x, final_y) for p in topo["inner"]]

            if any(polygons_overlap(pp, outer_t[0]) for pp in placed_polygons):
                all_valid = False
                break

            chars_meta.append({
                "char": word[i],
                "final_tl": (final_x, final_y),
                "rot_angle": angle,
                "orig_top_left": (ox, oy),
                "outer_polygons": outer_t,
                "inner_polygons": inner_t,
                "has_holes": topo["has_holes"],
            })

        if not all_valid or len(chars_meta) != len(word):
            continue

        bg_mask[
            paste_y : paste_y + rotated_bin.shape[0],
            paste_x : paste_x + rotated_bin.shape[1],
        ] |= rotated_bin
        for cm in chars_meta:
            placed_polygons.append(cm["outer_polygons"][0])

        return {
            "type": "straight",
            "canvas_size": canvas_size,
            "angle": angle,
            "chars_meta": chars_meta,
        }

    return None


# ============================================================
# Curved Word Placement (Fixed)
# ============================================================


def place_curved_word(bg_mask, word, font, config, placed_polygons):
    img_w = config.get("image", {}).get("width", 1024)
    img_h = config.get("image", {}).get("height", 1024)
    font_size = font.size if hasattr(font, "size") else 40

    try:
        advances = [font.getlength(word[:i]) for i in range(len(word) + 1)]
    except Exception:
        advances = [get_text_size(font, word[:i])[0] for i in range(len(word) + 1)]
    char_advances = [advances[i + 1] - advances[i] for i in range(len(word))]

    radius = random.randint(int(img_w * 0.15), int(min(img_w, img_h) * 0.4))
    start_angle = random.uniform(0, 360)
    circumference = 2 * math.pi * radius
    angular_widths = [
        (adv / max(1.0, circumference)) * 360.0 for adv in char_advances
    ]
    total_span = sum(angular_widths)
    cursor = start_angle - total_span / 2.0

    char_canvas_size = int(font_size * 4)
    char_center = char_canvas_size // 2

    char_data = []
    for i, ch in enumerate(word):
        char_angle = cursor + angular_widths[i] / 2.0
        rot_angle = char_angle + 90.0
        rad = math.radians(char_angle)
        arc_x = radius * math.cos(rad)
        arc_y = radius * math.sin(rad)

        mask = Image.new("L", (char_canvas_size, char_canvas_size), 0)
        ImageDraw.Draw(mask).text((char_center, char_center), ch, font=font, fill=255)
        rotated_mask = mask.rotate(rot_angle, expand=True, resample=Image.NEAREST)

        char_data.append({
            "char": ch,
            "arc_x": arc_x,
            "arc_y": arc_y,
            "rot_angle": rot_angle,
            "mask": mask,
            "rotated_mask": rotated_mask,
        })
        cursor += angular_widths[i]

    # Bounding box of all characters relative to arc center
    positions = []
    for cd in char_data:
        rw, rh = cd["rotated_mask"].size
        cx = cd["arc_x"] - rw / 2
        cy = cd["arc_y"] - rh / 2
        positions.append((cx, cy, cx + rw, cy + rh))

    min_x = min(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    max_x = max(p[2] for p in positions)
    max_y = max(p[3] for p in positions)

    word_w = int(max_x - min_x) + 4
    word_h = int(max_y - min_y) + 4
    if word_w > img_w or word_h > img_h or word_w <= 0 or word_h <= 0:
        return None

    # Build combined mask for placement
    combined_mask = np.zeros((word_h, word_w), dtype=np.uint8)
    char_offsets = []
    for cd, pos in zip(char_data, positions):
        local_x = int(pos[0] - min_x) + 2
        local_y = int(pos[1] - min_y) + 2
        rw, rh = cd["rotated_mask"].size
        rm_np = np.array(cd["rotated_mask"])

        paste_h = min(rh, word_h - local_y)
        paste_w = min(rw, word_w - local_x)
        if paste_h <= 0 or paste_w <= 0 or local_x < 0 or local_y < 0:
            return None

        combined_mask[
            local_y : local_y + paste_h, local_x : local_x + paste_w
        ] = np.maximum(
            combined_mask[local_y : local_y + paste_h, local_x : local_x + paste_w],
            rm_np[:paste_h, :paste_w],
        )
        char_offsets.append((local_x, local_y))

    combined_bin = (combined_mask > 0).astype(np.uint8) * 255
    placement = find_placement(bg_mask, combined_bin)
    if placement is None:
        return None

    place_x, place_y = placement

    # Extract polygons and check overlap
    chars_meta = []
    for cd, (local_x, local_y) in zip(char_data, char_offsets):
        final_x = place_x + local_x
        final_y = place_y + local_y

        topo = extract_topology_polygons(np.array(cd["rotated_mask"]))
        if topo is None:
            return None

        outer_t = [_translate_poly(p, final_x, final_y) for p in topo["outer"]]
        inner_t = [_translate_poly(p, final_x, final_y) for p in topo["inner"]]

        if any(polygons_overlap(pp, outer_t[0]) for pp in placed_polygons):
            return None

        chars_meta.append({
            "char": cd["char"],
            "final_tl": (final_x, final_y),
            "rot_angle": cd["rot_angle"],
            "mask": cd["mask"],
            "rotated_mask": cd["rotated_mask"],
            "outer_polygons": outer_t,
            "inner_polygons": inner_t,
            "has_holes": topo["has_holes"],
        })

    bg_mask[
        place_y : place_y + combined_bin.shape[0],
        place_x : place_x + combined_bin.shape[1],
    ] |= combined_bin
    for cm in chars_meta:
        placed_polygons.append(cm["outer_polygons"][0])

    return {"type": "curved", "chars_meta": chars_meta}


# ============================================================
# Image Generation
# ============================================================


def generate_image(image_id, fonts, config, bg_provider, augmenter):
    img_w = config.get("image", {}).get("width", 1024)
    img_h = config.get("image", {}).get("height", 1024)
    text_cfg = config.get("text", {})
    colors_cfg = config.get("colors", {})
    effects_cfg = config.get("effects", {})
    font_cfg = config.get("fonts", {})

    min_chars = text_cfg.get("min_chars_per_image", 80)
    max_attempts = text_cfg.get("max_placement_attempts", 600)
    curved_prob = text_cfg.get("curved_word_prob", 0.2)
    same_color_prob = colors_cfg.get("same_color_word_prob", 0.6)
    contrast_thresh = colors_cfg.get("contrast_threshold", 3.5)
    font_size_range = font_cfg.get("size_range", [18, 80])

    faker = Faker()
    bg = bg_provider.get().resize((img_w, img_h))
    bg_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    placed_polygons = []
    annotations = []
    word_id = 0
    placed_chars = 0
    placed_straight = False

    for _ in range(max_attempts):
        if placed_chars >= min_chars:
            break

        word = generate_text_content(faker, config)
        word = "".join(ch for ch in word if ch.isprintable() and ch not in "\n\r\t")
        if not word:
            continue

        font_path = random.choice(fonts)
        font_size = random.randint(*font_size_range)
        font = safe_truetype(font_path, font_size)

        if not font_supports_chars(font, word):
            continue

        do_curved = (
            random.random() < curved_prob if placed_straight else False
        )

        if not do_curved:
            result = place_straight_word(bg_mask, word, font, config, placed_polygons)
        else:
            result = place_curved_word(bg_mask, word, font, config, placed_polygons)

        if result is None:
            continue

        word_id += 1
        if result["type"] == "straight":
            placed_straight = True

        # Decide per-word rendering style
        use_same_color = random.random() < same_color_prob
        first_cm = result["chars_meta"][0]
        first_bg_color = sample_background_color(
            bg, first_cm["final_tl"][0], first_cm["final_tl"][1]
        )
        word_color = pick_contrasting_color(first_bg_color, threshold=contrast_thresh)

        apply_shadow = random.random() < effects_cfg.get("shadow", {}).get("prob", 0)
        shadow_offset = (
            random.randint(*effects_cfg.get("shadow", {}).get("offset_range", [2, 4]))
            if apply_shadow
            else 0
        )
        apply_outline = random.random() < effects_cfg.get("outline", {}).get("prob", 0)
        outline_width = (
            random.randint(*effects_cfg.get("outline", {}).get("width_range", [1, 2]))
            if apply_outline
            else 0
        )

        for idx, cm in enumerate(result["chars_meta"]):
            ch = cm["char"]
            final_x, final_y = cm["final_tl"]

            if use_same_color:
                color = word_color
            else:
                local_bg = sample_background_color(bg, final_x, final_y)
                color = pick_contrasting_color(local_bg, threshold=contrast_thresh)

            if result["type"] == "straight":
                canvas_w, canvas_h = result["canvas_size"]
                orig_ox, orig_oy = cm["orig_top_left"]
                single = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(single)

                if apply_shadow:
                    shadow_c = tuple(max(0, c - 80) for c in color) + (100,)
                    draw.text(
                        (orig_ox + shadow_offset, orig_oy + shadow_offset),
                        ch, font=font, fill=shadow_c,
                    )

                if apply_outline:
                    ol_c = (
                        (0, 0, 0, 255) if luminance(color) > 0.5
                        else (255, 255, 255, 255)
                    )
                    draw.text(
                        (orig_ox, orig_oy), ch, font=font,
                        fill=color + (255,),
                        stroke_width=outline_width, stroke_fill=ol_c,
                    )
                else:
                    draw.text((orig_ox, orig_oy), ch, font=font, fill=color + (255,))

                rotated = single.rotate(
                    cm["rot_angle"], expand=True, resample=Image.BICUBIC
                )
                bg.paste(rotated, (final_x, final_y), rotated)

            else:
                mask_img = cm["mask"]
                rot_angle = cm["rot_angle"]
                colored = Image.new("RGBA", mask_img.size, color + (255,))
                colored.putalpha(mask_img)
                rotated_colored = colored.rotate(
                    rot_angle, expand=True, resample=Image.BICUBIC
                )
                bg.paste(rotated_colored, (final_x, final_y), rotated_colored)

            poly_data = {
                "outer": cm["outer_polygons"],
                "inner": cm["inner_polygons"],
                "has_holes": cm["has_holes"],
            }
            annotations.append({
                "char": ch,
                "polygon_data": poly_data,
                "word_id": word_id,
                "bbox": _compute_bbox(cm["outer_polygons"][0]),
            })
            placed_chars += 1

    bg = augmenter.apply(bg)
    return bg, annotations, placed_chars


def _compute_bbox(flat_poly):
    xs = [flat_poly[i] for i in range(0, len(flat_poly), 2)]
    ys = [flat_poly[i] for i in range(1, len(flat_poly), 2)]
    x_min, y_min = min(xs), min(ys)
    return [x_min, y_min, max(xs) - x_min, max(ys) - y_min]


# ============================================================
# Annotation Writers
# ============================================================


def write_csv(path, annotations):
    with open(path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["char", "polygon_json", "word_id"])
        for ann in annotations:
            writer.writerow([
                ann["char"],
                json.dumps(ann["polygon_data"]),
                ann["word_id"],
            ])


def build_coco_json(all_images, all_annotations):
    coco = {
        "info": {
            "description": "CharSeg Synthetic Character Segmentation Dataset",
            "version": "7.0",
            "contributor": "CharSeg Generator v7",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "character", "supercategory": "text"}
        ],
    }

    ann_id = 1
    for img_entry, img_anns in zip(all_images, all_annotations):
        coco["images"].append(img_entry)
        for ann in img_anns:
            poly_data = ann["polygon_data"]
            area = _poly_area(poly_data["outer"][0]) if poly_data["outer"] else 0

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_entry["id"],
                "category_id": 1,
                "segmentation": poly_data["outer"],
                "area": round(area, 2),
                "bbox": [round(b, 2) for b in ann["bbox"]],
                "iscrowd": 0,
                "attributes": {
                    "character": ann["char"],
                    "word_id": ann["word_id"],
                    "has_holes": poly_data["has_holes"],
                    "inner_contours": poly_data["inner"],
                },
            })
            ann_id += 1

    return coco


# ============================================================
# Main Pipeline
# ============================================================


def generate_split(split_name, num_images, fonts, config, output_dir):
    split_img_dir = os.path.join(output_dir, "images", split_name)
    split_anno_dir = os.path.join(output_dir, "annotations", split_name)
    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_anno_dir, exist_ok=True)

    bg_provider = BackgroundProvider(config)
    augmenter = AugmentationPipeline(config)
    formats = config.get("output", {}).get("formats", ["coco_json", "csv"])
    log_every = config.get("pipeline", {}).get("log_every", 50)

    all_images = []
    all_annotations = []
    total_chars = 0

    log.info(f"Generating '{split_name}' split: {num_images} images")

    for i in tqdm(range(1, num_images + 1), desc=split_name):
        img_name = f"{i:05d}.png"
        bg, annotations, placed = generate_image(
            i, fonts, config, bg_provider, augmenter
        )
        bg.save(os.path.join(split_img_dir, img_name))

        if "csv" in formats:
            write_csv(
                os.path.join(split_anno_dir, f"{i:05d}.csv"), annotations
            )

        img_w = config.get("image", {}).get("width", 1024)
        img_h = config.get("image", {}).get("height", 1024)
        all_images.append({
            "id": i,
            "file_name": f"{split_name}/{img_name}",
            "width": img_w,
            "height": img_h,
        })
        all_annotations.append(annotations)
        total_chars += placed

        if i % log_every == 0:
            avg = total_chars / i
            log.info(f"  [{split_name}] {i}/{num_images}, avg {avg:.1f} chars/img")

    if "coco_json" in formats:
        coco = build_coco_json(all_images, all_annotations)
        coco_path = os.path.join(output_dir, "annotations", f"{split_name}.json")
        with open(coco_path, "w") as f:
            json.dump(coco, f)
        log.info(f"  COCO JSON saved: {coco_path}")

    avg = total_chars / max(1, num_images)
    log.info(
        f"  {split_name} complete: {total_chars} total chars, "
        f"avg {avg:.1f}/img"
    )
    return total_chars


def main():
    parser = argparse.ArgumentParser(description="CharSeg Dataset Generator v7")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-val", type=int, default=None)
    parser.add_argument("--num-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--font-dir", type=str, default=None)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)
        log.info(f"Loaded config from {args.config}")

    if args.output_dir:
        config.setdefault("output", {})["dir"] = args.output_dir
    if args.seed is not None:
        config.setdefault("pipeline", {})["seed"] = args.seed
    if args.font_dir:
        config.setdefault("fonts", {})["dir"] = args.font_dir

    splits = config.get("splits", {})
    if args.num_train is not None:
        splits["train"] = args.num_train
    if args.num_val is not None:
        splits["val"] = args.num_val
    if args.num_test is not None:
        splits["test"] = args.num_test

    seed = config.get("pipeline", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    log.info(f"Random seed: {seed}")

    font_dir = config.get("fonts", {}).get("dir", "fonts")
    fonts = list_fonts(font_dir)
    if not fonts:
        raise SystemExit(
            f"No fonts found in '{font_dir}'. "
            "Run prototype/font_download/download_curated_fonts.py first."
        )
    log.info(f"Found {len(fonts)} fonts in {font_dir}")

    output_dir = config.get("output", {}).get("dir", "dataset_v7")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    with open(os.path.join(output_dir, "generation_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    log.info(f"Config saved to {output_dir}/generation_config.yaml")

    total = 0
    for split_name, num_images in splits.items():
        if num_images <= 0:
            continue
        total += generate_split(split_name, num_images, fonts, config, output_dir)

    log.info(f"Generation complete! {total} total characters.")
    log.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
