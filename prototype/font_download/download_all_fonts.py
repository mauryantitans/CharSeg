"""
Font Download Script - Variant 1: ALL FONTS
Downloads the entire Google Fonts repository and extracts all .ttf files.

⚠️ WARNING: This downloads ~1GB and extracts thousands of fonts.
Use variant 2 (download_curated_fonts.py) for a smaller, curated selection.
"""
import os
import zipfile
import requests

# Configuration
FONT_DIR = "/content/drive/MyDrive/Project/fonts_all"  # Change this path as needed
os.makedirs(FONT_DIR, exist_ok=True)

# Step 1: Download the Google Fonts repo ZIP
print("Downloading Google Fonts ZIP (~1GB)...")
zip_url = "https://github.com/google/fonts/archive/main.zip"
zip_path = "google_fonts.zip"

if not os.path.exists(zip_path):
    with requests.get(zip_url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
else:
    print("ZIP file already exists, skipping download.")

# Step 2: Extract ALL .ttf font files
print("Extracting all fonts...")
with zipfile.ZipFile(zip_path, 'r') as z:
    for member in z.namelist():
        if member.lower().endswith(".ttf"):
            target_path = os.path.join(FONT_DIR, os.path.basename(member))
            if not os.path.exists(target_path):
                with open(target_path, 'wb') as out_file:
                    out_file.write(z.read(member))
                print(f"Saved: {target_path}")

print("\n--- All fonts download complete. ---")
print(f"Fonts saved to: {FONT_DIR}")
