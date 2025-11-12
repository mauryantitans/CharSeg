"""
Font Download Script - Variant 2: CURATED FONTS
Downloads a curated list of 100+ commonly used fonts from Google Fonts.

This is much faster and more practical than downloading all fonts.
Recommended for most use cases.
"""
import os
import zipfile
import requests

# Configuration
FONT_DIR = "fonts"  # Change this path as needed
os.makedirs(FONT_DIR, exist_ok=True)

# 100+ commonly used fonts folder names (Google Fonts repo folder names, lowercase, no spaces)
GOOGLE_FONTS_TO_DOWNLOAD = [
    "roboto", "opensans", "lato", "montserrat", "oswald",
    "sourcesanspro", "raleway", "ptsans", "lora", "nunito",
    "merriweather", "poppins", "work-sans", "inconsolata", "playfairdisplay",
    "ubuntu", "quicksand", "oxygen", "arimo", "titilliumweb",
    "karla", "bitter", "fira-sans", "noto-serif", "rubik",
    "hind", "cantarell", "cabin", "ptserif", "exo2",
    "domine", "crimsontext", "glegoo", "heebo", "josefin-sans",
    "pt-sans-narrow", "spectral", "libre-baskerville", "manrope", "ibm-plex-sans",
    "catamaran", "asap", "abril-fatface", "signika", "raleway-dots",
    "teko", "koho", "shadows-into-light", "courier-prime", "dosis",
    "balsamiq-sans", "mukta", "fjalla-one", "cormorant-garamond", "exo",
    "chivo", "droid-sans", "kaushan-script", "raleway", "cantora-one",
    "nunito-sans", "julius-sans-one", "hind-siliguri", "kaushan-script", "anton",
    "amaranth", "lobster", "lobster-two", "libre-franklin", "nunito-sans",
    "patua-one", "abril-fatface", "aleo", "baloo", "concert-one",
    "david-libre", "dm-sans", "fira-code", "francois-one", "glory",
    "imprima", "jura", "kalam", "kumamoto", "lexend",
    "meddon", "monoton", "montaga", "neuton", "old-standard-tt",
    "pacifico", "pt-mono", "roboto-condensed", "rock-salt", "segoe-ui"
]

print("Downloading Google Fonts ZIP (~1GB)...")
zip_url = "https://github.com/google/fonts/archive/main.zip"
zip_path = "google_fonts.zip"

if not os.path.exists(zip_path):
    r = requests.get(zip_url, stream=True)
    r.raise_for_status()
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
else:
    print("ZIP file already exists, skipping download.")

print("Extracting selected fonts (all styles)...")
extracted_count = 0
with zipfile.ZipFile(zip_path, 'r') as z:
    for member in z.namelist():
        for font_name in GOOGLE_FONTS_TO_DOWNLOAD:
            # Look for folder containing font_name, then .ttf files inside it
            if f"/{font_name}/" in member.lower() and member.lower().endswith(".ttf"):
                target_path = os.path.join(FONT_DIR, os.path.basename(member))
                if not os.path.exists(target_path):
                    with open(target_path, 'wb') as out_file:
                        out_file.write(z.read(member))
                    print(f"Saved: {target_path}")
                    extracted_count += 1

print(f"\n--- Font download process complete. ---")
print(f"Extracted {extracted_count} font files to: {FONT_DIR}")
