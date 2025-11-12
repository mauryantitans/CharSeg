# Font Download Scripts

Two variants for downloading Google Fonts for dataset generation.

## 📥 Variants

### Variant 1: `download_all_fonts.py`
Downloads **ALL** fonts from Google Fonts repository.

**Pros:**
- Maximum font variety
- Comprehensive coverage

**Cons:**
- ~1GB download
- Thousands of fonts (slow extraction)
- Many obscure/specialty fonts

**Usage:**
```bash
python download_all_fonts.py
```

### Variant 2: `download_curated_fonts.py` ⭐ RECOMMENDED
Downloads **100+ commonly used fonts** only.

**Pros:**
- Much faster
- Practical selection of popular fonts
- Smaller storage footprint

**Cons:**
- Limited to curated list

**Usage:**
```bash
python download_curated_fonts.py
```

## 🎯 Which to Use?

- **For most cases**: Use Variant 2 (curated)
- **For research/maximum diversity**: Use Variant 1 (all fonts)

## 📝 Configuration

Edit the `FONT_DIR` variable in each script to change the output directory:

```python
FONT_DIR = "fonts"  # Change this to your desired path
```

## ⚙️ Requirements

```bash
pip install requests
```

## 📤 Output

Both scripts create a folder with `.ttf` font files that can be used by the dataset generators.

Default output locations:
- Variant 1: `/content/drive/MyDrive/Project/fonts_all`
- Variant 2: `fonts/`
