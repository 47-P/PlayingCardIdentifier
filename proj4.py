
# otsu_mosaic_viewer.py
# Requires: opencv-python, numpy
# Usage: just edit INPUT_DIR and run:  python otsu_mosaic_viewer.py

import cv2
import numpy as np
from pathlib import Path

# ───────── CONFIG: EDIT THESE ─────────
INPUT_DIR  = Path(r"C:\DESKTOP SHIT\Image processing\Project 4\Playing Cards.v3-original_raw-images.coco")
FILE_GLOB  = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
PAUSE_MS   = 0   # 0 = wait for key each image; >0 = auto-advance (milliseconds)
# ──────────────────────────────────────

def iter_files(folder: Path, patterns):
    for pat in patterns:
        for p in sorted(folder.glob(pat)):
            yield p

def show_otsu_mosaic(img_path: Path):
    # Load grayscale
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"[skip] Could not read: {img_path}")
        return

    # Otsu threshold
    # (Using binary + OTSU; tweak to THRESH_BINARY_INV if you prefer inverted)
    _thr, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Make a side-by-side mosaic: original | binarized
    # Ensure both are single-channel uint8 and same size (they are).
    mosaic = cv2.hconcat([gray, bin_img])

    # Display
    title = f"{img_path.name}  |  left: original, right: Otsu"
    cv2.imshow(title, mosaic)
    key = cv2.waitKey(PAUSE_MS)
    cv2.destroyWindow(title)

    # Optional: exit early on ESC
    if key == 27:  # ESC
        raise KeyboardInterrupt     

def main():
    if not INPUT_DIR.exists():
        print(f"[error] INPUT_DIR does not exist: {INPUT_DIR}")
        return

    files = list(iter_files(INPUT_DIR, FILE_GLOB))
    if not files:
        print(f"[warn] No images found in {INPUT_DIR} with patterns {FILE_GLOB}")
        return

    try:
        for f in files:
            show_otsu_mosaic(f)     
    except KeyboardInterrupt:
        print("\n[info] Stopped by user.")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

