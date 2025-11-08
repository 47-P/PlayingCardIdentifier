# laplacian_edges.py
# Requires: pip install opencv-python numpy

import cv2
import numpy as np
from pathlib import Path

# ───────── CONFIG: EDIT THIS ─────────
IMAGE_PATH = r"C:\DESKTOP SHIT\Image processing\Project 4\vecteezy_set-of-poker-cards-with-isolated-on-white-background-poker_53654344.jpg"   # <- change me
OUT_DIR    = Path("./laplacian_output")
KSIZE      = 3   # Laplacian aperture size (must be 1, 3, 5, or 7 for cv2.Laplacian)
# ─────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load image
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Could not read image at: {IMAGE_PATH}")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- A) OpenCV's Laplacian (uses second derivatives) ---
# Use CV_16S to avoid clipping negatives, then convert to uint8
lap_cv16 = cv2.Laplacian(gray, ddepth=cv2.CV_16S, ksize=KSIZE, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
lap_cv = cv2.convertScaleAbs(lap_cv16)  # |value| and rescale to [0,255]

# --- B) Manual Laplacian via a 3×3 kernel and filter2D ---
# Standard 3×3 kernel (4-neighborhood). You can swap for 8-neighborhood if desired.
# 4-neighborhood:
kernel_3x3 = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]], dtype=np.float32)
# Alternative 8-neighborhood (uncomment to try):
# kernel_3x3 = np.array([[1,  1, 1],
#                        [1, -8, 1],
#                        [1,  1, 1]], dtype=np.float32)

filtered16 = cv2.filter2D(src=gray, ddepth=cv2.CV_16S, kernel=kernel_3x3, borderType=cv2.BORDER_DEFAULT)
lap_manual = cv2.convertScaleAbs(filtered16)

# Optional: normalize for visualization consistency (not strictly necessary)
# lap_cv = cv2.normalize(lap_cv, None, 0, 255, cv2.NORM_MINMAX)
# lap_manual = cv2.normalize(lap_manual, None, 0, 255, cv2.NORM_MINMAX)

# Build a mosaic for quick viewing
def to_bgr(u8):
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

mosaic = np.hstack([to_bgr(gray), to_bgr(lap_cv), to_bgr(lap_manual)])

# Save outputs
input_name = Path(IMAGE_PATH).stem
cv2.imwrite(str(OUT_DIR / f"{input_name}_gray.png"), gray)
cv2.imwrite(str(OUT_DIR / f"{input_name}_laplacian_cv.png"), lap_cv)
cv2.imwrite(str(OUT_DIR / f"{input_name}_laplacian_manual.png"), lap_manual)
cv2.imwrite(str(OUT_DIR / f"{input_name}_mosaic_gray_cv_manual.png"), mosaic)

# Show (press any key to close)
cv2.imshow("Laplacian: [Gray | OpenCV | Manual Kernel]", mosaic)
cv2.waitKey(0)
cv2.destroyAllWindows()
