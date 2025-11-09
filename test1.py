# card_identifier_menu_fixed.py
# PyQt6 app with reliable Menu <-> Viewer navigation using QStackedWidget
# Requirements: pip install opencv-python PyQt6 numpy

from __future__ import annotations
import os, sys, cv2, numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFileDialog, QHBoxLayout,
    QVBoxLayout, QPushButton, QStatusBar, QMessageBox, QToolBar, QStackedWidget
)

# ----------------------- CONFIG -----------------------
TEMPLATES_DIR = r"C:\DESKTOP SHIT\Image processing\Project 4\OpenCV-Playing-Card-Detector\Card_Imgs"  # <--- EDIT THIS PATH
BKG_THRESH = 60
CARD_THRESH = 30
CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000
RANK_WIDTH, RANK_HEIGHT = 70, 125
SUIT_WIDTH, SUIT_HEIGHT = 70, 100
CORNER_WIDTH, CORNER_HEIGHT = 32, 84
RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700
FONT = cv2.FONT_HERSHEY_SIMPLEX

# -------------------- Data holders --------------------
class TrainImage:
    def __init__(self, name, img):
        self.name = name
        self.img = img  # grayscale template

class QueryCard:
    def __init__(self):
        self.contour = None
        self.center = None
        self.warp = None
        self.rank_img = None
        self.suit_img = None
        self.best_rank = "Unknown"
        self.best_suit = "Unknown"
        self.rank_diff = 0
        self.suit_diff = 0

# -------------------- Core functions ------------------
def load_templates(templates_dir):
    ranks = ['Ace','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten','Jack','Queen','King']
    suits = ['Spades','Diamonds','Clubs','Hearts']
    def rd(p):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"Missing template: {p}")
        return im
    rank_t = [TrainImage(r, rd(os.path.join(templates_dir, f"{r}.jpg"))) for r in ranks]
    suit_t = [TrainImage(s, rd(os.path.join(templates_dir, f"{s}.jpg"))) for s in suits]
    return rank_t, suit_t

def preprocess_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    h, w = gray.shape
    bkg_level = gray[max(1, h//100), w//2]
    thresh_level = int(min(255, bkg_level + BKG_THRESH))
    _, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)
    return thresh

def find_card_contours(thresh_img):
    info = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(info) == 3:
        _, cnts, hier = info
    else:
        cnts, hier = info
    if not cnts or hier is None or len(hier) == 0:
        return [], []
    idx = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)
    cnts_sorted = [cnts[i] for i in idx]
    hier_sorted = [hier[0][i] for i in idx]
    is_card = np.zeros(len(cnts_sorted), dtype=int)
    for i, c in enumerate(cnts_sorted):
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01*peri, True)
        if (CARD_MIN_AREA < area < CARD_MAX_AREA) and (hier_sorted[i][3] == -1) and (len(approx) == 4):
            is_card[i] = 1
    return cnts_sorted, is_card

def _order_points_for_warp(pts, w, h):
    rect = np.zeros((4,2), dtype="float32")
    s = np.sum(pts, axis=2)
    tl = pts[np.argmin(s)][0]; br = pts[np.argmax(s)][0]
    diff = np.diff(pts, axis=-1)
    tr = pts[np.argmin(diff)][0]; bl = pts[np.argmax(diff)][0]
    if w <= 0.8*h:      rect[:] = [tl, tr, br, bl]            # vertical
    elif w >= 1.2*h:    rect[:] = [bl, tl, tr, br]            # horizontal
    else:               rect[:] = [tl, tr, br, bl]            # fallback
    return rect

def flattener(image_bgr, contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01*peri, True)
    pts = np.float32(approx)
    x, y, w, h = cv2.boundingRect(contour)
    rect = _order_points_for_warp(pts, w, h)
    dst = np.array([[0,0],[199,0],[199,299],[0,299]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image_bgr, M, (200, 300))
    return cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

def extract_rank_suit_from_warp(warp_gray):
    corner = warp_gray[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    zoom = cv2.resize(corner, (0,0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    white_level = int(zoom[15, (CORNER_WIDTH*4)//2])
    thresh_level = max(1, white_level - CARD_THRESH)
    _, q = cv2.threshold(zoom, thresh_level, 255, cv2.THRESH_BINARY_INV)
    Qrank = q[20:185, 0:128]; Qsuit = q[186:336, 0:128]

    def biggest(bin_img):
        info = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(info) == 3: _, cnts, _ = info
        else:              cnts, _   = info
        if not cnts: return None
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        x,y,w,h = cv2.boundingRect(cnts[0])
        return bin_img[y:y+h, x:x+w]

    rank_roi = biggest(Qrank); suit_roi = biggest(Qsuit)
    rank_img = cv2.resize(rank_roi, (RANK_WIDTH, RANK_HEIGHT), interpolation=cv2.INTER_NEAREST) if rank_roi is not None else None
    suit_img = cv2.resize(suit_roi, (SUIT_WIDTH, SUIT_HEIGHT), interpolation=cv2.INTER_NEAREST) if suit_roi is not None else None
    return rank_img, suit_img

def match_card(rank_img, suit_img, rank_templates, suit_templates):
    best_rank, best_suit = "Unknown", "Unknown"
    best_rd, best_sd = 10_000, 10_000
    if rank_img is not None:
        for tr in rank_templates:
            score = int(np.sum(cv2.absdiff(rank_img, tr.img)) / 255)
            if score < best_rd: best_rd, best_rank = score, tr.name
    if suit_img is not None:
        for ts in suit_templates:
            score = int(np.sum(cv2.absdiff(suit_img, ts.img)) / 255)
            if score < best_sd: best_sd, best_suit = score, ts.name
    if best_rd >= RANK_DIFF_MAX: best_rank = "Unknown"
    if best_sd >= SUIT_DIFF_MAX: best_suit = "Unknown"
    return best_rank, best_suit, best_rd, best_sd

def annotate_frame(image_bgr, cards):
    if not cards: return image_bgr
    cv2.drawContours(image_bgr, [c.contour for c in cards if c.contour is not None], -1, (255,0,0), 2)
    for c in cards:
        if c.center is None: continue
        x, y = c.center
        cv2.circle(image_bgr, (x,y), 5, (255,0,0), -1)
        t1 = f"{c.best_rank} of"; t2 = f"{c.best_suit}"
        cv2.putText(image_bgr, t1, (x-60, y-10), FONT, 1, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(image_bgr, t1, (x-60, y-10), FONT, 1, (50,200,200), 2, cv2.LINE_AA)
        cv2.putText(image_bgr, t2, (x-60, y+25), FONT, 1, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(image_bgr, t2, (x-60, y+25), FONT, 1, (50,200,200), 2, cv2.LINE_AA)
    return image_bgr

def identify_cards_in_frame(frame_bgr, rank_t, suit_t):
    thresh = preprocess_image(frame_bgr)
    cnts, is_card = find_card_contours(thresh)
    cards = []
    for i, c in enumerate(cnts):
        if i >= len(is_card) or is_card[i] != 1: continue
        qc = QueryCard()
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01*peri, True)
        avg = np.mean(approx.reshape(-1,2), axis=0).astype(int)
        qc.center = [int(avg[0]), int(avg[1])]
        qc.warp = flattener(frame_bgr, c)
        qc.rank_img, qc.suit_img = extract_rank_suit_from_warp(qc.warp)
        rn, sn, rd, sd = match_card(qc.rank_img, qc.suit_img, rank_t, suit_t)
        qc.best_rank, qc.best_suit, qc.rank_diff, qc.suit_diff = rn, sn, rd, sd
        qc.contour = c
        cards.append(qc)
    out = annotate_frame(frame_bgr, cards)
    return out, cards

# ------------------------ UI ---------------------------
class CardIdentifierApp(QMainWindow):
    IDX_MENU = 0
    IDX_VIEW = 1

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Card Identifier")
        self.resize(1100, 700)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # capture keys (C/Esc)

        # Load templates
        try:
            self.rank_t, self.suit_t = load_templates(TEMPLATES_DIR)
        except Exception as e:
            QMessageBox.critical(self, "Template Error", str(e))
            self.rank_t, self.suit_t = [], []

        # Status bar
        self.status = QStatusBar(); self.setStatusBar(self.status)

        # Toolbar: Home always available
        tb = QToolBar("Main"); self.addToolBar(tb)
        self.act_home = QAction("Home", self)
        self.act_home.triggered.connect(self.go_menu)
        tb.addAction(self.act_home)

        # Build screens
        self.stack = QStackedWidget(self)
        self.menu_screen = self._build_menu()
        self.viewer_screen = self._build_viewer()
        self.stack.addWidget(self.menu_screen)   # index 0
        self.stack.addWidget(self.viewer_screen) # index 1
        self.setCentralWidget(self.stack)
        self.stack.setCurrentIndex(self.IDX_MENU)

        # Camera state
        self.cap = None
        self.timer = QTimer(self); self.timer.timeout.connect(self._on_frame)
        self.last_frame = None

    # ----- Screens -----
    def _build_menu(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        title = QLabel("Choose Input"); title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24px;")
        btn_img = QPushButton("Load Image"); btn_img.setMinimumHeight(48)
        btn_cam = QPushButton("Use Webcam"); btn_cam.setMinimumHeight(48)
        btn_img.clicked.connect(self.menu_open_image)
        btn_cam.clicked.connect(self.menu_open_camera)
        lay.addWidget(title); lay.addSpacing(20)
        lay.addWidget(btn_img); lay.addWidget(btn_cam); lay.addStretch(1)
        return w

    def _build_viewer(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        self.video_label = QLabel("")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 450)
        lay.addWidget(self.video_label, stretch=1)

        info = QLabel("Webcam: press 'C' to capture and detect.  |  Press Esc or Home to return to Menu.")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet("color: gray;")
        lay.addWidget(info)

        btn_row = QHBoxLayout()
        self.btn_back = QPushButton("Back to Menu")
        self.btn_back.clicked.connect(self.go_menu)
        btn_row.addStretch(1); btn_row.addWidget(self.btn_back)
        lay.addLayout(btn_row)
        return w

    # ----- Navigation helpers -----
    def go_menu(self):
        # Always stop camera and reset viewer before switching
        self.stop_camera()
        self._clear_viewer()
        self.stack.setCurrentIndex(self.IDX_MENU)
        self.status.showMessage("Select an option to begin.")

    def go_viewer(self):
        self.stack.setCurrentIndex(self.IDX_VIEW)

    # ----- Menu actions -----
    def menu_open_image(self):
        if not self.rank_t or not self.suit_t:
            QMessageBox.warning(self, "Templates", "Templates not loaded. Fix TEMPLATES_DIR and restart.")
            return
        self.go_viewer()
        self.open_image()

    def menu_open_camera(self):
        if not self.rank_t or not self.suit_t:
            QMessageBox.warning(self, "Templates", "Templates not loaded. Fix TEMPLATES_DIR and restart.")
            return
        self.go_viewer()
        self.start_camera()

    # ----- Image flow -----
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if not path:
            self.status.showMessage("No file selected."); return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.critical(self, "Error", f"Failed to read: {path}"); return
        vis, cards = identify_cards_in_frame(img, self.rank_t, self.suit_t)
        self._show_bgr(vis)
        if cards:
            txt = f"{cards[0].best_rank} of {cards[0].best_suit}"
            self.status.showMessage(f"Detected: {txt}  |  Total cards: {len(cards)}")
        else:
            self.status.showMessage("No card-like contours detected.")

    # ----- Webcam flow -----
    def start_camera(self):
        self.stop_camera()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera", "Cannot open camera."); self.cap = None; return
        self.status.showMessage("Camera started. Press 'C' to capture and detect. Esc/Home to return to Menu.")
        self.timer.start(16)  # ~60 FPS

    def stop_camera(self):
        if self.timer.isActive(): self.timer.stop()
        if self.cap:
            self.cap.release(); self.cap = None

    def _on_frame(self):
        if not self.cap: return
        ret, frame = self.cap.read()
        if not ret: return
        self.last_frame = frame
        # Instruction bar
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 1 - 0.55, 0, frame)
        cv2.putText(frame, "Press 'C' to capture and detect | Esc/Home to Menu",
                    (12, 32), FONT, 0.7, (255,255,255), 2, cv2.LINE_AA)
        self._show_bgr(frame)

    # Capture on key 'C'; Esc goes Home
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_C:
            if self.cap and self.last_frame is not None:
                frame = self.last_frame.copy()
                self.stop_camera()              # freeze for result
                vis, cards = identify_cards_in_frame(frame, self.rank_t, self.suit_t)
                self._show_bgr(vis)
                if cards:
                    txt = f"{cards[0].best_rank} of {cards[0].best_suit}"
                    self.status.showMessage(f"[Captured] {txt}  |  Cards: {len(cards)}")
                else:
                    self.status.showMessage("[Captured] No card-like contours.")
                event.accept(); return
        elif key in (Qt.Key.Key_Escape,):
            self.go_menu(); event.accept(); return
        super().keyPressEvent(event)

    # ----- Utils -----
    def _show_bgr(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(),
                       Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        )

    def _clear_viewer(self):
        # Clear the label to avoid showing stale frames after returning to menu
        blank = np.zeros((450, 800, 3), dtype=np.uint8)
        self._show_bgr(blank)

# ------------------------ Main --------------------------
def main():
    app = QApplication(sys.argv)
    win = CardIdentifierApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
