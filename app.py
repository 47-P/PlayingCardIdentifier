import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

def order_quad(pts):
    pts = pts.reshape(4,2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

sobel_kernel_x = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]], dtype=np.float32)

sobel_kernel_y = np.array([
    [1,  2,  1],
    [0,  0,  0],
    [-1, -2, -1]], dtype=np.float32)

def process_single_image_array(img_data, display_name="image"):
    original_bgr = img_data.copy()
    gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    img_edgex = cv2.filter2D(blur.astype(np.float64), cv2.CV_64F, sobel_kernel_x)
    img_edgey = cv2.filter2D(blur.astype(np.float64), cv2.CV_64F, sobel_kernel_y)
    edges_mag = np.hypot(img_edgex, img_edgey)
    if edges_mag.max() > 0:
        edges_norm = (edges_mag / edges_mag.max() * 255.0).astype(np.uint8)
    else:
        edges_norm = edges_mag.astype(np.uint8)
    _, edges_bin = cv2.threshold(edges_norm, 50, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(edges_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print(f"No contours found: {display_name}")
        output_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    else:
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quad = order_quad(approx)
            (tl, tr, br, bl) = quad
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxW = int(max(widthA, widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxH = int(max(heightA, heightB))
            dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(quad, dst)
            warped = cv2.warpPerspective(img_data, M, (maxW, maxH))
            h,w = warped.shape[:2]
            if w > h:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            output_img = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            rect = cv2.minAreaRect(c)
            angle = rect[2]
            if rect[1][0] < rect[1][1]:
                angle = angle
            else:
                angle = angle + 90.0
            center = tuple(np.array(img_data.shape[1::-1]) / 2)
            Mrot = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_data, Mrot, (img_data.shape[1], img_data.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            pts = c.reshape(-1,2)
            ones = np.ones((pts.shape[0],1))
            pts_h = np.hstack([pts, ones])
            trans = (Mrot @ pts_h.T).T
            x,y,w_box,h_box = cv2.boundingRect(trans.astype(np.int32))
            crop = rotated[y:y+h_box, x:x+w_box]
            if crop.shape[1] > crop.shape[0]:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            output_img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    ax1.axis("off")
    ax1.set_title("Image")
    ax2.imshow(output_img, "grey")
    ax2.axis("off")
    ax2.set_title("Upright & Cropped")
    plt.suptitle(display_name)
    plt.show()

def select_single_image_and_process():
    path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
    if not path:
        return
    root.destroy()
    img = cv2.imread(path)
    if img is None:
        messagebox.showerror("Error", f"Failed to read {path}")
        return
    process_single_image_array(img, display_name=os.path.basename(path))

def run_camera_mode():
    try:
        root.destroy()
    except Exception:
        pass
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Cannot open camera.")
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    instructions = ["Press 'c' to capture", "Press 'q' to quit"]
    rect_height = 60
    alpha = 0.55
    window_name = 'Camera (press c to capture, q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, rect_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        y0 = 20
        for i, line in enumerate(instructions):
            y = y0 + i * 20
            cv2.putText(frame, line, (10, y), font, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, line, (10, y), font, 0.6, (0,0,0), 1, cv2.LINE_AA)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            captured = frame.copy()
            cv2.destroyWindow(window_name)
            if captured is not None:
                process_single_image_array(captured, display_name="captured_frame")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Choose input")
    root.geometry("320x120")
    root.resizable(False, False)
    label = tk.Label(root, text="Choose input source:", font=("Arial", 12))
    label.pack(pady=(12,6))
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=6)
    btn_file = tk.Button(btn_frame, text="Select Image (single)", width=18, command=select_single_image_and_process)
    btn_file.grid(row=0, column=0, padx=6)
    btn_cam = tk.Button(btn_frame, text="Use Camera", width=20, command=run_camera_mode)
    btn_cam.grid(row=0, column=1, padx=6)
    root.mainloop()
