#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eye-controlled mouse + HUD; opens the on-screen keyboard when the AT-SPI flag file appears.
"""

import argparse
import cv2
import numpy as np
import math
import time
import pyautogui
import json
import os
import threading

from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

pyautogui.FAILSAFE = False

SCREEN_W, SCREEN_H = pyautogui.size()
ASPECT = 3 / 4

DEAD_R = 0.30
MAX_SPEED = 80
ALPHA = 0.27
SAFE_MARGIN = 3

mirror_preview = True

CALIB_FILE = "eye_dir_calib5.json"
CALIB_DIRS = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]
calib_points = {}
mapping_ready = False

FLAT_OPEN_THR = 0.45
FLAT_CLOSED_THR = 0.55
HYST_OPEN = 3
HYST_CLOSED = 3

IGNORE_THR = 0.20
LEFT_MAX = 0.80
RIGHT_MAX = 2.00
EXIT_MAX = 5.00

GPU_AVAILABLE = False
MIN_CONTOUR_AREA = 1200

shared_lock = threading.Lock()
shared_frame_bgr = None
shared_eye_state = "OPEN"
shared_flat = 0.0
shared_mouse_enabled = False
shared_status_text = "Baslatiliyor..."
terminate_flag = False

THEME_QSS = """
QWidget#Root {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #2a2a2a, stop:0.45 #1b1b1b, stop:1 #0f0f0f);
}
QLabel {
    color: #d6d6d6;
    font-size: 12px;
}
QLabel#Status {
    color: #00bcd4;
    font-size: 12px;
}
QLabel#Cam {
    background-color: #101010;
    border: 2px solid #3d3d3d;
    border-radius: 10px;
}
QPushButton#Toggle {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #2a2a2a, stop:1 #171717);
    border: 1px solid #3b3b3b;
    border-radius: 10px;
    padding: 6px 10px;
    color: #eaeaea;
    font-size: 12px;
}
QPushButton#Toggle:hover { border: 1px solid #4a4a4a; }
QPushButton#Toggle:pressed { background: #121212; }
"""

def crop_to_aspect(image, w, h):
    ih, iw = image.shape[:2]
    target = w / h
    cur = iw / ih
    if cur > target:
        new_w = int(target * ih)
        off = (iw - new_w) // 2
        img = image[:, off:off + new_w]
    else:
        new_h = int(iw / target)
        off = (ih - new_h) // 2
        img = image[off:off + new_h, :]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

def update_mapping_ready():
    global mapping_ready
    mapping_ready = all(d in calib_points for d in CALIB_DIRS)

def map_with_calib(px, py):
    if not mapping_ready:
        return None
    Cx, Cy = calib_points["CENTER"]

    if px >= Cx and "RIGHT" in calib_points:
        denom = calib_points["RIGHT"][0] - Cx
        nx = 0.0 if abs(denom) < 1e-6 else (px - Cx) / denom
    elif px < Cx and "LEFT" in calib_points:
        denom = Cx - calib_points["LEFT"][0]
        nx = 0.0 if abs(denom) < 1e-6 else (px - Cx) / denom
    else:
        nx = 0.0

    if py >= Cy and "DOWN" in calib_points:
        denom = calib_points["DOWN"][1] - Cy
        ny = 0.0 if abs(denom) < 1e-6 else (py - Cy) / denom
    elif py < Cy and "UP" in calib_points:
        denom = Cy - calib_points["UP"][1]
        ny = 0.0 if abs(denom) < 1e-6 else (py - Cy) / denom
    else:
        ny = 0.0

    nx = max(-1.5, min(1.5, nx))
    ny = max(-1.5, min(1.5, ny))
    return nx, ny

def load_calib(path=CALIB_FILE):
    global calib_points
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        calib_points = {k: (float(v[0]), float(v[1])) for k, v in data.items()}
        update_mapping_ready()
    except Exception:
        pass

def get_darkest_point(gray, ignore=8, step=8, search=20, inner=4):
    H, W = gray.shape
    min_sum = float("inf")
    best = None
    for y in range(ignore, H - ignore - search, step):
        for x in range(ignore, W - ignore - search, step):
            patch = gray[y:y + search:inner, x:x + search:inner]
            s = int(np.sum(patch))
            if s < min_sum:
                min_sum = s
                best = (x + search // 2, y + search // 2)
    return best

def detect_pupil_and_flatness(bgr, use_gpu=False):
    if use_gpu and GPU_AVAILABLE:
        try:
            gmat = cv2.cuda_GpuMat()
            gmat.upload(bgr)
            ggray = cv2.cuda.cvtColor(gmat, cv2.COLOR_BGR2GRAY)
            gray = ggray.download()
        except Exception:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    try:
        gray = cv2.equalizeHist(gray)
    except Exception:
        pass

    try:
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)
    except Exception:
        pass

    darkest = get_darkest_point(gray)
    if darkest is None:
        return None

    cx, cy = int(darkest[0]), int(darkest[1])
    H, W = gray.shape

    best_center = None
    best_score = 0.0
    best_axes = None

    try:
        base_v = int(gray[cy, cx])
        thresholds = [base_v + 6, base_v + 16, base_v + 30]
        for thr in thresholds:
            _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

            x0 = max(0, cx - 48)
            x1 = min(W, cx + 48)
            y0 = max(0, cy - 48)
            y1 = min(H, cy + 48)
            mask = th[y0:y1, x0:x1]
            if mask.size == 0:
                continue

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area < MIN_CONTOUR_AREA / 6.0:
                continue

            if len(largest) >= 5:
                ellipse = cv2.fitEllipse(largest)
                (cx_l, cy_l), axes, _ = ellipse
                ax0, ax1 = float(axes[0]), float(axes[1])
            else:
                M = cv2.moments(largest)
                if M["m00"] == 0:
                    continue
                cx_l = M["m10"] / M["m00"]
                cy_l = M["m01"] / M["m00"]
                ax0, ax1 = 0.0, 0.0

            if area > best_score:
                best_score = area
                best_center = (cx_l + x0, cy_l + y0)
                best_axes = (ax0, ax1)

        if best_center is not None:
            bx, by = best_center
            flat = 0.0
            if best_axes is not None:
                a, b = best_axes
                if a > 1e-6 and b > 1e-6:
                    ratio = min(a, b) / max(a, b)
                    flat = 1.0 - ratio
            return int(bx), int(by), float(flat)

    except Exception:
        pass

    return cx, cy, 0.0

def open_any_camera(candidate_indices, frame_w, frame_h):
    for idx in candidate_indices:
        if idx is None:
            continue
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        except Exception:
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
            except Exception:
                pass
            return cap, idx
        cap.release()
    return None, None

def draw_blink_bar(frame_bgr, eye_state: str, closed_elapsed: float):
    h, w = frame_bgr.shape[:2]
    bar_h = 14
    bar_y2 = h - 6
    bar_y1 = bar_y2 - bar_h

    cv2.rectangle(frame_bgr, (6, bar_y1), (w - 6, bar_y2), (80, 80, 80), 1)

    if eye_state != "CLOSED":
        cv2.putText(frame_bgr, "Blink: L / R / Exit", (8, bar_y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)
        return

    blink = closed_elapsed
    if blink <= IGNORE_THR:
        color = (140, 140, 140)
    elif blink <= LEFT_MAX:
        color = (0, 255, 110)
    elif blink <= RIGHT_MAX:
        color = (255, 160, 0)
    elif blink <= EXIT_MAX:
        color = (0, 60, 255)
    else:
        color = (0, 0, 160)

    prog = max(0.0, min(1.0, blink / EXIT_MAX))
    fill_w = int((w - 12) * prog)
    cv2.rectangle(frame_bgr, (6, bar_y1), (6 + fill_w, bar_y2), color, -1)
    cv2.putText(frame_bgr, f"{blink:.2f}s", (8, bar_y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

def mouse_worker(args, use_gpu):
    global terminate_flag, shared_frame_bgr, shared_eye_state, shared_flat, shared_status_text, mirror_preview

    frame_w = args.frame
    frame_h = int(frame_w * ASPECT)

    cap, cam_idx = open_any_camera(
        [args.cam_index] if args.cam_index is not None else [0, 1, 2, 3],
        frame_w, frame_h
    )
    if cap is None or not cap.isOpened():
        with shared_lock:
            shared_status_text = "Kamera acilamadi"
        return

    mx, my = SCREEN_W / 2.0, SCREEN_H / 2.0
    try:
        pyautogui.moveTo(int(mx), int(my))
    except Exception:
        pass

    eye_state = "OPEN"
    open_cnt = 0
    closed_cnt = 0
    closed_start_time = None
    closed_elapsed = 0.0

    with shared_lock:
        shared_status_text = f"Kamera OK (index={cam_idx})"

    while not terminate_flag:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame = crop_to_aspect(frame, frame_w, frame_h)
        if mirror_preview:
            frame = cv2.flip(frame, 1)

        with shared_lock:
            mouse_enabled = bool(shared_mouse_enabled)

        pupil = detect_pupil_and_flatness(frame, use_gpu=use_gpu)

        have_pupil = False
        flat = 1.0
        px = py = None
        if pupil is not None:
            px, py, flat = pupil
            have_pupil = True
            cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 110), -1)

        if flat >= FLAT_CLOSED_THR:
            closed_cnt += 1
            open_cnt = 0
        elif flat <= FLAT_OPEN_THR:
            open_cnt += 1
            closed_cnt = 0

        if closed_cnt >= HYST_CLOSED:
            eye_state = "CLOSED"
        elif open_cnt >= HYST_OPEN:
            eye_state = "OPEN"

        blink_dur = None
        now = time.time()
        if eye_state == "CLOSED":
            if closed_start_time is None:
                closed_start_time = now
            closed_elapsed = now - closed_start_time
        else:
            if closed_start_time is not None:
                blink_dur = closed_elapsed
                closed_start_time = None
                closed_elapsed = 0.0

        if blink_dur is not None and mouse_enabled:
            if blink_dur <= IGNORE_THR:
                pass
            elif blink_dur <= LEFT_MAX:
                try:
                    pyautogui.click()
                except Exception:
                    pass
            elif blink_dur <= RIGHT_MAX:
                try:
                    pyautogui.rightClick()
                except Exception:
                    pass
            else:
                terminate_flag = True
                break

        if have_pupil:
            if mapping_ready:
                mapped = map_with_calib(px, py)
                if mapped is not None:
                    nx, ny = mapped
                else:
                    nx = ny = 0.0
            else:
                cx, cy = frame_w / 2, frame_h / 2
                nx = (px - cx) / (frame_w * 0.5)
                ny = (py - cy) / (frame_h * 0.5)
                nx = max(-1.5, min(1.5, nx))
                ny = max(-1.5, min(1.5, ny))

            r = math.sqrt(nx * nx + ny * ny)
            if (r < DEAD_R) or (not mouse_enabled) or (eye_state != "OPEN"):
                vx = vy = 0.0
            else:
                k = (r - DEAD_R) / max(1e-6, (1.0 - DEAD_R))
                k = max(0.0, min(1.0, k))
                v = MAX_SPEED * k

                ax = abs(nx)
                ay = abs(ny)
                if ax > ay * 1.4:
                    ny_eff = 0.0
                    nx_eff = math.copysign(1.0, nx)
                elif ay > ax * 1.4:
                    nx_eff = 0.0
                    ny_eff = math.copysign(1.0, ny)
                else:
                    nx_eff = math.copysign(1.0, nx)
                    ny_eff = math.copysign(1.0, ny)

                vx = v * nx_eff
                vy = v * ny_eff

            if mouse_enabled and eye_state == "OPEN":
                mx = mx + ALPHA * vx
                my = my + ALPHA * vy
                mx = max(SAFE_MARGIN, min(SCREEN_W - SAFE_MARGIN, mx))
                my = max(SAFE_MARGIN, min(SCREEN_H - SAFE_MARGIN, my))
                try:
                    pyautogui.moveTo(int(mx), int(my))
                except Exception:
                    pass

        draw_blink_bar(frame, eye_state, closed_elapsed)

        with shared_lock:
            shared_frame_bgr = frame
            shared_eye_state = eye_state
            shared_flat = float(flat)

        time.sleep(0.001)

    cap.release()

class EyeHUD(QWidget):
    def __init__(self, preview_w: int, preview_h: int):
        super().__init__()

        self.setObjectName("Root")
        self.setStyleSheet(THEME_QSS)

        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowType.Tool, True)

        self.setFixedSize(520, preview_h + 44)
        scr = QApplication.primaryScreen().geometry()
        self.move(scr.width() // 2 - self.width() // 2, scr.height() - self.height() - 20)

        main = QVBoxLayout(self)
        main.setContentsMargins(10, 8, 10, 8)
        main.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(10)

        self.lbl_open = QLabel("OPEN")
        self.lbl_open.setFixedWidth(110)
        self.lbl_open.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.cam = QLabel()
        self.cam.setObjectName("Cam")
        self.cam.setFixedSize(preview_w, preview_h)
        self.cam.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_closed = QLabel("CLOSED")
        self.lbl_closed.setFixedWidth(110)
        self.lbl_closed.setAlignment(Qt.AlignmentFlag.AlignCenter)

        top.addWidget(self.lbl_open)
        top.addWidget(self.cam)
        top.addWidget(self.lbl_closed)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        self.status = QLabel("Status: ...")
        self.status.setObjectName("Status")
        self.status.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        self.btn_toggle = QPushButton("Mouse: OFF")
        self.btn_toggle.setObjectName("Toggle")
        self.btn_toggle.setFixedSize(140, 28)
        self.btn_toggle.clicked.connect(self.toggle_mouse)

        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.btn_toggle, 0)

        main.addLayout(top)
        main.addLayout(bottom)

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(40)

        self.raise_timer = QTimer()
        self.raise_timer.timeout.connect(self.raise_)
        self.raise_timer.start(1500)

    def toggle_mouse(self):
        global shared_mouse_enabled
        with shared_lock:
            shared_mouse_enabled = not shared_mouse_enabled

    def tick(self):
        global shared_frame_bgr, shared_eye_state, shared_flat, shared_status_text, shared_mouse_enabled

        with shared_lock:
            frame = None if shared_frame_bgr is None else shared_frame_bgr.copy()
            state = shared_eye_state
            flat = float(shared_flat)
            text = str(shared_status_text)
            mouse_on = bool(shared_mouse_enabled)

        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(q).scaled(
                    self.cam.width(),
                    self.cam.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.cam.setPixmap(pix)
            except Exception:
                self.cam.setText("cam err")
        else:
            self.cam.setText("no cam")

        if state == "OPEN":
            self.lbl_open.setStyleSheet("color:#00ff6e; font-size:22px; font-weight:800;")
            self.lbl_closed.setStyleSheet("color:#555; font-size:22px; font-weight:800;")
        else:
            self.lbl_open.setStyleSheet("color:#555; font-size:22px; font-weight:800;")
            self.lbl_closed.setStyleSheet("color:#ff3b3b; font-size:22px; font-weight:800;")

        if mouse_on:
            self.btn_toggle.setText("Mouse: ON")
            self.btn_toggle.setStyleSheet(
                "QPushButton{background:#00bcd4;color:#001014;border-radius:10px;"
                "padding:6px 10px;font-size:12px;font-weight:700;}"
                "QPushButton:pressed{background:#00a6bb;}"
            )
        else:
            self.btn_toggle.setText("Mouse: OFF")
            self.btn_toggle.setStyleSheet("")  # QSS'e geri dön

        self.status.setText(f"{text} | eye={state} flat={flat:.2f}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        global terminate_flag
        terminate_flag = True
        event.accept()

def main():
    parser = argparse.ArgumentParser(description="Gozle mouse + kucuk HUD (dark theme)")
    parser.add_argument("--cam-index", type=int, default=None, help="kamera index (0,1,2...)")
    parser.add_argument("--frame", type=int, default=320, help="isleme frame genisligi (CPU icin)")
    parser.add_argument("--gpu", action="store_true", help="varsa OpenCV-CUDA ile hizlandirma")
    parser.add_argument("--mouse-on", action="store_true", help="baslangicta mouse kontrolu acik baslasin")
    parser.add_argument("--preview-w", type=int, default=200, help="HUD kamera onizleme genisligi")
    parser.add_argument("--preview-h", type=int, default=110, help="HUD kamera onizleme yuksekligi")
    args = parser.parse_args()

    global GPU_AVAILABLE, terminate_flag, shared_mouse_enabled

    load_calib()

    use_gpu = False
    if args.gpu:
        try:
            cnt = cv2.cuda.getCudaEnabledDeviceCount()
            if cnt and cnt > 0:
                GPU_AVAILABLE = True
                use_gpu = True
        except Exception:
            pass

    with shared_lock:
        shared_mouse_enabled = bool(args.mouse_on)

    terminate_flag = False
    t = threading.Thread(target=mouse_worker, args=(args, use_gpu), daemon=True)
    t.start()

    app = QApplication([])
    w = EyeHUD(preview_w=args.preview_w, preview_h=args.preview_h)
    w.show()

    try:
        app.exec()
    finally:
        terminate_flag = True
        t.join(timeout=2.0)

if __name__ == "__main__":
    main()
