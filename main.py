#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project entry point (startup script).
"""

import cv2
import numpy as np
import pyautogui
import sys
import math
import time
import threading
import subprocess
import os
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap


SCREEN_W, SCREEN_H = pyautogui.size()
MARGIN_RATIO = 0.02
MARGIN_X = SCREEN_W * MARGIN_RATIO
MARGIN_Y = SCREEN_H * MARGIN_RATIO

EYE_CLOSED_TIME = 1.2

DEAD_ZONE_X = 160        # piksel
HOLD_TIME_FIRST = 0.55   # saniye
HOLD_TIME_REPEAT = 0.56  # saniye

MOVE_THRESH_PX = 30.0

AREA_MIN = 500.0

HYST_OPEN = 4
HYST_CLOSED = 4

FLAT_OPEN_MAX = 0.40
FLAT_CLOSED_MIN = 0.60

FRAME_W, FRAME_H = 640, 480

THR_OFFSETS = [6, 16, 30]

MASK_SIZE = 200

# sağlar.
class LowPass:
    def __init__(self, a, init=0.0):
        self.a = a
        self.s = init
        self.init = False

    def filter(self, x, a):
        if not self.init:
            self.s = x
            self.init = True
            return x
        self.s = a * x + (1 - a) * self.s
        return self.s


def alpha_from_cutoff(c, dt):
    tau = 1.0 / (2 * math.pi * c)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, freq=30, minc=0.5, beta=0.02, dcut=1.0):
        self.f = freq
        self.mc = minc
        self.b = beta
        self.dc = dcut
        self.lp_x = LowPass(alpha_from_cutoff(self.mc, 1 / self.f))
        self.lp_dx = LowPass(alpha_from_cutoff(self.dc, 1 / self.f))
        self.prev = None
        self.last = None

    def __call__(self, x, t=None):
        now = t or time.time()
        dt = 1 / self.f if not self.last else max(1e-6, now - self.last)
        self.last = now
        dx = 0 if self.prev is None else (x - self.prev) / dt
        edx = self.lp_dx.filter(dx, alpha_from_cutoff(self.dc, dt))
        cutoff = self.mc + self.b * abs(edx)
        a = alpha_from_cutoff(cutoff, dt)
        xf = self.lp_x.filter(x, a)
        self.prev = xf
        return xf


euro_x = OneEuro(freq=30, minc=0.5, beta=0.02, dcut=1.0)
euro_y = OneEuro(freq=30, minc=0.6, beta=0.02, dcut=1.0)


def crop_to_aspect_ratio(image, width=FRAME_W, height=FRAME_H):
    """
    Verilen görüntüyü istenilen en/boy oranında kırpar ve yeniden boyutlandırır.
    Kırpma işlemi görüntünün merkezine odaklı yapılır.
    """
    h, w = image.shape[:2]
    target_ratio = width / height
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = int(target_ratio * h)
        off = (w - new_w) // 2
        img = image[:, off : off + new_w]
    else:
        new_h = int(w / target_ratio)
        off = (h - new_h) // 2
        img = image[off : off + new_h, :]
    return cv2.resize(img, (width, height))


def get_darkest_point(gray, ignore=8, step=8, search=20, inner=4):
    """
    Göz bebeğinin konumunu yaklaşık olarak bulmak için görüntüyü
    tarar ve en düşük toplam parlaklığa sahip küçük bölgeyi döndürür.
    """
    H, W = gray.shape
    min_sum = float("inf")
    best = None
    for y in range(ignore, H - ignore - search, step):
        for x in range(ignore, W - ignore - search, step):
            patch = gray[y : y + search : inner, x : x + search : inner]
            s = int(np.sum(patch))
            if s < min_sum:
                min_sum = s
                best = (x + search // 2, y + search // 2)
    return best


def mask_outside_square(img, center, size):
    """
    Verilen merkez ve kare boyutuna göre maskenin dışında kalan
    bölgeleri sıfırlar. Böylece pupil araması sadece bu bölge içinde yapılır.
    """
    x, y = center
    mask = np.zeros_like(img)
    half = size // 2
    y0 = max(0, y - half)
    y1 = min(img.shape[0], y + half)
    x0 = max(0, x - half)
    x1 = min(img.shape[1], x + half)
    mask[y0:y1, x0:x1] = 255
    return cv2.bitwise_and(img, mask)


def detect_pupil_and_flatness(frame, thr_offsets=THR_OFFSETS):
    """
    Verilen renkli karede pupil merkezini ve yassılık değerini tahmin eder.
    Dönüş: (center_x, center_y, flatness, area)

    * `flatness` değeri 0–1 aralığındadır ve elipsin ne kadar dairesel
      olduğunun bir göstergesidir: 0 → daire, 1 → yassı (göz kapalı adayı).
    * `area` değeri elipsin alanıdır. Alan çok küçükse tespit başarısız
      olabilir.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None, 0.0, 0.0, 0.0
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
        return None, 0.0, 0.0, 0.0
    cx, cy = int(darkest[0]), int(darkest[1])
    H, W = gray.shape
    best_center = None
    best_area = 0.0
    best_axes = None
    base_v = int(gray[cy, cx])
    offsets = thr_offsets
    for off in offsets:
        thr_val = base_v + off
        _, th = cv2.threshold(gray, thr_val, 255, cv2.THRESH_BINARY_INV)
        th_mask = mask_outside_square(th, (cx, cy), MASK_SIZE)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        th_mask = cv2.morphologyEx(th_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(th_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < AREA_MIN / 8.0:
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
        if area > best_area:
            best_area = area
            best_center = (cx_l + (cx - MASK_SIZE // 2), cy_l + (cy - MASK_SIZE // 2))
            best_axes = (ax0, ax1)
    if best_center is None:
        return None, 0.0, 0.0, 0.0
    bx, by = best_center
    flat = 0.0
    if best_axes is not None:
        a, b = best_axes
        if a > 1e-3 and b > 1e-3:
            ratio = min(a, b) / max(a, b)
            flat = 1.0 - ratio
    return (float(bx), float(by)), float(flat), float(best_area)


def open_camera():
    """
    0–5 arası indexleri sırayla dener ve ilk açılan kamerayı döndürür.
    Kamera açılınca temel parametreler (autofocus, focus, exposure, gain,
    brightness) ayarlanır. GC0308 sensörlü kameralar sabit odaklıdır ancak
    lens hafifçe döndürülerek odak düzeltilebilir.
    """
    for idx in range(0, 6):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cap.set(cv2.CAP_PROP_FOCUS, 10)
                cap.set(cv2.CAP_PROP_EXPOSURE, -6)
                cap.set(cv2.CAP_PROP_GAIN, 0)
                cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
            except Exception:
                pass
            print(f"[gaze] camera opened on index {idx}")
            return cap
        else:
            cap.release()
    print("[gaze] camera error: hiçbir index açılamadı")
    return None


shared_lock = threading.Lock()
shared_frame = None
shared_smoothed = (SCREEN_W / 2, SCREEN_H / 2)
shared_eye_closed_elapsed = 0.0
shared_conf = 0.0
shared_area = 0.0
shared_flatness = 0.0
shared_state = "OPEN"
terminate_flag = False
gaze_thread = None


def gaze_loop():

    global shared_frame, shared_smoothed, shared_eye_closed_elapsed, terminate_flag
    global shared_conf, shared_area, shared_state, shared_flatness

    cap = open_camera()
    if cap is None:
        return
    prev_center = None
    open_cnt = 0
    closed_cnt = 0
    state = "OPEN"
    closed_start_t = None
    last_print = 0.0
    sharpen_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    while not terminate_flag:
        ret, f = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        f = cv2.flip(f, 1)
        f = cv2.filter2D(f, -1, sharpen_k)
        frame = crop_to_aspect_ratio(f, FRAME_W, FRAME_H)
        center_data = detect_pupil_and_flatness(frame, thr_offsets=THR_OFFSETS)
        if center_data is None:
            move = 999.0
            area = 0.0
            flatness = 1.0
            center_to_map = None
        else:
            (cx, cy), flatness, area = center_data
            if prev_center is None:
                move = 0.0
            else:
                dx = abs(cx - prev_center[0])
                dy = abs(cy - prev_center[1])
                move = math.hypot(dx, dy)
            prev_center = (cx, cy)
            center_to_map = (cx, cy)
        if center_to_map is None or area < AREA_MIN or move > MOVE_THRESH_PX:
            is_closed_candidate = True
        else:
            is_closed_candidate = False
        if flatness >= FLAT_CLOSED_MIN:
            is_closed_candidate = True
        elif flatness <= FLAT_OPEN_MAX:
            is_closed_candidate = False
        if is_closed_candidate:
            closed_cnt += 1
            open_cnt = 0
        else:
            open_cnt += 1
            closed_cnt = 0
        if closed_cnt > HYST_CLOSED:
            state = "CLOSED"
        elif open_cnt > HYST_OPEN:
            state = "OPEN"
        now = time.time()
        if state == "CLOSED":
            if closed_start_t is None:
                closed_start_t = now
            closed_elapsed = now - closed_start_t
        else:
            closed_start_t = None
            closed_elapsed = 0.0
        with shared_lock:
            shared_eye_closed_elapsed = float(closed_elapsed)
        if center_to_map is not None:
            px, py = center_to_map
            sx = MARGIN_X + (SCREEN_W - 2 * MARGIN_X) * (px / FRAME_W)
            sy = MARGIN_Y + (SCREEN_H - 2 * MARGIN_Y) * (py / FRAME_H)
            sx = euro_x(sx)
            sy = euro_y(sy)
            with shared_lock:
                shared_smoothed = (sx, sy)
        area_score = 0.0
        if area > 0:
            area_score = max(0.0, min(1.0, (area - AREA_MIN) / (AREA_MIN * 2.0)))
        move_score = 1.0 - max(0.0, min(1.0, move / (MOVE_THRESH_PX * 2.0)))
        flat_score = 1.0 - max(0.0, min(1.0, flatness))
        conf_open = max(0.0, min(1.0, 0.4 * area_score + 0.4 * move_score + 0.2 * flat_score))
        conf = conf_open if state == "OPEN" else (1.0 - conf_open)
        if area < 7000 or area > 25000:
            conf = 0.0
            state = "CLOSED"
        elif area < 8000:
            conf *= 0.5
        with shared_lock:
            shared_conf = float(conf)
            shared_area = float(area)
            shared_state = state
            shared_flatness = float(flatness)
        draw = frame.copy()
        if center_to_map is not None and area > 0:
            cv2.circle(draw, (int(center_to_map[0]), int(center_to_map[1])), 4, (255, 255, 0), -1)
        color = (0, 255, 0) if state == "OPEN" else (0, 0, 255)
        cv2.putText(draw, f"{state}  conf={conf:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        with shared_lock:
            shared_frame = draw
        if now - last_print > 0.05:
            print(
                f"state={state:6s}  closed_s={closed_elapsed:4.2f}  move={move:5.1f}  area={area:6.1f}  flat={flatness:4.2f}  conf={conf:0.2f}  open_cnt={open_cnt}  closed_cnt={closed_cnt}"
            )
            last_print = now
        time.sleep(0.005)
    cap.release()


class Launcher(QWidget):
    """
    Üç butonlu basit bir launcher. Göz pozisyonuna göre sağ/sol seçim yapar
    ve göz kapatma ile seçili programı çalıştırır. Kamera önizlemesi ve
    durum göstergesi içerir.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Launcher (Stable)")
        self.setStyleSheet("background-color:#000;")
        self.setFixedSize(820, 150)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        scr = QApplication.primaryScreen().geometry()
        self.move(scr.width() // 2 - 410, scr.height() - 170)
        main = QHBoxLayout(self)
        left = QHBoxLayout()
        self.buttons = []
        names = ["Klavye", "Fare", "CUA"]
        for n in names:
            b = QPushButton(n)
            b.setFixedSize(180, 120)
            b.setStyleSheet(
                """
                QPushButton{
                    background-color:#111;
                    color:#09f;
                    font-size:26px;
                    border-radius:14px;
                }
                """
            )
            self.buttons.append(b)
            left.addWidget(b)
        main.addLayout(left)
        self.selected = 1  # Başlangıçta ortadaki (Fare) seçili
        self.update_visual()
        right = QVBoxLayout()
        self.cam = QLabel()
        self.cam.setFixedSize(200, 110)
        self.cam.setStyleSheet("background-color:#000;border:2px solid #333;")
        self.cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status = QLabel("Status: bekleniyor…")
        self.status.setStyleSheet("color:#09f;font-size:12px;")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.cam)
        right.addWidget(self.status)
        main.addLayout(right)
        self.dead_zone = DEAD_ZONE_X
        self.hold_first = HOLD_TIME_FIRST
        self.hold_repeat = HOLD_TIME_REPEAT
        self.nav_dir = None
        self.nav_last = 0
        self.nav_first = True
        self.dwell_thresh = EYE_CLOSED_TIME
        self.require_open = False
        self.busy = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_from_gaze)
        self.timer.start(40)
    def update_visual(self):
        for i, b in enumerate(self.buttons):
            if i == self.selected:
                b.setStyleSheet(
                    """
                    QPushButton{
                        background-color:#09f;
                        color:#000;
                        font-size:26px;
                        border-radius:14px;
                    }
                    """
                )
            else:
                b.setStyleSheet(
                    """
                    QPushButton{
                        background-color:#111;
                        color:#09f;
                        font-size:26px;
                        border-radius:14px;
                    }
                    """
                )
    def move_sel(self, direction):
        if direction == "left":
            self.selected = (self.selected - 1) % 3
        else:
            self.selected = (self.selected + 1) % 3
        self.update_visual()
    def update_from_gaze(self):
        with shared_lock:
            sx, sy = shared_smoothed
            closed_elapsed = shared_eye_closed_elapsed
            state = shared_state
            frame = shared_frame.copy() if shared_frame is not None else None
            conf = shared_conf
            area = shared_area
        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                q = QImage(
                    rgb.data,
                    rgb.shape[1],
                    rgb.shape[0],
                    rgb.strides[0],
                    QImage.Format.Format_RGB888,
                )
                pix = QPixmap.fromImage(q).scaled(
                    self.cam.width(),
                    self.cam.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
                self.cam.setPixmap(pix)
            except Exception:
                self.cam.setText("cam err")
        else:
            self.cam.setText("no cam")
        if self.busy:
            self.status.setText("Status: dış program çalışıyor…")
            self.status.setStyleSheet("color:#ff0;font-size:12px;")
        else:
            self.status.setText(f"Status: {state} conf={conf:.2f} area={area:.0f}")
            self.status.setStyleSheet(
                "color:#0f0;font-size:12px;" if state == "OPEN" else "color:#f33;font-size:12px;"
            )
        if self.busy:
            return
        if closed_elapsed >= self.dwell_thresh and not self.require_open:
            self.activate()
            self.require_open = True
        elif closed_elapsed == 0.0:
            self.require_open = False
        if state != "OPEN":
            self.nav_dir = None
            self.nav_first = True
            return
        dx = sx - SCREEN_W / 2
        if abs(dx) <= self.dead_zone:
            self.nav_dir = None
            self.nav_first = True
            return
        direction = "left" if dx < -self.dead_zone else "right"
        now = time.time()
        if self.nav_dir != direction:
            self.nav_dir = direction
            self.nav_first = True
            self.nav_last = now
            return
        wait = self.hold_first if self.nav_first else self.hold_repeat
        if now - self.nav_last >= wait:
            self.move_sel(direction)
            self.nav_last = now
            self.nav_first = False
    def activate(self):
        """Seçili programı çalıştır."""
        if self.busy:
            return
        idx = self.selected
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if idx == 0:
            script = os.path.join(base_dir, "klavye_kullanma.py")
            extra_args = []
        elif idx == 1:
            script = os.path.join(base_dir, "mouse_kullanma.py")
            extra_args = ["--mouse-on"]
        else:
            script = os.path.join(base_dir, "CUA_main.py")
            extra_args = []
        cmd = [sys.executable, script, *extra_args]
        self.buttons[idx].setStyleSheet("background-color:#0f0;color:#000;")
        QTimer.singleShot(300, self.update_visual)
        self.busy = True
        self.hide()
        def worker():
            global terminate_flag, gaze_thread
            terminate_flag = True
            if gaze_thread is not None:
                gaze_thread.join(timeout=2.0)
            try:
                subprocess.run(cmd, check=False)
            finally:
                terminate_flag = False
                gaze_thread = threading.Thread(target=gaze_loop, daemon=True)
                gaze_thread.start()
                self.busy = False
                QTimer.singleShot(0, self.show)
        threading.Thread(target=worker, daemon=True).start()


def main():
    global terminate_flag, gaze_thread
    terminate_flag = False
    gaze_thread = threading.Thread(target=gaze_loop, daemon=True)
    gaze_thread.start()
    app = QApplication(sys.argv)
    w = Launcher()
    w.show()
    try:
        sys.exit(app.exec())
    finally:
        terminate_flag = True
        if gaze_thread is not None:
            gaze_thread.join(2)


if __name__ == "__main__":
    main()
