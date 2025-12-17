#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("GGML_CUDA_DISABLE_GRAPHS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import time
import math
import threading
import gc
import subprocess

import numpy as np
import cv2
import pyautogui

from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QVBoxLayout,
    QPushButton, QLabel, QTextEdit, QProgressBar, QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

try:
    cv2.setNumThreads(1)
except Exception:
    pass

COMMAND_LAYOUT = [
    [
        ("Firefox",  "open firefox"),
        ("YouTube",  "Click youtube icon"),
        ("Dosyalar", "Open file manager"),
        ("Haberler", "Search for latest news"),
        ("Terminal", "Open terminal"),
        ("Ne Var?",  "Describe what is on the screen"),
        ("Yenile",   "Refresh page"),
        ("Kapat",    "EXIT_APP"),
    ]
]


SCREEN_W, SCREEN_H = pyautogui.size()
MARGIN_RATIO = 0.02
MARGIN_X = SCREEN_W * MARGIN_RATIO
MARGIN_Y = SCREEN_H * MARGIN_RATIO

EYE_CLOSED_TIME  = 2.0
FRAME_W, FRAME_H = 400, 300

DEAD_ZONE_X       = 120
HOLD_TIME_FIRST   = 0.45
HOLD_TIME_REPEAT  = 0.35

MOVE_THRESH_PX = 30.0
AREA_MIN       = 400.0
HYST_OPEN      = 3
HYST_CLOSED    = 3
FLAT_OPEN_MAX   = 0.35
FLAT_CLOSED_MIN = 0.55
THR_ADD_MED    = 15
MASK_SIZE      = 250

shared_lock = threading.Lock()
shared_frame = None
shared_smoothed = (SCREEN_W/2, SCREEN_H/2)
shared_eye_closed_elapsed = 0.0
shared_state = "OPEN"
terminate_flag = False
gaze_thread = None

def force_cleanup():
    gc.collect()

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
        self.s = a*x + (1-a)*self.s
        return self.s

def alpha_from_cutoff(c, dt):
    tau = 1.0 / (2 * math.pi * c)
    return 1.0 / (1.0 + tau / dt)

class OneEuro:
    def __init__(self, freq=30, minc=0.4, beta=0.01, dcut=1.0):
        self.f=freq
        self.mc=minc
        self.b=beta
        self.dc=dcut
        self.lp_x=LowPass(alpha_from_cutoff(self.mc,1/self.f))
        self.lp_dx=LowPass(alpha_from_cutoff(self.dc,1/self.f))
        self.prev=None
        self.last=None
    def __call__(self, x, t=None):
        now = t or time.time()
        dt  = 1/self.f if not self.last else max(1e-6, now-self.last)
        self.last = now
        dx  = 0 if self.prev is None else (x-self.prev)/dt
        edx = self.lp_dx.filter(dx, alpha_from_cutoff(self.dc, dt))
        cutoff = self.mc + self.b*abs(edx)
        a = alpha_from_cutoff(cutoff, dt)
        xf = self.lp_x.filter(x, a)
        self.prev = xf
        return xf

euro_x = OneEuro(freq=30, minc=0.4, beta=0.01, dcut=1.0)
euro_y = OneEuro(freq=30, minc=0.6, beta=0.01, dcut=1.0)

def crop_to_aspect_ratio(image, width=FRAME_W, height=FRAME_H):
    h, w = image.shape[:2]
    target_ratio  = width / height
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = int(target_ratio * h)
        off = (w - new_w) // 2
        img = image[:, off:off+new_w]
    else:
        new_h = int(w / target_ratio)
        off = (h - new_h) // 2
        img = image[off:off+new_h, :]
    return cv2.resize(img, (width, height))

def get_darkest_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = 9_999_999
    point=None
    for y in range(20, gray.shape[0]-20, 10):
        for x in range(20, gray.shape[1]-20, 10):
            area = gray[y:y+20, x:x+20]
            val  = int(np.sum(area))
            if val < min_sum:
                min_sum = val
                point = (x+10, y+10)
    return point

def mask_outside_square(img, center, size):
    x, y = center
    mask  = np.zeros_like(img)
    half  = size//2
    y0 = max(0, y-half); y1 = min(img.shape[0], y+half)
    x0 = max(0, x-half); x1 = min(img.shape[1], x+half)
    mask[y0:y1, x0:x1] = 255
    return cv2.bitwise_and(img, mask)

def open_any_camera(candidate_indices=(0, 1, 2, 3, 4, 5)):
    print("[DEBUG] Kamera açma denemesi, aday indexler:", candidate_indices)
    for idx in candidate_indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[DEBUG] Kamera {idx} indexinde açıldı.")
            return cap
        else:
            print(f"[DEBUG] Kamera index {idx} açılamadı.")
            cap.release()
    print("[HATA] Hiçbir kamera açılamadı.")
    return None

def gaze_loop():
    global shared_frame, shared_smoothed, shared_eye_closed_elapsed, terminate_flag
    global shared_state

    cap = open_any_camera()
    if cap is None:
        print("[GazeLoop] Kamera bulunamadı, gaze_loop çıkıyor.")
        return

    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    except Exception as e:
        print("[DEBUG] Kamera property set ederken hata:", e)

    prev_center = None
    open_cnt = 0
    closed_cnt = 0
    state = "OPEN"
    closed_start_t = None
    sharpen_k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)

    print("[GazeLoop] Başladı.")

    while not terminate_flag:
        ret, f = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        f = cv2.flip(f, 1)
        f = cv2.filter2D(f, -1, sharpen_k)
        frame = crop_to_aspect_ratio(f, FRAME_W, FRAME_H)

        darkest_point = get_darkest_area(frame)
        draw = frame.copy()

        is_closed_candidate = True
        center_to_map = None
        area = 0.0
        move = 999.0

        if darkest_point is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            base = int(gray[darkest_point[1], darkest_point[0]])
            _, thr_med = cv2.threshold(gray, base+THR_ADD_MED, 255, cv2.THRESH_BINARY_INV)
            thr_med = mask_outside_square(thr_med, darkest_point, MASK_SIZE)

            k = np.ones((5,5), np.uint8)
            dil = cv2.dilate(thr_med, k, 2)
            contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) >= 5:
                    ellipse = cv2.fitEllipse(largest)
                    (cx, cy), axes, _ = ellipse
                    area = math.pi * (axes[0]/2.0) * (axes[1]/2.0)
                    major = max(axes[0], axes[1]); minor = min(axes[0], axes[1])
                    flatness = 1.0 - (minor / major) if major > 1e-3 else 0.0

                    if prev_center is None:
                        move = 0.0
                    else:
                        move = math.hypot(cx - prev_center[0], cy - prev_center[1])
                    prev_center = (cx, cy)

                    is_closed_candidate = (move > MOVE_THRESH_PX) or (area < AREA_MIN)
                    if flatness >= FLAT_CLOSED_MIN:
                        is_closed_candidate = True
                    elif flatness <= FLAT_OPEN_MAX:
                        is_closed_candidate = False

                    if is_closed_candidate:
                        center_to_map = None
                    else:
                        center_to_map = (cx, cy)
                        cv2.ellipse(draw, ellipse, (0,255,255), 2)
                        cv2.circle(draw, (int(cx), int(cy)), 4, (255,255,0), -1)

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

        if center_to_map is not None and state == "OPEN":
            px, py = center_to_map
            sx = MARGIN_X + (SCREEN_W - 2*MARGIN_X) * (px / FRAME_W)
            sy = MARGIN_Y + (SCREEN_H - 2*MARGIN_Y) * (py / FRAME_H)
            sx = euro_x(sx)
            sy = euro_y(sy)
            with shared_lock:
                shared_smoothed = (sx, sy)

        if area < 7000 or area > 25000:
            state = "CLOSED"

        with shared_lock:
            shared_state = state

        color = (0,255,0) if state=="OPEN" else (0,0,255)
        cv2.putText(draw, f"{state}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        with shared_lock:
            shared_frame = draw

        time.sleep(0.005)

    print("[GazeLoop] terminate_flag geldi, çıkılıyor.")
    cap.release()

class AgentCore:
    """
    Modelleri ve Sandbox'ı TEK SEFERDE yükler.
    Gaze UI içinden run_command() ile doğrudan çağrılır.
    """
    def __init__(self):
        self.initialized = False
        self.sandbox = None
        self.client = None
        self.run_planner = None

    def _init_if_needed(self):
        if self.initialized:
            return

        print("[AgentCore] Modeller ve Sandbox yükleniyor...")
        from os_computer_use.streaming import Sandbox, DisplayClient
        from os_computer_use.sandbox_agent import run_planner

        self.sandbox = Sandbox()
        self.client = DisplayClient()
        self.run_planner = run_planner

        stream_url = self.sandbox.start_stream()
        print(f"[AgentCore] Sandbox stream: {stream_url}")

        self.initialized = True
        print("[AgentCore] Hazır.")

    def run_command(self, prompt: str) -> str:
        """
        Göz arayüzünden gelen komutu çalıştırır.
        Önce bazı temel komutlar için lokal fallback dener (örn. Firefox),
        sonra genel planner'a düşer.
        """
        self._init_if_needed()

        p = prompt.strip().lower()

        # ---- LOKAL FIREFOX FALLBACK ----
        if "firefox" in p:
            try:
                subprocess.Popen(["firefox"])
                return "[Local] Firefox açma komutu çalıştırıldı."
            except Exception as e:
                return f"[LocalHATA] Firefox açılamadı: {e}"

        # ---- Genel durumda planner'ı çalıştır ----
        try:
            res = self.run_planner(prompt)
        except Exception as e:
            res = f"[AgentHATA] {e}"
        finally:
            gc.collect()
        return str(res)


    def shutdown(self):
        if not self.initialized:
            return
        print("[AgentCore] Kapatılıyor...")
        try:
            self.sandbox.stop_stream()
        except Exception as e:
            print(f"[AgentCore] sandbox.stop_stream hatası: {e}")
        try:
            self.client.save_stream()
        except Exception as e:
            print(f"[AgentCore] client.save_stream hatası: {e}")
        self.initialized = False

agent_core = AgentCore()

class AgentWorker(QThread):
    finished_signal = pyqtSignal(str)

    def __init__(self, prompt: str):
        super().__init__()
        self.prompt = prompt

    def run(self):
        result = agent_core.run_command(self.prompt)
        self.finished_signal.emit(result)

class AgentButton(QWidget):
    def __init__(self, label, prompt, parent, size_h=40):
        super().__init__()
        self.prompt = prompt
        self.label_text = label
        self.low_start = None

        self.btn = QPushButton(label)
        self.btn.setFixedHeight(size_h)
        self.btn.setStyleSheet(
            "background-color:#222; color:#ffffff; font-size:14px; "
            "border-radius:6px; font-weight: bold;"
        )

        self.prog = QProgressBar()
        self.prog.setRange(0, 100)
        self.prog.setValue(0)
        self.prog.setTextVisible(False)
        self.prog.setFixedHeight(3)
        self.prog.setStyleSheet(
            "QProgressBar::chunk{background-color:#00bfff;} "
            "QProgressBar{background-color:#444;border-radius:2px;}"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(1, 1, 1, 1)
        lay.setSpacing(1)
        lay.addWidget(self.btn)
        lay.addWidget(self.prog)

    def set_selected(self, sel):
        bg = "#00bfff" if sel else "#222"
        fg = "#000000" if sel else "#ffffff"
        self.btn.setStyleSheet(
            f"background-color:{bg}; color:{fg}; font-size:14px; "
            "border-radius:6px; font-weight: bold;"
        )

class AgentKeyboardUI(QWidget):
    def __init__(self):
        super().__init__()

        self.dwell_time_ms = int(EYE_CLOSED_TIME * 1000)
        self.full_thresh = self.dwell_time_ms / 1000.0

        self.setWindowTitle("Gaze Agent Bar")
        self.setStyleSheet("background-color:#111;")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)

        # ---- Sağ alt, ince bar ----
        screen = QApplication.primaryScreen().geometry()
        kbd_width = int(screen.width() * 0.55)   # genişlik: ekranın ~%55'i
        kbd_height = 80                          # ince yükseklik
        self.setFixedSize(kbd_width, kbd_height)
        self.move(screen.width() - kbd_width - 20,
                  screen.height() - kbd_height - 20)

        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ---- Sol taraf: status + tek satır butonlar ----
        left_col = QVBoxLayout()
        left_col.setSpacing(2)

        self.buffer_label = QLabel("Agent: HAZIR")
        self.buffer_label.setStyleSheet(
            "color:#0bf;font-size:11px;font-weight:bold;"
        )
        left_status = QHBoxLayout()
        left_status.addWidget(self.buffer_label)
        left_status.addStretch(1)
        left_col.addLayout(left_status)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.hide()

        self.gridw = QWidget()
        self.grid = QHBoxLayout(self.gridw)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(4)
        left_col.addWidget(self.gridw)

        root.addLayout(left_col, stretch=1)

        # ---- Sağ taraf: küçük kamera önizleme ----
        self.cam = QLabel()
        self.cam.setFixedSize(130, 60)
        self.cam.setStyleSheet("background-color:#000;border:1px solid #333;")
        self.cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.cam)

        # ---- Butonları tek satıra yerleştir ----
        self.buttons_mtx = [[]]
        for item in COMMAND_LAYOUT[0]:
            label, prompt = item
            b = AgentButton(label, prompt, self, size_h=40)
            self.grid.addWidget(b)
            self.buttons_mtx[0].append(b)

        self.rows = 1
        self.sel_row = 0
        self.sel_col = 0
        self.update_visual()

        self.nav_dir = None
        self.nav_first = True
        self.nav_last_t = 0.0
        self.last_commit = 0.0
        self.require_open = False
        self.is_processing = False

        self.poll = QTimer()
        self.poll.timeout.connect(self.poll_update)
        self.poll.start(40)

        self.worker = None


    def update_log(self, text):
        self.text_area.append(f">> {text}")
        sb = self.text_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_visual(self):
        for r, row in enumerate(self.buttons_mtx):
            for c, btn in enumerate(row):
                btn.set_selected(r == self.sel_row and c == self.sel_col)

    def move_selection(self, dir_):
        if dir_ == "right":
            if self.sel_col < len(self.buttons_mtx[self.sel_row]) - 1:
                self.sel_col += 1
            else:
                self.sel_row = (self.sel_row + 1) % self.rows
                self.sel_col = 0
        elif dir_ == "left":
            if self.sel_col > 0:
                self.sel_col -= 1
            else:
                self.sel_row = (self.sel_row - 1) if self.sel_row > 0 else self.rows - 1
                self.sel_col = len(self.buttons_mtx[self.sel_row]) - 1

        self.update_visual()
        for r in self.buttons_mtx:
            for b in r:
                b.prog.setValue(0)

    def execute_command(self, btn: AgentButton):
        if btn.prompt == "EXIT_APP":
            QApplication.quit()
            return

        if self.is_processing:
            return

        self.is_processing = True
        self.buffer_label.setText("Agent: ÇALIŞIYOR...")
        self.buffer_label.setStyleSheet(
            "color:#ff9900;font-size:14px;font-weight:bold;"
        )
        self.update_log(f"Komut: {btn.label_text}")

        force_cleanup()

        self.worker = AgentWorker(btn.prompt)
        self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self, result: str):
        self.is_processing = False
        self.buffer_label.setText("Agent: HAZIR")
        self.buffer_label.setStyleSheet(
            "color:#0bf;font-size:14px;font-weight:bold;"
        )
        self.update_log(result)
        self.require_open = True

    def poll_update(self):
        with shared_lock:
            f = shared_frame.copy() if shared_frame is not None else None
            sx, sy = shared_smoothed
            closed_elapsed = shared_eye_closed_elapsed
            state = shared_state   # state'i şimdilik sadece debug için alıyoruz

        if f is not None:
            try:
                disp_w, disp_h = 200, 110
                f_resized = cv2.resize(f, (disp_w, disp_h))
                rgb = cv2.cvtColor(f_resized, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                h, w, ch = rgb.shape
                q = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888).copy()
                self.cam.setPixmap(QPixmap.fromImage(q))
            except Exception as e:
                print("[DEBUG] Kamera görüntüsü QLabel'a aktarılırken hata:", e)
        else:
            self.cam.setText("no cam")

        if self.is_processing:
            return

        if self.require_open:
            if closed_elapsed == 0.0:
                self.require_open = False
            for r in self.buttons_mtx:
                for b in r:
                    b.prog.setValue(0)
            return

        try:
            cur_btn = self.buttons_mtx[self.sel_row][self.sel_col]
        except IndexError:
            return

        if closed_elapsed > 0.0:
            if cur_btn.low_start is None:
                cur_btn.low_start = time.time() - min(closed_elapsed, self.full_thresh)

            el = time.time() - cur_btn.low_start
            prog = min(100, int((el / self.full_thresh) * 100))
            cur_btn.prog.setValue(prog)

            # dwell tamamlanınca komutu çalıştır
            if el >= self.full_thresh and (time.time() - self.last_commit > 1.0):
                self.last_commit = time.time()
                cur_btn.prog.setValue(0)
                self.execute_command(cur_btn)
                return

            self.nav_dir = None
            self.nav_first = True
            return
        else:
            cur_btn.low_start = None
            cur_btn.prog.setValue(0)

        dx = sx - SCREEN_W / 2

        if abs(dx) <= DEAD_ZONE_X:
            self.nav_dir = None
            self.nav_first = True
            return

        direction = "left" if dx < -DEAD_ZONE_X else "right"
        now = time.time()

        if self.nav_dir != direction:
            self.nav_dir = direction
            self.nav_first = True
            self.nav_last_t = now
            return

        wait = HOLD_TIME_FIRST if self.nav_first else HOLD_TIME_REPEAT
        if now - self.nav_last_t >= wait:
            self.move_selection(direction)
            self.nav_last_t = now
            self.nav_first = False

def main():
    global terminate_flag, gaze_thread

    print("[MAIN] gaze_agent_ui.py (tek process) başlatılıyor...")
    terminate_flag = False
    gaze_thread = threading.Thread(target=gaze_loop, daemon=True)
    gaze_thread.start()

    app = QApplication(sys.argv)
    ui = AgentKeyboardUI()
    ui.show()

    try:
        sys.exit(app.exec())
    finally:
        print("[MAIN] Qt event loop bitti, gaze_thread sonlandırılıyor...")
        terminate_flag = True
        if gaze_thread:
            gaze_thread.join(2)
        try:
            agent_core.shutdown()
        except Exception as e:
            print(f"[MAIN] agent_core.shutdown hatası: {e}")
        print("[MAIN] Çıkış.")

if __name__ == "__main__":
    main()
