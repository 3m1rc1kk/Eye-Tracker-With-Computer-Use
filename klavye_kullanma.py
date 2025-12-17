

import cv2, numpy as np, pyautogui, sys, math, time, threading, subprocess, os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QGridLayout, QVBoxLayout,
    QPushButton, QLabel, QTextEdit, QProgressBar, QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

import re
from collections import Counter, defaultdict

SUGGESTION_ENABLED = True
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as e:
    print("[suggest] torch/transformers import edilemedi, öneriler kapalı:", e)
    SUGGESTION_ENABLED = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

tokenizer = None
model = None
PREFIX_TREE = {}
device = "cpu"

WORD_MIN_FREQ       = 10
WORD_MAX_VOCAB      = 8000
WORD_MAX_PREFIX_LEN = 6
WORD_MAX_PER_PREFIX = 20

def build_word_pool(text_file, min_freq=5, max_vocab=10000, max_prefix_len=6, max_per_prefix=20):
    try:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read().lower()
        words = re.findall(r"[a-zçğıöşü]{2,}", text)
        word_counts = Counter(words)
        most_common = word_counts.most_common(max_vocab)
        filtered = [(w, c) for (w, c) in most_common if c >= min_freq]
        prefix_tree = defaultdict(list)
        for word, count in filtered:
            L = min(len(word), max_prefix_len)
            for i in range(1, L + 1):
                prefix = word[:i]
                prefix_tree[prefix].append((word, count))
        for prefix in list(prefix_tree.keys()):
            entries = prefix_tree[prefix]
            entries.sort(key=lambda x: x[1], reverse=True)
            prefix_tree[prefix] = entries[:max_per_prefix]
        return dict(prefix_tree)
    except FileNotFoundError:
        print("⚠️ Metin dosyası bulunamadı, sadece GPT-2 kullanılacak")
        return {}

def suggest_char_by_char(text, limit=5):
    if not PREFIX_TREE: return []
    text = text.lower().strip()
    if not text: return []
    words = text.split()
    if not words: return []
    last_word = words[-1]
    prefix_part = " ".join(words[:-1])
    if last_word in PREFIX_TREE:
        candidates = PREFIX_TREE[last_word][:limit]
        if prefix_part: return [f"{prefix_part} {w}" for w, _ in candidates]
        else: return [w for w, _ in candidates]
    return []

def suggest_gpt2(prompt, max_tokens=8, num_sugg=3):
    if not SUGGESTION_ENABLED or tokenizer is None or model is None: return []
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_tokens, num_beams=5,
                num_return_sequences=num_sugg, no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
            )
        results = []
        input_len = inputs["input_ids"].shape[-1]
        for out in outputs:
            text = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            text = text.split(".")[0].strip()
            if text: results.append(text)
        return results
    except Exception as e:
        print("[suggest] GPT-2 generate hatası:", e)
        return []

def smart_suggest(text, limit=5):
    if not SUGGESTION_ENABLED: return []
    text = text.strip()
    if not text: return []
    try:
        if len(text) <= 10:
            sugg = suggest_char_by_char(text, limit=limit)
            if sugg: return sugg
            return suggest_gpt2(text, max_tokens=5, num_sugg=limit)
        else:
            conts = suggest_gpt2(text, max_tokens=8, num_sugg=limit)
            return [f"{text} {c}" for c in conts]
    except Exception as e:
        print("[suggest] smart_suggest hatası:", e)
        return []

if SUGGESTION_ENABLED and torch is not None and AutoTokenizer is not None:
    try:
        MODEL_PATH = "./gpt2_FINAL_COMPLETE"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[suggest] Model yükleniyor: {MODEL_PATH} ({device})")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
        model.eval()
        print("[suggest] GPT-2 model yüklendi")
        PREFIX_TREE = build_word_pool("./combined_cleaned_texts.txt", min_freq=WORD_MIN_FREQ, max_vocab=WORD_MAX_VOCAB)
        if PREFIX_TREE: print(f"✅ {len(PREFIX_TREE)} prefix yüklendi")
    except Exception as e:
        print("[suggest] init hatası:", e)
        SUGGESTION_ENABLED = False
else:
    SUGGESTION_ENABLED = False
    print("[suggest] Öneri sistemi devre dışı")

SCREEN_W, SCREEN_H = pyautogui.size()
MARGIN_RATIO = 0.02
MARGIN_X = SCREEN_W * MARGIN_RATIO
MARGIN_Y = SCREEN_H * MARGIN_RATIO

PRINT_INTERVAL = 0.05
EYE_CLOSED_TIME = 1.2      # dwell süresi (s)
EAR_HALF_RATIO = 0.5

OPEN_RELEASE_TIME = 0.25
FRAME_W, FRAME_H = 640, 480

DEAD_ZONE_X = 120
HOLD_TIME_FIRST = 0.45
HOLD_TIME_REPEAT = 0.35

MOVE_THRESH_PX = 30.0
AREA_MIN = 400.0
HYST_OPEN = 3
HYST_CLOSED = 3

FLAT_OPEN_MAX = 0.35
FLAT_CLOSED_MIN = 0.55

THR_ADD_STRICT = 5
THR_ADD_MED = 15
THR_ADD_RELAX = 25
MASK_SIZE = 250

EXIT_KEY_LABEL = "Kapat"
ENTER_KEY_LABEL = "Enter"
BACKSPACE_KEY_LABEL = "Sil"

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

class LowPass:
    def __init__(self, a, init=0.0):
        self.a = a; self.s = init; self.init = False
    def filter(self, x, a):
        if not self.init: self.s = x; self.init = True; return x
        self.s = a * x + (1 - a) * self.s
        return self.s

def alpha_from_cutoff(c, dt):
    tau = 1.0 / (2 * math.pi * c)
    return 1.0 / (1.0 + tau / dt)

class OneEuro:
    def __init__(self, freq=30, minc=0.4, beta=0.01, dcut=1.0):
        self.f = freq; self.mc = minc; self.b = beta; self.dc = dcut
        self.lp_x = LowPass(alpha_from_cutoff(self.mc, 1 / self.f))
        self.lp_dx = LowPass(alpha_from_cutoff(self.dc, 1 / self.f))
        self.prev = None; self.last = None
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

euro_x = OneEuro(freq=30, minc=0.4, beta=0.01, dcut=1.0)
euro_y = OneEuro(freq=30, minc=0.6, beta=0.01, dcut=1.0)

def crop_to_aspect_ratio(image, width=FRAME_W, height=FRAME_H):
    h, w = image.shape[:2]
    target_ratio = width / height
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = int(target_ratio * h); off = (w - new_w) // 2
        img = image[:, off : off + new_w]
    else:
        new_h = int(w / target_ratio); off = (h - new_h) // 2
        img = image[off : off + new_h, :]
    return cv2.resize(img, (width, height))

def get_darkest_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = 9_999_999; point = None
    for y in range(20, gray.shape[0] - 20, 10):
        for x in range(20, gray.shape[1] - 20, 10):
            area = gray[y : y + 20, x : x + 20]
            val = int(np.sum(area))
            if val < min_sum: min_sum = val; point = (x + 10, y + 10)
    return point

def open_any_camera(candidate_indices=(2, 0, 1, 3)):
    for idx in candidate_indices:
        if idx is None: continue
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened(): print(f"[keyboard] camera opened {idx}"); return cap
        cap.release()
    print("[keyboard] camera error"); return None

def apply_binary_threshold(gray, base, add):
    _, thr = cv2.threshold(gray, base + add, 255, cv2.THRESH_BINARY_INV)
    return thr

def mask_outside_square(img, center, size):
    x, y = center
    mask = np.zeros_like(img)
    half = size // 2
    y0 = max(0, y - half); y1 = min(img.shape[0], y + half)
    x0 = max(0, x - half); x1 = min(img.shape[1], x + half)
    mask[y0:y1, x0:x1] = 255
    return cv2.bitwise_and(img, mask)

def gaze_loop():
    global shared_frame, shared_smoothed, shared_eye_closed_elapsed, terminate_flag
    global shared_conf, shared_area, shared_state, shared_flatness

    cap = open_any_camera((2, 0, 1, 3))
    if cap is None: return

    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    except: pass

    prev_center = None
    open_cnt = 0; closed_cnt = 0
    state = "OPEN"
    closed_start_t = None
    sharpen_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

    while not terminate_flag:
        ret, f = cap.read()
        if not ret: time.sleep(0.01); continue
        f = cv2.flip(f, 1)
        f = cv2.filter2D(f, -1, sharpen_k)
        frame = crop_to_aspect_ratio(f, FRAME_W, FRAME_H)

        darkest_point = get_darkest_area(frame)
        draw = frame.copy()

        move = 999.0; area = 0.0; flatness = 0.0
        is_closed_candidate = True
        center_to_map = None

        if darkest_point is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            base = int(gray[darkest_point[1], darkest_point[0]])
            thr_med = apply_binary_threshold(gray, base, THR_ADD_MED)
            thr_med = mask_outside_square(thr_med, darkest_point, MASK_SIZE)
            k = np.ones((5, 5), np.uint8)
            dil = cv2.dilate(thr_med, k, 2)
            contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) >= 5:
                    ellipse = cv2.fitEllipse(largest)
                    (cx, cy), axes, _ = ellipse
                    area = math.pi * (axes[0] / 2.0) * (axes[1] / 2.0)
                    major = max(axes[0], axes[1]); minor = min(axes[0], axes[1])
                    flatness = 1.0 - (minor / major) if major > 1e-3 else 0.0

                    if prev_center is None: move = 0.0
                    else: move = math.hypot(cx - prev_center[0], cy - prev_center[1])
                    prev_center = (cx, cy)

                    is_closed_candidate = (move > MOVE_THRESH_PX) or (area < AREA_MIN)
                    if flatness >= FLAT_CLOSED_MIN: is_closed_candidate = True
                    elif flatness <= FLAT_OPEN_MAX: is_closed_candidate = False

                    center_to_map = (cx, cy)
                    cv2.ellipse(draw, ellipse, (0, 255, 255), 2)
                    cv2.circle(draw, (int(cx), int(cy)), 4, (255, 255, 0), -1)

        if is_closed_candidate: closed_cnt += 1; open_cnt = 0
        else: open_cnt += 1; closed_cnt = 0

        if closed_cnt > HYST_CLOSED: state = "CLOSED"
        elif open_cnt > HYST_OPEN: state = "OPEN"

        now = time.time()
        if state == "CLOSED":
            if closed_start_t is None: closed_start_t = now
            closed_elapsed = now - closed_start_t
        else:
            closed_start_t = None; closed_elapsed = 0.0

        with shared_lock: shared_eye_closed_elapsed = float(closed_elapsed)

        if center_to_map is not None:
            px, py = center_to_map
            sx = MARGIN_X + (SCREEN_W - 2 * MARGIN_X) * (px / FRAME_W)
            sy = MARGIN_Y + (SCREEN_H - 2 * MARGIN_Y) * (py / FRAME_H)
            sx = euro_x(sx); sy = euro_y(sy)
            with shared_lock: shared_smoothed = (sx, sy)

        conf = 0.0
        if area > 0: conf = 1.0
        if state=="CLOSED": conf = 0.0

        with shared_lock:
            shared_conf = float(conf); shared_area = float(area)
            shared_state = state; shared_flatness = float(flatness)

        cv2.putText(draw, f"{state} F:{flatness:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0) if state=="OPEN" else (0,0,255), 2)
        with shared_lock: shared_frame = draw
        time.sleep(0.005)
    cap.release()

class DwellButton(QWidget):
    def __init__(self, label, parent, size=52):
        super().__init__()
        self.label_text = label
        self.low_start = None
        self.btn = QPushButton(label)
        self.btn.setFixedSize(size, size)
        self.btn.setStyleSheet("background-color:#222;color:#ffffff;font-size:18px;border-radius:6px;")
        self.prog = QProgressBar()
        self.prog.setRange(0, 100); self.prog.setValue(0); self.prog.setTextVisible(False)
        self.prog.setFixedHeight(4)
        self.prog.setStyleSheet("QProgressBar::chunk{background-color:#00bfff;}QProgressBar{background-color:#444;border-radius:2px;}")
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(1)
        lay.addWidget(self.btn); lay.addWidget(self.prog)

    def set_selected(self, sel):
        if sel: self.btn.setStyleSheet("background-color:#00bfff;color:#000000;font-size:18px;border-radius:6px;")
        else: self.btn.setStyleSheet("background-color:#222;color:#ffffff;font-size:18px;border-radius:6px;")

class GazeKeyboard(QWidget):
    def __init__(self, dwell_time_ms=1000):
        super().__init__()
        self.SUGG_KEY_LABELS = ["[S1]", "[S2]", "[S3]"]
        self.keys = [
            self.SUGG_KEY_LABELS,
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "Ğ", "Ü"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L", "Ş", "İ"],
            ["Z", "X", "C", "V", "B", "N", "M", "Ö", "Ç", BACKSPACE_KEY_LABEL, "Space", ENTER_KEY_LABEL, EXIT_KEY_LABEL],
        ]
        self.rows = len(self.keys)
        BUTTON_SIZE = 52
        self.setWindowTitle("Gaze Keyboard (Direct Type)")
        self.setStyleSheet("background-color:#111;")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)

        screen = QApplication.primaryScreen().geometry()

        max_cols = max(len(r) for r in self.keys)
        base_width = max_cols * (BUTTON_SIZE + 6) + 60
        kbd_width_calc = int(base_width * 1.4) # Float çıkmasın diye int
        kbd_width = int(min(kbd_width_calc, screen.width() - 40)) # Garanti olsun diye int
        kbd_height = 430

        self.setFixedSize(kbd_width, kbd_height)
        self.move(int((screen.width()-kbd_width)/2), int(screen.height()-kbd_height-20))

        self.lay = QVBoxLayout(self); self.lay.setContentsMargins(4, 4, 4, 4); self.lay.setSpacing(2)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setFixedHeight(40)
        self.text_area.setStyleSheet("background-color:#222;color:#0bf;font-size:16px;font-weight:bold;")
        self.lay.addWidget(self.text_area)

        top = QHBoxLayout(); top.setSpacing(4)
        self.status = QLabel("Ready"); self.status.setStyleSheet("color:#aaa;font-size:11px;")
        top.addWidget(self.status)
        self.cam = QLabel(); self.cam.setFixedSize(200, 110)
        self.cam.setStyleSheet("background-color:#000;border:1px solid #333;")
        top.addWidget(self.cam)
        self.lay.addLayout(top)

        self.dwell_time_ms = dwell_time_ms
        self.full_thresh = dwell_time_ms / 1000.0

        self.buttons = []
        for r, row_keys in enumerate(self.keys):
            row_btns = []
            for c, key in enumerate(row_keys):
                b = DwellButton(key, self, BUTTON_SIZE)
                self.add_btn(b, r, c)
                row_btns.append(b)
            self.buttons.append(row_btns)

        self.suggestion_texts = ["", "", ""]; self.last_text_for_sugg = ""
        self.sel_row = 2; self.sel_col = 2
        self.update_visual(); self.update_suggestion_buttons()
        self.require_open = False

        self.poll = QTimer(); self.poll.timeout.connect(self.poll_update); self.poll.start(40)
        self.nav_dir = None; self.nav_first = True; self.nav_last_t = 0.0; self.last_commit = 0.0
        self.COMMIT_DEB = 0.5
        self._blink_started_at = None; self._open_stable_since = None

    def add_btn(self, b, r, c):
        if not hasattr(self, "gridw"):
            self.gridw = QWidget(); self.grid = QGridLayout(self.gridw); self.grid.setSpacing(2)
            self.lay.addWidget(self.gridw)
        self.grid.addWidget(b, r, c)

    def update_visual(self):
        for r, row in enumerate(self.buttons):
            for c, btn in enumerate(row):
                btn.set_selected((r == self.sel_row and c == self.sel_col))

    def move_one_step(self, dir_):
        if dir_ == "right":
            if self.sel_col < len(self.buttons[self.sel_row]) - 1: self.sel_col += 1
            else: self.sel_row = (self.sel_row + 1) % self.rows; self.sel_col = 0
        elif dir_ == "left":
            if self.sel_col > 0: self.sel_col -= 1
            else: self.sel_row = self.rows - 1 if self.sel_row == 0 else self.sel_row - 1; self.sel_col = len(self.buttons[self.sel_row]) - 1
        self.update_visual()
        for row in self.buttons:
            for b in row: b.prog.setValue(0); b.low_start = None

    def backspace_action(self):
        current = self.text_area.toPlainText()
        if current: self.text_area.setPlainText(current[:-1])
        try: pyautogui.press("backspace")
        except: pass
        self.last_text_for_sugg = "";
        if SUGGESTION_ENABLED: self.update_suggestions()

    def update_suggestion_buttons(self):
        if not self.buttons or len(self.buttons[0]) < 3: return
        for i in range(3):
            btn = self.buttons[0][i]; text = self.suggestion_texts[i]
            btn.btn.setText(text[:17]+"…" if len(text)>18 else text if text else "...")

    def update_suggestions(self):
        if not SUGGESTION_ENABLED: return
        text = self.text_area.toPlainText().strip()
        if text == self.last_text_for_sugg: return
        self.last_text_for_sugg = text
        if not text: self.suggestion_texts = ["", "", ""]
        else:
            suggs = smart_suggest(text, limit=3)
            self.suggestion_texts = []
            for i in range(3): self.suggestion_texts.append(suggs[i] if i<len(suggs) else "")
        self.update_suggestion_buttons()

    def apply_suggestion(self, idx):
        if not (0 <= idx < len(self.suggestion_texts)):
            return
        suggestion = (self.suggestion_texts[idx] or "").strip()
        if not suggestion:
            return

        current = self.text_area.toPlainText()

        m = re.search(r"([A-Za-zÇĞİÖŞÜçğıöşü]+)$", current)

        if m:
            last_word = m.group(1)
            start = m.start(1)

            new_text = current[:start] + suggestion
            self.text_area.setPlainText(new_text)

            cursor = self.text_area.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.text_area.setTextCursor(cursor)

            try:
                for _ in range(len(last_word)):
                    pyautogui.press("backspace")
                pyautogui.typewrite(suggestion)
            except:
                pass
        else:
            prefix = current
            add_space = (len(prefix) > 0 and not prefix.endswith((" ", "\n")))
            if add_space:
                prefix += " "
                try:
                    pyautogui.typewrite(" ")
                except:
                    pass

            self.text_area.setPlainText(prefix + suggestion)
            cursor = self.text_area.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.text_area.setTextCursor(cursor)

            try:
                pyautogui.typewrite(suggestion)
            except:
                pass

        self.last_text_for_sugg = self.text_area.toPlainText()
        if SUGGESTION_ENABLED:
            self.update_suggestions()

    def stable_closed_elapsed(self, raw_closed_elapsed: float, now: float) -> float:
        if raw_closed_elapsed > 0.0:
            if self._blink_started_at is None: self._blink_started_at = now - raw_closed_elapsed
            self._open_stable_since = None
            return now - self._blink_started_at
        if self._blink_started_at is None: return 0.0
        if self._open_stable_since is None: self._open_stable_since = now
        if (now - self._open_stable_since) < OPEN_RELEASE_TIME: return now - self._blink_started_at
        self._blink_started_at = None; self._open_stable_since = None
        return 0.0

    def poll_update(self):
        with shared_lock:
            f = shared_frame.copy() if shared_frame is not None else None
            sx, sy = shared_smoothed; closed_elapsed = shared_eye_closed_elapsed
            state = shared_state

        now = time.time()
        closed_eff = self.stable_closed_elapsed(closed_elapsed, now)
        self.status.setText(f"{state}"); self.status.setStyleSheet("color:#0f0;" if state=="OPEN" else "color:#f33;")

        if f is not None:
            try:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                q = QImage(rgb.data, f.shape[1], f.shape[0], 3*f.shape[1], QImage.Format.Format_RGB888)
                self.cam.setPixmap(QPixmap.fromImage(q).scaled(200, 110, Qt.AspectRatioMode.KeepAspectRatio))
            except: pass

        if self.require_open:
            if closed_eff == 0.0: self.require_open = False
            self.buttons[self.sel_row][self.sel_col].prog.setValue(0)
            return

        current_key = self.keys[self.sel_row][self.sel_col]
        is_sugg = (self.sel_row == 0 and current_key in self.SUGG_KEY_LABELS)
        is_exit = (current_key == EXIT_KEY_LABEL)
        is_enter = (current_key == ENTER_KEY_LABEL)
        is_back = (current_key == BACKSPACE_KEY_LABEL)
        btn = self.buttons[self.sel_row][self.sel_col]

        # --- DWELL MANTIĞI (BUFFERSIZ) ---
        if closed_eff > 0.0:
            if btn.low_start is None: btn.low_start = now - min(closed_eff, self.full_thresh)
            el = now - btn.low_start
            prog = min(100, int((el / self.full_thresh) * 100))
            btn.prog.setValue(prog)

            if el >= self.full_thresh and now - self.last_commit > self.COMMIT_DEB:
                self.last_commit = now

                btn.prog.setValue(0)

                if is_sugg:
                    self.apply_suggestion(self.sel_col)
                elif is_back:
                    self.backspace_action()
                elif is_exit:
                    QApplication.instance().quit()
                elif is_enter:
                    self.text_area.insertPlainText("\n")
                    try: pyautogui.press("enter")
                    except: pass
                else:
                    txt = " " if current_key == "Space" else current_key
                    self.text_area.insertPlainText(txt)

                    sb = self.text_area.verticalScrollBar()
                    sb.setValue(sb.maximum())

                    try:
                        if current_key == "Space": pyautogui.press("space")
                        else: pyautogui.typewrite(txt)
                    except: pass

                if SUGGESTION_ENABLED and not is_sugg: self.update_suggestions()
                self.require_open = True # İşlem bitti, göz açılana kadar bekle
                return

            self.nav_dir = None; self.nav_first = True
            return

        btn.low_start = None; btn.prog.setValue(0)

        dx = sx - SCREEN_W / 2
        if abs(dx) <= DEAD_ZONE_X:
            self.nav_dir = None; self.nav_first = True
            if SUGGESTION_ENABLED: self.update_suggestions()
            return

        direction = "left" if dx < -DEAD_ZONE_X else "right"
        if self.nav_dir != direction:
            self.nav_dir = direction; self.nav_first = True; self.nav_last_t = now
        else:
            wait = HOLD_TIME_FIRST if self.nav_first else HOLD_TIME_REPEAT
            if now - self.nav_last_t >= wait:
                self.move_one_step(direction)
                self.nav_last_t = now; self.nav_first = False

        if SUGGESTION_ENABLED: self.update_suggestions()

def main():
    global terminate_flag, gaze_thread
    terminate_flag = False
    gaze_thread = threading.Thread(target=gaze_loop, daemon=True)
    gaze_thread.start()
    app = QApplication(sys.argv)
    kb = GazeKeyboard(dwell_time_ms=int(EYE_CLOSED_TIME * 1000))
    kb.show()
    try: sys.exit(app.exec())
    finally: terminate_flag = True; gaze_thread.join(2)

if __name__ == "__main__":
    main()
