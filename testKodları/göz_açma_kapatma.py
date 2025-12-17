#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import time
import argparse
import pyautogui


FRAME_W, FRAME_H = 640, 480

PRINT_INTERVAL   = 0.05  # saniye

MOVE_THRESH_PX = 30.0      # pupil merkez hareket eşiği
AREA_MIN       = 400.0     # ellipse alan eşiği
HYST_OPEN      = 3         # ardışık "open" kare eşiği
HYST_CLOSED    = 3         # ardışık "closed" kare eşiği

# --- YASSILIK (FLATNESS) TABANLI EŞİKLER (ELLE AYARLANACAK) ---
FLAT_OPEN_MAX     = 0.35   # bunun ALTINDA ise "göz açık lehine" ipucu
FLAT_CLOSED_MIN   = 0.55   # bunun ÜSTÜNDE ise "göz kapalı lehine" ipucu (normal histerezisli)
FLAT_HARD_CLOSED  = 0.70   # bunun ÜSTÜNDE ise histerezis BYPASS → direkt CLOSED

THR_ADD_STRICT = 5
THR_ADD_MED    = 15
THR_ADD_RELAX  = 25
MASK_SIZE      = 250       # kare maske kenarı (piksel)

SCREEN_W, SCREEN_H = pyautogui.size()
MARGIN_RATIO = 0.02
MARGIN_X = SCREEN_W * MARGIN_RATIO
MARGIN_Y = SCREEN_H * MARGIN_RATIO


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
    def __init__(self, freq=30, minc=0.4, beta=0.01, dcut=1.0):
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


euro_x = OneEuro(freq=30, minc=0.4, beta=0.01, dcut=1.0)
euro_y = OneEuro(freq=30, minc=0.6, beta=0.01, dcut=1.0)


def crop_to_aspect_ratio(image, width=FRAME_W, height=FRAME_H):
    h, w = image.shape[:2]
    target_ratio = width / height
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = int(target_ratio * h)
        off = (w - new_w) // 2
        img = image[:, off:off + new_w]
    else:
        new_h = int(w / target_ratio)
        off = (h - new_h) // 2
        img = image[off:off + new_h, :]
    return cv2.resize(img, (width, height))


def get_darkest_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = 9_999_999
    point = None
    for y in range(20, gray.shape[0] - 20, 10):
        for x in range(20, gray.shape[1] - 20, 10):
            area = gray[y:y + 20, x:x + 20]
            val = int(np.sum(area))
            if val < min_sum:
                min_sum = val
                point = (x + 10, y + 10)
    return point


def apply_binary_threshold(gray, base, add):
    _, thr = cv2.threshold(gray, base + add, 255, cv2.THRESH_BINARY_INV)
    return thr


def mask_outside_square(img, center, size):
    x, y = center
    mask = np.zeros_like(img)
    half = size // 2
    y0 = max(0, y - half)
    y1 = min(img.shape[0], y + half)
    x0 = max(0, x - half)
    x1 = min(img.shape[1], x + half)
    mask[y0:y1, x0:x1] = 255
    return cv2.bitwise_and(img, mask)


def open_camera(cam_index=None):
    """
    --cam-index verilirse onu dener,
    verilmezse 0..5 arası indexleri sırayla dener.
    """
    if cam_index is not None:
        candidates = [cam_index]
    else:
        candidates = [0, 1, 2, 3, 4, 5]

    for idx in candidates:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"[debug] camera opened on index {idx}")
            return cap
        cap.release()
    print("[debug] camera error: hiçbir index açılamadı")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-index", type=int, default=None,
                        help="Kamera index (vermezsen 0-5 arası otomatik dener)")
    args = parser.parse_args()

    cap = open_camera(args.cam_index)
    if cap is None:
        return

    # Kamera ayarları (deneme, hata verirse sorun değil)
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, 10)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        cap.set(cv2.CAP_PROP_GAIN, 0)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    except Exception as e:
        print("camera props set error:", e)

    print("[debug] kamera ok, tespit başlıyor. Çıkış için 'q' ya da ESC.")

    prev_center = None
    open_cnt = 0
    closed_cnt = 0
    state = "OPEN"
    closed_start_t = None

    last_print = 0.0

    sharpen_k = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], dtype=np.float32)

    while True:
        ret, f = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        f = cv2.flip(f, 1)
        f = cv2.filter2D(f, -1, sharpen_k)
        frame = crop_to_aspect_ratio(f, FRAME_W, FRAME_H)

        darkest_point = get_darkest_area(frame)
        draw = frame.copy()

        move = 999.0
        area = 0.0
        flatness = 0.0
        is_closed_candidate = True   # default “kapalı adayı”
        hard_closed = False          # FLAT_HARD_CLOSED ile histerezisi bypass et
        reasons = []
        sx = None
        sy = None

        center_to_map = None

        if darkest_point is None:
            reasons.append("NO_DARKEST_POINT (pupil bulunamadı)")
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            base = int(gray[darkest_point[1], darkest_point[0]])

            thr_strict = apply_binary_threshold(gray, base, THR_ADD_STRICT)
            thr_med    = apply_binary_threshold(gray, base, THR_ADD_MED)
            thr_relax  = apply_binary_threshold(gray, base, THR_ADD_RELAX)

            thr_strict = mask_outside_square(thr_strict, darkest_point, MASK_SIZE)
            thr_med    = mask_outside_square(thr_med,    darkest_point, MASK_SIZE)
            thr_relax  = mask_outside_square(thr_relax,  darkest_point, MASK_SIZE)

            k = np.ones((5, 5), np.uint8)
            dil = cv2.dilate(thr_med, k, 2)
            contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                reasons.append("NO_CONTOUR (pupil konturu yok)")
            else:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) < 5:
                    reasons.append("FEW_POINTS (ellipse fit için yetersiz nokta)")
                else:
                    ellipse = cv2.fitEllipse(largest)
                    (cx, cy), axes, _ = ellipse
                    area = math.pi * (axes[0] / 2.0) * (axes[1] / 2.0)

                    major = max(axes[0], axes[1])
                    minor = min(axes[0], axes[1])
                    if major > 1e-3:
                        flatness = 1.0 - (minor / major)   # 0 = daire, 1 = çok yassı
                    else:
                        flatness = 0.0

                    if prev_center is None:
                        move = 0.0
                    else:
                        dx = abs(cx - prev_center[0])
                        dy = abs(cy - prev_center[1])
                        move = math.hypot(dx, dy)
                    prev_center = (cx, cy)

                    center_to_map = (cx, cy)

                    # --- Orlosky kapalı adayı kararı (hareket + küçük alan) ---
                    is_closed_candidate = (move > MOVE_THRESH_PX) or (area < AREA_MIN)
                    if move > MOVE_THRESH_PX:
                        reasons.append(f"MOVE_LARGE (move={move:.1f} > {MOVE_THRESH_PX})")
                    if area < AREA_MIN:
                        reasons.append(f"AREA_SMALL (area={area:.1f} < {AREA_MIN})")

                    # --- FLATNESS TABANLI KARAR ---
                    if flatness >= FLAT_CLOSED_MIN:
                        is_closed_candidate = True
                        reasons.append(
                            f"FLATNESS_HIGH (flat={flatness:.2f} >= {FLAT_CLOSED_MIN})"
                        )
                    elif flatness <= FLAT_OPEN_MAX:
                        is_closed_candidate = False
                        reasons.append(
                            f"FLATNESS_ROUND (flat={flatness:.2f} <= {FLAT_OPEN_MAX})"
                        )

                    cv2.ellipse(draw, ellipse, (0, 255, 255), 2)
                    cv2.circle(draw, (int(cx), int(cy)), 4, (255, 255, 0), -1)

        if center_to_map is not None:
            px, py = center_to_map
            sx = MARGIN_X + (SCREEN_W - 2 * MARGIN_X) * (px / FRAME_W)
            sy = MARGIN_Y + (SCREEN_H - 2 * MARGIN_Y) * (py / FRAME_H)
            sx = euro_x(sx)
            sy = euro_y(sy)

        if hard_closed:
            state = "CLOSED"
            closed_cnt = HYST_CLOSED + 1  # ilerleyen framelerde de kapalı kalmaya devam etsin
            open_cnt = 0
        else:
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

        # --- MERKEZ NOKTA EKRAN DIŞINDA İSE KAPALI SAY ---
        if sx is not None and sy is not None:
            if sx < 0 or sx > SCREEN_W or sy < 0 or sy > SCREEN_H:
                state = "CLOSED"
                reasons.append(f"CENTER_OUT_OF_SCREEN (sx={sx:.1f}, sy={sy:.1f})")

        color = (0, 255, 0) if state == "OPEN" else (0, 0, 255)
        cv2.putText(draw,
                    f"{state}  area={area:.0f}  flat={flatness:.2f}  move={move:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

        cv2.imshow("eye_close_flat_debug", draw)

        # ---- TERMINAL LOG: sadece kapalıysa sebebi yaz ----
        if state == "CLOSED" and (now - last_print > PRINT_INTERVAL):
            if not reasons:
                reasons_str = "UNKNOWN_REASON"
            else:
                reasons_str = " | ".join(reasons)
            sx_str = f"{sx:.1f}" if sx is not None else "None"
            sy_str = f"{sy:.1f}" if sy is not None else "None"
            print(f"[CLOSED] elapsed={closed_elapsed:.2f}s  area={area:.1f}  "
                  f"flat={flatness:.3f}  move={move:.1f}  "
                  f"sx={sx_str} sy={sy_str}  reasons: {reasons_str}")
            last_print = now

        # ---- KLAVYE GİRİŞLERİ ----
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q veya ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
