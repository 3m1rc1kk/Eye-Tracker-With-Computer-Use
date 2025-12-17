#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import time
import pyautogui
import json
import os

pyautogui.FAILSAFE = False

SCREEN_W, SCREEN_H = pyautogui.size()
MARGIN_RATIO = 0.02
MARGIN_X = SCREEN_W * MARGIN_RATIO
MARGIN_Y = SCREEN_H * MARGIN_RATIO

FRAME_W = 320
ASPECT = 3 / 4
FRAME_H = int(FRAME_W * ASPECT)

DEAD_R = 0.30      # büyüdükçe merkez daha geniş, daha az kıpırdar
MAX_SPEED = 120     # hareket hâlâ hızlı ise bunu daha da düşürebilirsin

ALPHA = 0.27

SAFE_MARGIN = 3  # köşeden 3 piksel içeride tut

mirror_preview = True

# ---- 5 yönlü kalibrasyon ----
CALIB_FILE = "eye_dir_calib5.json"
CALIB_DIRS = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]
calib_samples = {d: [] for d in CALIB_DIRS}
calib_points = {}   # {dir: (mx,my)}
mapping_ready = False


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
    return cv2.resize(img, (w, h))


def detect_pupil(gray):
    """
    Çok basit, minimum test için:
    - Gauss blur
    - En karanlık nokta = pupil merkezi varsay
    """
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    min_val, _, min_loc, _ = cv2.minMaxLoc(blur)
    cx, cy = min_loc
    return cx, cy, 0.0  # flatness şimdilik 0


def update_mapping_ready():
    global mapping_ready
    mapping_ready = all(d in calib_points for d in CALIB_DIRS)


def map_with_calib(px, py):
    """
    Kalibre edilmiş 5 nokta (CENTER, LEFT, RIGHT, UP, DOWN) ile
    px,py -> norm (-1..1) döndür.
    """
    if not mapping_ready:
        return None

    Cx, Cy = calib_points["CENTER"]

    # ----- X ekseni -----
    if px >= Cx and "RIGHT" in calib_points:
        denom = calib_points["RIGHT"][0] - Cx
        if abs(denom) < 1e-6:
            nx = 0.0
        else:
            nx = (px - Cx) / denom
    elif px < Cx and "LEFT" in calib_points:
        denom = Cx - calib_points["LEFT"][0]
        if abs(denom) < 1e-6:
            nx = 0.0
        else:
            nx = (px - Cx) / denom
    else:
        nx = 0.0

    # ----- Y ekseni (aşağı +, yukarı -) -----
    if py >= Cy and "DOWN" in calib_points:
        denom = calib_points["DOWN"][1] - Cy
        if abs(denom) < 1e-6:
            ny = 0.0
        else:
            ny = (py - Cy) / denom
    elif py < Cy and "UP" in calib_points:
        denom = Cy - calib_points["UP"][1]
        if abs(denom) < 1e-6:
            ny = 0.0
        else:
            ny = (py - Cy) / denom
    else:
        ny = 0.0

    nx = max(-1.5, min(1.5, nx))
    ny = max(-1.5, min(1.5, ny))
    return nx, ny


def save_calib(path=CALIB_FILE):
    data = {k: v for k, v in calib_points.items()}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print("[calib] saved to", path)
    except Exception as e:
        print("[calib] save error:", e)


def load_calib(path=CALIB_FILE):
    global calib_points
    if not os.path.exists(path):
        print("[calib] file not found:", path)
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        calib_points = {k: tuple(v) for k, v in data.items()}
        update_mapping_ready()
        print("[calib] loaded:", calib_points)
    except Exception as e:
        print("[calib] load error:", e)


def collect_dir(cap, direction, n_samples=40):
    """
    Belirli yöne (CENTER/LEFT/RIGHT/UP/DOWN) bakarken
    n_samples adet pupil örneği topla, median'ı calib_points'e yaz.
    """
    global calib_samples

    dir_name_tr = {
        "CENTER": "MERKEZ",
        "LEFT": "SOL",
        "RIGHT": "SAĞ",
        "UP": "YUKARI",
        "DOWN": "AŞAĞI",
    }.get(direction, direction)

    print(f"[calib] {direction} için {n_samples} örnek toplanacak.")
    print(f"        Lütfen ekranda {dir_name_tr} bölgesine bak ve bekle.")

    samples = []
    while len(samples) < n_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = crop_to_aspect(frame, FRAME_W, FRAME_H)
        if mirror_preview:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p = detect_pupil(gray)
        if p is not None:
            px, py, _ = p
            samples.append((float(px), float(py)))
            cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)

        cv2.putText(
            frame,
            f"CALIB {direction} {len(samples)}/{n_samples}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "q/ESC: iptal",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

        cv2.imshow("eye_dir_test", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            print("[calib] iptal edildi.")
            return

    if samples:
        arr = np.array(samples, dtype=float)
        med = np.median(arr, axis=0)
        calib_samples[direction] = samples
        calib_points[direction] = (float(med[0]), float(med[1]))
        update_mapping_ready()
        print(f"[calib] {direction} median -> {calib_points[direction]}")
    else:
        print("[calib] yeterli örnek yok.")


def main():
    global mirror_preview

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("camera error")
        return

    mx, my = SCREEN_W / 2, SCREEN_H / 2
    pyautogui.moveTo(int(mx), int(my))

    mouse_enabled = False

    print(
        """\
Kısayollar:
  SPACE : mouse kontrolünü aç/kapat
  m     : aynalama aç/kapat
  5     : merkez kalibrasyonu
  4     : sol kalibrasyonu
  6     : sağ kalibrasyonu
  8     : yukarı kalibrasyonu
  2     : aşağı kalibrasyonu
  v     : kalibrasyonu dosyaya kaydet
  l     : dosyadan kalibrasyonu yükle
  c     : kalibrasyonu sıfırla
  q/ESC : çık
"""
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = crop_to_aspect(frame, FRAME_W, FRAME_H)
        if mirror_preview:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = detect_pupil(gray)

        if res is not None:
            px, py, _ = res
            cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)

            # ----- norm vektör -----
            if mapping_ready:
                nx, ny = map_with_calib(px, py)
            else:
                cx, cy = FRAME_W / 2, FRAME_H / 2
                nx = (px - cx) / (FRAME_W * 0.5)
                ny = (py - cy) / (FRAME_H * 0.5)
                nx = max(-1.5, min(1.5, nx))
                ny = max(-1.5, min(1.5, ny))

            # ----- hız hesabı -----
            r = math.sqrt(nx * nx + ny * ny)
            if r < DEAD_R or not mouse_enabled:
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

            if mouse_enabled:
                mx = mx + ALPHA * vx
                my = my + ALPHA * vy

                mx = max(SAFE_MARGIN, min(SCREEN_W - SAFE_MARGIN, mx))
                my = max(SAFE_MARGIN, min(SCREEN_H - SAFE_MARGIN, my))

                pyautogui.moveTo(int(mx), int(my))

            cv2.putText(
                frame,
                f"nx={nx:.2f} ny={ny:.2f} r={r:.2f} map={'Y' if mapping_ready else 'N'}  mouse={'ON' if mouse_enabled else 'OFF'}",
                (10, FRAME_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        cv2.imshow("eye_dir_test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break
        elif key == ord(" "):  # SPACE
            mouse_enabled = not mouse_enabled
            print("[info] mouse enabled:", mouse_enabled)
        elif key == ord("m"):
            mirror_preview = not mirror_preview
            print("[info] mirror:", mirror_preview)
        elif key == ord("5"):
            collect_dir(cap, "CENTER")
        elif key == ord("4"):
            collect_dir(cap, "LEFT")
        elif key == ord("6"):
            collect_dir(cap, "RIGHT")
        elif key == ord("8"):
            collect_dir(cap, "UP")
        elif key == ord("2"):
            collect_dir(cap, "DOWN")
        elif key == ord("v"):
            save_calib()
        elif key == ord("l"):
            load_calib()
        elif key == ord("c"):
            calib_points.clear()
            update_mapping_ready()
            print("[calib] cleared")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
