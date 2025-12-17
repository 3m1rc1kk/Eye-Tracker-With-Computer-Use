#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Target trainer overlay for testing gaze mouse accuracy.
"""

import sys
import time
import random
import math
import argparse

import pyautogui
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QPainter, QPen, QFont
from PyQt6.QtWidgets import QApplication, QWidget

pyautogui.FAILSAFE = False


class TargetOverlay(QWidget):
    def __init__(self, hold_s: float, radius_px: int, size_px: int,
                 margin_left: int, margin_top: int, margin_right: int, margin_bottom: int):
        super().__init__()

        self.hold_s = float(hold_s)
        self.radius = int(radius_px)
        self.size = int(size_px)

        self.margin_left = int(margin_left)
        self.margin_top = int(margin_top)
        self.margin_right = int(margin_right)
        self.margin_bottom = int(margin_bottom)

        self.screen_w, self.screen_h = pyautogui.size()

        self.target_x = self.screen_w // 2
        self.target_y = self.screen_h // 2
        self.hold_start = None
        self.hits = 0

        self.setWindowTitle("Target Trainer")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self.setFixedSize(self.size, self.size)
        self.spawn_new_target()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)  # ~33 FPS

    def spawn_new_target(self):
        x0 = self.margin_left
        y0 = self.margin_top
        x1 = max(x0 + 1, self.screen_w - self.margin_right)
        y1 = max(y0 + 1, self.screen_h - self.margin_bottom)

        self.target_x = random.randint(x0, x1)
        self.target_y = random.randint(y0, y1)

        self.move(int(self.target_x - self.size / 2), int(self.target_y - self.size / 2))

        self.hold_start = None
        self.update()

    def tick(self):
        mx, my = pyautogui.position()
        d = math.hypot(mx - self.target_x, my - self.target_y)

        now = time.time()
        if d <= self.radius:
            if self.hold_start is None:
                self.hold_start = now
            elapsed = now - self.hold_start
            if elapsed >= self.hold_s:
                self.hits += 1
                self.spawn_new_target()
                return
        else:
            self.hold_start = None

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        prog = 0.0
        if self.hold_start is not None:
            prog = min(1.0, max(0.0, (time.time() - self.hold_start) / self.hold_s))

        pad = 6
        rect = QRectF(pad, pad, self.size - 2 * pad, self.size - 2 * pad)

        pen_bg = QPen()
        pen_bg.setWidth(6)
        pen_bg.setColor(Qt.GlobalColor.darkGray)
        painter.setPen(pen_bg)
        painter.drawEllipse(rect)

        pen_prog = QPen()
        pen_prog.setWidth(6)
        pen_prog.setColor(Qt.GlobalColor.cyan)
        painter.setPen(pen_prog)

        start_angle = 90 * 16
        span_angle = int(-360 * 16 * prog)
        painter.drawArc(rect, start_angle, span_angle)

        dot_r = 6
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Qt.GlobalColor.green)
        painter.drawEllipse(self.size/2 - dot_r, self.size/2 - dot_r, dot_r*2, dot_r*2)

        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(QFont("Arial", 9))
        pct = int(prog * 100)
        painter.drawText(0, self.size - 8, self.size, 12, Qt.AlignmentFlag.AlignCenter, f"{pct}%  |  hits: {self.hits}")

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            QApplication.quit()
            return
        super().keyPressEvent(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=float, default=2.0, help="Hedef üzerinde bekleme süresi (sn)")
    ap.add_argument("--radius", type=int, default=45, help="Hedef kabul yarıçapı (px)")
    ap.add_argument("--size", type=int, default=92, help="Hedef overlay pencere boyutu (px)")
    ap.add_argument("--margin-left", type=int, default=80)
    ap.add_argument("--margin-top", type=int, default=80)
    ap.add_argument("--margin-right", type=int, default=80)
    ap.add_argument("--margin-bottom", type=int, default=220, help="Alttan güvenli alan (klavye/HUD için)")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    w = TargetOverlay(
        hold_s=args.hold,
        radius_px=args.radius,
        size_px=args.size,
        margin_left=args.margin_left,
        margin_top=args.margin_top,
        margin_right=args.margin_right,
        margin_bottom=args.margin_bottom,
    )
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
