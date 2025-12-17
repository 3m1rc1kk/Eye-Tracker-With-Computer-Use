#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300
CAMERA_INDEX = 0  # Kameranız 0, 1 veya 2 olabilir. Açılmazsa burayı değiştirin.

class CameraAligner(QWidget):
    def __init__(self):
        super().__init__()

        # --- PENCERE AYARLARI ---
        self.setWindowTitle("Kamera Hizalama")

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground) # İsteğe bağlı şeffaflık

        self.setStyleSheet("background-color: #000; border: 2px solid #00ff00; border-radius: 10px;")

        # --- KONUMLANDIRMA (Aşağı Orta) ---
        screen = QApplication.primaryScreen().geometry()
        screen_w, screen_h = screen.width(), screen.height()

        pos_x = (screen_w - WINDOW_WIDTH) // 2
        pos_y = screen_h - WINDOW_HEIGHT - 60

        self.setGeometry(pos_x, pos_y, WINDOW_WIDTH, WINDOW_HEIGHT)

        # --- ARAYÜZ ---
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(2, 2, 2, 2)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)

        # --- KAMERA BAŞLATMA ---
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if self.cap.isOpened():
            self.configure_camera(self.cap)
        else:
            self.image_label.setText("KAMERA AÇILAMADI!")
            self.image_label.setStyleSheet("color: red; font-size: 20px;")

        # --- GÜNCELLEME ZAMANLAYICISI ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) # 30 FPS

    def configure_camera(self, cap):
        """Kamerayı göz takibi için optimize eder (Manuel Ayarlar)"""
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            cap.set(cv2.CAP_PROP_FOCUS, 0) # Sonsuza odakla (veya 10-20 deneyin)

            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 1=Manual (bazı kameralarda 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, 100) # Ortama göre artır/azalt

            cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
            cap.set(cv2.CAP_PROP_CONTRAST, 64)
        except Exception:
            pass

    def update_frame(self):
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        h, w, ch = frame.shape

        # --- HİZALAMA REHBERİ ÇİZİMLERİ ---
        center_x, center_y = w // 2, h // 2

        cv2.line(frame, (center_x, 0), (center_x, h), (100, 100, 100), 1)
        cv2.line(frame, (0, center_y), (w, center_y), (100, 100, 100), 1)

        box_w, box_h = 140, 180
        x1 = center_x - box_w // 2
        y1 = center_y - box_h // 2
        x2 = center_x + box_w // 2
        y2 = center_y + box_h // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        eye_level_y = center_y - 20
        cv2.line(frame, (x1, eye_level_y), (x2, eye_level_y), (255, 255, 0), 1)

        cv2.putText(frame, "Gozlerinizi cizgiye hizalayin", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Cikmak icin: ESC", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # --- GÖRÜNTÜYÜ PYQT'YE ÇEVİR ---
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = QPixmap.fromImage(convert_to_qt_format)

        self.image_label.setPixmap(p)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraAligner()
    window.show()
    sys.exit(app.exec())
