# -*- coding: utf-8 -*-
"""
hand_recognizer.py — 手语识别公共核心模块（高内聚，低耦合）

职责：
    只做一件事：对一张画面 "定位手 + 裁剪 + 分类字母"。
    不关心是谁在用（camera.py 显示用、camera_text.py 打字用），
    不包含任何界面/累积/防抖逻辑 → 低耦合。

对外暴露 3 个东西：
    HandRecognizer         类：封装模型加载与识别
    HandRecognizer.recognize(frame)  → (存在手?, 手部框, 字母, 置信度)
    cleanup 后记得 close() 释放资源

这样 camera.py / camera_text.py / predict.py 都 import 它，
代码不再重复，改动一处全脚本生效 → 高内聚。
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

IMGSZ = 224          # 分类模型输入尺寸
PADDING = 0.35       # 手部框外扩比例
MOVE_THRESHOLD = 25  # 运动检测最小位移阈值（像素）


class HandRecognizer:
    """手语识别器：定位手 + 裁剪 + 字母分类 的封装。"""

    def __init__(self, model_path, hand_model_path="hand_landmarker.task", num_hands=1):
        # 分类模型
        self.classifier = YOLO(model_path)
        # 手部关键点模型（Tasks API）
        base_options = python.BaseOptions(model_asset_path=hand_model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=num_hands)
        self.detector = vision.HandLandmarker.create_from_options(options)
        # 运动检测状态：上一帧手部中心 + 位移阈值
        self._last_center = None
        self.MOVE_THRESHOLD = MOVE_THRESHOLD

    def recognize(self, frame):
        """对一帧画面识别手语。

        返回 dict：
            detected : bool  是否检测到手
            box      : (x1,y1,x2,y2)  手部框（像素坐标），无手为 None
            letter   : str   识别出的字母/类别，无手为 None
            conf     : float 置信度(0~1)，无手为 0
        """
        h, w = frame.shape[:2]

        # 1. 定位手（MediaPipe 要 RGB，OpenCV 是 BGR）
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return {"detected": False, "box": None, "letter": None, "conf": 0.0,
                    "moving": False}

        # 2. 算手部框
        box = self._hand_box(result.hand_landmarks[0], w, h)
        x1, y1, x2, y2 = box

        # 2.5 运动检测：手部中心点相对上一帧的位移
        #     位移大 = 手在移动（过渡手势不可靠），位移小 = 手基本静止（可稳定识别）
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        moving = False
        if self._last_center is not None:
            dx, dy = cx - self._last_center[0], cy - self._last_center[1]
            # 位移阈值（像素）：超过则判定在动。画面越大阈值应越大，用相对框宽更好
            threshold = max(self.MOVE_THRESHOLD, int((x2 - x1) * 0.15))
            if abs(dx) > threshold or abs(dy) > threshold:
                moving = True
        self._last_center = (cx, cy)

        # 3. 裁剪 + 分类
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (IMGSZ, IMGSZ))
        results = self.classifier.predict(resized, imgsz=IMGSZ, verbose=False)

        names = results[0].names
        probs = results[0].probs.data.tolist()
        idx = probs.index(max(probs))

        return {"detected": True, "box": box,
                "letter": names[idx], "conf": probs[idx],
                "moving": moving}

    @staticmethod
    def _hand_box(hand_landmarks, frame_w, frame_h, pad=PADDING):
        """由 21 个手部关键点算出带外扩的包围框。"""
        xs = [lm.x for lm in hand_landmarks]
        ys = [lm.y for lm in hand_landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bw = (max_x - min_x) * frame_w
        bh = (max_y - min_y) * frame_h
        x1 = max(0, int((min_x * frame_w) - bw * pad))
        y1 = max(0, int((min_y * frame_h) - bh * pad))
        x2 = min(frame_w - 1, int((max_x * frame_w) + bw * pad))
        y2 = min(frame_h - 1, int((max_y * frame_h) + bh * pad))
        return x1, y1, x2, y2

    def close(self):
        """释放资源（模型/摄像头句柄）"""
        self.detector.close()
