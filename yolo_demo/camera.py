# -*- coding: utf-8 -*-
"""
camera.py — 摄像头实时手语识别（显示版）

定位手 + 分类的逻辑在 hand_recognizer.py（公共核心，高内聚）。
本脚本只负责：开摄像头 → 每帧调 recognizer → 在画面上显示结果。
不含累积/防抖等额外逻辑（那是 camera_text.py 的事）。

用法：
    python camera.py [--model <权重>] [--conf <阈值>] [--cam <编号>]
    按键：q 退出
"""

import argparse
import cv2
from hand_recognizer import HandRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description="摄像头实时手语识别（显示版）")
    parser.add_argument("--model", type=str, default="models/asl_demo-6-best.pt",
                        help="分类模型权重路径")
    parser.add_argument("--hand", type=str, default="hand_landmarker.task",
                        help="MediaPipe 手部模型路径")
    parser.add_argument("--conf", type=float, default=0.5, help="分类置信度阈值")
    parser.add_argument("--cam", type=int, default=0, help="摄像头编号")
    return parser.parse_args()


def main():
    args = parse_args()

    recognizer = HandRecognizer(args.model, args.hand)
    print(f"模型加载完成: {args.model}")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("无法打开摄像头，请检查连接或换 --cam")
        recognizer.close()
        return
    print("摄像头已打开，按 q 退出")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 识别（核心逻辑在公共模块里）
        r = recognizer.recognize(frame)
        frame_h, frame_w = frame.shape[:2]

        if r["detected"]:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{r['letter']} {r['conf']:.0%}" if r["conf"] >= args.conf else f"? {r['conf']:.0%}"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No hand", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "q: quit", (frame_w - 120, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("ASL Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()
    print("已退出")


if __name__ == "__main__":
    main()
