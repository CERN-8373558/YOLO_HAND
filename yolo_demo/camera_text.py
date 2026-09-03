# -*- coding: utf-8 -*-
"""
camera_text.py — 摄像头手语打字机（文字串累积）

定位手 + 分类的逻辑在 hand_recognizer.py（公共核心，高内聚）。
本脚本只负责累积相关的逻辑（防抖、追加、退格、清空）→ 与 camera.py 解耦。

    特殊手势：space→空格  del→退格  nothing→忽略
    防抖：同一手势连续 STABLE_FRAMES 帧才输出，过滤手抖。

用法：
    python camera_text.py [--model <权重>] [--conf <阈值>] [--cam <编号>]
    按键：q 退出 | c 清空 | 空格键 手动加空格
"""

import argparse
import cv2
from hand_recognizer import HandRecognizer

# 连续多少帧识别同一字母才输出（越大越稳但越慢）
STABLE_FRAMES = 8
# 类别 → 动作映射
ACTION_MAP = {"space": " ", "del": "\b", "nothing": None}


def parse_args():
    parser = argparse.ArgumentParser(description="摄像头手语识别 + 文字串累积")
    parser.add_argument("--model", type=str, default="models/asl_demo-6-best.pt",
                        help="分类模型权重路径")
    parser.add_argument("--hand", type=str, default="hand_landmarker.task",
                        help="MediaPipe 手部模型路径")
    parser.add_argument("--conf", type=float, default=0.5, help="分类置信度阈值")
    parser.add_argument("--cam", type=int, default=0, help="摄像头编号")
    parser.add_argument("--stable", type=int, default=STABLE_FRAMES,
                        help="连续稳定帧数，防抖参数")
    return parser.parse_args()


class TextBuffer:
    """文字累积缓冲区。"""

    def __init__(self):
        self.chars = []

    def append(self, ch):
        self.chars.append(ch)

    def backspace(self):
        if self.chars:
            self.chars.pop()

    def clear(self):
        self.chars.clear()

    def text(self):
        return "".join(self.chars)


def main():
    args = parse_args()

    recognizer = HandRecognizer(args.model, args.hand)
    print(f"模型加载完成: {args.model}")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("无法打开摄像头，请检查连接或换 --cam")
        recognizer.close()
        return
    print("摄像头已打开 | 按键: q=退出 c=清空")

    buffer = TextBuffer()
    # 防抖状态
    current_cls = None    # 当前识别的字母
    stable_count = 0      # 连续帧数
    # 光标闪烁计时基准（单位：秒）
    t0 = cv2.getTickCount() / cv2.getTickFrequency()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_w = frame.shape[1]

        # 识别（核心在公共模块）
        r = recognizer.recognize(frame)

        # ---- 防抖确认 + 累积 ----
        if not r["detected"]:
            # 手放下 → 手势段结束，重置
            stable_count = 0
            current_cls = None
        else:
            if r["conf"] < args.conf:
                pass  # 低置信度忽略
            elif r["letter"] != current_cls:
                current_cls = r["letter"]
                stable_count = 1
            else:
                stable_count += 1
                if stable_count == args.stable:
                    action = ACTION_MAP.get(r["letter"], r["letter"])
                    if action is None:
                        pass
                    elif action == "\b":
                        buffer.backspace()
                    else:
                        buffer.append(action)
                    print(f"输入: [{action}] → {buffer.text()}")

        # ---- 显示 ----
        if r["detected"]:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{r['letter']} {r['conf']:.0%}" if r["conf"] >= args.conf else f"? {r['conf']:.0%}"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        text = buffer.text()
        # 光标闪烁：每 0.5 秒交替显示/隐藏末尾的 |（类似输入框）
        elapsed_s = cv2.getTickCount() / cv2.getTickFrequency() - t0
        cursor_visible = int(elapsed_s * 2) % 2 == 0  # 每秒闪 2 次
        display = (text + ("|" if cursor_visible else " ")) if text else ("|" if cursor_visible else " ")
        cv2.rectangle(frame, (0, 0), (frame_w, 40), (40, 40, 40), -1)
        cv2.putText(frame, display, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "q:quit c:clear spc:space", (frame_w - 220, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("ASL Text Input", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            buffer.clear()
            print("已清空")
        elif key == ord(" "):  # 空格键手动加空格（space 手势摆不出来时的备选）
            buffer.append(" ")
            print(f"手动空格 → {buffer.text()}")

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()
    print(f"\n最终文字: {buffer.text() if buffer.text() else '(空)'}")
    print("已退出")


if __name__ == "__main__":
    main()
