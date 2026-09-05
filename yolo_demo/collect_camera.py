# -*- coding: utf-8 -*-
"""
collect_camera.py — 摄像头额外数据采集脚本

用途：
    补充现有训练集缺少的真实背景（暗/黑背景）手语图。
    与 confusion_camera.py 形式一致：键盘切换采集类，逐帧从摄像头
    检测框裁剪保存。用帧差异阈值去重，避免存大量几乎相同的帧。

用法：
    python collect_camera.py [--model ...] [--hand ...] [--cam 0]
                             [--out 目录] [--per-class 300]
                             [--min-diff 2000] [--size 200]

输出：
    <out>/<类名>/xxx.jpg   每类一个文件夹，如 collect/B/B_000123.jpg
    重启脚本会自动续采（扫描已有张数，不从 0 覆盖）。

操作：
    a-z     切换到要采集的字母（含 q）
    1 2 3   切到 nothing / space / del（凑齐 29 类用）
            · nothing = 不伸手的背景画面（存整帧缩小）
            · space   = 空格手势（正常裁剪）
            · del     = 退格手势（正常裁剪）
    Space   暂停/继续当前类（暂停期间不存图）
    Esc     退出
"""

import argparse
import os
import time
import cv2
import numpy as np
from hand_recognizer import HandRecognizer

LETTERS = "abcdefghijklmnopqrstuvwxyz"
SPECIAL_NAMES = {"nothing": ord("1"), "space": ord("2"), "del": ord("3")}
# 去重：相邻已存帧像素差阈值（mean abs diff * 1000），画面几乎不动则不存
DEFAULT_MIN_DIFF = 2000


def parse_args():
    p = argparse.ArgumentParser(description="摄像头额外数据采集")
    p.add_argument("--model", default="models/asl_demo_dark-best.pt")
    p.add_argument("--hand", default="hand_landmarker.task")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--out", default="collect", help="输出目录（每类一个子文件夹）")
    p.add_argument("--per-class", type=int, default=300, help="每类目标张数(到量暂停等按键)")
    p.add_argument("--min-diff", type=float, default=DEFAULT_MIN_DIFF,
                   help="帧差异阈值(千分位 mean abs diff)，越大越少存")
    p.add_argument("--size", type=int, default=200, help="裁剪后保存尺寸(正方形)")
    return p.parse_args()


def main():
    args = parse_args()
    recognizer = HandRecognizer(args.model, args.hand)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("无法打开摄像头，请检查连接或换 --cam")
        recognizer.close()
        return

    os.makedirs(args.out, exist_ok=True)

    cur = None            # 当前采集类（小写字母）
    counts = {}           # letter -> 已在本会话存的张数
    paused = False
    last_saved = {}       # letter -> 上一张已存裁剪图(灰度) 用于帧差去重
    seq = {}              # letter -> 下一个文件编号

    def dir_name(cls):
        """目录名：字母用大写(A-Z)，特殊类(nothing/space/del)保持小写。"""
        return cls.upper() if len(cls) == 1 else cls

    def class_count(letter):
        """该类文件夹已有文件数（含历史，用于续采/编号）。"""
        d = os.path.join(args.out, dir_name(letter))
        if os.path.isdir(d):
            return len([f for f in os.listdir(d) if f.lower().endswith(".jpg")])
        return 0

    print(f"输出目录: {os.path.abspath(args.out)} | 每类目标 {args.per_class} 张")
    print("按 a-z 切字母类 | 1=nothing 2=space 3=del | Space 暂停/继续 | Esc 退出")

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_h, frame_w = frame.shape[:2]

        r = recognizer.recognize(frame, topk=1)

        # ---- 采集 ----
        # nothing 类：无需检测到手，直接存整帧缩小为背景样本
        # space/del/字母：需检测到手 且 不在移动 才裁剪存
        if cur and not paused:
            if cur == "nothing":
                src = frame  # 整帧背景
            elif r["detected"] and not r.get("moving"):
                x1, y1, x2, y2 = r["box"]
                src = frame[y1:y2, x1:x2]
            else:
                src = None

            if src is not None and src.size > 0:
                resized = cv2.resize(src, (args.size, args.size))
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                # 帧差去重：与当前类上一张差异不足 → 跳过
                prev = last_saved.get(cur)
                if prev is None or np.mean(np.abs(gray.astype(int) - prev.astype(int))) * 1000 >= args.min_diff:
                    n = counts.get(cur, 0)
                    n += 1
                    d = os.path.join(args.out, dir_name(cur))
                    os.makedirs(d, exist_ok=True)
                    if seq.get(cur) is None:
                        seq[cur] = class_count(cur) + 1
                    fname = f"{dir_name(cur)}_{seq[cur]:06d}.jpg"
                    seq[cur] += 1
                    cv2.imwrite(os.path.join(d, fname), resized)
                    counts[cur] = n
                    last_saved[cur] = gray
                    # 存够目标 → 暂停提醒
                    if n >= args.per_class:
                        paused = True
                        print(f"[完成] {cur} 已存 {n} 张，达目标 {args.per_class}，按 Space 继续或按键切类")

        # ---- 显示 ----
        cv2.rectangle(frame, (0, 0), (frame_w, 40), (40, 40, 40), -1)
        cur_disp = cur if cur else "-"
        cur_text = f"class: {cur_disp} (a-z / 1=nothing 2=space 3=del)"
        if paused:
            cur_text += "  [暂停]"
        cv2.putText(frame, cur_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (0, 40), (frame_w, 62), (20, 20, 20), -1)
        if cur:
            n = counts.get(cur, 0)
            total = class_count(cur)
            status = f"{cur}: 本次 {n}/{args.per_class} | 该文件夹共 {total} 张"
            cv2.putText(frame, status, (10, 56), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1, cv2.LINE_AA)

        if r["detected"]:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{r['letter']} {r['conf']:.0%}"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No hand", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "Space:pause Esc:quit", (frame_w - 200, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("ASL Data Collect", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc 退出
            break
        elif key == ord(" "):   # Space 暂停/继续
            paused = not paused
            print("[暂停]" if paused else "[继续]")
        elif ord("a") <= key <= ord("z"):
            cur = chr(key)
            paused = False
            seq[cur] = class_count(cur) + 1
            counts[cur] = 0
            last_saved[cur] = None
            print(f"[采集] 切到 {cur.upper()}（文件夹现有 {seq[cur]-1} 张，从此续采）")
        elif key in (ord("1"), ord("2"), ord("3")):
            # 1=nothing 2=space 3=del
            cur = next(name for name, k in SPECIAL_NAMES.items() if k == key)
            paused = False
            seq[cur] = class_count(cur) + 1
            counts[cur] = 0
            last_saved[cur] = None
            hint = "不伸手的背景" if cur == "nothing" else f"{cur} 手势"
            print(f"[采集] 切到 {cur}（{hint}，文件夹现有 {seq[cur]-1} 张）")

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()

    # ---- 汇总 ----
    print("\n=== 本次采集汇总 ===")
    for ch in list(LETTERS) + ["nothing", "space", "del"]:
        d = os.path.join(args.out, ch)
        if os.path.isdir(d):
            print(f"  {ch}: {len([f for f in os.listdir(d) if f.lower().endswith('.jpg')])} 张")
    print(f"输出目录: {os.path.abspath(args.out)}")
    print("已退出")


if __name__ == "__main__":
    main()
