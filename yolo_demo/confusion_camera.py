# -*- coding: utf-8 -*-
"""
confusion_camera.py — 字母混淆统计工具（摄像头实测）

目的：
    统计"用户自己的手型下，模型容易把哪些字母互相认错"。
    因手语手势与其字母无形态关联，混淆只能实测，结果供大模型判断参考。

用法：
    python confusion_camera.py [--model <权重>] [--threshold 0.05] [--cam 0]

操作：
    a-z    切到意图字母（我"想测"的是这个，之后手势都归到它名下统计）
    Space  手动停止当前意图并清零重采（等同换下一类前重置计数）
    Esc    结束并输出统计（JSON 落盘 + 控制台可读摘要）

统计规则（详见 需求分析_混淆统计.md）：
    - 仅 26 字母 A-Z 参与混淆；top1 落 del/space/nothing 计为 "other"
    - 手未检测到 / 手在移动(moving) → 该帧跳过不统计
    - 低置信帧(top1 < 阈值) 单列为 low_conf，不进具体字母混淆
    - 混淆率 = 意图 X 的有效帧中 top1=Y 的帧数 / 意图 X 有效帧总数（行归一化）
"""

import argparse
import os
import sys
import time
import json
import cv2
from hand_recognizer import HandRecognizer

TOPK = 5               # 每帧记录前 k 个候选
LETTERS = "abcdefghijklmnopqrstuvwxyz"
LOW_CONF = 0.5         # top1 低于此视为低置信（可被 --conf 覆盖）


def parse_args():
    p = argparse.ArgumentParser(description="摄像头字母混淆统计工具")
    p.add_argument("--model", default="models/asl_demo-6-best.pt")
    p.add_argument("--hand", default="hand_landmarker.task")
    p.add_argument("--conf", type=float, default=LOW_CONF, help="低置信阈值")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="易混淆对判定阈值(混淆率) 默认0.05")
    p.add_argument("--cam", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    low_conf_th = args.conf

    recognizer = HandRecognizer(args.model, args.hand)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("无法打开摄像头，请检查连接或换 --cam")
        recognizer.close()
        return

    intent = ""            # 当前意图字母（按 a-z 切换），空=不统计
    # 统计容器: intent -> {total, low_conf, top1: {letter: n}, topk: {letter: n}}
    stats = {}
    # top5 跨意图整体累计（每帧记一次 top1 到意图，topk 只算字母共现于 top2..5）
    topk_acc = {}          # letter -> 出现在 top-k 的次数（用于 top5_dist 参考）

    print("摄像头已打开 | a-z 切意图 空格=重采 Esc=结束输出")
    print("提示：先按想测的字母（如 b），再摆 B 手势保持，看统计变化")
    t0 = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_h, frame_w = frame.shape[:2]

        r = recognizer.recognize(frame, topk=TOPK)

        # ---- 统计（仅手已检测到且不在移动）----
        if intent and r["detected"] and not r.get("moving"):
            if intent not in stats:
                stats[intent] = {"total": 0, "low_conf": 0,
                                 "top1": {}, "topk": {}}
            st = stats[intent]
            st["total"] += 1
            tk = r.get("topk") or []
            # low_conf：top1 概率过低，单列
            if r["conf"] < low_conf_th:
                st["low_conf"] += 1
            else:
                top1_letter = (r["letter"] or "").lower()
                # 只统计 26 字母；del/space/nothing 落 "other"
                top1_key = top1_letter if top1_letter in LETTERS else "other"
                st["top1"][top1_key] = st["top1"].get(top1_key, 0) + 1
            # top-k 分布累计（仅字母，用于分析"差点认成别的"）
            for item in tk:
                ch = (item["letter"] or "").lower()
                if ch in LETTERS:
                    st["topk"][ch] = st["topk"].get(ch, 0) + 1
                    topk_acc[ch] = topk_acc.get(ch, 0) + 1

        # ---- 显示 ----
        # 顶部条1：意图字母 + 累计
        cv2.rectangle(frame, (0, 0), (frame_w, 40), (40, 40, 40), -1)
        intent_text = f"intent: {intent.upper() if intent else '-'}   (a-z 切换)"
        cv2.putText(frame, intent_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # 顶部条2：当前意图的累计统计
        cv2.rectangle(frame, (0, 40), (frame_w, 62), (20, 20, 20), -1)
        if intent and intent in stats:
            st = stats[intent]
            t = st["total"]
            if t:
                correct = st["top1"].get(intent, 0)
                correct_rate = correct / t
                # 找出 top1 错分最多的两个
                wrong = {k: v / t for k, v in st["top1"].items() if k != intent}
                wrong_sorted = sorted(wrong.items(), key=lambda x: -x[1])[:2]
                wrong_str = " ".join(f"{k.upper()}:{v:.0%}" for k, v in wrong_sorted)
                stat_line = f"正确 {correct_rate:.0%}  ({correct}/{t})   误: {wrong_str or '-'}   低置信 {st['low_conf']}"
                cv2.putText(frame, stat_line, (10, 56), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "摆好手势后稍等，帧会累积到上方统计", (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

        # 第三行：当前帧 top-k
        cv2.rectangle(frame, (0, 64), (frame_w, 102), (30, 20, 20), -1)
        if r["detected"] and r.get("topk"):
            tk_str = "  ".join(
                f"{i+1}.{it['letter'].upper()} {it['prob']:.0%}"
                for i, it in enumerate(r["topk"])
            )
            cv2.putText(frame, tk_str, (10, 92), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No hand / moving", (10, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # 检测框
        if r["detected"]:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{r['letter']} {r['conf']:.0%}" if r["conf"] >= low_conf_th else f"? {r['conf']:.0%}"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, "Esc:end Space:reset", (frame_w - 180, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("ASL Confusion Stat", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc 退出（q 留给字母意图）
            break
        elif key == ord(" "):   # 空格 = 手动停止当前意图（重采清零）
            if intent:
                print(f"[重置] 意图 {intent.upper()} 已清零，可重新采或换字母")
                stats.pop(intent, None)
        elif ord("a") <= key <= ord("z"):
            ch = chr(key)
            intent = ch
            print(f"[意图] 切换到 {ch.upper()} —— 摆 {ch.upper()} 手势开始统计")

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()

    # ---- 输出统计 ----
    data = build_output(stats, topk_acc, args)
    out_path = save_output(data)
    print_summary(data)

    print(f"\nJSON 已存: {out_path}")
    print("已退出")


def build_output(stats, topk_acc, args):
    """汇总为 JSON 结构（供落盘 + 控制台）。"""
    confusion = {}
    total_all = 0
    for intent, st in stats.items():
        t = st["total"]
        if t <= 0:
            continue
        total_all += t
        correct = st["top1"].get(intent, 0)
        targets = {}
        for k, n in st["top1"].items():
            # 排除意图自身与 "other"(del/space/nothing)，只留字母间混淆
            if k != intent and k != "other":
                targets[k.upper()] = round(n / t, 4)
        confusion[intent.upper()] = {
            "total": t,
            "correct_rate": round(correct / t, 4),
            "low_conf_rate": round(st["low_conf"] / t, 4),
            "targets": dict(sorted(targets.items(), key=lambda x: -x[1])),
        }

    top5_dist = {}
    for ch, n in topk_acc.items():
        # 出现该字母于 top-k 的次数/总有效帧（粗略参考）
        top5_dist[ch.upper()] = n

    return {
        "meta": {
            "model": os.path.basename(args.model),
            "threshold": args.threshold,
            "low_conf_th": args.conf,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_frames": total_all,
        },
        "confusion": confusion,
        "top5_appear": top5_dist,
    }


def save_output(data):
    """存 data/confusion_<时间戳>.json，返回路径。"""
    os.makedirs("data", exist_ok=True)
    path = f"data/confusion_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return os.path.abspath(path)


def print_summary(data):
    """控制台人读摘要（可直接复制喂 LLM）。"""
    th = data["meta"]["threshold"]
    sep = "=" * 46
    print("\n" + sep)
    print(f"字母混淆统计（易混淆对, 阈值>{th:.0%}）")
    print(sep)
    cf = data["confusion"]
    if not cf:
        print("无有效统计（未按 a-z 设意图或未采到帧）")
        return
    unstable = []
    for intent in sorted(cf):
        d = cf[intent]
        t = d["total"]
        if t < 10:
            continue  # 样本太少不判定
        targets = {k: v for k, v in d["targets"].items() if v >= th}
        if not targets:
            print(f"{intent}  稳定 (正确 {d['correct_rate']:.0%}, 样本 {t})")
        else:
            pair_str = "  ".join(f"{k}({v:.0%})" for k, v in targets.items())
            print(f"{intent} → {pair_str}   [样本 {t}]")
            unstable.append(intent)
    # 低置信提示
    low_conf_list = [
        f"{i}({cf[i]['low_conf_rate']:.0%})"
        for i in sorted(cf)
        if cf[i]["total"] >= 10 and cf[i]["low_conf_rate"] >= 0.1
    ]
    if low_conf_list:
        print(sep)
        print("低置信帧较多（模型拿不准，可能字母本身不稳）: " + " ".join(low_conf_list))
    if unstable:
        print(sep)
        print("易混淆字母（建议注入大模型提示词）: " + " ".join(unstable))
    print(f"总有效帧: {data['meta']['total_frames']}")


if __name__ == "__main__":
    main()
