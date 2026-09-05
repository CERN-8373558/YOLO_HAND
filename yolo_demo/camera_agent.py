# -*- coding: utf-8 -*-
"""
camera_agent.py — 摄像头手语 AI 补全打字机【E10】

与 camera_text.py 的区别（AI 补全模式）：
    camera_text.py:  每个字母手势稳定后"直接上屏"（逐字母打字）
    camera_agent.py: 每个词先累积字母到 pending，收手停顿后
                     由 Agent(本地+n-gram+可选大模型) 生成中文候选，
                     用户按数字键选候选上屏 → 类输入法体验

交互：
    1. 手语比字母 → 顶部显示当前拼音串（如 nihao）
    2. 收手停顿约 0.5s → 触发 Agent → 显示候选 [1.你好 2.你好呀...]
    3. 按数字键选候选 → 上屏到正文
    4. q 退出 / c 清空 / 空格=候选1 / Enter=确定当前候选

用法：
    python camera_agent.py [--model <权重>] [--conf] [--cam]
    --use-llm  初始开启大模型增强（运行中按 m 可切换）
    --demo     无摄像头演示（用预置输入序列模拟，验证逻辑）

按键：q退出 | c清空 | 1-9选候选 | m切LLM开关 | S=单词补全(发pending字母) | T=整句润色
"""

import argparse
import time
import cv2
from hand_recognizer import HandRecognizer

from pinyin.agent import Agent
from pinyin import tool_dict

# 收手后等待多少秒再触发补全
TRIGGER_DELAY = 0.6
# 连续多少帧确认一个字母
STABLE_FRAMES = 8


def parse_args():
    p = argparse.ArgumentParser(description="手语 AI 补全打字机")
    p.add_argument("--model", default="models/asl_demo-6-best.pt")
    p.add_argument("--hand", default="hand_landmarker.task")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--stable", type=int, default=STABLE_FRAMES)
    p.add_argument("--use-llm", action="store_true", help="初始开启大模型增强")
    p.add_argument("--demo", action="store_true", help="无摄像头演示")
    return p.parse_args()


class TextBuffer:
    """正文（已上屏中文）+ 候选选择后的结果。"""

    def __init__(self):
        self.chars = []

    def append(self, s):
        self.chars.extend(list(s))

    def backspace(self, n=1):
        for _ in range(n):
            if self.chars:
                self.chars.pop()

    def text(self):
        return "".join(self.chars)


def demo_main(args):
    """无摄像头演示：模拟打词序列，验证补全逻辑。"""
    from pinyin.agent import Agent
    a = Agent()
    llm_hook = None
    if args.use_llm:
        from pinyin import llm_client as llm
        llm_hook = llm.complete

    print("=== DEMO：模拟手语打词，观察补全 ===")
    for letters in ["nihao", "wo", "xihuan", "xuexi"]:
        print(f"\n手语字母串: {letters}")
        r = a.decide(letters, llm_hook=llm_hook)
        print(f"Agent: {r['action']} | best={r['best']}")
        for i, c in enumerate(r["candidates"][:5], 1):
            print(f"  {i}. {c['text']}")
        print("→ 选 1 上屏:", r["best"])


def camera_main(args):
    recognizer = HandRecognizer(args.model, args.hand)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("无法打开摄像头")
        recognizer.close()
        return

    a = Agent()
    # 运行时开关：初始由命令行决定，按 m 切换
    use_llm = args.use_llm
    buffer = TextBuffer()
    pending = []            # 当前拼音串（累积中）
    current_cls = None
    stable_count = 0
    stable_energy = 0.0     # 同字母累积置信度能量（>=STABLE 且帧数够 → 确认）
    confirmed_letter = None # 已确认的字母：同字母持续期间不再重复确认，换字母/放手才解锁
    last_hand_time = time.time()   # 上次检测到手的时间
    candidates = []         # 本地候选列表（第1行）
    llm_cands = []          # 大模型候选列表（第2行，自动追加）
    pending_text = ""
    last_pending = ""       # 上一次触发补全的拼音（防重复触发）
    # 记录最后一次上屏的词及其长度（供 LLM 替换用）
    last_committed = ""     # 最后一次上屏的词
    cur_letters = ""        # 最近一次触发补全的字母串（供原文回显）
    hist_log = []           # 最近上屏记录 [{letters, text}]，最多3条（原文回显栏用）
    # 整句润色状态（s 键触发）
    polished = [""]          # 润色结果
    polish_done = [False]    # 是否完成
    # LLM 请求状态容器（线程写入，主循环读取）
    llm_status_holder = {"val": ""}
    llm_gen = {"n": 0}          # LLM 请求代数：忽略/新请求时 +1，使旧结果作废
    t0 = cv2.getTickCount() / cv2.getTickFrequency()

    print("摄像头已打开 | q退出 c清空 1-9选 m切LLM | S=单词补全 T=整句润色")

    # 大模型结果用线程异步获取，避免卡主循环
    import threading

    def request_llm(letters, gen):
        """后台请求大模型，完成后写入 llm_cands 和状态。

        gen: 发起时的请求代数。完成后若与当前代数不一致
             （用户按 Esc 忽略 / 又发了新请求），则丢弃结果。
        """
        from pinyin import llm_client as llm
        llm_status_holder["val"] = "busy"
        res = llm.complete(letters)
        if llm_gen["n"] != gen:
            return  # 已过期：用户忽略或换了新请求，结果作废
        if res:
            llm_cands[:] = res  # 线程安全：主循环只在显示时读取
            llm_status_holder["val"] = "done"
            print(f"[LLM] 收到结果: {[c['text'] for c in res[:3]]}")
        else:
            llm_status_holder["val"] = "empty"
            print(f"[LLM] '{letters}' 无法推断出结果（试试更长/更明确的字母串）")

    def request_polish(sentence):
        """后台整句润色，完成后写入 polished 变量。"""
        from pinyin import llm_client as llm
        result = llm.polish_sentence(sentence)
        polished[0] = result if result else sentence
        polish_done[0] = True

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_w = frame.shape[1]
        frame_h = frame.shape[0]

        # 润色完成检测：结果就绪则替换 buffer 并提示
        if polish_done[0]:
            new_text = polished[0]
            buffer = TextBuffer()
            buffer.append(new_text)
            hist_log.clear()   # 整句被重写，逐词原字母映射不再对应
            print(f"[整句] 润色完成，已替换: {new_text}")
            polished[0] = ""
            polish_done[0] = False

        r = recognizer.recognize(frame)

        # ---- 识别 + 拼音累积 ----
        if r["detected"]:
            last_hand_time = time.time()
            # 手在移动时（过渡手势）跳过累积，防止误判乱字母
            if r.get("moving"):
                current_cls = None
                stable_count = 0
                stable_energy = 0.0
                confirmed_letter = None
            elif r["conf"] >= args.conf:
                letter = r["letter"].lower()
                if letter in "abcdefghijklmnopqrstuvwxyz":
                    if letter != current_cls:
                        # 换字母 → 重新开始能量累积
                        current_cls = letter
                        stable_count = 1
                        stable_energy = r["conf"]
                        # 换字母会解锁：该字母可能后续需要再确认
                        if confirmed_letter != letter:
                            confirmed_letter = None
                    else:
                        stable_count += 1
                        stable_energy += r["conf"]
                        # 已确认过且未换字母 → 不重复确认（锁）
                        if confirmed_letter != letter:
                            energy_need = args.stable * 0.9
                            if stable_count >= args.stable and stable_energy >= energy_need:
                                # 字母确认 → 若在补全状态先取消候选
                                candidates = []
                                if letter not in "".join(pending[-3:]):  # 防连打重复
                                    pending.append(letter)
                                pending_text = "".join(pending)
                                confirmed_letter = letter   # 加锁：同字母不再重复
                                stable_energy = 0.0
        else:
            # 手放下：重置字母级状态（解锁，允许下次重新确认同字母）
            current_cls = None
            stable_count = 0
            stable_energy = 0.0
            confirmed_letter = None

        # 进度条填充比例 0~1：同字母稳定帧数/目标帧数 × 平均置信度加权
        progress = 0.0
        if current_cls and stable_count > 0:
            avg_conf = stable_energy / stable_count
            frame_ratio = stable_count / args.stable
            progress = min(1.0, max(0.0, (frame_ratio * 0.6 + avg_conf * 0.4) * frame_ratio))

        # ---- 收手停顿触发补全 ----
        if pending and not r["detected"]:
            elapsed = time.time() - last_hand_time
            if elapsed >= TRIGGER_DELAY and "".join(pending) != last_pending:
                letters = "".join(pending)
                last_pending = letters
                llm_cands.clear()

                # 1. 本地决策（不带 llm_hook，本地行先确定）
                dec = a.decide(letters, llm_hook=None)
                candidates = dec.get("candidates", [])

                if dec["action"] == "output":
                    # 高置信直接上屏本地结果
                    if dec["best"]:
                        buffer.append(dec["best"])
                        last_committed = dec["best"]
                        hist_log.append({"letters": letters, "text": dec["best"]})
                        print(f"[本地] 自动上屏: {dec['best']}  (原字母: {letters})")
                    pending, candidates = [], []
                    pending_text = ""
                elif dec["action"] == "fail":
                    print(f"无法识别 '{letters}'（可能是英文/未打完），c=清空后重打")
                    candidates = []
                # confirm → 等用户从本地候选里选

                # 2. 本地确定后（无论 output/confirm/fail），若开启 LLM → 异步请求第2行
                if use_llm:
                    print(f"[触发] 本地完成({dec['action']})，请求 LLM 增强: {letters}")
                    llm_gen["n"] += 1
                    threading.Thread(target=request_llm,
                                     args=(letters, llm_gen["n"]), daemon=True).start()

        # ---- 显示 ----
        # 顶部条1：正文 + 闪烁光标（每 0.5s 交替显示/隐藏末尾的 |，同 camera_text.py）
        cv2.rectangle(frame, (0, 0), (frame_w, 40), (40, 40, 40), -1)
        body = buffer.text() + "".join(pending)
        elapsed_s = cv2.getTickCount() / cv2.getTickFrequency() - t0
        cursor_visible = int(elapsed_s * 2) % 2 == 0
        cv2.putText(frame, body + ("|" if cursor_visible else " "), (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        # 顶部条2：原文回显（最近上屏字母串→中文，防中文上屏后丢失原字母）
        cv2.rectangle(frame, (0, 40), (frame_w, 62), (20, 20, 20), -1)
        hist_str = "  |  ".join(f"{h['letters']} → {h['text']}" for h in hist_log[-3:][::-1])
        if not hist_str:
            hist_str = "（打完字母后这里显示 nihao → 你好，方便核对原字母）"
        cv2.putText(frame, "原: " + hist_str, (10, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)
        # 第三行：本地候选（深蓝底）
        if candidates:
            cv2.rectangle(frame, (0, 64), (frame_w, 100), (20, 20, 60), -1)
            cand_str = "  ".join(f"{i+1}.{c['text']}" for i, c in enumerate(candidates[:5]))
            cv2.putText(frame, cand_str, (10, 92), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
        # 第四行：大模型候选（LLM 行，绿字/黄色标）
        llm_stat = llm_status_holder["val"]
        if llm_cands:
            cv2.rectangle(frame, (0, 102), (frame_w, 138), (20, 60, 20), -1)
            llm_str = "  ".join(f"{i+1}.{c['text']}" for i, c in enumerate(llm_cands[:4]))
            cv2.putText(frame, "LLM " + llm_str, (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
        elif llm_stat == "busy":
            cv2.rectangle(frame, (0, 102), (frame_w, 138), (40, 40, 40), -1)
            cv2.putText(frame, "LLM 请求中...", (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)
        elif llm_stat == "empty":
            cv2.rectangle(frame, (0, 102), (frame_w, 138), (40, 20, 20), -1)
            cv2.putText(frame, "LLM 无结果（字母串太模糊），按 s 重发或继续拼", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, "q:quit c:clear 1-9:pick S:word T:sent", (frame_w - 300, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        # 左上角显示大模型开关状态
        llm_color = (0, 255, 0) if use_llm else (150, 150, 150)
        llm_text = f"LLM: {'ON' if use_llm else 'OFF'} (m)"
        cv2.putText(frame, llm_text, (10, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, llm_color, 2, cv2.LINE_AA)
        if r["detected"]:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 在框上方画识别出的字母和置信度
            label = f"{r['letter']} {r['conf']:.0%}"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            # 框下方进度条：显示字母确认程度，满格(黄)即成功累积
            bar_w = max(60, x2 - x1)
            bar_h = 8
            bx1, by1 = x1, min(frame_h - bar_h - 4, y2 + 6)
            bx2, by2 = x1 + bar_w, by1 + bar_h
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (80, 80, 80), -1)
            fill_w = int(bar_w * min(1.0, progress))
            if fill_w > 0:
                col = (0, 200, 255) if progress >= 1.0 else (0, 255, 0)
                cv2.rectangle(frame, (bx1, by1), (bx1 + fill_w, by2), col, -1)

        cv2.imshow("ASL Agent Input", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):  # 运行时切换大模型
            use_llm = not use_llm
            print(f"[大模型增强] {'开' if use_llm else '关'}")
        elif key == ord("c"):
            buffer = TextBuffer()
            pending, candidates, llm_cands = [], [], []
            pending_text = ""
            polished[0] = ""
            polish_done[0] = False
            hist_log.clear()
            print("已清空")
        elif key == 27:  # Esc = 忽略本轮 LLM 候选：作废在途结果，保留本地已上屏词
            llm_gen["n"] += 1          # 使迟到的旧请求结果作废
            llm_cands.clear()
            llm_status_holder["val"] = ""
            last_committed = ""        # 结束本词替换窗口，避免误删本地词
            print("[Esc] 已忽略 LLM 候选，保留当前已上屏文字")
        elif key == ord("s"):  # S = 单次补全：把当前 pending 字母串发大模型推断成词
            if pending:
                letters = "".join(pending)
                print(f"[发送] 把字母串发给大模型推断: {letters}")
                llm_cands.clear()
                llm_status_holder["val"] = ""
                # S 手动发送 = 新词（追加），不是替换——清空 last_committed 标记
                last_committed = ""
                cur_letters = letters   # 记录原文，数字键选 LLM 时回显用
                pending, candidates = [], []
                pending_text = ""
                threading.Thread(target=request_llm,
                                 args=(letters, llm_gen["n"] + 1), daemon=True).start()
                llm_gen["n"] += 1
            else:
                print("S 发送单次补全：当前没有待发送的字母串（先比字母拼词）")
        elif key == ord("t"):  # T = 整句润色：把已上屏整句发大模型润色成通顺句子
            if buffer.text():
                text = buffer.text()
                polished[0] = ""
                polish_done[0] = False
                print(f"[整句] 发送大模型润色: {text}")
                threading.Thread(target=request_polish, args=(text,), daemon=True).start()
            else:
                print("T 整句润色：当前没有已上屏的内容")
        elif 49 <= key <= 57:  # 1-9：优先选 LLM 候选，无则选本地候选
            idx = key - 49
            chosen = None
            letters_used = last_pending or cur_letters
            if llm_cands and idx < len(llm_cands):
                # 选 LLM 行候选：若上一词未上屏则是追加新词，否则替换
                chosen = llm_cands[idx]["text"]
                if last_committed:
                    for _ in range(len(last_committed)):
                        buffer.backspace()
                    if hist_log:
                        hist_log[-1]["text"] = chosen
                else:
                    hist_log.append({"letters": letters_used, "text": chosen})
                buffer.append(chosen)
                last_committed = chosen
                print(f"[LLM] 选中: {chosen}")
            elif candidates and idx < len(candidates):
                # 选本地候选
                chosen = candidates[idx]["text"]
                buffer.append(chosen)
                last_committed = chosen
                hist_log.append({"letters": last_pending or cur_letters, "text": chosen})
                print(f"[本地] 选中: {chosen}")
                if use_llm and last_pending:
                    llm_gen["n"] += 1
                    threading.Thread(target=request_llm,
                                     args=(last_pending, llm_gen["n"]), daemon=True).start()
            if chosen:
                pending, candidates, llm_cands = [], [], []
                pending_text = ""
            llm_status_holder["val"] = ""
        elif key == ord("0"):  # 0 = 采纳 LLM 第一候选（替换刚上屏的词）
            if llm_cands:
                choice = llm_cands[0]["text"]
                letters_used = last_pending or cur_letters
                if last_committed:
                    for _ in range(len(last_committed)):
                        buffer.backspace()
                    if hist_log:
                        hist_log[-1]["text"] = choice
                else:
                    hist_log.append({"letters": letters_used, "text": choice})
                buffer.append(choice)
                last_committed = choice
                print(f"[LLM] 采纳替换: {choice}")
            llm_cands.clear()

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()
    print(f"\n最终文字: {buffer.text()}")
    print("已退出")


def main():
    global USE_LLM_FLAG
    args = parse_args()
    USE_LLM_FLAG = args.use_llm
    if args.demo:
        demo_main(args)
    else:
        camera_main(args)


if __name__ == "__main__":
    main()
