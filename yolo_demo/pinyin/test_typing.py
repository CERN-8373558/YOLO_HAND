# -*- coding: utf-8 -*-
"""
test_typing.py — 模拟手语打字全流程测试（E8 效果验证）

模拟用户在摄像头前依次打词（每个词 = 一串字母 + 停顿）。
真实场景里 camera_text 检测到"收手停顿"会触发 Agent。
这里直接喂"词级字母串"，观察 Agent 决策和候选。

用法: python test_typing.py [--use-llm]
"""
import sys
sys.path.insert(0, r'C:\Users\24405\Pictures\Yolo_test\yolo_demo')

from pinyin.agent import Agent

USE_LLM = "--use-llm" in sys.argv


def main():
    a = Agent()

    # 模拟用户依次打出这些词（每个词停一下，收手后再打下一个）
    # 每个元素 = (手语字母串, 描述)
    words = [
        ("nihao", "你好"),
        ("wo", "我"),
        ("xihuan", "喜欢"),
        ("xuexi", "学习"),
        ("hello", "打个英文试试"),
    ]

    for letters, desc in words:
        print(f"\n=== 用户打: {letters!r}  (期望: {desc}) ===")
        # 词停顿触发一次 Agent 决策
        llm_hook = None
        if USE_LLM:
            from pinyin import llm_client as llm
            llm_hook = llm.complete
        r = a.decide(letters, llm_hook=llm_hook)

        print(f"  决策: {r['action']}")
        if r["candidates"]:
            top = [c["text"] for c in r["candidates"][:3]]
            print(f"  候选: {top}")
        if r.get("best"):
            mark = "✓" if r["best"] == desc else "?"
            print(f"  best: {r['best']} {mark}  (期望{desc})")
        print(f"  说明: {r['message']}")


if __name__ == "__main__":
    main()
