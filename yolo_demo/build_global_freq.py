# -*- coding: utf-8 -*-
"""
build_global_freq.py — E6 补充数据：全局字/词频率表

背景：
    E3 的 char_pinyin.json 中 freq 是"音节内归一化"（ni 里你占 97%），
    不能跨音节比较。E6 语言模型需要"全局词频"（好 vs 号 在整个中文里谁常见）。
    本脚本从 jieba dict.txt 提取单字/双字词的全局词频，固化成 JSON。

产出：
    pinyin/data/global_freq.json
      {
        "char_freq":  {字: 全局词频},            # 单字（1.1万+）
        "word_freq":  {词: 全局词频}             # 全部词（20万）
      }
"""

import json
import os
import jieba

OUT_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "global_freq.json"))
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

DICT = os.path.join(os.path.dirname(jieba.__file__), "dict.txt")


def main():
    char_freq = {}
    word_freq = {}
    with open(DICT, encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            w, f = parts[0], int(parts[1])
            if not all("\u4e00" <= ch <= "\u9fff" for ch in w):
                continue  # 只保留纯中文
            word_freq[w] = f
            if len(w) == 1:
                char_freq[w] = f

    with open(OUT_PATH, "w", encoding="utf-8") as fp:
        json.dump({"char_freq": char_freq, "word_freq": word_freq}, fp, ensure_ascii=False)
    print(f"已生成: {OUT_PATH}")
    print(f"单字数: {len(char_freq)} | 总词数: {len(word_freq)} | 大小: {os.path.getsize(OUT_PATH)/1024:.0f} KB")
    for ch in ("你", "好", "号", "泥"):
        print(f"  字频 {ch}: {char_freq.get(ch, 0)}")


if __name__ == "__main__":
    main()
