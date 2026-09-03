# -*- coding: utf-8 -*-
"""
build_pinyin_data.py — E3 构建拼音词典与词频语料

目标：
    生成两个静态数据文件（数据与逻辑分离，后续工具层直接加载）：
      1) 拼音 → 汉字 候选表（同音字，按词频排序）
      2) 常用词 → 拼音 词表（用于整词匹配 / 候选生成）

数据来源：
    - jieba 内置词典（34.9 万词条，含词频）：提供常用词与词频
    - pypinyin：把汉字词转成拼音（自动按词取音，多音字处理准）

产出：
    pinyin_/data/
      ├── char_pinyin.json   # 拼音 → 汉字候选[{char,freq}]（按词频降序）
      ├── word_pinyin.json   # 词 → 拼音   （整词匹配用）
      └── pinyin_word.json   # 拼音 → 词候选[{word,freq}]（按词频降序）

说明：
    E3 只产出数据文件，不写任何拼音切分/打分逻辑（那是 E4~E7）。
    运行一次即可，产物提交进项目，运行时无需再依赖 jieba/pypinyin。
"""

import json
import os
from collections import defaultdict

import jieba
from pypinyin import lazy_pinyin, Style

# 输出目录
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)

# jieba 默认词典路径（含 词,词频,词性）
JIEBA_DICT = os.path.join(os.path.dirname(jieba.__file__), "dict.txt")

# 保留全部中文词（jieba 词频是分词权重，不是真实语言频率，
# 无法用词频阈值区分常用/生僻；全量保留靠 tool_lm 的整词加成取胜）
# 数据文件会更大（pinyin_word.json ~30MB），换取覆盖率。
# 说明：注释掉的 MIN_FREQ 方案因 jieba 词频失真（如"谢谢"仅1089）不可用。
# MIN_FREQ = 200


def load_jieba_words():
    """读 jieba 词典：返回 [(word, freq, pos), ...]，按词频排序（自带）"""
    words = []
    with open(JIEBA_DICT, encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) >= 2:
                word, freq = parts[0], int(parts[1])
                words.append((word, freq))
    return words


def main():
    print("读取 jieba 词典...")
    words = load_jieba_words()
    print(f"总词条: {len(words)}")

    # 过滤：只保留中文词（去掉含字母数字的英文/符号词）
    # 全量保留，不做数量/词频截断（jieba 词频失真，详见文件头注释）
    cn_words = []
    for w, f in words:
        if all("\u4e00" <= ch <= "\u9fff" for ch in w):
            cn_words.append((w, f))
    print(f"中文词: {len(cn_words)} 个")

    # ---- 统计结构 ----
    # pinyin -> {char: freq}      （取单字）
    # pinyin -> {word: freq}      （取整词）
    # word   -> pinyin
    char_pinyin = defaultdict(lambda: defaultdict(int))   # 拼音 -> 字 -> 频
    pinyin_word = defaultdict(lambda: defaultdict(int))   # 拼音串 -> 词 -> 频
    word_pinyin = {}                                      # 词 -> 拼音串

    # ---- 逐词处理 ----
    for w, f in cn_words:
        py_list = lazy_pinyin(w)  # ['ni','hao']
        py_str = "".join(py_list)  # 'nihao'

        word_pinyin[w] = py_str

        if len(w) == 1:
            # 单字：计入 char_pinyin（每个拼音下）
            char_pinyin[py_str][w] += f
        # 整词计入 pinyin_word（拼音串 -> 词）
        pinyin_word[py_str][w] += f

        # 对多字词里的每个字也建索引（用于"逐字拼音"场景）
        # 注意多音字以词为单位取音，这里单字索引可能有偏差，暂不强求

    print(f"不同拼音: {len(char_pinyin)} | 词条映射: {len(word_pinyin)}")

    # ---- 组装成输出格式，按词频降序 ----
    def to_sorted_list(d):
        """把 {k: {inner: freq}} 转成排序列表结构"""
        result = {}
        for key, inner in d.items():
            items = sorted(inner.items(), key=lambda kv: -kv[1])
            total = sum(inner.values()) or 1
            result[key] = [{"char": k, "freq": round(v / total, 6)} for k, v in items]
        return result

    char_data = to_sorted_list(char_pinyin)
    pword_data = to_sorted_list(pinyin_word)

    # ---- 写文件 ----
    with open(os.path.join(OUT_DIR, "char_pinyin.json"), "w", encoding="utf-8") as fp:
        json.dump(char_data, fp, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT_DIR, "pinyin_word.json"), "w", encoding="utf-8") as fp:
        json.dump(pword_data, fp, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT_DIR, "word_pinyin.json"), "w", encoding="utf-8") as fp:
        json.dump(word_pinyin, fp, ensure_ascii=False, indent=1)

    print("数据文件已生成:")
    for name in ("char_pinyin.json", "pinyin_word.json", "word_pinyin.json"):
        path = os.path.join(OUT_DIR, name)
        print(f"  {name}: {os.path.getsize(path)/1024:.0f} KB")

    # ---- 抽样验证 ----
    print("\n验证样例:")
    for py in ("nihao", "zhongguo", "xiexie"):
        if py in char_data:
            print(f"  拼音 '{py}':", [c["char"] for c in char_data[py][:5]])
        if py in pword_data:
            print(f"  拼音串 '{py}':", [w["char"] for w in pword_data[py][:5]])


if __name__ == "__main__":
    main()
