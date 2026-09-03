# -*- coding: utf-8 -*-
"""
tool_seg.py — 工具层：拼音切分器【E5】

功能：
    pinyin_cut(letters)：把无空格的字母串切成合法的拼音音节序列，
    找出所有可能的切分方式，并用动态规划选最优路径。

例：  "nihao"
    全切分候选:
      ni|hao        (合法, 常用)
      nian|hao      (合法, 但 nian 吃掉更多字母)
      n|i|h|a|o     (单字母，大多不是合法拼音)
      nihao         (不合法, 无此音节)
    → 动态规划取"覆盖完整 + 路径代价最小"的最优切分

切分合法性依据：
    E3 数据的拼音集合（398 个标准音节，含 lv/nv 等 v 代替 ü 的写法）

设计（对应 E2 文档工具 1）：
    纯函数；返回 Top-N 条切分路径，每条带 score（本版用"路径长度惩罚+
    常用度加权"，真正的 n-gram 打分在 E6 tool_lm 做，这里先给几何/词频权重）。
"""

from . import loader
from . import tool_dict

# 合法拼音音节集合（仅供切分判断：只用单音节表 char_pinyin 的 key。
# 注意不能混入 pinyin_word 的整词拼音串 key，否则 "nihao" 会被当单个音节）
_PINYIN_SET = set(loader.char_pinyin().keys())


def _is_valid_syl(s):
    """判断 s 是否是合法拼音音节。"""
    if not s:
        return False
    # 合法音节集里直接有（如 "ni"、"hao"）
    if s in _PINYIN_SET:
        return True
    # ü 的 v 写法已在数据中（lv/nv 等）
    return False


def _full_cut(letters, max_syl_len=6):
    """全切分：枚举所有"每一段都是合法拼音"的分割方式。

    返回所有分割的 [(syllable1, syllable2, ...), ...]。
    max_syl_len: 最长的合法拼音一般不超过 6 个字母(如 zhuang)。
    """
    n = len(letters)
    results = []

    def dfs(start, path):
        if start >= n:
            results.append(tuple(path))
            return
        # 尝试从 start 取 1..max_syl_len 个字母作为一段
        for length in range(1, min(max_syl_len, n - start) + 1):
            syl = letters[start:start + length]
            if _is_valid_syl(syl):
                dfs(start + length, path + [syl])

    dfs(0, [])
    return results


def _path_score(path):
    """给一条切分路径打分。

    简单加权策略（先不引入 n-gram）：
      1) 段数越少越好（段数多说明把词拆碎了，通常更不合理）
      2) 每段若是"整词"则加分（说明该音节能独立成词，如 ni→你）
      3) 段长越符合常见词越好——这里简单用"长度接近 2~3"轻微加分

    本版用于初筛排序，精细语言模型打分在 E6。
    """
    score = 0.0
    for syl in path:
        # 段数惩罚合并进来：每段基础给 1 分，段越多总分越低会偏差，
        # 改用"平均"- 简单起见：段少则总分离
        score -= 1.0  # 每出现一段，扣 1（越少越好）
        # 若该音节能直接成词（有候选），给予小奖励
        if tool_dict.expand(syl)["candidates"]:
            score += 0.3
        elif tool_dict.expand_words(syl)["words"]:
            score += 0.2
    return score


def pinyin_cut(letters, top_k=5):
    """把字母串切成拼音音节，返回 Top-N 切分。

    参数:
        letters: str 连续字母（手语累积得到），如 "nihao"
        top_k:   返回前几条路径
    返回:
        {"letters": ..., "cuts": [{"syllables": [...], "score": float}], "count": int}
        cuts 按 score 降序。
    """
    letters = letters.strip().lower()
    if not letters:
        return {"letters": letters, "cuts": [], "count": 0}

    all_cuts = _full_cut(letters)
    if not all_cuts:
        return {"letters": letters, "cuts": [], "count": 0}

    scored = sorted(
        ({"syllables": list(p), "score": round(_path_score(p), 4)} for p in all_cuts),
        key=lambda c: -c["score"],
    )
    return {
        "letters": letters,
        "cuts": scored[:top_k],
        "count": len(all_cuts),
    }
