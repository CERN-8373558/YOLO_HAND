# -*- coding: utf-8 -*-
"""
tool_cand.py — 工具层：候选输出【E7】

功能：
    build(letters, top_k)：组合 E4 词典 + E5 切分 + E6 语言模型，
    把字母串转成 Top-N 完整汉字候选（统一 JSON 结构），供 Agent/界面消费。

输入：   手语累积的字母串，如 "nihao"
输出：   候选列表，每个候选 = 完整中文词/句 + 切分 + 得分

    {
      "letters": "nihao",
      "candidates": [
        {"text": "你好", "score": 0.635, "seg": "ni|hao", "source": "local"},
        {"text": "泥号", "score": 0.540, "seg": "ni|hao", "source": "local"},
        ...
      ],
      "best": "你好",
      "ambiguous": false
    }

生成流程：
  切分(E5) → 每条切分逐音节展开(E4) → 用语言模型(E6)给组合打分 → 排序取 Top-K

说明：
  本版采用"贪心展开"（每个音节只取词频最高的 1~2 个候选字），
  避免全组合爆炸（ni 有 21 候选 × hao 有 18 候选 = 378 组合×切分）。
  top1 高频字生成主候选，再适度生成对比候选。
"""

from . import tool_seg
from . import tool_dict
from . import tool_lm

# 每音节最多取的高频候选数（控制组合规模）
_MAX_CHARS_PER_SYL = 2
# 是否让语言模型对"首候选 vs 次候选"做对比——本版直接按每音节 top 组合


def _expend_syllables(syllables):
    """对一条切分的各音节，取每音节 top-N 高频候选字，组合出候选词序列。

    返回: [ [字序列,...], ...]  如 [['你','好'], ['你','号'], ...]
    """
    per_syl = []
    for syl in syllables:
        r = tool_dict.expand(syl)
        cands = r["candidates"][:_MAX_CHARS_PER_SYL]
        if not cands:
            # 音节无同音字记录，跳过该音节（此切分基本无效）
            per_syl.append([])
        else:
            per_syl.append([c["char"] for c in cands])

    # 笛卡尔组合（各音节候选组合）
    from itertools import product
    combos = []
    for combo in product(*per_syl):
        combos.append(list(combo))
    return combos


def _whole_word_candidates(letters):
    """整串直接命中词典里的词（整词匹配）。

    手语打出完整词拼音（如 xiexie）时，优先查"拼音串→词"表，
    直接得到"谢谢"这类整词候选，避免走单字拼凑（单字频率对成词语境失真）。
    返回 [{text, score(基于全局词频), seg}], 可能为空。
    """
    r = tool_dict.expand_words(letters)  # {拼音串: [{char(词),freq},...]}
    words = r.get("words", [])
    cands = []
    for item in words[:5]:  # 最多取前 5 个高频整词
        w = item["char"]
        score = tool_lm.score_sequence([w])
        cands.append({"text": w, "score": round(score, 4),
                      "seg": letters, "source": "local"})
    return cands


def build(letters, top_k=5):
    """把字母串转成 Top-N 汉字候选。"""
    letters = letters.strip().lower()
    if not letters:
        return {"letters": letters, "candidates": [], "best": None, "ambiguous": False}

    # 1. 优先整词匹配（如 xiexie→谢谢 整词直接命中）
    candidates = _whole_word_candidates(letters)

    # 2. 若整词命中不足，再走切分+单字/多字拼凑
    if len(candidates) < top_k:
        seg_result = tool_seg.pinyin_cut(letters, top_k=3)
        cuts = seg_result["cuts"]
        if cuts:
            scored_cands = []
            for cut in cuts:
                syls = cut["syllables"]
                for combo in _expend_syllables(syls):
                    if not combo:
                        continue
                    text = "".join(combo)
                    if any(c["text"] == text for c in candidates):
                        continue  # 整词已命中，跳过重复
                    score = tool_lm.score_sequence(combo)
                    scored_cands.append({
                        "text": text, "score": round(score, 4),
                        "seg": "|".join(syls), "source": "local",
                    })
            candidates.extend(scored_cands)

    if not candidates:
        return {"letters": letters, "candidates": [], "best": None,
                "ambiguous": False, "note": "no_valid_pinyin_cut"}

    # 3. 去重 + 排序取 Top-K
    seen = set()
    uniq = []
    for c in candidates:
        if c["text"] not in seen:
            seen.add(c["text"])
            uniq.append(c)
    uniq.sort(key=lambda c: -c["score"])
    candidates = uniq[:top_k]

    # 4. 判定是否有歧义（前两名接近）
    ambiguous = False
    if len(candidates) >= 2:
        ambiguous = (candidates[0]["score"] - candidates[1]["score"]) < 0.02

    return {
        "letters": letters,
        "candidates": candidates,
        "best": candidates[0]["text"] if candidates else None,
        "ambiguous": ambiguous,
    }
