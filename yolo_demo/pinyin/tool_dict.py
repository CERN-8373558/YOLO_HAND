# -*- coding: utf-8 -*-
"""
tool_dict.py — 工具层：词典（拼音 → 汉字候选映射）【E4】

功能：
    expand(syllable)：输入一个拼音音节（如 "ni"），返回它的同音汉字候选列表，
    按词频降序排列，如：
        {"syllable": "ni"}
        → [{"char": "你", "freq": 0.97}, {"char": "泥", "freq": 0.012}, ...]

    是 Agent 补全时的"字表"来源，数据来自 E3 生成的 char_pinyin.json。

接口设计（对应 E2 架构文档工具 2）：
    纯函数、JSON 兼容结构、无副作用 → 便于 Agent 调用与单元测试。
"""

from . import loader


def expand(syllable):
    """返回拼音音节的同音汉字候选（按词频降序）。

    参数:
        syllable: str，如 "ni"、"hao"
    返回:
        {"syllable": ..., "candidates": [{"char","freq"}, ...]}，
        空/无记录时 candidates 为空列表。
    """
    syllable = syllable.strip().lower()
    table = loader.char_pinyin()  # {拼音: [{char,freq},...]}
    candidates = table.get(syllable, [])

    return {
        "syllable": syllable,
        "candidates": candidates,
    }


def expand_words(syllable):
    """整音节直接是词的候选（用于"一个字就是一个词"的场景）。

    参数:
        syllable: str，如 "wo"
    返回:
        {"syllable": ..., "words": [{"char"(词),"freq"}, ...]}
    """
    syllable = syllable.strip().lower()
    word_table = loader.pinyin_word()  # {拼音串: [{char,freq},...]}
    words = word_table.get(syllable, [])

    return {
        "syllable": syllable,
        "words": words,
    }


def lookup_word(word):
    """词 → 拼音串（反向查询，整词匹配用）。

    参数:
        word: str，如 "你好"
    返回:
        str 拼音串，如 "nihao"；查不到返回 None。
    """
    return loader.word_pinyin().get(word)
