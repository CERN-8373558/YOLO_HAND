# -*- coding: utf-8 -*-
"""
loader.py — 数据加载器（pinyin 包共用）

职责：
    负责把 E3 生成的 JSON 数据文件加载进内存，并做一次加载缓存。
    其他工具（tool_dict / tool_seg / tool_lm）都从这里取数据，
    不自己读文件 → 数据访问统一在这一层，改动文件路径只动这里。

数据文件（E3 产出）：
    data/char_pinyin.json  拼音 → 单字候选[{char,freq}]（按词频降序）
    data/pinyin_word.json  拼音串 → 词候选[{char,freq}]（注意键名叫 char）
    data/word_pinyin.json  词 → 拼音串
"""

import json
import os

# data 目录固定相对本文件路径
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA_DIR = os.path.normpath(DATA_DIR)

# 缓存：只读一次文件，后续复用，避免每次调用都读磁盘
_cache = {}


def _load(filename):
    """读 JSON，带缓存。文件名相同只读一次。"""
    if filename not in _cache:
        path = os.path.join(DATA_DIR, filename)
        with open(path, encoding="utf-8") as fp:
            _cache[filename] = json.load(fp)
    return _cache[filename]


def char_pinyin():
    """拼音 → 单字候选列表 [{char, freq}]"""
    return _load("char_pinyin.json")


def pinyin_word():
    """拼音串 → 词候选列表 [{char(实为词), freq}]"""
    return _load("pinyin_word.json")


def word_pinyin():
    """词 → 拼音串 的反向表 {word: pinyin}"""
    return _load("word_pinyin.json")


def global_freq():
    """全局字/词频表（E6 补）：
        {"char_freq": {字: 词频}, "word_freq": {词: 词频}}"""
    return _load("global_freq.json")
