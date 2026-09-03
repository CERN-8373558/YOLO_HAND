# -*- coding: utf-8 -*-
"""
tool_lm.py — 工具层：语言模型打分【E6】

功能：
    score_sequence(words)：给一个候选"字/词序列"打分，分数越高越像人话。
    供 tool_cand（E7）对切分后的不同选字组合排序。

评分模型（初版 unigram）：
    P(你好) ∝ 全局词频(你好)  如果能整词命中
            否则 ≈ Σ 单字全局词频   （跨音节可比！E6 关键改进）
    → 归一化到 [0,1] 再返回。

    说明：真正的 bigram（P(好|你) 共现概率）需要句子级语料，
    本版先做 unigram；bigram 列入 E12 优化项。

数据（E6 补）：
    global_freq.json  {char_freq: {字:词频}, word_freq: {词:词频}}
"""

from . import loader


class LM:
    """unigram 语言模型。初始化加载一次全局词频。"""

    def __init__(self):
        gf = loader.global_freq()
        self.char_freq = gf["char_freq"]   # {字: 词频}
        self.word_freq = gf["word_freq"]   # {词: 词频}

    def word_freq_of(self, word):
        return self.word_freq.get(word, 0)

    def char_freq_of(self, ch):
        return self.char_freq.get(ch, 0)

    def score_sequence(self, words):
        """给候选词序列打分，返回 0~1 分数。

        策略：
          1) 若是词典中的真实整词（word_freq 命中）→ 显著加成，
             因为"能整体成词"说明语义合理，优于单字拼凑
          2) 否则按单字/词条全局频率累加
          3) 归一化映射 [0,1]
        """
        if not words:
            return 0.0

    def score_sequence(self, words):
        """给候选词序列打分，返回 0~1 分数。

        核心原则：
          "是词典里的真实词"是强语义证据 → 整词直接进高分区 [0.6, 1.0]
          单字拼凑（查不到整词）→ 只能进低分区 [0, 0.6)，按单字频率细分
        这样保证真实词（你好/谢谢）恒胜过拼凑（你号/些些）。

        分区内再用词频做细排序。
        """
        if not words:
            return 0.0

        full = "".join(words)
        wf_full = self.word_freq.get(full, 0)

        if wf_full > 0:
            # 整词命中（含高频单字如"我/你"）→ 高分区 0.6~1.0
            return 0.6 + 0.4 * self._rank_in_range(wf_full)

        # 非整词（单字或拼凑不出真实词）→ 低分区 0~0.6
        total = 0.0
        for w in words:
            if len(w) == 1:
                total += self.char_freq.get(w, 0)
            else:
                total += self.word_freq.get(w, 0)
        return 0.6 * self._rank_in_range(total)

    @staticmethod
    def _rank_in_range(raw):
        """把原始频数映射到 0~1（log 压缩大数差）。"""
        import math
        if raw <= 0:
            return 0.0
        return min(math.log1p(raw) / 16.0, 1.0)


# 模块级单例（工具层共享一个模型，避免每次重建）
_lm = None


def score_sequence(words):
    """给候选词序列打分（0~1）。"""
    global _lm
    if _lm is None:
        _lm = LM()
    return _lm.score_sequence(words)


def compare(a_words, b_words):
    """比较两个候选序列，返回谁更像人话。
    返回: (更好序列, 分数A, 分数B)"""
    sa, sb = score_sequence(a_words), score_sequence(b_words)
    return (a_words, sa, sb) if sa >= sb else (b_words, sb, sa)
