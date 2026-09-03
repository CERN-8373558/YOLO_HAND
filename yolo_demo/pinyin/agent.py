# -*- coding: utf-8 -*-
"""
agent.py — Agent 决策引擎【E8a】

功能：
    把工具层（tool_cand 等）封装成"决策流程"。输入一段字母串（+可选上文），
    输出一个决策动作：直接输出 / 候选确认 / 请求增强 / 等待。

决策动作（action）：
    defer       字母串太短/在输入中 → 等待，不动作
    output      best 足够确定 → 直接输出这个候选（可自动上屏）
    confirm     存在歧义 → 展示候选列表等用户选（UI 层用）
    ask_llm     本地候选都不好 → 请求大模型语义增强（E8b 接入）
    fail        无可识别候选（如英文/空）→ 提示，交给上层切模式

规则（对应 E2 文档 4 节，阈值参数化）：
    1. letters 长度 < min_len 或 无停顿 → defer
    2. best.score >= output_th → output
    3. 前两名得分差 < ambig_gap → confirm（歧义）
    4. 所有候选 score < local_th（本地都不可靠）→ ask_llm
    5. 无候选（英文等）→ fail
    6. （可选）手语识别 conf 低时直接 ask_llm

本文件只做"决策逻辑"，不碰界面/网络。大模型调用在 E8b 由
decision 里的 ask_llm 触发（此处留出 hook 接口）。
"""

from . import tool_cand


class Agent:
    """Agent 决策引擎。阈值可通过参数调整（E12 调优）。"""

    def __init__(self, output_th=0.75, ambig_gap=0.02, local_th=0.5, min_len=2):
        self.output_th = output_th    # best 足够确定直接输出
        self.ambig_gap = ambig_gap    # 前两名差距阈值：小于则歧义
        self.local_th = local_th      # 本地候选低于此分 → 请求增强
        self.min_len = min_len        # 太短不处理

    def decide(self, letters, context="", hand_conf=1.0, llm_hook=None):
        """核心决策入口。

        参数:
            letters:  手语累积的字母串（无空格），如 "nihao"
            context:  上文/已上屏文本（供后续增强用，本版暂不消费）
            hand_conf: 最近一次手语识别的置信度(0~1)，低则倾向 ask_llm
            llm_hook: 可选的"大模型调用函数"(letters, context)->candidates，
                      由 E8b 注入；若本决策判定需要增强且 hook 可用则调用
        返回:
            {"action": str, "candidates": [...], "best": str, "reason": str,
             "message": str}
        """
        letters = (letters or "").strip().lower()

        # 规则1：太短 / 空 → defer
        if len(letters) < self.min_len:
            return {"action": "defer", "candidates": [], "best": None,
                    "reason": "too_short", "message": f"输入太短({len(letters)})，等待继续输入"}

        # 取本地候选
        result = tool_cand.build(letters, top_k=5)
        candidates = result.get("candidates", [])
        best = result.get("best")
        ambiguous = result.get("ambiguous", False)

        # 规则5：无候选（英文/无法切分）→ fail
        if not candidates or best is None:
            return {"action": "fail", "candidates": [], "best": None,
                    "reason": "no_candidate", "message": "无法切分为中文拼音（疑似英文或输入有误）",
                    "note": result.get("note", "")}

        best_score = candidates[0]["score"]
        second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0

        # 规则6：手语置信度低 → 直接请求增强（怀疑识别错字母）
        if hand_conf < 0.7:
            return self._ask_llm_or_confirm(letters, context, candidates,
                                            reason="low_hand_conf", llm_hook=llm_hook)

        # 规则2：best 足够确定 → output
        if best_score >= self.output_th:
            return {"action": "output", "candidates": candidates, "best": best,
                    "reason": "high_score",
                    "message": f"识别为：{best}（置信 {best_score:.2f}）"}

        # 规则3：前两名接近 → confirm（歧义）
        if (best_score - second_score) < self.ambig_gap or ambiguous:
            return {"action": "confirm", "candidates": candidates, "best": best,
                    "reason": "ambiguous",
                    "message": f"存在歧义，请选择候选（1={candidates[0]['text']}）"}

        # 规则4：本地都不够好 → 请求增强
        if best_score < self.local_th:
            return self._ask_llm_or_confirm(letters, context, candidates,
                                            reason="low_local_score", llm_hook=llm_hook)

        # 兜底：分数中等但无明显歧义 → 仍 output 第一候选（低分置信提示）
        return {"action": "output", "candidates": candidates, "best": best,
                "reason": "moderate",
                "message": f"识别为：{best}（置信 {best_score:.2f}，请确认）"}

    def _ask_llm_or_confirm(self, letters, context, candidates, reason, llm_hook):
        """请求增强：若有 llm_hook 则调用，否则退化为 confirm。"""
        if llm_hook is not None:
            try:
                llm_cands = llm_hook(letters, context)
                if llm_cands:
                    return {"action": "confirm", "candidates": llm_cands,
                            "best": llm_cands[0]["text"],
                            "reason": reason + "->llm",
                            "message": f"本地候选不可靠，已请求大模型增强：{llm_cands[0]['text']}"}
            except Exception as e:
                # 大模型失败 → 回落本地 confirm，不崩溃
                pass
        return {"action": "confirm", "candidates": candidates,
                "best": candidates[0]["text"] if candidates else None,
                "reason": reason,
                "message": "本地候选不可靠，请手动确认候选"}


# 模块级默认实例
_default_agent = None


def get_agent(**kwargs):
    """获取默认 Agent（参数可覆盖）。"""
    global _default_agent
    if _default_agent is None:
        _default_agent = Agent(**kwargs)
    elif kwargs:
        _default_agent = Agent(**kwargs)  # 有参数则重建（允许 E12 调优）
    return _default_agent
