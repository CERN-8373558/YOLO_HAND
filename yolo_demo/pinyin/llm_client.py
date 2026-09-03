# -*- coding: utf-8 -*-
"""
llm_client.py — E8b 大模型 API 客户端（DeepSeek / OpenAI 兼容）

功能：
    complete(letters, context)：把手语字母串 + top候选 + 上文发给大模型，
    请求它推断最可能的 Top-3 整句候选，返回统一格式。

    Agent（E8a）把本模块的 complete 作为 llm_hook 注入。
    设计目标：
      - 词级别调用（由上层 Agent 控制时机，本模块只做单次请求）
      - 失败/超时/无 Key → 返回空列表，调用方回落本地候选（不阻塞、不崩溃）

协议：OpenAI Chat Completions（DeepSeek 兼容）。
配置：config.py（LLM_API_URL / LLM_MODEL / LLM_API_KEY）
"""

import json

import requests

from . import config


def _build_prompt(letters, context="", local_cands=None):
    """构造发给大模型的 prompt。"""
    local_part = ""
    if local_cands:
        texts = [c["text"] for c in local_cands[:5]]
        local_part = "本地候选: " + "、".join(texts) + "\n"
    ctx_part = f"上文: {context}\n" if context else ""
    return (
        "你是一个手语识别纠错助手。用户通过手语输入中文，手语按字母拼出内容，"
        "可能包含两种情况：\n"
        "  1) 完整汉语拼音（如 nihao → 你好）\n"
        "  2) 拼音首字母缩写/简写（像输入法，如 jtzwcsm → 今天中午吃什么，nxhlb → 你想喝凉白开 等）\n"
        "字母可能因手型近似识别错误。请根据字母串推断最可能的汉语词或句子。\n"
        "即使字母串看起来模糊、太短或不像完整拼音，也请直接给出你认为最合理的猜测，"
        "不要花时间反复推理，不要返回空结果。如果没把握，就给出最常见的同音/同首字母词。\n\n"
        "示例：\n"
        '输入 "jtzwcsm" → {"candidates": [{"text": "今天中午吃什么", "reason": "首字母缩写"}]}\n'
        '输入 "nihao" → {"candidates": [{"text": "你好", "reason": "完整拼音"}]}\n\n'
        f"字母串: {letters}\n"
        f"{local_part}"
        f"{ctx_part}"
        "请给出最可能的 3 个候选词/句，只输出 JSON，格式:\n"
        '{"candidates": [{"text": "候选1", "reason": "简短原因"}, ...]}\n'
        "text 必须全部是中文。"
    )


def _parse_response(text):
    """解析模型返回，提取 candidates 列表。兼容各种格式包裹。

    LLM 可能返回：
      - 纯 JSON {candidates:[...]}
      - 前后带说明文字（"以下是结果：" + JSON）
      - markdown 代码块 ```json ... ```
    逐级尝试提取。
    """
    if not text:
        return []
    t = text.strip()

    # 尝试 1: 提取第一个 { 到最后一个 }（容纳前后说明文字）
    try:
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end > start:
            data = json.loads(t[start:end + 1])
            cands = data.get("candidates", [])
            out = []
            for c in cands[:3]:
                if isinstance(c, dict) and c.get("text"):
                    out.append({"text": c["text"], "score": 0.9,
                                "source": "llm", "reason": c.get("reason", "")})
            if out:
                return out
    except Exception:
        pass

    # 尝试 2: 整串直接解析
    try:
        data = json.loads(t)
        cands = data.get("candidates", [])
        return [{"text": c["text"], "score": 0.9, "source": "llm",
                 "reason": c.get("reason", "")}
                for c in cands[:3] if isinstance(c, dict) and c.get("text")]
    except Exception:
        pass

    # 尝试 3: 截断 JSON 部分恢复——提取所有 "text":"xxx" 片段
    import re
    texts = re.findall(r'"text"\s*:\s*"([^"]{1,20})"', t)
    out = [{"text": x, "score": 0.9, "source": "llm", "reason": ""}
           for x in texts if x.strip()]
    if out:
        return out[:3]
    return []


def complete(letters, context="", local_cands=None):
    """调用大模型补全，返回候选列表；失败返回空列表。

    返回: [{"text","score","source","reason"}, ...]
    """
    if not config.LLM_API_KEY or config.LLM_API_KEY.startswith("sk-REPLACE"):
        return []  # 未配置 Key

    prompt = _build_prompt(letters, context, local_cands)
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": "你是手语识别纠错助手，只输出指定 JSON。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1000,
    }
    if config.LLM_THINKING == "disabled":
        payload["thinking"] = {"type": "disabled"}
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    url = config.LLM_API_URL.rstrip("/") + "/v1/chat/completions"

    # 最多尝试 3 次（应对 DeepSeek 对模糊简写的偶发空响应）
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, headers=headers,
                                 timeout=config.LLM_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            parsed = _parse_response(content)
            if parsed:
                return parsed
            # content 为空/解析失败 → 重试
            if attempt < 2:
                print(f"[llm] 第{attempt+1}次响应为空，重试...")
        except Exception as e:
            if attempt == 2:
                print(f"[llm] 调用失败，回落本地候选: {e}")
            else:
                print(f"[llm] 第{attempt+1}次调用异常，重试... {e}")
    return []


def _chat(prompt, system="你是中文润色助手，只输出指定 JSON。"):
    """通用单次对话（结构化 JSON 输出），返回原始响应文本；失败返回 None。"""
    if not config.LLM_API_KEY or config.LLM_API_KEY.startswith("sk-REPLACE"):
        return None
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"},
    }
    if config.LLM_THINKING == "disabled":
        payload["thinking"] = {"type": "disabled"}
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    url = config.LLM_API_URL.rstrip("/") + "/v1/chat/completions"
    for attempt in range(2):
        try:
            resp = requests.post(url, json=payload, headers=headers,
                                 timeout=config.LLM_TIMEOUT)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if content and content.strip():
                return content.strip()
        except Exception as e:
            if attempt == 1:
                print(f"[llm] 调用失败: {e}")
    return None


def _extract_json_field(text, field):
    """从 JSON 文本中提取指定字段（容错：容忍前后说明文字）。"""
    if not text:
        return None
    try:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            data = json.loads(text[start:end + 1])
            val = data.get(field)
            return val if val not in (None, "") else None
    except Exception:
        pass
    return None


def polish_sentence(sentence):
    """把已上屏的整句中文字符串润色成通顺句子（加标点/纠错）。

    参数: sentence: str，如 "你好学习在建"
    返回: str 润色后的句子（如 "你好，学习再建。"），失败返回原句。
    """
    text = (sentence or "").strip()
    if not text:
        return text

    prompt = (
        "用户用手语逐词输入了一句话，可能缺少标点、有同音错字、语序需调整。"
        "请润色成一句通顺的中文句子，加好标点。\n"
        f"用户输入: {text}\n"
        '只输出 JSON: {"sentence": "润色后的句子"}'
    )
    content = _chat(prompt)
    result = _extract_json_field(content, "sentence") if content else None
    return result if result else text
