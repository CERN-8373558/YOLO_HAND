# RAG 模块（混淆知识记忆库）—— 需求分析文档

> 基于《手语字母识别 + AI 拼音补全》项目扩展
> 版本：v1.0 | 日期：2026-09-05 | 状态：草案

---

## 1. 背景与动机

### 1.1 现状
- 字母混淆统计工具（confusion_camera.py）能产出 JSON：各意图字母 top1 错分成别的字母的比率
- 这些数据揭示了"用户手型下哪些字母互相易混"（如 B→C 12%）
- 目前这些 JSON 只是落盘文件，**没有任何模块消费它们**
- milvus-lite 已装好（yolo_env311, Python 3.11），一直闲置

### 1.2 目的
把混淆统计数据**向量化存进 milvus-lite**，形成"混淆知识记忆库"（RAG 数据源），
提供独立检索接口——未来可被 Agent/补全逻辑查询（如"B 手势容易认成谁"）。
**本次范围：不接入 LLM、不接入 camera_agent 流程**，仅独立实现 store/retrieve 能力并验证。

### 1.3 决策记录（需求澄清结论）
| 问题 | 结论 |
|------|------|
| RAG 定位 | 独立能力模块，暂不接 LLM/主流程 |
| 数据源 | 字母混淆统计 JSON（先用现有 data/confusion_*.json，后续用新模型重采完整 A-Z） |
| 数据形态 | **每条混淆关系 = 一条向量记录** |
| 存储内容 | 意图字母 → 易混淆字母（含混淆率），如 "B 的手势常被认成 C" |
| embedding | **先留可替换接口 + 占位实现**（跑通链路），真实中文模型后续接入 |
| 向量库 | **milvus-lite 真接上**（pymilvus MilvusClient 指向本地文件） |
| 检索方 | 先只验证"存进去能查出来"（测试脚本），不接业务 |
| 入库过滤 | 混淆率 ≥ 1% 入库（滤噪声） |
| 交付形态 | `pinyin/rag.py` + 测试脚本 |

---

## 2. 环境约束
- yolo_env311（Python 3.11.16）
- milvus-lite + pymilvus 3.0.1（已验证 MilvusClient 本地文件可用）
- 混淆数据：data/confusion_*.json（现有多份，仅测个别字母；工具已支持完整 A-Z 采集）

---

## 3. 功能需求

### 3.1 核心功能
| 编号 | 功能 | 说明 | 优先级 |
|------|------|------|--------|
| R1 | embedding 抽象接口 | 可替换的 embed(text)->vector；先给占位实现 | 高 |
| R2 | 建库 + 存记录 | 从 confusion JSON 解析，逐条混淆关系向量化入库 milvus-lite | 高 |
| R3 | 检索 | 输入查询（如 "B 手势 认成"）→ 召回最相关混淆知识 | 高 |
| R4 | 数据解析 | 从 confusion JSON 生成记录（过滤 ≥1%） | 高 |
| R5 | 测试脚本 | rag_test.py：建库 → 存 → 检索 → 打印，证明链路 | 高 |

### 3.2 数据模型
**一条向量记录** = 一条"意图字母 X 会错认成 Y"的混淆知识：
```json
{
  "id": "conf_B_C_001",
  "text": "字母 B 的手势有时会被识别成 C（概率 12%）",
  "intent": "B",        // 意图字母（作为标量字段，便于过滤）
  "target": "C",        // 错认成的字母
  "rate": 0.12,         // 混淆率（元数据保留）
  "vector": [0.1, ...]  // embedding 结果（维度由实现定）
}
```
> 一条数据只代表一个具体 X→Y 关系。意图字母做 scalar field，检索可带过滤。

### 3.3 embedding 接口（R1）
```python
class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: ...
    @property
    def dim(self) -> int: ...

class HashEmbedder(Embedder):
    """占位实现：字符 n-gram 哈希成固定维向量，跑通链路用。
       后续替换为 bge-small-zh 等真实中文 embedding（同接口即可）。"""
```

### 3.4 入库（R2 + R4）
- 输入：confusion JSON 路径（或 dict）
- 解析：对每个 intent 的 targets（rate ≥ 0.01），生成一条记录 text + 向量
- 存到 milvus-lite collection：字段 id/text/intent/target/rate/vector
- 幂等：同库重复跑需清空重建（或按 id 去重），测试阶段直接 drop+create

### 3.5 检索（R3）
- 输入：query 文本（如 "B 手势被认成"）+ top_k
- embedding query → 向量检索 top_k → 返回 [{text,intent,target,rate,similarity}]
- 检索纯向量即可（milvus 默认内积/余弦），scalar 过滤可选进阶

### 3.6 milvus-lite 用法
```python
from pymilvus import MilvusClient
client = MilvusClient("data/rag_confusion.db")   # 本地文件即库
client.create_collection("confusion", dimension=EMB_DIM)
client.insert("confusion", rows)                 # 逐条/批量
res = client.search("confusion", data=[vec], limit=top_k)
```

---

## 4. 非功能需求
| 编号 | 类别 | 需求 |
|------|------|------|
| N1 | 可替换 | embedding 与存储接口清晰，换真模型只动 Embedder 实现 |
| N2 | 幂等 | 建库脚本可重复运行（重建） |
| N3 | 编码 | JSON 读 UTF-8；文本含中文正常 |
| N4 | 规模 | 26 字母×多目标 ≈ 数百条，毫秒级检索足够 |

---

## 5. 验收标准
1. `rag_test.py` 跑通：读 confusion JSON → 建库 → 全部混淆关系入库
2. 检索 "B 的手势" → 能召回 B 相关的混淆记录（如 B→C 那条）
3. 不接 LLM、不接 camera_agent，camera 系列程序无任何行为变化
4. embedding 换成真实模型时，仅替换 Embedder 实现即可（接口不变）
