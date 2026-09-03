# 项目进度总结（重启恢复用）

> 更新：2026-09-03 | 项目：手语字母识别 + AI 拼音补全

---

## 一、项目全貌

两条主线：
1. **手语字母识别**（YOLOv8n 分类模型）：摄像头识别手语字母 → 拼词
2. **AI 拼音补全**（阶段二扩展）：字母串 → 本地引擎 + 大模型 → 中文候选/句子

---

## 二、环境

### 本机（CPU）C:\Users\24405
- conda 环境：`yolo_env311`（Python 3.11.16，**新主力**，已装 milvus-lite 可跑 E9 RAG）
  - 库：torch 2.8.0+cpu、ultralytics 8.4.137、opencv 5.0.0、mediapipe 1.0.1、pymilvus 3.0.1 + milvus-lite、jieba、pypinyin
  - 运行：`conda activate yolo_env311`；test_typing.py 4/4 通过
- 旧环境：`yolo_env`（Python 3.9.25，备份保留，无 milvus-lite）
- 主目录：`C:\Users\24405\Pictures\Yolo_test`
  - `yolo_demo\`：全部代码 + 运行时数据（~86MB）
  - 文档：`需求分析.md`、`方案浓缩分析.md`、`任务清单.md`
- ⚠️ **训练大数据已移至 `C:\Users\24405\Pictures\YOLO_resource\`**（PyCharm 文件数过多打不开）：
  - `YOLO_resource\ASL\`（原始训练集，2.1GB / 8.7万文件）
  - `YOLO_resource\CE-CSL\`（连续手语，未用，9.4GB）
  - `YOLO_resource\dataset\`（yolo_demo\dataset 划分后 train/val，1.06GB / 8.7万文件，重训前需移回 yolo_demo\ 或改 train.py 路径）

### 3060 电脑（F:\PYcharmTEST\YOLO——test）
- 系统 Python D:\python（3.13）、torch 2.14.0+cu130
- 曾训练 asl_demo-6（GPU 完整训练），best.pt 已拷回本机

---

## 三、任务清单状态（详见 任务清单.md）

主线 T1~T12：
- [X] T1~T10：环境、数据、训练、推理、摄像头、打字机全部完成
- [ ] T11 联调优化、T12 验收（挂起）

阶段二 E1~E12（AI 拼音补全）：
- [X] E1 原理笔记、E2 架构设计、E3 词典数据、E4 词典工具、E5 切分器、E6 语言模型、E7 候选生成、E8a Agent、E8b 大模型API、E10 界面层
- [-] E9 RAG（搁置：Milvus Lite 需 py≥3.10，本机 py3.9 装不了）
- [ ] E11 端到端联调、E12 评估

---

## 四、当前正在做的（重启后从这里继续）

**E10 界面层 camera_agent.py 正在打磨**，最近刚完成的改动：
1. ✅ S 键 = 单词补全（发 pending 字母串给大模型）、T 键 = 整句润色（分开了）
2. ✅ 数字键 1-9 优先选 LLM 候选，无则选本地候选
3. ✅ S 手动发送 = 追加；本地 output 后的 LLM = 替换（修了"覆盖不对"）
4. ✅ 运动检测：手移动过渡期跳过识别（防乱字母）
5. ✅ 大模型 response 修复：`thinking: {type: "disabled"}` 关闭深度思考，解决空响应（max_tokens=1000）

**下一步待做**：运行 `camera_agent.py --use-llm` 实测运动检测 + S/T 键体验

---

## 五、关键文件说明（yolo_demo\）

| 文件 | 作用 | 状态 |
|------|------|------|
| `camera.py` | 摄像头纯识别显示 | 完成 |
| `camera_text.py` | 打字机（逐字母上屏） | 完成 |
| `camera_agent.py` | **AI 补全打字机（主战场）** | 打磨中 |
| `hand_recognizer.py` | 公共核心：定位手+分类+运动检测 | 完成 |
| `models\asl_demo-6-best.pt` | 正式模型（GPU 训练，验证集100%） | 就绪 |
| `hand_landmarker.task` | MediaPipe 手部模型 | 就绪 |
| `data\*.json` | 拼音数据（char/pinyin_word/word_pinyin/global_freq） | 就绪 |
| `pinyin\` | 阶段二拼音引擎包 | 完成 |
| `build_pinyin_data.py` | 生成拼音数据 | 完成 |
| `build_global_freq.py` | 生成全局词频 | 完成 |
| 各 `README_*.md` | 学习讲解文档 | 完成 |
| `E1_拼音输入法原理.md` | E1 原理笔记 | 完成 |
| `E2_架构设计.md` | E2 架构文档 | 完成 |

### pinyin\ 包结构
```
pinyin/
├── config.py        # 大模型 API 配置（URL/model/key/thinking）
├── loader.py        # 数据加载（缓存）
├── tool_dict.py     # E4 拼音→汉字映射
├── tool_seg.py      # E5 字母串切分器
├── tool_lm.py       # E6 unigram 语言模型打分
├── tool_cand.py     # E7 候选生成（整词优先）
├── agent.py         # E8a Agent 决策
├── llm_client.py    # E8b 大模型调用（complete/polish_sentence）
└── test_typing.py   # 打字流程测试
```

---

## 六、踩过的坑速查（避免重复）

1. **deepseek v4-flash 空响应** = 深度思考耗尽 token → 加 `thinking: {type:"disabled"}`
2. **milvus-lite 装不上** = 需 Python≥3.10，py3.9 环境放弃 RAG
3. **jieba 词频失真**（"你好"频725 < "你"频23万）→ tool_lm 用"整词高分区[0.6,1] vs 拼凑低分区[0,0.6)"方案
4. **jieba dict 不按词频排序** → 不能按前N条截断，全量保留
5. **PowerShell 中文乱码** = 测试脚本直接写 UTF-8 文件，勿用 2>$null 管道（会吞输出）
6. **摄像头乱字母** = 手移动过渡误判 → 运动检测（moving 跳过累积）
7. 3060 训练 workers=8 会崩（bad allocation），用 workers=3

---

## 七、模型对比

| 模型 | 来源 | 状态 |
|------|------|------|
| asl_demo-6 (models\asl_demo-6-best.pt) | 3060 GPU | **当前正式模型** |
| asl_demo-2 | 本机 CPU | 旧 |
| E:\2\best.pt | = asl_demo-6 | 已复制 |

---

## 八、运行命令速记

```powershell
conda activate yolo_env311
cd C:\Users\24405\Pictures\Yolo_test\yolo_demo

python camera.py                 # 纯识别测试
python camera_text.py            # 打字机
python camera_agent.py --use-llm # AI 补全打字机（主）
python pinyin/test_typing.py     # 引擎测试

# camera_agent 按键
# q退出 c清空 1-9选候选 m切LLM开关 S=单词补全(发字母) T=整句润色
```

---

## 九、重启后下一步建议

1. 跑 `camera_agent.py --use-llm` 实测（运动检测 + S/T + 选择追加）
2. 手感 OK 后做 E11（端到端联调完善）
3. E12（评估：本地 vs 本地+大模型）
4. E9 RAG 需换 py3.10+ 环境或 ChromaDB 才可启
