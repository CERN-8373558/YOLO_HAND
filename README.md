# 项目进度总结（重启恢复用）

> 更新：2026-09-05 | 项目：手语字母识别 + AI 补全（中文拼音 / 英文）

---

## 一、项目全貌

**一句话**：摄像头前比手语字母 → 识别成字母串 → 本地引擎/大模型补全成中文或英文 → 上屏。

能力主线：
1. **手语字母识别**（YOLOv8n 分类 + MediaPipe 定位手）
2. **中文拼音补全**（本地词典/切分/LM + Agent 决策 + 大模型）
3. **英文模式**（纯字母累积 → 大模型纠错组词/润色）
4. **配套工具**：自采数据、混淆统计、增量训练、模型选择器
5. **RAG**（混淆知识记忆库，milvus-lite，规划中）

---

## 二、环境

### 本机（CPU）C:\Users\24405
- conda 环境：`yolo_env311`（Python 3.11.16，**主力**）
  - 库：torch 2.8.0+cpu、ultralytics 8.4.137、opencv 5.0.0、mediapipe 1.0.1、pymilvus 3.0.1 + milvus-lite、jieba、pypinyin
  - 运行：`conda activate yolo_env311`
- 旧环境：`yolo_env`（Python 3.9.25，备份保留）
- 主目录：`C:\Users\24405\Pictures\Yolo_test`
  - `yolo_demo\`：全部代码 + 运行时数据
  - 文档：`需求分析.md`、`方案浓缩分析.md`、`任务清单.md`、`需求分析_混淆统计.md`、`需求分析_英文模式.md`、`需求分析_RAG混淆库.md`
- ⚠️ **训练大数据在 `C:\Users\24405\Pictures\YOLO_resource\`**（不入项目，PyCharm 才打得开）：
  - `ASL\`、`CE-CSL\`、`dataset\`（原始训练集）
  - `collect_deleted_backup\`（曾删除的自采数据备份）
- 大模型 API Key：`pinyin\config.py`（gitignore，不入库）或环境变量 `DEEPSEEK_API_KEY`

### 3060 电脑（GPU 训练用）
- torch 2.14.0+cu130；曾训练 asl_demo-6；大数据量重训建议搬 GPU 机器

---

## 三、任务清单状态（详见 任务清单.md）

主线 T1~T12：
- [X] T1~T10：环境、数据、训练、推理、摄像头、打字机
- [ ] T11 联调优化、T12 验收（挂起）

阶段二 E1~E12：
- [X] E1~E8b、E10：拼音引擎全链路（词典/切分/LM/候选/Agent/LLM/界面）
- [-] E9 RAG：已拆分重构为 R 系列（见下），不再沿用 E9
- [ ] E11 端到端联调、E12 评估（挂起）

辅助工具 M 系列：
- [X] M1~M4：混淆统计工具开发
- [ ] M5：混淆实测（待新模型重采）

英文模式 EN 系列：
- [X] EN1~EN4：英文补全 prompt、英文润色、模式切换(e)、跳过本地直发 LLM
- [ ] EN5：英文整句润色键实测 + 中文回归

RAG R 系列（混淆知识库，独立实现）：
- [ ] R1~R5：待开工（需求已定：需求分析_RAG混淆库.md）

---

## 四、当前状态（2026-09-05）

**第一版整体功能已完成**，处于"打磨 + 验证"阶段：

1. ✅ 识别模型已迭代到 **asl_demo_dark-v2**（基于用户自采 3.5 万张暗背景数据增量训练，验证集 100%）
2. ✅ camera_agent：中/英模式切换、置信度能量累积确认、进度条可视化、防重复锁
3. ✅ 模型三选一启动器 run_camera.py
4. ✅ 混淆统计、数据采集工具可用（采集模型默认 v2）
5. ✅ 英文模式：纯字母 + LLM 优化（hello/world 组词验证通过）
6. ⏳ RAG 混淆知识库（R1~R5）需求已定，待开发
7. ⏳ 需实测项：英文模式手感、双字母(ll)确认、M5 混淆重采

---

## 五、关键文件说明（yolo_demo\）

| 文件/目录 | 作用 | 状态 |
|-----------|------|------|
| `camera_agent.py` | **AI 补全打字机（主程序）**：中/英模式、能量确认、进度条 | 打磨中 |
| `camera_text.py` | 逐字母上屏打字机 | 完成 |
| `camera.py` | 摄像头纯识别显示 | 完成 |
| `hand_recognizer.py` | 公共核心：定位手+分类+运动检测+top-k | 完成 |
| `confusion_camera.py` | 字母混淆统计工具（键盘意图+top5采集） | 完成 |
| `collect_camera.py` | 摄像头数据采集（a-z + 1=nothing 2=space 3=del） | 完成 |
| `run_camera.py` | 模型选择器（列出 models/*.pt 供选跑 camera_agent） | 完成 |
| `prepare_data.py` | 数据集划分 train/val | 完成 |
| `train.py` | 训练脚本（支持 --model 增量 / --name 实验名） | 完成 |
| `predict.py` | 图片推理 | 完成 |
| `pinyin\` | 引擎包 + LLM 客户端（中文+英文接口） | 完成 |
| `models\` | 见下方模型表 | — |
| `collect\` | 自采数据（A-Z+del/space/nothing，各~1000-1300） | 大文件不入库 |
| `dataset_dark_v2\` | 增量训练集（train 3万 / val 5千） | 大文件不入库 |
| `hand_landmarker.task` | MediaPipe 手部模型 | 就绪 |

### 模型（models\）
| 模型 | 说明 |
|------|------|
| `asl_demo-6-best.pt` | 旧：8.7万张标准浅背景 ASL 训练（GPU）|
| `asl_demo_dark-best.pt` | v1：asl_demo-6 + 8千张自采暗背景增量 |
| `asl_demo_dark_v2-best.pt` | **v2 当前最优**：+ 3.5万张自采，验证集 100%（浅背景会退化，仅自用暗环境）|
| `asl_demo_dark_v2-best.onnx` | v2 的 ONNX 版（安卓 ONNX Runtime 迁移用）|

### pinyin\ 包结构
```
pinyin/
├── config.py        # LLM API 配置（gitignore 不入库）
├── loader.py        # 数据加载（缓存）
├── tool_dict.py     # 拼音→汉字映射
├── tool_seg.py      # 字母串切分器
├── tool_lm.py       # unigram 语言模型
├── tool_cand.py     # 候选生成
├── agent.py         # Agent 决策
├── llm_client.py    # LLM 客户端（中文 complete/polish + 英文 complete_en/polish_sentence_en）
└── test_typing.py   # 引擎测试
```

---

## 六、踩过的坑速查（避免重复）

1. **deepseek v4-flash 空响应** = 深度思考耗尽 token → 加 `thinking:{type:"disabled"}`
2. **milvus-lite** = 需 py≥3.10；yolo_env311 可用（pymilvus MilvusClient 指向本地文件）
3. **jieba 词频失真** → tool_lm 用整词高分区 vs 拼凑低分区方案
4. **jieba dict 不按词频排序** → 全量保留，不截断
5. **PowerShell 中文乱码** = `$env:PYTHONIOENCODING='utf-8'` 或 Python 直读
6. **摄像头乱字母** = 手移动过渡 → 运动检测跳过；抖动 → 稳定帧+能量确认
7. **tflite 转换 Windows 不行** = ultralytics 8.4 LiteRT 仅 Linux/Mac → 改用 ONNX Runtime（安卓迁移路线）
8. **git key 泄露** = config.py 已 gitignore；key 用环境变量或本地文件
9. **增量训练遗忘** = 只在暗背景数据上微调会忘浅背景（v2 浅背景仅 ~16%）——需混训才两全
10. **采集重复字母(hello ll)** = 中/英都靠"进度条清零=手势段结束=解锁"防重复，不再按模式特判

---

## 七、运行命令速记

```powershell
conda activate yolo_env311
cd C:\Users\24405\Pictures\Yolo_test\yolo_demo

python run_camera.py                  # 模型选择器 → camera_agent（推荐入口）
python camera_agent.py --use-llm      # 手动指定（默认模型请用 --model）
python camera_text.py                 # 打字机
python camera.py                      # 纯识别
python confusion_camera.py            # 混淆统计
python collect_camera.py --per-class 1000   # 采数据（默认新模型）
python pinyin/test_typing.py          # 中文引擎测试
python train.py --data <dir> --model <weights> --name <exp>  # 训练/增量
```

### camera_agent 按键
`e`=中/英模式  `m`=LLM开关  `q`=退出  `c`=清空
`1-9`=选候选(LLM优先)  `0`=采纳LLM第1  `Esc`=忽略LLM候选
`S`=发当前字母串给LLM补全  `T`=整句润色

### confusion_camera / collect_camera 按键
`a-z`=切意图/采集类(含q)  `1/2/3`=nothing/space/del(采集用)
`Space`=暂停(采集)或重置(混淆)  `Esc`=退出

---

## 八、重启后下一步建议

1. **英文模式实测**：EN5 —— e 切 EN 比 hello/world，验证双字母(ll)确认 + 整句润色
2. **M5 混淆重采**：用新模型(v2)重采完整 A-Z 混淆表（当前 JSON 都是旧模型且不全）
3. **R1~R5 RAG**：按 需求分析_RAG混淆库.md 实现混淆知识记忆库（独立，不接 LLM）
4. **安卓迁移（可选）**：模型已转 ONNX，ONNX Runtime Mobile 可在安卓跑
5. 若需浅背景+暗背景两全：混合新旧数据集重训

---

# 附录：功能总览 · 用途 · 使用场景 · 未来计划

## A. 功能总览

### 核心功能（手语输入系统）
| 功能 | 说明 |
|------|------|
| 手语字母实时识别 | MediaPipe 定位手 + YOLOv8n 分类 29 类（A-Z + del/space/nothing）|
| 中文拼音补全 | 手语字母串→本地拼音引擎(切分/词典/LM/候选)→Agent 决策→大模型增强→中文候选上屏 |
| 英文模式 | 字母串直接发大模型→英文纠错组词/润色（无本地猜词）|
| 逐字母打字机 | camera_text.py：字母确认即上屏（含退格/空格手势）|
| 整句润色 | T 键：整句发给大模型润色成通顺句子（中/英文各自 prompt）|

### 识别质量机制
| 机制 | 作用 |
|------|------|
| 运动检测(moving) | 手移动过渡期跳过累积，防乱字母 |
| 置信度能量累积确认 | 连续 8 帧 + 平均置信≈0.9 才确认字母（比单纯帧数更稳）|
| 确认进度条 | 检测框下方实时显示确认程度，满格=确认成功 |
| 防重复锁 | 进度条清零(手势段结束)才解锁，同一手势只确认一次 |
| 原文回显栏 | 上屏后保留"字母→文本"映射，便于核对/纠错 |

### 工具链
| 工具 | 用途 |
|------|------|
| run_camera.py | 一键选模型启动 camera_agent（多模型对比）|
| collect_camera.py | 自采手势数据（a-z / nothing / space / del，自动续采+去重）|
| confusion_camera.py | 统计"我手型下哪个字母易被认成谁"（产出给 LLM/Agent 的混淆知识）|
| prepare_data.py | 数据集 train/val 划分 |
| train.py | 训练/增量训练（--model 续训、--name 多实验）|

### 引擎模块（pinyin\）
工具层(tool_dict/seg/lm/cand) → Agent(agent) → 大模型(llm_client 中/英) → 界面(camera_agent)

## B. 用途

1. **手语→文字的实时输入法**：比手语拼词，得到中文或英文文本（省去键盘）
2. **手语学习辅助**：识别自己的手势是否正确（camera.py 纯识别 + 置信度）
3. **混淆分析**：量化"我的哪个字母手势容易被系统认错"，指导针对性训练/纠正
4. **数据自采**：非技术用户也能通过按键+摄像头积累个性化训练数据
5. **模型增量迭代**：新数据→增量训练→run_camera 对比新旧模型效果

## C. 使用场景

| 场景 | 适用功能/工具 |
|------|--------------|
| 个人固定暗环境使用 | asl_demo_dark_v2 + camera_agent（CN 拼中文 / EN 拼英文）|
| 公开演示（答辩/展示） | camera.py 纯识别直观展示、run_camera 选浅背景模型 |
| 想用自己手型定制模型 | collect_camera 采数 → prepare/train 增量 → 对比 |
| 排查"为什么老识别错" | confusion_camera 统计某字母的错认目标 |
| 语音/键盘不便的场景 | 手语→中文/英文补全（如静音环境交流）|
| 英文单词练习 | EN 模式拼英文词给大模型纠错 |

## D. 未来计划

| 方向 | 内容 | 依赖/备注 |
|------|------|----------|
| RAG 混淆知识库 | R1~R5：混淆 JSON 向量化入 milvus-lite，提供检索（独立，不接 LLM）| 需求已定 |
| E11/E12 | 中文端到端联调 + 本地 vs 本地+LLM 评估 | 主线收尾 |
| 英文字母双字母/整句手感 | EN5 实测打磨 | 待实机 |
| 浅背景兼容 | 混合新旧数据重训，使模型同时适应浅/暗背景 | 需重训 |
| 完整混淆表 | 用 v2 重采全 A-Z 混淆，替换旧 JSON | 供 RAG/LLM |
| 手机当摄像头 | 手机推流 → 电脑复用识别系统（IP Webcam MJPEG）| 局域网 |
| 安卓 App | 模型已转 ONNX → ONNX Runtime Mobile 真机识别 | 需 Android 环境 |
| 多语种 | 提示词工程扩展（现 EN 已支持，可推广其他语种）| 纯 prompt |
