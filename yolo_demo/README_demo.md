# ASL 手语字母数据集准备 Demo —— 讲解文档

> 配套脚本：`prepare_data.py` | 项目：手语字母识别（学习向）

---

## 1. 这个脚本在做什么？

把 ASL Alphabet 原始数据（每类一个文件夹）**按比例随机划分**成训练集和验证集，输出成 YOLO 分类模式要求的目录结构。

### 原始数据结构
```
asl_alphabet_train/
├── A/      (3000张图，都是字母A的手势)
├── B/      (3000张图)
├── ...
├── Z/
├── del/    (删除手势)
├── space/  (空格手势)
└── nothing/(无手势/背景)
```

### 处理后结构
```
dataset_small/
├── train/
│   ├── A/  ├── B/  ...  ├── Z/
├── val/
│   ├── A/  ├── B/  ...  ├── Z/
└── classes.txt
```

**类别名 = 目录名**，这就是 YOLO 分类模式读取标签的方式——**不用人工标注**，省掉了 LabelImg 的环节。

---

## 2. 为什么用"分类模式"而不是"检测模式"？

| 对比 | 分类 (Classification) | 检测 (Detection) |
|------|----------------------|------------------|
| 回答的问题 | "这张图**是什么**？" | "图里**哪里有**什么？" |
| 需要标注框吗 | 不需要 | 需要 (x,y,w,h) |
| 适合场景 | 整张图一个主体 | 图里多个物体 |
| 本数据集 | ✅ 单手势居中特写 | ❌ 没有现成框标注 |

ASL 数据每张是 200×200 的单手势特写，手势居中，**整张图就代表一个字母**，所以分类模式最合适。检测模式要手动画框，属于"小题大做"。

> 这是工程里常见的权衡：**先选最简单匹配数据的方式**，而不是盲目套用复杂模型。你之前学的 YOLO 检测在这里用不上，是正常且合理的。

---

## 3. 脚本关键点逐行讲解

### 3.1 随机划分的完整逻辑
```python
random.seed(args.seed)          # 固定随机种子，结果可复现
...
files = sorted(...)             # 列出该类所有图片
random.shuffle(files)           # 打乱顺序
n_val = int(len(files) * args.val_ratio)  # 算验证集张数
val_files = files[:n_val]       # 前 N 张给验证集
train_files = files[n_val:]     # 其余给训练集
```

**为什么必须打乱？** 原始文件按 A1, A2, A3... 排列，A1 可能是同一时间段拍的，相似度高。不打乱会让训练集和验证集"长得太像"，模型评估成绩虚高（叫**数据泄漏**）。

**为什么需要验证集？** 训练集用来让模型学习，验证集用来**检测模型有没有学过头**（过拟合）。如果训练准、验证不准，就是过拟合了。

### 3.2 为什么支持 `--limit`？
```bash
python prepare_data.py --src ... --out dataset_small --limit 20
```
每类只取 20 张 → 总共才 580 张，CPU 训练几分钟就能跑完一轮。**学习阶段先跑通，再上全量数据**，这是验证"流程对不对"的高效策略。

### 3.3 `classes.txt` 是干嘛的？
把类别名顺序写下来，供后续训练脚本统一管理标签顺序，避免字母顺序混乱。

---

## 4. 常用运行命令

```bash
# 完整数据（每类 3000 张，约 8.7 万张）
python prepare_data.py --src "ASL\...\asl_alphabet_train" --out dataset

# 小样本快速测试（每类 20 张）
python prepare_data.py --src "ASL\...\asl_alphabet_train" --out dataset_small --limit 20

# 自定义验证集比例（如 10%）
python prepare_data.py --src "ASL\...\asl_alphabet_train" --out dataset --val-ratio 0.1
```

---

## 5. 下一步预告（T6 训练脚本）

数据准备好了，接下来就是：
```
加载数据 → YOLOv8n 分类模型 → CPU 训练 → 得到模型文件 best.pt
```

训练脚本会用到 ultralytics 的 `YOLO.train()` 接口，只需配好 `data` 路径和几个超参数即可。你准备好后我们继续。
