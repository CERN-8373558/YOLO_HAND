# 移植到 RTX 3060 电脑的说明

本压缩包含代码、模型权重和文档，可移植到另一台有 RTX 3060 的电脑上运行。

## 压缩包含什么

```
yolo_demo/
├── camera.py              # 摄像头识别（MediaPipe 定位手 + YOLO 分类）
├── predict.py             # 图片/视频推理
├── prepare_data.py        # 数据准备（划分 train/val）
├── train.py               # 训练脚本（注意：默认 device="cpu"，见下文）
├── hand_landmarker.task   # MediaPipe 手部模型
├── yolov8n-cls.pt         # YOLO 分类预训练权重
├── runs/classify/runs/classify/asl_demo-2/weights/best.pt  # 全量训练好的最优模型
├── README_demo.md / README_predict.md / README_camera.md
└── README_移植说明.md
```

## 第 1 步：安装 Python 环境（在 3060 电脑上）

用 Anaconda 新建环境并装 **CUDA 版** PyTorch（关键！）：

```bash
conda create -n yolo_env python=3.9 -y
conda activate yolo_env

# CUDA 版 PyTorch（3060 用这个，别装 +cpu）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# YOLO 和配套
pip install ultralytics opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

# MediaPipe
pip install mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 第 2 步：启用 GPU 训练（只改一行）

`train.py` 中这一行目前写死 CPU：

```python
model.train(..., device="cpu")
```

改成：

```python
model.train(..., device=0)
```

`device=0` 表示用第一块 GPU（即你的 RTX 3060）。

## 第 3 步：准备数据集（二选一）

数据集约 1GB、不在压缩包内：

**方式 A：直接复制**
把原电脑的 `C:\Users\24405\Pictures\Yolo_test\yolo_demo\dataset` 整个文件夹复制到新电脑同目录。

**方式 B：重新生成**
复制原始 ASL 数据后运行：
```bash
python prepare_data.py --src <ASL原始目录> --out dataset
```

## 第 4 步：快速验证

```bash
# 用已有全量模型推理一张测试图
python predict.py "图片路径"

# 摄像头实时识别
python camera.py
```

## 第 5 步：用 3060 重新全量训练

```bash
python train.py --data dataset --epochs 50 --imgsz 224 --batch 32
```

预期单轮约 1~3 分钟（原 CPU 约 50 分钟/轮），50 轮约 1~2 小时。

## 性能对比参考

| 项目 | i7-12700 CPU | RTX 3060 |
|------|--------------|----------|
| 单轮全量训练 | ~50 分钟 | ~1-3 分钟 |
| 单张推理 | ~6ms | ~2-3ms |
