# -*- coding: utf-8 -*-
"""
train.py — YOLOv8n 分类模式训练脚本（T6）

功能：
    用 ultralytics 训练一个"手语字母分类"模型。
    输入：prepare_data.py 生成的 train/val 目录
    输出：runs/classify/... 下的 best.pt（训练好的模型）

为什么是"分类"而不是"检测"？
    详见 README_demo.md——ASL 数据是单手势特写，整图一个类别，分类最合适。

CPU 环境提示：
    纯 CPU 训练请保持模型为 nano、imgsz 别太大、epochs 先少跑几轮验证流程。

用法：
    python train.py [--data <数据集目录>] [--epochs 50] [--imgsz 224] [--batch 32]
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8n 分类训练")
    parser.add_argument("--data", type=str, default="dataset_small", help="数据集根目录（含 train/ 和 val/）")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=224, help="输入图片尺寸（分类一般用 224 或 192）")
    parser.add_argument("--batch", type=int, default=32, help="批大小（CPU 建议 16~32）")
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt", help="预训练分类模型权重")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--patience", type=int, default=20, help="早停轮数，防止过拟合")
    parser.add_argument("--name", type=str, default="asl_demo", help="输出实验名(runs/classify/<name>)")
    return parser.parse_args()


def main():
    args = parse_args()
    data = Path(args.data).resolve()

    # 1. 校验数据集目录结构
    for split in ("train", "val"):
        d = data / split
        if not d.is_dir():
            raise RuntimeError(f"缺少目录: {d}（请先运行 prepare_data.py）")

    print(f"数据集: {data}")
    print(f"训练配置: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}")

    # 2. 加载预训练分类模型（nano 版，CPU 友好）
    #    yolov8n-cls.pt 是 YOLOv8 在 ImageNet 上预训练的分类模型
    #    用它作为起点（迁移学习）比从零训练收敛更快、效果更好
    model = YOLO(args.model)

    # 3. 训练
    #    data 传数据集根目录即可，ultralytics 自动识别 train/val 下的类别文件夹
    #    device="cpu" 强制 CPU
    #    project/name 指定输出目录，便于管理多次实验
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        patience=args.patience,
        device="cpu",
        project="runs/classify",
        name=args.name,
    )

    print(f"\n训练完成！模型保存在 runs/classify/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
