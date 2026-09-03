# -*- coding: utf-8 -*-
"""
prepare_data.py — ASL 手语字母数据集准备脚本（T5）

功能：
    1. 读取 ASL Alphabet 原始目录（每类一个文件夹，目录名即类别标签）
    2. 按比例随机划分 train / val
    3. 输出成 YOLO 分类模式需要的数据集结构

为什么用"分类模式"而不是"检测模式"？
    - ASL 数据是 200x200 单手势特写、手势居中，没有人工画框标注。
    - 分类模式 = 整张图属于一个类别，无需标注框，训练更快，CPU 也能跑。
    - 检测模式需要每张图有 (x,y,w,h) 框，适合"图中多个物体"的场景，
      对"整张图一个手势"来说分类是更合理的选择。

用法：
    python prepare_data.py --src <源目录> --out <输出目录> [--val-ratio 0.2]
    可选：--limit 每类最多取多少张（学习/测试时减小样本量，加快速度）
"""

import os
import random
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ASL 数据集划分工具")
    parser.add_argument("--src", type=str, required=True, help="ASL 原始数据目录（含各类子目录）")
    parser.add_argument("--out", type=str, required=True, help="输出数据集根目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，默认 0.2")
    parser.add_argument("--limit", type=int, default=0, help="每类最多取多少张（0=不限制，全部使用）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # 1. 找出所有类别（子目录名就是类别标签）
    classes = sorted(
        d for d in os.listdir(args.src)
        if os.path.isdir(os.path.join(args.src, d))
    )
    if not classes:
        raise RuntimeError("源目录中没有找到任何类别子目录")

    print(f"检测到 {len(classes)} 个类别: {classes}")

    # 2. 创建输出目录结构: out/train/<类别>/  和  out/val/<类别>/
    for split in ("train", "val"):
        for cls in classes:
            os.makedirs(os.path.join(args.out, split, cls), exist_ok=True)

    # 3. 逐类划分并复制文件
    total_train = 0
    total_val = 0
    for cls in classes:
        src_cls_dir = os.path.join(args.src, cls)
        files = sorted(
            f for f in os.listdir(src_cls_dir)
            if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")
        )

        # 若设置了 --limit，只取前 N 张（先打乱再截取，保证随机）
        if args.limit and len(files) > args.limit:
            random.shuffle(files)
            files = files[: args.limit]

        random.shuffle(files)  # 打乱后再按比例切分
        n_val = int(len(files) * args.val_ratio)
        val_files = files[:n_val]
        train_files = files[n_val:]

        for f in train_files:
            shutil.copy2(
                os.path.join(src_cls_dir, f),
                os.path.join(args.out, "train", cls, f),
            )
        for f in val_files:
            shutil.copy2(
                os.path.join(src_cls_dir, f),
                os.path.join(args.out, "val", cls, f),
            )

        total_train += len(train_files)
        total_val += len(val_files)
        print(f"  {cls}: train={len(train_files)} val={len(val_files)}")

    print("-" * 40)
    print(f"完成! 训练集 {total_train} 张, 验证集 {total_val} 张")
    print(f"输出目录: {args.out}")

    # 4. 生成 data.yaml（YOLO 分类模式用不到，但为了后续训练脚本统一管理类别名，先导出一份类别清单）
    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as fp:
        for cls in classes:
            fp.write(cls + "\n")
    print("已生成类别清单: classes.txt")


if __name__ == "__main__":
    main()
