# -*- coding: utf-8 -*-
"""
predict.py — 手语字母识别推理脚本（T8）

功能：
    用训练好的 best.pt 模型识别图片/文件夹/视频/摄像头里的手语字母，
    显示类别和置信度，并可以把结果画在图上保存。

用法：
    python predict.py <图片或文件夹路径>
    python predict.py <视频路径>
    python predict.py 0                  # 0 表示摄像头
    可选：--model <权重路径>  --conf <置信度阈值>  --save

学习要点：
    推理 = 加载模型 → 喂一张图 → 得到每个类别的概率 → 取最高的作为答案
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="手语字母识别")
    parser.add_argument("source", type=str, help="图片/文件夹/视频路径，或 0（摄像头）")
    parser.add_argument("--model", type=str, default="models/asl_demo-6-best.pt",
                        help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值，低于此值不算识别成功")
    parser.add_argument("--save", action="store_true", help="是否保存结果图（到 runs/predict/）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载模型
    #    分类模型推理一行代码即可：model(source)
    #    内部自动完成：图像缩放 → 归一化 → 网络前向 → softmax 概率
    model = YOLO(args.model)
    print(f"模型: {args.model} 加载完成")

    # 2. 推理
    #    stream=True 表示按流式逐张/逐帧处理，内存占用小，适合大文件夹和视频
    results = model(
        args.source,
        conf=args.conf,
        stream=True,
        save=args.save,
        project="runs/predict",
        name="results",
    )

    # 3. 逐个结果输出
    for i, r in enumerate(results):
        # r.names 是类别名列表，r.probs 是每个类别的概率
        names = r.names
        probs = r.probs.data.tolist()  # 与 names 一一对应的概率

        # 找出概率最高的类别
        top_idx = probs.index(max(probs))
        top_cls = names[top_idx]
        top_prob = probs[top_idx]

        # 显示结果（也可改成中文提示）
        print(f"[{i+1}] 识别结果: {top_cls}  (置信度 {top_prob:.2%})")

        # 低于阈值的提示（模型"不确定"）
        if top_prob < args.conf:
            print(f"     ⚠ 置信度低于阈值，模型可能不确定，请换更清晰的手势")


if __name__ == "__main__":
    main()
