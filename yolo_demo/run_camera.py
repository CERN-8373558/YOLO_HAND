# -*- coding: utf-8 -*-
"""
run_camera.py — 模型选择器：选一个模型跑 camera_agent.py

用法：
    python run_camera.py

控制台会列出 models/ 下所有 .pt 模型，输入序号选择，
再把所选模型传给 camera_agent.py 启动（自动带上 --use-llm）。

退出后用同命令再选另一个模型即可对比。
"""

import glob
import os
import subprocess
import sys


def pick_model():
    """扫描 models/*.pt，让用户选择，返回所选路径。"""
    models = sorted(glob.glob(os.path.join("models", "*.pt")))
    if not models:
        print("models/ 下没有 .pt 模型文件")
        return None

    print("可选模型：")
    for i, m in enumerate(models, 1):
        name = os.path.basename(m)
        tag = ""
        if "v2" in name:
            tag = "  <- 暗背景v2(30000张, 最新)"
        elif "dark" in name:
            tag = "  <- 暗背景v1(8000张)"
        elif "asl_demo-6" in name:
            tag = "  <- 旧版浅背景(标准ASL)"
        print(f"  {i}. {name}{tag}")

    while True:
        try:
            s = input(f"\n输入序号(1-{len(models)})，Enter 回车运行，q 取消: ").strip()
        except EOFError:
            return None
        if s.lower() in ("q", "quit", ""):
            return None
        if s.isdigit() and 1 <= int(s) <= len(models):
            return models[int(s) - 1]
        print("输入无效，请重新输入")


def main():
    # 确保工作目录是 yolo_demo
    if not os.path.isdir("models"):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    model = pick_model()
    if not model:
        print("已取消")
        return

    print(f"\n使用模型: {model}")
    print("启动 camera_agent.py（LLM 增强已开），按 q 退出…\n")

    cmd = [sys.executable, "camera_agent.py",
           "--model", model, "--use-llm"]
    try:
        subprocess.call(cmd)
    except KeyboardInterrupt:
        pass
    print("\n已退出。重新运行本脚本可换模型。")


if __name__ == "__main__":
    main()
