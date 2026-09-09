"""睡姿识别模块全局配置。

传感器矩阵为 44 行 x 24 列（新版采集格式），每帧由连续 44 行、每行 24 个
压力值构成。本模块与 server.py 共用 MATTRESS_DATA_DIR 环境变量定位原始数据
根目录（即包含各用户子目录的 `睡姿数据` 目录）。
"""
from __future__ import annotations

import os
from pathlib import Path

# 传感器矩阵尺寸：44 行 x 24 列。原始 txt 每行 24 个值，44 行合成一帧。
ROWS = 44
COLS = 24

# 固定随机种子，保证用户划分、数据增强、权重初始化可复现。
SEED = 42

# 用户级训练/验证划分比例（课程要求 70% 用户训练、30% 新用户测试）。
TRAIN_RATIO = 0.7

# 睡姿粗分类类别数：仰卧 / 俯卧 / 左侧卧 / 右侧卧。
N_CLASSES = 4

# 原始压力值为 10-bit 量程；实测课程数据最大值可超过 255，不能按 8-bit 处理。
PRESSURE_FULL_SCALE = 1023.0

# 原始数据目录，通过环境变量 MATTRESS_DATA_DIR 指定（与 server.py 一致）。
DATA_DIR = Path(os.environ.get("MATTRESS_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))

# 训练结果与模型输出目录（不提交到公开仓库，见 .gitignore）。
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"
