"""数据解析、清洗、标签映射、数据增强与按用户划分。

原始数据位于 MATTRESS_DATA_DIR（即包含 dgs、lcy、nys 等用户子目录的 `睡姿数据`
目录）。每个采集文本每行 24 个逗号分隔的非负整数（对应一行 24 个压力传感器），
连续 44 行合成一帧 44x24 的压力矩阵。

只加载 1-21 号静态睡姿用于分类；`空载`(0) 与 `动态`(22) 不参与静态睡姿分类。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, labels

# 压力值归一化上限：传感器为 8-bit 量程，除以 255 映射到 [0,1]。
NORMALIZE = 255.0


def list_users(data_dir: Path | None = None) -> list[str]:
    """返回按名称排序的用户目录列表（仅目录，排除空载/动态等非目录项）。"""
    root = data_dir or config.DATA_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{root}（请设置 MATTRESS_DATA_DIR）")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def parse_frames(path: Path) -> list[np.ndarray]:
    """把单个采集文本解析为若干 44x24 的 float32 帧（已归一化到 [0,1]）。

    忽略列数不等于 24 的行（例如动态采集文本中的标注行）。尾部不足 44 行的
    残帧被丢弃。
    """
    rows: list[list[int]] = []
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != config.COLS:
            continue
        try:
            rows.append([int(p) for p in parts])
        except ValueError:
            continue
    frames: list[np.ndarray] = []
    for start in range(0, len(rows) - config.ROWS + 1, config.ROWS):
        frame = np.asarray(rows[start:start + config.ROWS], dtype=np.float32) / NORMALIZE
        frames.append(frame)
    return frames


@dataclass
class Dataset:
    """静态睡姿分类数据集（按用户隔离）。"""
    X: np.ndarray          # [N, 44, 24] float32，归一化到 [0,1]
    y: np.ndarray          # [N] int32，粗分类索引 0-3
    posture: np.ndarray    # [N] int32，采集协议序号 1-21
    user_idx: np.ndarray   # [N] int32，用户索引（对应 users 列表）
    users: list[str]       # 排序后的用户名称列表
    files: list[str]       # 每帧来源文件名，便于追溯


def load_dataset(data_dir: Path | None = None) -> Dataset:
    """加载全部用户的静态睡姿帧，返回 Dataset。

    空载与动态采集不纳入；缺失的采集文本会被统计并记录（不中断）。

    文件前缀不一定等于目录名（例如目录 `dwk1` 下文件为 `dwk_1.txt`），因此按
    文件名的数字后缀（1-21）识别静态睡姿序号，而非假设 `{目录名}_{序号}`。
    """
    users = list_users(data_dir)
    root = data_dir or config.DATA_DIR
    X, y, posture, user_idx, files = [], [], [], [], []
    missing: list[str] = []
    for uid, user in enumerate(users):
        folder = root / user
        index_files: dict[int, Path] = {}
        for path in folder.iterdir():
            if path.suffix != ".txt":
                continue
            suffix = path.stem.rsplit("_", 1)[-1]
            if suffix.isdigit() and int(suffix) in labels.STATIC_POSTURES:
                index_files[int(suffix)] = path
        for idx in labels.STATIC_POSTURES:
            if idx not in index_files:
                missing.append(f"{user}/{idx}")
                continue
            candidate = index_files[idx]
            for frame in parse_frames(candidate):
                X.append(frame)
                y.append(labels.posture_to_class(idx))
                posture.append(idx)
                user_idx.append(uid)
                files.append(candidate.name)
    return Dataset(
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int32),
        posture=np.asarray(posture, dtype=np.int32),
        user_idx=np.asarray(user_idx, dtype=np.int32),
        users=users,
        files=files,
    )


def split_users(users: list[str], seed: int = config.SEED, ratio: float = config.TRAIN_RATIO) -> tuple[list[str], list[str]]:
    """按用户级随机切分训练/测试名单（同一用户不会被同时分到两边）。

    返回 (train_users, test_users)。使用固定种子保证可复现。
    """
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(users))
    n_train = max(1, int(round(len(users) * ratio)))
    train = [users[i] for i in order[:n_train]]
    test = [users[i] for i in order[n_train:]]
    return train, test


def split_dataset(dataset: Dataset, seed: int = config.SEED, ratio: float = config.TRAIN_RATIO) -> tuple[dict, dict]:
    """按用户级切分 Dataset，返回 (train_dict, test_dict)。

    每个 dict 含 X / y / posture / user_idx / users。
    """
    users = dataset.users
    train_users, test_users = split_users(users, seed, ratio)
    train_set = {u for u in train_users}
    test_set = {u for u in test_users}
    train_mask = np.array([dataset.users[i] in train_set for i in dataset.user_idx])
    test_mask = ~train_mask
    return (
        {"X": dataset.X[train_mask], "y": dataset.y[train_mask],
         "posture": dataset.posture[train_mask], "user_idx": dataset.user_idx[train_mask],
         "users": train_users},
        {"X": dataset.X[test_mask], "y": dataset.y[test_mask],
         "posture": dataset.posture[test_mask], "user_idx": dataset.user_idx[test_mask],
         "users": test_users},
    )


def augment(X: np.ndarray, seed: int | None = None, noise_sigma: float = 0.012, scale_range: tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
    """对训练帧做标签无关的数据增强。

    1) 高斯噪声：加 N(0, noise_sigma)（约 ±3/255 的传感器噪声）；
    2) 乘性缩放：整帧乘以 [scale_range] 内的随机系数，模拟不同体重压强调制。
    返回与输入同形状的新数组（不修改原数组），随后裁剪回 [0,1]。
    """
    rng = np.random.RandomState(seed)
    out = X.astype(np.float32).copy()
    out += rng.normal(0.0, noise_sigma, size=out.shape).astype(np.float32)
    scale = rng.uniform(*scale_range)
    out *= scale
    return np.clip(out, 0.0, 1.0)
