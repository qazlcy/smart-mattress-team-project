"""睡姿识别训练与评估入口。

用法（PowerShell，需先设置数据目录）：
    $env:MATTRESS_DATA_DIR = 'D:\\path\\to\\睡姿数据'
    python -m sleep_posture.train

会按 70% 用户训练 / 30% 新用户测试，训练 CNN、MLP、SVM 三种模型，
输出准确率、精确率、召回率、F1 与混淆矩阵，结果写入 results/。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config, data, evaluate, labels, models


def _build_split(seed: int, max_users: int | None):
    ds = data.load_dataset()
    if max_users is not None:
        keep = ds.users[:max_users]
        mask = np.array([u in keep for u in [ds.users[i] for i in ds.user_idx]])
        ds = data.Dataset(ds.X[mask], ds.y[mask], ds.posture[mask], ds.user_idx[mask], keep,
                          [f for i, f in enumerate(ds.files) if mask[i]])
    train, test = data.split_dataset(ds, seed=seed)
    return ds, train, test


def _augment_train(train: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """把训练集与其一次增强副本拼接（高斯噪声 + 乘性缩放），标签不变。"""
    X = np.concatenate([train["X"], data.augment(train["X"], seed=seed)], axis=0)
    y = np.concatenate([train["y"], train["y"]], axis=0)
    return X, y


def run(max_users: int | None = None, epochs: int = 15, seed: int = config.SEED,
        output_dir: Path | None = None) -> dict:
    output_dir = output_dir or config.OUTPUT_DIR
    ds, train, test = _build_split(seed, max_users)

    stats = {
        "data_dir": str(config.DATA_DIR),
        "total_users": len(ds.users),
        "total_frames": int(ds.X.shape[0]),
        "class_names": labels.CLASS_NAMES,
        "train_users": train["users"],
        "test_users": test["users"],
        "train_frames": int(train["X"].shape[0]),
        "test_frames": int(test["X"].shape[0]),
        "seed": seed,
        "rows": config.ROWS, "cols": config.COLS,
    }
    print(f"[data] 用户 {len(ds.users)} 训练 {len(train['users'])} 测试 {len(test['users'])}；"
          f"训练帧 {stats['train_frames']} 测试帧 {stats['test_frames']}")

    X_tr, y_tr = _augment_train(train, seed)
    results = {"stats": stats, "models": {}}

    # 1) CNN
    print("[model] 训练 CNN ...")
    r = models.fit_evaluate_torch(models.CNN(), X_tr, y_tr, test["X"], test["y"], cnn=True, epochs=epochs, seed=seed)
    results["models"]["cnn"] = evaluate.classification_metrics(test["y"], r["predictions"])

    # 2) MLP
    print("[model] 训练 MLP ...")
    r = models.fit_evaluate_torch(models.MLP(), X_tr, y_tr, test["X"], test["y"], cnn=False, epochs=epochs, seed=seed)
    results["models"]["mlp"] = evaluate.classification_metrics(test["y"], r["predictions"])

    # 3) SVM（下采样特征，无增强）
    print("[model] 训练 SVM ...")
    r = models.fit_evaluate_svm(train["X"], train["y"], test["X"], test["y"], seed=seed)
    results["models"]["svm"] = evaluate.classification_metrics(test["y"], r["predictions"])

    # 输出结果
    for name, m in results["models"].items():
        print(f"\n===== {name} 测试准确率 {m['accuracy'] * 100:.2f}% =====")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluate.save_json(results, output_dir / "sleep_posture_results.json")
    for name, m in results["models"].items():
        evaluate.save_confusion_matrix_figure(m, output_dir / f"confusion_matrix_{name}.png")
    print(f"\n结果已写入 {output_dir}")
    return results


def main():
    ap = argparse.ArgumentParser(description="睡姿识别训练与评估")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-users", type=int, default=None, help="仅用前 N 个用户（快速验证用）")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    run(max_users=args.max_users, epochs=args.epochs, seed=args.seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
