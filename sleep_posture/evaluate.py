"""分类评估：准确率、精确率、召回率、F1 与混淆矩阵。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from . import labels


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算整体与各类别指标。

    返回 dict：accuracy、macro/weighted 精确率/召回率/F1、各类别逐项、
    混淆矩阵（按 CLASS_NAMES 顺序）。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    class_names = labels.CLASS_NAMES
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)
    macro_p, macro_r, macro_f1 = float(p.mean()), float(r.mean()), float(f1.mean())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return {
        "accuracy": acc,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "per_class": [
            {"class": class_names[i], "precision": float(p[i]), "recall": float(r[i]),
             "f1": float(f1[i]), "support": int(support[i])}
            for i in range(len(class_names))
        ],
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def format_report(metrics: dict) -> str:
    """把指标 dict 渲染为可读的 Markdown 表格文本。"""
    lines = [f"整体准确率：**{metrics['accuracy'] * 100:.2f}%**",
             "",
             "| 类别 | 精确率 | 召回率 | F1 | 样本数 |",
             "| --- | --- | --- | --- | --- |"]
    for row in metrics["per_class"]:
        lines.append(f"| {row['class']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |")
    m = metrics["macro"]
    lines.append(f"| 宏平均 | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | — |")
    lines.append("")
    lines.append("混淆矩阵（行=真实，列=预测）：")
    lines.append("")
    header = "| 真实\\预测 | " + " | ".join(metrics["class_names"]) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join(["---"] * len(metrics["class_names"])) + " |")
    for i, row in enumerate(metrics["confusion_matrix"]):
        lines.append(f"| {metrics['class_names'][i]} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def save_confusion_matrix_figure(metrics: dict, path: Path) -> None:
    """把混淆矩阵渲染为 PNG（供报告引用）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Windows 下使用中文字体，避免混淆矩阵标签显示为方框。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sum, 1)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics["class_names"])), metrics["class_names"])
    ax.set_yticks(range(len(metrics["class_names"])), metrics["class_names"])
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_json(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
