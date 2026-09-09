"""睡姿识别模型定义与训练入口。

提供三种算法：
1. CNN —— 2D 卷积神经网络（PyTorch），输入 44x24 压力图；
2. MLP —— 前馈神经网络（PyTorch），输入展平后的 1056 维；
3. SVM —— 支持向量机（scikit-learn），输入 2x2 下采样后的 264 维特征。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import config

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    """轻量 2D 卷积神经网络：输入 [N,1,44,24]，输出 [N,4] 类别 logits。"""

    def __init__(self, n_classes: int = config.N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MLP(nn.Module):
    """前馈神经网络：输入 [N,1056]，输出 [N,4] 类别 logits。"""

    def __init__(self, n_classes: int = config.N_CLASSES, in_features: int = config.ROWS * config.COLS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _as_input_tensor(X: np.ndarray, cnn: bool) -> torch.Tensor:
    t = torch.from_numpy(X)
    if cnn:
        return t.unsqueeze(1)  # [N,1,44,24]
    return t.view(t.shape[0], -1)  # [N,1056]


def fit_evaluate_torch(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray, cnn: bool,
                       epochs: int = 20, batch_size: int = 128, lr: float = 1e-3,
                       seed: int = config.SEED) -> dict:
    """训练并评估一个 PyTorch 模型，返回预测结果与训练历史。"""
    torch.manual_seed(seed)
    model = model.to(_DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    X_train_t = _as_input_tensor(X_train, cnn)
    n = X_train_t.shape[0]
    history: list[float] = []
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train_t[idx].to(_DEVICE)
            yb = y_train_t[idx].to(_DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * len(idx)
        history.append(epoch_loss / n)
    model.eval()
    pred = _predict_torch(model, X_test, cnn)
    # Keep a CPU copy so training can export a portable inference checkpoint.
    state_dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return {"predictions": pred, "history": history, "state_dict": state_dict}


def _predict_torch(model: nn.Module, X: np.ndarray, cnn: bool, batch_size: int = 256) -> np.ndarray:
    model = model.to(_DEVICE)
    X_t = _as_input_tensor(X, cnn)
    out = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            logits = model(X_t[i:i + batch_size].to(_DEVICE))
            out.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.int64)


def downsample_features(X: np.ndarray, factor: int = 2) -> np.ndarray:
    """2x2 分块平均下采样，44x24 -> 22x12（264 维），供 SVM 使用。"""
    n, r, c = X.shape
    nr, nc = r // factor, c // factor
    X = X[:, :nr * factor, :nc * factor]
    return X.reshape(n, nr, factor, nc, factor).mean(axis=(2, 4)).reshape(n, nr * nc)


def fit_evaluate_svm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                     seed: int = config.SEED, max_train: int = 8000) -> dict:
    """训练并评估 SVM（RBF 核，2x2 下采样特征，标准化）。"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # 特征下采样 + 标准化
    Xtr = downsample_features(X_train)
    Xte = downsample_features(X_test)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)

    # 训练样本过多时均匀子采样，控制训练时间（4 类高度可分，子采样足够）。
    if Xtr.shape[0] > max_train:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(Xtr.shape[0])[:max_train]
        Xtr, y_train = Xtr[idx], y_train[idx]

    clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
    clf.fit(Xtr, y_train)
    return {"predictions": clf.predict(Xte).astype(np.int64), "history": [], "clf": clf}
