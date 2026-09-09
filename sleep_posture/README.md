# sleep_posture —— 数据处理与睡姿识别

第三组成员宁耀硕的“数据处理与睡姿识别”模块。完成课程原始数据的解析、清洗、
标签映射、数据增强与按用户划分，并训练对比三种睡姿识别算法（CNN、MLP、SVM）。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `config.py` | 传感器矩阵尺寸（44×24）、随机种子、用户划分比例 |
| `labels.py` | 标签字典：21 种细分类睡姿 → 4 种粗分类（仰卧/俯卧/左侧卧/右侧卧） |
| `data.py` | 解析、清洗、数据增强、按用户 70/30 划分 |
| `models.py` | CNN / MLP / SVM 三种模型及其训练入口 |
| `evaluate.py` | 准确率、精确率、召回率、F1 与混淆矩阵 |
| `train.py` | 训练与评估总入口 |

## 环境

Python 3.9 + numpy + torch（CPU）+ scikit-learn + matplotlib。

## 运行

```powershell
$env:MATTRESS_DATA_DIR = 'D:\path\to\睡姿数据'
python -m sleep_posture.train --epochs 15
```

结果写入 `results/`：`sleep_posture_results.json`（完整指标）、
`confusion_matrix_*.png`（混淆矩阵图）和 `cnn_state_dict.pt`（供集成端通过
`MATTRESS_CNN_WEIGHTS` 加载的模型权重；不提交到公开仓库）。

快速验证（仅用前 6 个用户、2 个 epoch）：

```powershell
python -m sleep_posture.train --max-users 6 --epochs 2
```

## 数据说明

- 传感器矩阵为 **44 行 × 24 列**，原始文本每行 24 个值，连续 44 行合成一帧。
- 只使用 1-21 号静态睡姿；`空载`（0）与 `动态`（22）不参与静态分类。
- 部分目录的文件前缀与目录名不一致（`dwk1/` → `dwk_1.txt`），解析时按数字后缀识别。
- 训练/测试按 **用户隔离** 划分（70% 用户训练、30% 新用户测试），随机种子 42。
