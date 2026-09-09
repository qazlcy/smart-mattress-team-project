"""睡姿标签字典：细分类（采集协议序号）-> 粗分类（sleep_pose）。

采集协议（见《睡姿采集2026.docx》）定义了 21 种静态睡姿，分为四大类；
新版区域划分数据集的 `sleep_pose` 字段同样采用四分类：
    0 - 仰卧  1 - 俯卧  2 - 左侧卧  3 - 右侧卧

本模块提供细分类名称、粗分类映射，以及稳定有序的类别输出顺序。
"""
from __future__ import annotations

# 粗分类（sleep_pose）：类别索引 -> 中文名称。顺序即模型输出顺序。
CLASS_NAMES = ["仰卧", "俯卧", "左侧卧", "右侧卧"]

# 采集协议序号 -> 细分类睡姿名称。0 为空载，22 为动态采集，均不参与静态分类。
POSTURE_NAMES = {
    0: "空载",
    1: "标准仰卧", 2: "左腿支撑", 3: "支撑翘腿", 4: "双腿交叉", 5: "双腿支撑", 6: "单腿外展",
    7: "直腿俯卧", 8: "张腿俯卧", 9: "双臂支撑",
    10: "左直腿卧", 11: "左屈腿卧", 12: "左腿展开", 13: "左婴儿卧", 14: "左腿前屈", 15: "左侧俯卧",
    16: "右直腿卧", 17: "右屈腿卧", 18: "右腿展开", 19: "右婴儿卧", 20: "右腿前屈", 21: "右侧俯卧",
    22: "动态采集",
}

# 细分类序号 -> 粗分类索引。仰卧 1-6、俯卧 7-9、左侧卧 10-15、右侧卧 16-21。
_POSTURE_TO_CLASS = {}
for _i in range(1, 7):    # 仰卧类
    _POSTURE_TO_CLASS[_i] = 0
for _i in range(7, 10):   # 俯卧类
    _POSTURE_TO_CLASS[_i] = 1
for _i in range(10, 16):  # 左侧卧类
    _POSTURE_TO_CLASS[_i] = 2
for _i in range(16, 22):  # 右侧卧类
    _POSTURE_TO_CLASS[_i] = 3


def posture_to_class(index: int) -> int:
    """把采集协议序号（1-21）映射为粗分类索引（0-3）。"""
    if index not in _POSTURE_TO_CLASS:
        raise ValueError(f"无对应粗分类的采集序号: {index}")
    return _POSTURE_TO_CLASS[index]


def class_name(index: int) -> str:
    return CLASS_NAMES[index]


# 参与静态睡姿分类的采集序号列表（1-21）。
STATIC_POSTURES = sorted(_POSTURE_TO_CLASS)
