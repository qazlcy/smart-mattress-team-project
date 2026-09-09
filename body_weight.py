"""Body-region segmentation and weight prediction helpers.

The module is deliberately standard-library only.  It can run as a transparent
baseline without local course data, and improves calibration when
MATTRESS_DATA_DIR points at the private course dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import re
import json
from statistics import mean, median, pstdev

ROWS, COLS = 44, 24
BODY_PARTS = [
    ("shoulder", "肩部", 0.00, 0.13),
    ("back", "背部", 0.13, 0.27),
    ("waist", "腰部", 0.27, 0.40),
    ("hip", "臀部", 0.40, 0.58),
    ("thigh", "大腿", 0.58, 0.78),
    ("calf", "小腿", 0.78, 1.00),
]
WEIGHT_BINS = [
    (0, 55, "45-55 kg"),
    (55, 65, "55-65 kg"),
    (65, 75, "65-75 kg"),
    (75, 85, "75-85 kg"),
    (85, 95, "85-95 kg"),
    (95, 105, "95-105 kg"),
    (105, 200, "105 kg以上"),
]


def corrected_frame(frame: list[list[int]], empty_baseline: list[list[float]] | None = None) -> list[list[float]]:
    if not empty_baseline:
        return [[float(value) for value in row] for row in frame]
    return [
        [max(0.0, float(value) - empty_baseline[row_index][col_index]) for col_index, value in enumerate(row)]
        for row_index, row in enumerate(frame)
    ]


def pressure_features(frame: list[list[int]], empty_baseline: list[list[float]] | None = None) -> dict[str, float]:
    corrected = corrected_frame(frame, empty_baseline)
    flat = [value for row in corrected for value in row]
    max_pressure = max(flat) if flat else 0.0
    threshold = max(15.0, max_pressure * 0.08)
    active = [(r, c, value) for r, row in enumerate(corrected) for c, value in enumerate(row) if value >= threshold]
    total = sum(value for _, _, value in active)
    area = len(active)
    if not active:
        return {
            "total_load": 0.0, "sqrt_load": 0.0, "active_area": 0.0, "mean_pressure": 0.0,
            "max_pressure": max_pressure, "row_min": 0.0, "row_max": float(ROWS - 1),
            "col_min": 0.0, "col_max": float(COLS - 1), "center_row": ROWS / 2, "center_col": COLS / 2,
        }
    row_min, row_max = min(r for r, _, _ in active), max(r for r, _, _ in active)
    col_min, col_max = min(c for _, c, _ in active), max(c for _, c, _ in active)
    return {
        "total_load": total,
        "sqrt_load": sqrt(total),
        "active_area": float(area),
        "mean_pressure": total / area,
        "max_pressure": max_pressure,
        "row_min": float(row_min),
        "row_max": float(row_max),
        "col_min": float(col_min),
        "col_max": float(col_max),
        "center_row": sum(r * value for r, _, value in active) / max(total, 1.0),
        "center_col": sum(c * value for _, c, value in active) / max(total, 1.0),
    }


def feature_vector(frame: list[list[int]], empty_baseline: list[list[float]] | None = None) -> list[float]:
    features = pressure_features(frame, empty_baseline)
    return [
        features["sqrt_load"],
        features["active_area"],
        features["mean_pressure"],
        features["max_pressure"],
        features["center_row"],
        features["center_col"],
    ]


def estimate_body_regions(frame: list[list[int]], empty_baseline: list[list[float]] | None = None) -> list[dict]:
    corrected = corrected_frame(frame, empty_baseline)
    features = pressure_features(frame, empty_baseline)
    row_min, row_max = int(features["row_min"]), int(features["row_max"])
    col_min, col_max = int(features["col_min"]), int(features["col_max"])
    body_height = max(1, row_max - row_min + 1)
    body_width = max(1, col_max - col_min + 1)
    total_load = max(sum(sum(row) for row in corrected), 1.0)
    regions = []
    for key, label, start_ratio, end_ratio in BODY_PARTS:
        start_row = max(0, min(ROWS - 1, round(row_min + body_height * start_ratio)))
        end_row = max(start_row + 1, min(ROWS, round(row_min + body_height * end_ratio)))
        pad = 1 if key in {"hip", "thigh", "calf"} else 0
        start_col = max(0, col_min - pad)
        end_col = min(COLS, col_max + 1 + pad)
        values = [corrected[r][c] for r in range(start_row, end_row) for c in range(start_col, end_col)]
        load = sum(values)
        regions.append({
            "key": key,
            "label": label,
            "startRow": start_row,
            "endRow": end_row,
            "startCol": start_col,
            "endCol": end_col,
            "meanPressure": round(load / max(len(values), 1), 1),
            "maxPressure": round(max(values) if values else 0.0, 1),
            "loadShare": round(load / total_load * 100, 1),
        })
    return regions


@dataclass
class BodyRegionCalibrator:
    samples: list[tuple[list[float], list[tuple[int, int, int, int]]]]
    means: list[float]
    scales: list[float]
    train_users: list[str]
    test_users: list[str]

    @classmethod
    def fallback(cls) -> "BodyRegionCalibrator":
        return cls([], [], [], [], [])

    @classmethod
    def from_dataset(cls, data_root: Path) -> "BodyRegionCalibrator":
        json_path = find_region_json(data_root)
        if not json_path:
            return cls.fallback()
        records = json.loads(json_path.read_text(encoding="utf-8"))
        parsed = []
        for record in records:
            regions = parse_region_label(record.get("region", ""))
            if not regions:
                continue
            frame = parse_json_frame(record["data"])
            parsed.append((record.get("people_name", ""), feature_vector(frame), regions))
        users = sorted({user for user, _, _ in parsed if user})
        if not parsed:
            return cls.fallback()
        split = max(1, round(len(users) * 0.7))
        train_users, test_users = users[:split], users[split:]
        train = [(features, regions) for user, features, regions in parsed if not train_users or user in train_users]
        if not train:
            train = [(features, regions) for _, features, regions in parsed]
        means = [mean(features[i] for features, _ in train) for i in range(len(train[0][0]))]
        scales = [pstdev(features[i] for features, _ in train) or 1.0 for i in range(len(train[0][0]))]
        return cls(train, means, scales, train_users, test_users)

    def predict(self, frame: list[list[int]], empty_baseline: list[list[float]] | None = None) -> list[dict]:
        if not self.samples:
            return estimate_body_regions(frame, empty_baseline)
        vector = feature_vector(frame, empty_baseline)
        _, label_regions = min(
            self.samples,
            key=lambda sample: sum(((vector[i] - sample[0][i]) / self.scales[i]) ** 2 for i in range(len(vector))),
        )
        corrected = corrected_frame(frame, empty_baseline)
        total_load = max(sum(sum(row) for row in corrected), 1.0)
        result = []
        for (key, label, _, _), (start_row, end_row, start_col, end_col) in zip(BODY_PARTS, label_regions):
            start_row, end_row = max(0, start_row), min(ROWS, end_row)
            start_col, end_col = max(0, start_col), min(COLS, end_col)
            values = [corrected[r][c] for r in range(start_row, end_row) for c in range(start_col, end_col)]
            load = sum(values)
            result.append({
                "key": key,
                "label": label,
                "startRow": start_row,
                "endRow": end_row,
                "startCol": start_col,
                "endCol": end_col,
                "meanPressure": round(load / max(len(values), 1), 1),
                "maxPressure": round(max(values) if values else 0.0, 1),
                "loadShare": round(load / total_load * 100, 1),
            })
        return result


def weight_interval(weight: float) -> dict:
    for index, (low, high, label) in enumerate(WEIGHT_BINS):
        if low <= weight < high:
            return {"index": index, "label": label, "low": low, "high": high}
    low, high, label = WEIGHT_BINS[-1]
    return {"index": len(WEIGHT_BINS) - 1, "label": label, "low": low, "high": high}


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for pivot in range(n):
        best = max(range(pivot, n), key=lambda r: abs(aug[r][pivot]))
        aug[pivot], aug[best] = aug[best], aug[pivot]
        divisor = aug[pivot][pivot] or 1e-9
        aug[pivot] = [value / divisor for value in aug[pivot]]
        for row in range(n):
            if row == pivot:
                continue
            factor = aug[row][pivot]
            aug[row] = [value - factor * aug[pivot][col] for col, value in enumerate(aug[row])]
    return [aug[row][-1] for row in range(n)]


@dataclass
class WeightCalibrator:
    coefficients: list[float]
    heights: dict[str, float]
    train_users: list[str]
    test_users: list[str]
    metadata_count: int

    @classmethod
    def fallback(cls) -> "WeightCalibrator":
        return cls([47.0, 0.28, 0.0, 0.0, 0.0, 0.0], {}, [], [], 0)

    @classmethod
    def from_dataset(cls, data_dir: Path, metadata_path: Path | None = None) -> "WeightCalibrator":
        metadata = parse_body_metadata(metadata_path or find_metadata_file(data_dir))
        weights = {user: item["weight"] for user, item in metadata.items()}
        heights = {user: item["height"] for user, item in metadata.items()}
        users = sorted(user for user in weights if (data_dir / user).is_dir())
        if len(users) < 4:
            return cls.fallback()
        split = max(1, round(len(users) * 0.7))
        train_users, test_users = users[:split], users[split:]
        samples = []
        for user in train_users:
            feature_rows = sample_user_features(data_dir / user)
            if not feature_rows:
                continue
            med = median_feature_row(feature_rows)
            height = heights.get(user, 175.0)
            samples.append((weight_feature_row(med, height), weights[user]))
        if len(samples) < 4:
            return cls.fallback()
        width = len(samples[0][0])
        xtx = [[0.0 for _ in range(width)] for _ in range(width)]
        xty = [0.0 for _ in range(width)]
        for features, target in samples:
            for r in range(width):
                xty[r] += features[r] * target
                for c in range(width):
                    xtx[r][c] += features[r] * features[c]
        for i in range(width):
            xtx[i][i] += 0.01
        return cls(solve_linear_system(xtx, xty), heights, train_users, test_users, len(weights))

    def predict(self, frame: list[list[int]], empty_baseline: list[list[float]] | None = None, user: str = "") -> dict:
        features = pressure_features(frame, empty_baseline)
        row = weight_feature_row(features, self.heights.get(user, 175.0))
        weight = sum(coef * value for coef, value in zip(self.coefficients, row))
        weight = min(130.0, max(40.0, weight))
        interval = weight_interval(weight)
        confidence = "calibrated" if self.metadata_count else "fallback"
        return {
            "kg": round(weight, 1),
            "interval": interval,
            "source": "用户级压力回归+空载校正" if self.metadata_count else "演示压力启发式估计",
            "confidence": confidence,
        }


def parse_weights(path: Path | None) -> dict[str, float]:
    return {user: item["weight"] for user, item in parse_body_metadata(path).items()}


def parse_body_metadata(path: Path | None) -> dict[str, dict[str, float]]:
    if not path or not path.is_file():
        return {}
    metadata: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 3 and re.fullmatch(r"[A-Za-z0-9]+", parts[0]):
            try:
                metadata[parts[0]] = {"height": float(parts[1]), "weight": float(parts[2])}
            except ValueError:
                pass
    return metadata


def weight_feature_row(features: dict[str, float], height_cm: float) -> list[float]:
    return [
        1.0,
        features["sqrt_load"],
        features["max_pressure"],
        features["sqrt_load"] / max(height_cm, 1.0),
        features["center_row"],
        features["center_col"],
    ]


def find_metadata_file(data_dir: Path) -> Path | None:
    for candidate in (data_dir / "readme", data_dir.parent / "readme"):
        if candidate.is_file():
            return candidate
    return None


def find_region_json(data_root: Path) -> Path | None:
    roots = [data_root, data_root.parent]
    for root in roots:
        for path in root.rglob("data.json"):
            if path.is_file():
                return path
    return None


def parse_region_label(value: str) -> list[tuple[int, int, int, int]] | None:
    parts = value.split()
    if len(parts) != 24 or any(part.lower() == "na" for part in parts):
        return None
    nums = [int(part) for part in parts]
    return [(nums[12 + 2 * i], nums[13 + 2 * i], nums[2 * i], nums[2 * i + 1]) for i in range(6)]


def parse_json_frame(value: str) -> list[list[int]]:
    nums = [int(float(part)) for part in value.split(",")]
    return [nums[i:i + COLS] for i in range(0, ROWS * COLS, COLS)]


def parse_course_frames(path: Path, limit: int = 12) -> list[list[list[int]]]:
    rows: list[list[int]] = []
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        values = [int(value) for value in re.findall(r"\d+", line)]
        if len(values) == COLS:
            rows.append(values)
    return [[rows[i + r] for r in range(ROWS)] for i in range(0, len(rows) - ROWS + 1, ROWS)][:limit]


def average_empty_baseline(user_dir: Path) -> list[list[float]] | None:
    files = list(user_dir.glob("*空载*.txt"))
    if not files:
        return None
    frames = parse_course_frames(files[0], limit=8)
    if not frames:
        return None
    return [
        [sum(frame[r][c] for frame in frames) / len(frames) for c in range(COLS)]
        for r in range(ROWS)
    ]


def sample_user_features(user_dir: Path) -> list[dict[str, float]]:
    baseline = average_empty_baseline(user_dir)
    frames: list[list[list[int]]] = []
    for path in sorted(user_dir.glob("*.txt")):
        if "空载" in path.name or "动态" in path.name:
            continue
        frames.extend(parse_course_frames(path, limit=3))
        if len(frames) >= 18:
            break
    return [pressure_features(frame, baseline) for frame in frames]


def median_feature_row(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: median(row[key] for row in rows) for key in keys}
