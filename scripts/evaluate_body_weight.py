from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from body_weight import BodyRegionCalibrator, WeightCalibrator, parse_body_metadata, weight_feature_row, weight_interval


def parse_region(value: str) -> list[tuple[int, int, int, int]] | None:
    parts = value.split()
    if len(parts) != 24 or any(part.lower() == "na" for part in parts):
        return None
    nums = [int(part) for part in parts]
    return [(nums[12 + 2 * i], nums[13 + 2 * i], nums[2 * i], nums[2 * i + 1]) for i in range(6)]


def parse_frame(value: str) -> list[list[int]]:
    nums = [int(float(part)) for part in value.split(",")]
    return [nums[i:i + 24] for i in range(0, 56 * 24, 24)]


def locate_default_data_root(project_root: Path) -> Path | None:
    for candidate in project_root.parent.glob("*data"):
        if (candidate / "readme").is_file() and list(candidate.rglob("data.json")):
            return candidate
    return None


def evaluate_regions(data_root: Path) -> dict:
    json_path = next(data_root.rglob("data.json"))
    records = json.loads(json_path.read_text(encoding="utf-8"))
    model = BodyRegionCalibrator.from_dataset(data_root)
    test_users = set(model.test_users) or {record.get("people_name", "") for record in records}
    total = 0
    hits = 0
    known_total = 0
    known_hits = 0
    for record in records:
        expected = parse_region(record.get("region", ""))
        if not expected:
            continue
        predicted = model.predict(parse_frame(record["data"]))
        is_test = record.get("people_name", "") in test_users
        for item, target in zip(predicted, expected):
            row_start, row_end, col_start, col_end = target
            matched = (
                abs(item["startRow"] - row_start) <= 3
                and abs(item["endRow"] - row_end) <= 4
                and abs(item["startCol"] - col_start) <= 4
                and abs(item["endCol"] - col_end) <= 4
            )
            if is_test:
                total += 1
            else:
                known_total += 1
            if matched and is_test:
                hits += 1
            elif matched:
                known_hits += 1
    return {
        "known_users": model.train_users,
        "known_user_parts": known_total,
        "known_user_accuracy": round(known_hits / known_total * 100, 2) if known_total else 0.0,
        "test_users": sorted(test_users),
        "parts": total,
        "new_user_accuracy": round(hits / total * 100, 2) if total else 0.0,
    }


def evaluate_weight(data_root: Path) -> dict:
    data_dir = data_root / "睡姿数据"
    if not data_dir.is_dir():
        data_dir = next(path for path in data_root.iterdir() if path.is_dir() and any(path.glob("*/*.txt")))
    calibrator = WeightCalibrator.from_dataset(data_dir)
    metadata = parse_body_metadata(data_root / "readme")
    weights = {user: item["weight"] for user, item in metadata.items()}
    errors = []
    interval_hits = 0
    adjacent_hits = 0
    far_misses = 0
    users = [user for user in calibrator.test_users if user in weights]
    for user in users:
        feature_rows = []
        from body_weight import sample_user_features, median_feature_row

        feature_rows = sample_user_features(data_dir / user)
        if not feature_rows:
            continue
        med = median_feature_row(feature_rows)
        row = weight_feature_row(med, metadata.get(user, {}).get("height", 175.0))
        predicted = min(130.0, max(40.0, sum(coef * value for coef, value in zip(calibrator.coefficients, row))))
        actual = weights[user]
        errors.append(abs(predicted - actual))
        pred_bin = weight_interval(predicted)["index"]
        actual_bin = weight_interval(actual)["index"]
        diff = abs(pred_bin - actual_bin)
        if diff == 0:
            interval_hits += 1
        if diff <= 1 or abs(predicted - actual) <= 3:
            adjacent_hits += 1
        if diff >= 2:
            far_misses += 1
    count = len(errors)
    return {
        "test_users": count,
        "mae_kg": round(sum(errors) / count, 2) if count else 0.0,
        "interval_or_adjacent_hit_rate": round(adjacent_hits / count * 100, 2) if count else 0.0,
        "far_miss_rate": round(far_misses / count * 100, 2) if count else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).parents[1]
    data_root = args.data_root or locate_default_data_root(project_root)
    if not data_root:
        print("未找到本地课程数据。请使用 --data-root 指向包含 readme、睡姿数据、区域划分 的目录。")
        return 2
    result = {
        "data_root": str(data_root),
        "region": evaluate_regions(data_root),
        "weight": evaluate_weight(data_root),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
