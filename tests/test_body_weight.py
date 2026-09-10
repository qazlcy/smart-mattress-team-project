import tempfile
import unittest
from pathlib import Path

from body_weight import (COLS, ROWS, WeightCalibrator, corrected_frame,
                         augment_region_sample, find_region_json, weight_interval)


class BodyWeightContractTest(unittest.TestCase):
    def test_frame_shape_is_validated(self):
        with self.assertRaises(ValueError):
            corrected_frame([[0] * COLS for _ in range(ROWS - 1)])

    def test_weight_bins_cover_upper_range(self):
        interval = weight_interval(250.0)
        self.assertEqual(interval["index"], 6)
        self.assertIsNone(interval["high"])

    def test_out_of_range_height_uses_load_only_guard(self):
        model = WeightCalibrator(
            coefficients=[100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            load_only_coefficients=[50.0, 0.001],
            interval_samples=[],
            interval_scales=[],
            heights={"new": 195.0},
            train_users=["known"],
            test_users=["new"],
            metadata_count=2,
            training_height_range=(165.0, 190.0),
        )
        features = {
            "active_area": 10.0,
            "total_load": 20_000.0,
            "sqrt_load": 141.4,
            "max_pressure": 100.0,
            "center_row": 20.0,
            "center_col": 12.0,
        }
        result = model.predict_features(features, "new")
        self.assertEqual(result["kg"], 70.0)
        self.assertEqual(result["confidence"], "calibrated_extrapolation_guard")

    def test_height_below_training_range_does_not_use_load_guard(self):
        model = WeightCalibrator(
            coefficients=[68.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            load_only_coefficients=[100.0, 0.0],
            interval_samples=[],
            interval_scales=[],
            heights={"new": 160.0},
            train_users=["known"],
            test_users=["new"],
            metadata_count=2,
            training_height_range=(165.0, 190.0),
        )
        features = {
            "active_area": 10.0, "total_load": 20_000.0, "sqrt_load": 141.4,
            "max_pressure": 100.0, "center_row": 20.0, "center_col": 12.0,
        }
        self.assertEqual(model.predict_features(features, "new")["kg"], 68.0)

    def test_ordinal_calibration_only_moves_to_one_adjacent_interval(self):
        features = {
            "active_area": 200.0, "total_load": 40_000.0, "sqrt_load": 200.0,
            "mean_pressure": 200.0, "max_pressure": 800.0,
            "center_row": 20.0, "center_col": 12.0,
            "row_min": 5.0, "row_max": 35.0, "col_min": 3.0, "col_max": 21.0,
        }
        interval_sample = [
            200.0, 200.0, 200.0, 800.0, 20.0, 12.0,
            5.0, 35.0, 3.0, 21.0, 175.0, 200.0 / 175.0, 200.0 / 175.0,
        ]
        model = WeightCalibrator(
            coefficients=[78.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            load_only_coefficients=[78.0, 0.0],
            interval_samples=[(interval_sample, 4)],
            interval_scales=[1.0] * len(interval_sample),
            heights={"new": 175.0},
            train_users=["known"],
            test_users=["new"],
            metadata_count=2,
            training_height_range=(165.0, 190.0),
        )
        result = model.predict_features(features, "new")
        self.assertEqual(result["kg"], 85.0)
        self.assertEqual(result["interval"]["index"], 4)
        self.assertTrue(result["intervalCalibrationApplied"])
        self.assertEqual(result["confidence"], "calibrated_ordinal")

    def test_find_region_json_accepts_course_versioned_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "区域划分"
            folder.mkdir()
            expected = folder / "区域划分2026（30人）.json"
            expected.write_text("[]", encoding="utf-8")
            self.assertEqual(find_region_json(root), expected)

    def test_region_augmentation_is_deterministic_and_moves_labels(self):
        frame = [[10] * COLS for _ in range(ROWS)]
        regions = [(2, 4, 2, 4)] * 6
        augmented = augment_region_sample(frame, regions)
        self.assertEqual(len(augmented), 4)
        self.assertEqual(augmented[1][1][0], (2, 4, 1, 3))
        self.assertEqual(augmented[2][0][2][3], 10)


if __name__ == "__main__":
    unittest.main()
