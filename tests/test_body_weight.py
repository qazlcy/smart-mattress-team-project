import unittest

from body_weight import COLS, ROWS, WeightCalibrator, corrected_frame, weight_interval


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


if __name__ == "__main__":
    unittest.main()
