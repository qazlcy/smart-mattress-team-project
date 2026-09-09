import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from sleep_posture import config, data, labels  # noqa: E402


class LabelTest(unittest.TestCase):
    def test_posture_to_class(self):
        self.assertEqual(labels.posture_to_class(1), 0)   # 仰卧
        self.assertEqual(labels.posture_to_class(6), 0)
        self.assertEqual(labels.posture_to_class(7), 1)   # 俯卧
        self.assertEqual(labels.posture_to_class(9), 1)
        self.assertEqual(labels.posture_to_class(10), 2)  # 左侧卧
        self.assertEqual(labels.posture_to_class(15), 2)
        self.assertEqual(labels.posture_to_class(16), 3)  # 右侧卧
        self.assertEqual(labels.posture_to_class(21), 3)
        with self.assertRaises(ValueError):
            labels.posture_to_class(0)


class ParseTest(unittest.TestCase):
    def test_parse_frames_shape_and_normalize(self):
        # 44 行 x 24 列 -> 一帧；外加 1 行残帧被丢弃。
        lines = [",".join(["10"] * config.COLS) for _ in range(config.ROWS + 1)]
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(lines))
            path = Path(f.name)
        try:
            frames = data.parse_frames(path)
        finally:
            path.unlink()
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].shape, (config.ROWS, config.COLS))
        self.assertAlmostEqual(float(frames[0][0, 0]), 10 / 255.0, places=5)

    def test_parse_frames_skips_bad_rows(self):
        # 列数不等于 24 的行应被跳过。
        lines = [",".join(["1"] * 10), ", ".join(["2"] * config.COLS)]
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(lines))
            path = Path(f.name)
        try:
            self.assertEqual(len(data.parse_frames(path)), 0)  # 不足一帧
        finally:
            path.unlink()


class SplitTest(unittest.TestCase):
    def test_split_is_user_isolated_and_deterministic(self):
        users = [f"u{i}" for i in range(20)]
        tr1, te1 = data.split_users(users, seed=config.SEED)
        tr2, te2 = data.split_users(users, seed=config.SEED)
        self.assertEqual(tr1, tr2)
        self.assertEqual(te1, te2)
        self.assertEqual(set(tr1) & set(te1), set())
        self.assertEqual(len(tr1) + len(te1), len(users))
        self.assertEqual(len(tr1), int(round(len(users) * config.TRAIN_RATIO)))


class AugmentTest(unittest.TestCase):
    def test_augment_preserves_shape_and_range(self):
        X = np.random.rand(8, config.ROWS, config.COLS).astype(np.float32)
        out = data.augment(X, seed=0)
        self.assertEqual(out.shape, X.shape)
        self.assertTrue((out >= 0).all() and (out <= 1).all())


if __name__ == "__main__":
    unittest.main()
