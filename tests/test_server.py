import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from server import COLS, ROWS, demo_frames, metrics

class ReplayContractTest(unittest.TestCase):
    def test_demo_shape_and_metrics_contract(self):
        frame = demo_frames()[0]
        self.assertEqual((len(frame), len(frame[0])), (ROWS, COLS))
        result = metrics(frame)
        self.assertEqual(len(result["airbags"]), 4)
        self.assertIn(result["posture"], {"仰卧", "左侧卧", "右侧卧"})

if __name__ == "__main__": unittest.main()
