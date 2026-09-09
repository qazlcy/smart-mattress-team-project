import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from server import COLS, ROWS, airbag_states, demo_frames, metrics, parse_frames

class ReplayContractTest(unittest.TestCase):
    def test_demo_shape_and_metrics_contract(self):
        frame = demo_frames()[0]
        self.assertEqual((len(frame), len(frame[0])), (ROWS, COLS))
        result = metrics(frame)
        self.assertEqual(len(result["airbags"]), 3)
        self.assertIn(result["posture"], {"仰卧", "左侧卧", "右侧卧"})
        self.assertIn(result["postureSource"], {"CNN 睡姿模型", "规则回退（未找到 CNN 权重）"})
        self.assertEqual(len(result["bodyRegions"]), 6)
        self.assertIn("kg", result["weightPrediction"])
        self.assertIn("interval", result["weightPrediction"])
        self.assertGreaterEqual(result["weightPrediction"]["interval"]["index"], 0)

    def test_parser_uses_44_row_frames(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join([",".join(["1"] * COLS) for _ in range(ROWS)]))
            path = Path(f.name)
        try:
            frames = parse_frames(path)
        finally:
            path.unlink()
        self.assertEqual((len(frames), len(frames[0]), len(frames[0][0])), (1, ROWS, COLS))

    def test_airbags_follow_the_documented_channel_groups(self):
        states = airbag_states([[100] * COLS for _ in range(ROWS)])
        self.assertEqual([state["id"] for state in states], ["A_RED", "A_GREEN", "A_YELLOW"])
        self.assertEqual(states[0]["sensorChannels"], list(range(20)))

if __name__ == "__main__": unittest.main()
